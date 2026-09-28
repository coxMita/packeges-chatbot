#!/usr/bin/env python3
"""Static feature extraction for PyPI packages.

This module is the core of the project. It turns an unpacked package directory
into a fixed-width vector of ~70 named numeric features, derived entirely from
Python source parsed as an AST plus lightweight text statistics.

Two properties matter and are deliberate:

* **Nothing is executed.** `setup.py` is parsed, never run. That is the whole
  point -- install-time code execution is the dominant PyPI attack, so running
  the file we are trying to judge would be self-defeating.

* **Every feature has a name and a human-readable description** (see
  FEATURE_DESCRIPTIONS). The model's SHAP attributions are looked up against
  those descriptions and handed to the LLM, which is how the chatbot explains a
  verdict without ever being asked to classify anything itself.

Feature groups:
    install_*     code that runs at `pip install` time -- the highest-signal group
    call_*        dangerous callables (exec, subprocess, socket, pickle, ...)
    decode_*      base64/hex/zlib unpacking, and decode-then-exec chains
    obf_*         obfuscation signals: entropy, long strings, odd identifiers
    exfil_*       references to credentials, keys, wallets, browser stores
    net_*         hardcoded IPs, URLs, webhook and paste hosts
    pkg_*         package shape: file counts, README/LICENSE, binaries, typosquat
"""

from __future__ import annotations

import ast
import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# --- what counts as dangerous ----------------------------------------------

# Callables that hand control to attacker-supplied data.
EXEC_CALLS = {"eval", "exec", "compile", "__import__"}
IMPORT_CALLS = {"__import__", "import_module"}
# Home-directory lookups and filesystem walks: reconnaissance when they sit next
# to decoding and hidden imports, routine when they don't -- hence the
# worst-file features below rather than package-wide counts.
HOME_CALLS = {"expanduser", "home", "getuser"}
FS_WALK_CALLS = {"listdir", "walk", "scandir", "glob", "iglob", "rglob", "iterdir"}
FILE_DELETE_CALLS = {"remove", "unlink", "rmtree", "rmdir"}

# Module-qualified calls, matched on the dotted tail of the call expression.
PROCESS_CALLS = {
    "subprocess.run", "subprocess.call", "subprocess.Popen", "subprocess.check_output",
    "subprocess.check_call", "subprocess.getoutput", "os.system", "os.popen",
    "os.execv", "os.execve", "os.spawnv", "pty.spawn",
}
NETWORK_CALLS = {
    "socket.socket", "socket.create_connection", "requests.get", "requests.post",
    "requests.put", "urllib.request.urlopen", "urllib.urlopen", "httpx.get",
    "httpx.post", "http.client.HTTPConnection", "http.client.HTTPSConnection",
    "ftplib.FTP", "smtplib.SMTP", "telnetlib.Telnet",
}
DESERIALIZE_CALLS = {
    "pickle.loads", "pickle.load", "marshal.loads", "marshal.load",
    "dill.loads", "shelve.open", "yaml.load",
}
DECODE_CALLS = {
    "base64.b64decode", "base64.b64encode", "base64.b85decode", "base64.b32decode",
    "base64.a85decode", "base64.urlsafe_b64decode", "codecs.decode", "zlib.decompress",
    "gzip.decompress", "bz2.decompress", "lzma.decompress", "binascii.unhexlify",
    "binascii.a2b_base64",
}

# Modules whose mere import is worth noting.
SUSPICIOUS_IMPORTS = {
    "subprocess", "socket", "ctypes", "marshal", "pickle", "telnetlib",
    "cryptography", "Crypto", "win32api", "win32crypt", "pty",
}

# Paths that only malware has a reason to read.
EXFIL_PATTERNS = {
    "exfil_ssh": [r"\.ssh[/\\]", r"id_rsa", r"id_ed25519", r"authorized_keys", r"known_hosts"],
    "exfil_cloud": [r"\.aws[/\\]", r"credentials", r"\.kube[/\\]", r"\.docker[/\\]config",
                    r"\.netrc", r"gcloud"],
    "exfil_env": [r"\.env\b", r"\.pypirc", r"\.npmrc", r"\.git-credentials"],
    "exfil_browser": [r"Login Data", r"Cookies", r"cookies\.sqlite", r"key4\.db",
                      r"logins\.json", r"Local State", r"AppData.*Roaming",
                      r"Chrome[/\\]User Data", r"Opera Software", r"Mozilla[/\\]Firefox"],
    "exfil_wallet": [r"wallet\.dat", r"Exodus", r"Electrum", r"MetaMask", r"Atomic[/\\]",
                     r"Ethereum[/\\]keystore", r"\.bitcoin"],
    "exfil_token": [r"discord.*token", r"Telegram Desktop", r"tdata",
                    r"[Ss]team[/\\]config", r"Minecraft"],
}

# Hosts that show up in exfiltration payloads far more than in real packages.
SUSPICIOUS_HOSTS = [
    r"discord(?:app)?\.com/api/webhooks", r"api\.telegram\.org", r"pastebin\.com",
    r"paste\.ee", r"hastebin", r"transfer\.sh", r"ngrok\.io", r"ngrok-free\.app",
    r"requestbin", r"webhook\.site", r"anonfiles", r"gofile\.io", r"file\.io",
    r"\.onion\b", r"burpcollaborator", r"interact\.sh", r"oast\.fun", r"dnslog",
]
SUSPICIOUS_TLDS = [r"\.tk\b", r"\.ml\b", r"\.ga\b", r"\.cf\b", r"\.gq\b",
                   r"\.xyz\b", r"\.top\b", r"\.ru\b", r"\.su\b"]

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")
HEXISH_RE = re.compile(r"^[0-9a-fA-F]{40,}$")
B64ISH_RE = re.compile(r"^[A-Za-z0-9+/=_-]{60,}$")

SOURCE_SUFFIXES = {".py"}
BINARY_SUFFIXES = {".so", ".dll", ".exe", ".dylib", ".pyd", ".bin", ".node", ".msi", ".scr"}
DOC_NAMES = {"readme", "readme.md", "readme.rst", "readme.txt"}
LICENSE_NAMES = {"license", "license.txt", "license.md", "licence", "copying"}

# Parsing caps -- a handful of packages contain enormous generated files, and a
# single pathological file should not stall a 15k-package extraction run.
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_PY_FILES = 400

# Total source text held in memory per package. Without this, a package of 400
# large files would build an 800MB string in text_signals() and then run ~40
# regexes over it -- enough to OOM a worker and take the whole extraction run
# with it. Files are visited setup-first, so the budget is spent where the
# signal is.
MAX_TOTAL_TEXT_BYTES = 8 * 1024 * 1024


def shannon_entropy(s: str) -> float:
    """Shannon entropy in bits/char. High values indicate encoded or packed data."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


@dataclass
class _Acc:
    """Mutable accumulator threaded through the AST walk of every file."""
    exec_calls: int = 0
    process_calls: int = 0
    network_calls: int = 0
    deserialize_calls: int = 0
    decode_calls: int = 0
    decode_then_exec: int = 0
    dynamic_attr: int = 0
    dunder_access: int = 0
    suspicious_imports: int = 0
    import_inside_function: int = 0
    env_reads: int = 0
    file_writes: int = 0
    chmod_calls: int = 0
    lambda_count: int = 0
    try_except_pass: int = 0
    computed_import: int = 0
    obfuscated_import: int = 0
    char_code_build: int = 0
    home_access: int = 0
    fs_walk: int = 0
    file_delete: int = 0
    string_literals: list[str] = None
    identifiers: list[str] = None

    def __post_init__(self):
        if self.string_literals is None:
            self.string_literals = []
        if self.identifiers is None:
            self.identifiers = []


def dotted_name(node: ast.AST) -> str:
    """Render an attribute/name chain as a dotted string, e.g. `os.path.join`."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


GENERIC_BARE_NAMES = {"open", "load", "loads", "get", "post", "put", "run", "call",
                      "decode", "socket", "decompress"}


def _tail_matches(dotted: str, targets: set[str]) -> bool:
    """True if `dotted` matches a target, allowing aliased/partial prefixes.

    `base64.b64decode`, `b64decode` and `mod.base64.b64decode` all match the
    target `base64.b64decode` -- malware routinely aliases its imports.
    """
    if dotted in targets:
        return True
    for t in targets:
        if dotted.endswith("." + t):
            return True
        # A bare call (`from base64 import b64decode`) matches only when the
        # name is distinctive: bare `open`, `get` or `loads` is almost always
        # the builtin, requests or json, not shelve, requests or pickle.
        if dotted == t.split(".")[-1] and dotted not in GENERIC_BARE_NAMES:
            return True
    return False


def _is_int_seq(node: ast.AST) -> bool:
    return (isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 2
            and all(isinstance(e, ast.Constant) and type(e.value) is int for e in node.elts))


def _is_char_code_build(node: ast.Call) -> bool:
    """A string spelled as character codes: `chr(111)`, `map(chr, [...])`,
    `bytes([111, 115]).decode()`. Real code writes the string."""
    short = dotted_name(node.func).split(".")[-1]
    if short == "chr" and node.args and isinstance(node.args[0], ast.Constant) \
            and type(node.args[0].value) is int:
        return True
    if short == "map" and node.args and dotted_name(node.args[0]) == "chr":
        return True
    if short in {"bytes", "bytearray"} and node.args and _is_int_seq(node.args[0]):
        return True
    return False


def _first_str_arg(node: ast.Call) -> str:
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    return ""


# Kinds of suspicious operation. Legitimate libraries spread these across
# modules; malware bolted onto a real library packs them into one function.
FILE_CATEGORIES = {
    "exec": ("exec_calls",),
    "process": ("process_calls",),
    "network": ("network_calls",),
    "deserialize": ("deserialize_calls",),
    "decode": ("decode_calls",),
    "dynamic_import": ("computed_import",),
    "hidden_name": ("char_code_build", "obfuscated_import"),
    "home": ("home_access",),
    "fs_walk": ("fs_walk",),
    "delete": ("file_delete",),
    "env": ("env_reads",),
    "chmod": ("chmod_calls",),
    "silent_except": ("try_except_pass",),
}
# How strongly each kind points at malware when found in a file (as in
# ml/file_scan.py); summed over the kinds present, maxed over files.
DANGER_WEIGHTS = {"hidden_name": 4, "exec": 3, "process": 3, "network": 2, "deserialize": 2,
                  "decode": 2, "chmod": 2, "dynamic_import": 1, "home": 1, "fs_walk": 1,
                  "delete": 1, "env": 1, "silent_except": 1}
# What counts toward a file's suspicious-call density. Walks, deletes and
# swallowed exceptions are too common alone to count; they only add categories.
DENSITY_FIELDS = ("exec_calls", "process_calls", "network_calls", "deserialize_calls",
                  "decode_calls", "computed_import", "char_code_build", "home_access")
# Floor on a file's size for the density, so a 3-line shim with one
# `__import__` does not read as 333 calls per kLOC.
MIN_DENSITY_LOC = 50
_COUNTER_FIELDS = sorted({f for fs in FILE_CATEGORIES.values() for f in fs} | set(DENSITY_FIELDS))


FIELD_CATEGORY = {f: cat for cat, fields in FILE_CATEGORIES.items() for f in fields}


def _counters(acc: "_Acc") -> dict[str, int]:
    return {f: getattr(acc, f) for f in _COUNTER_FIELDS}


class _Visitor(ast.NodeVisitor):
    def __init__(self, acc: _Acc, hits: list | None = None):
        self.acc = acc
        self._func_depth = 0
        # When given, every suspicious operation is also recorded as
        # (category, line, call name) -- the per-file scan's receipts.
        self.hits = hits

    # -- calls ---------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        before = _counters(self.acc) if self.hits is not None else None
        self._count_call(node, name)
        if before is not None:
            for field, value in _counters(self.acc).items():
                if value > before[field]:
                    self.hits.append((FIELD_CATEGORY[field], node.lineno, name))
        self.generic_visit(node)

    def _count_call(self, node: ast.Call, name: str) -> None:
        short = name.split(".")[-1]
        a = self.acc

        if short in EXEC_CALLS:
            a.exec_calls += 1
            # A decode call nested inside exec/eval is the classic packed-payload
            # shape: exec(base64.b64decode("...")). Worth its own feature.
            if any(
                isinstance(sub, ast.Call) and _tail_matches(dotted_name(sub.func), DECODE_CALLS)
                for sub in ast.walk(node)
            ):
                a.decode_then_exec += 1

        if _tail_matches(name, PROCESS_CALLS):
            a.process_calls += 1
        if _tail_matches(name, NETWORK_CALLS):
            a.network_calls += 1
        if _tail_matches(name, DESERIALIZE_CALLS):
            a.deserialize_calls += 1
        if _tail_matches(name, DECODE_CALLS):
            a.decode_calls += 1
        # `__import__(name)` with anything but a literal hides which module is
        # loaded -- `__import__(''.join(map(chr, [111, 115])))` is `import os`.
        if short in IMPORT_CALLS and node.args and not (
                isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            a.computed_import += 1
            # Plugin loaders compute module names from config; nothing
            # legitimate spells one out in character codes or decodes it.
            if any(isinstance(sub, ast.Call) and (
                    _is_char_code_build(sub) or _tail_matches(dotted_name(sub.func), DECODE_CALLS)
                    or dotted_name(sub.func).endswith("decode"))
                   for sub in ast.walk(node.args[0])):
                a.obfuscated_import += 1
        if _is_char_code_build(node):
            a.char_code_build += 1
        if short in HOME_CALLS and (short != "expanduser" or _first_str_arg(node).startswith("~")):
            a.home_access += 1
        if short in FS_WALK_CALLS:
            a.fs_walk += 1
        # `remove` is also list.remove, so only count it on os/shutil; the
        # others are file operations whatever the receiver.
        if short in FILE_DELETE_CALLS and (short != "remove" or name in {"os.remove", "shutil.remove"}):
            a.file_delete += 1
        if short in {"getattr", "setattr"}:
            a.dynamic_attr += 1
        if name.endswith("os.chmod") or short == "chmod":
            a.chmod_calls += 1
        if short == "open":
            for arg in node.args[1:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if any(m in arg.value for m in ("w", "a", "x")):
                        a.file_writes += 1
        if name.endswith("environ.get") or name.endswith("os.getenv") or short == "getenv":
            a.env_reads += 1

    # -- imports -------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in SUSPICIOUS_IMPORTS:
                self.acc.suspicious_imports += 1
        if self._func_depth:
            self.acc.import_inside_function += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] in SUSPICIOUS_IMPORTS:
            self.acc.suspicious_imports += 1
        if self._func_depth:
            self.acc.import_inside_function += 1
        self.generic_visit(node)

    # -- shape ---------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.acc.lambda_count += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        # `except: pass` around a payload is how malware stays quiet when it
        # fails on an unexpected host.
        for handler in node.handlers:
            if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                self.acc.try_except_pass += 1
                if self.hits is not None:
                    self.hits.append(("silent_except", handler.lineno, "except: pass"))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.acc.string_literals.append(node.value)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.acc.identifiers.append(node.id)
        if node.id.startswith("__") and node.id.endswith("__"):
            self.acc.dunder_access += 1
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.acc.dunder_access += 1
        self.generic_visit(node)


# --- install-time analysis --------------------------------------------------

INSTALL_HOOK_CLASSES = {"install", "develop", "egg_info", "build_py", "sdist", "bdist_wheel"}
SETUP_FILES = {"setup.py", "setup.cfg", "pyproject.toml", "conftest.py", "__init__.py"}


def analyse_setup(tree: ast.Module) -> dict[str, int]:
    """Look for code that runs at `pip install` time.

    This is the highest-signal feature group in the whole model. A legitimate
    setup.py declares metadata and stops. A malicious one either runs its payload
    at module level (so merely importing setup.py fires it) or registers a
    cmdclass override so the payload fires during the install command.
    """
    out = {
        "install_has_cmdclass": 0,
        "install_hook_subclass": 0,
        "install_toplevel_stmts": 0,
        "install_toplevel_call": 0,
        "install_exec_at_toplevel": 0,
        "install_net_at_toplevel": 0,
        "install_proc_at_toplevel": 0,
    }

    for node in tree.body:
        # setup(..., cmdclass={"install": Evil}) -- the registered override.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if dotted_name(call.func).split(".")[-1] == "setup":
                for kw in call.keywords:
                    if kw.arg == "cmdclass":
                        out["install_has_cmdclass"] = 1

        # class Evil(install): def run(self): ...
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if dotted_name(base).split(".")[-1] in INSTALL_HOOK_CLASSES:
                    out["install_hook_subclass"] = 1

        # Statements at module level that are neither imports nor declarations
        # execute the moment setup.py is read.
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                                 ast.AnnAssign, ast.Expr, ast.If)):
            out["install_toplevel_stmts"] += 1

        # Walk every top-level statement for calls that should never run here.
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            name = dotted_name(sub.func)
            short = name.split(".")[-1]
            if short == "setup":
                continue
            if isinstance(node, ast.Expr):
                out["install_toplevel_call"] += 1
            if short in EXEC_CALLS:
                out["install_exec_at_toplevel"] += 1
            if _tail_matches(name, NETWORK_CALLS):
                out["install_net_at_toplevel"] += 1
            if _tail_matches(name, PROCESS_CALLS):
                out["install_proc_at_toplevel"] += 1

    return out


# --- text-level signals -----------------------------------------------------


def text_signals(blobs: list[str]) -> dict[str, float]:
    """Regex/statistical signals computed over raw file text."""
    joined = "\n".join(blobs)
    out: dict[str, float] = {}

    for key, patterns in EXFIL_PATTERNS.items():
        out[key] = float(sum(
            len(re.findall(p, joined, re.IGNORECASE)) for p in patterns
        ))
    out["exfil_total"] = float(sum(out[k] for k in EXFIL_PATTERNS))

    out["net_suspicious_host"] = float(sum(
        len(re.findall(p, joined, re.IGNORECASE)) for p in SUSPICIOUS_HOSTS
    ))
    out["net_suspicious_tld"] = float(sum(
        len(re.findall(p, joined, re.IGNORECASE)) for p in SUSPICIOUS_TLDS
    ))
    ips = IP_RE.findall(joined)
    # Loopback and RFC1918 addresses are ordinary in test fixtures and configs;
    # only routable literals are interesting.
    out["net_hardcoded_ip"] = float(sum(
        1 for ip in ips
        if not ip.startswith(("127.", "0.", "10.", "192.168.", "255."))
    ))
    out["net_url_count"] = float(len(URL_RE.findall(joined)))

    lines = joined.split("\n")
    out["obf_max_line_len"] = float(max((len(ln) for ln in lines), default=0))
    out["obf_long_line_count"] = float(sum(1 for ln in lines if len(ln) > 500))
    non_ascii = sum(1 for c in joined if ord(c) > 127)
    out["obf_non_ascii_ratio"] = non_ascii / max(len(joined), 1)

    return out


def string_signals(strings: list[str]) -> dict[str, float]:
    """Entropy and shape statistics over string literals found in the AST."""
    if not strings:
        return {
            "obf_str_entropy_mean": 0.0, "obf_str_entropy_max": 0.0,
            "obf_str_len_max": 0.0, "obf_str_len_mean": 0.0,
            "obf_hexish_count": 0.0, "obf_b64ish_count": 0.0,
            "obf_high_entropy_count": 0.0,
        }

    # Short strings are noisy; entropy only means something above ~20 chars.
    considered = [s for s in strings if len(s) >= 20] or strings
    entropies = [shannon_entropy(s) for s in considered]
    lengths = [len(s) for s in strings]

    return {
        "obf_str_entropy_mean": sum(entropies) / len(entropies),
        "obf_str_entropy_max": max(entropies),
        "obf_str_len_max": float(max(lengths)),
        "obf_str_len_mean": sum(lengths) / len(lengths),
        "obf_hexish_count": float(sum(1 for s in strings if HEXISH_RE.match(s))),
        "obf_b64ish_count": float(sum(1 for s in strings if B64ISH_RE.match(s))),
        # 4.5 bits/char is well above English or code, and typical of encoded blobs.
        "obf_high_entropy_count": float(sum(
            1 for s, e in zip(considered, entropies) if e > 4.5 and len(s) > 100
        )),
    }


# --- typosquatting ----------------------------------------------------------


def _edit_distance_within(a: str, b: str, limit: int) -> int:
    """Levenshtein distance, giving up early once it exceeds `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def typosquat_distance(name: str, popular: set[str]) -> int:
    """Edit distance from `name` to the nearest popular package name.

    0 means the name *is* popular. 1-2 is the typosquat danger zone
    (`requsts`, `python-dateutil` vs `python-dateutils`). We cap the search at 3
    because anything further apart is not a confusion attack.
    """
    n = name.lower()
    if n in popular:
        return 0
    best = 4
    for p in popular:
        d = _edit_distance_within(n, p, min(best - 1, 3))
        if d < best:
            best = d
            if best == 1:
                break
    return best


# --- package-level extraction ----------------------------------------------


def is_non_source_file(path: Path) -> bool:
    """Files that say nothing about the author's code, excluded from shape counts.

    Two kinds, both of which leaked the label into pkg_n_files:
      * pipeline artifacts -- DataDog wraps every malicious sample with a
        package_info-*.json, and acquire_benign.py writes .meta.json. Neither
        exists in a package fetched live from PyPI, so counting them is both a
        label leak and train/serve skew.
      * build-tool metadata -- PKG-INFO and the *.egg-info/ directory are
        generated by setuptools. Their presence tracks which build backend
        (and which era) produced the sdist, not what the code does.
    """
    name = path.name
    if name == ".meta.json" or (name.startswith("package_info-") and name.endswith(".json")):
        return True
    if name == "PKG-INFO":
        return True
    return any(part.endswith(".egg-info") for part in path.parts[:-1])


def iter_python_files(root: Path) -> list[Path]:
    """Python files in a package, setup-relevant ones first, capped for runtime."""
    files = [p for p in root.rglob("*.py") if p.is_file()]
    files.sort(key=lambda p: (p.name not in SETUP_FILES, len(p.parts), str(p)))
    return files[:MAX_PY_FILES]


def read_text(path: Path) -> str | None:
    """Read a source file as text, or None if it is too large or not decodable."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


RISKY_TOKENS = re.compile(r"subprocess|os\.system|popen|exec\(|eval\(|b64decode|urlopen|"
                         r"urlretrieve|requests\.(get|post)|socket\.|marshal|__import__|chmod|"
                         r"startfile|decompress|fromhex|getenv|environ")
MAX_EXTRA_PY_FILES = 20_000
MAX_EXTRA_BYTES = 256 * 1024 * 1024


def _add_behaviour(beh: dict, file_cats: list[str], tree: ast.Module, path: Path, root: Path) -> None:
    import behaviour
    beh["worst_file_danger"] = max(beh["worst_file_danger"],
                                   sum(DANGER_WEIGHTS.get(c, 1) for c in file_cats))
    beh["n_danger_files"] += len(file_cats) >= 3
    try:
        fb = behaviour.analyse_file(tree, str(path.relative_to(root)), path.name == "setup.py")
    except RecursionError:
        return
    for k, v in fb.items():
        if k == "_auto_categories":
            beh["autorun_categories"] = max(beh["autorun_categories"], v)
        else:
            beh[k] += v


def _scan_beyond_cap(beh: dict, root: Path, done: set[Path]) -> None:
    budget, n = MAX_EXTRA_BYTES, 0
    for path in root.rglob("*.py"):
        if path in done or not path.is_file():
            continue
        n += 1
        if n > MAX_EXTRA_PY_FILES or budget <= 0:
            break
        text = read_text(path)
        if not text or not RISKY_TOKENS.search(text):
            continue
        budget -= len(text)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(text)
            acc = _Acc()
            _Visitor(acc).visit(tree)
        except (SyntaxError, ValueError, RecursionError):
            continue
        counts = _counters(acc)
        cats = [c for c, fields in FILE_CATEGORIES.items() if any(counts[f] for f in fields)]
        _add_behaviour(beh, cats, tree, path, root)


def extract_features(
    root: Path,
    package_name: str = "",
    popular_names: set[str] | None = None,
) -> dict[str, float]:
    """Turn an unpacked package directory into a named feature vector.

    `root` is a directory containing the unpacked sdist. No file in it is ever
    executed -- everything below is `ast.parse` and regex over text.
    """
    acc = _Acc()
    setup_feats = {k: 0 for k in (
        "install_has_cmdclass", "install_hook_subclass", "install_toplevel_stmts",
        "install_toplevel_call", "install_exec_at_toplevel",
        "install_net_at_toplevel", "install_proc_at_toplevel",
    )}

    py_files = iter_python_files(root)
    blobs: list[str] = []
    n_parsed = n_syntax_errors = 0
    total_loc = 0
    has_setup_py = 0

    text_budget = MAX_TOTAL_TEXT_BYTES
    worst_categories = 0
    worst_density = 0.0

    # Behaviour features (ml/behaviour.py): per file, then summed or pooled so
    # a payload in one file of a large package is not averaged away.
    import behaviour
    beh: dict[str, float] = dict.fromkeys(behaviour.BEHAVIOUR_FEATURES, 0.0)

    for path in py_files:
        text = read_text(path)
        if text is None:
            continue
        total_loc += text.count("\n") + 1

        # Keep counting lines for every file, but only retain text for the
        # regex pass while the budget lasts.
        if text_budget > 0:
            blobs.append(text)
            text_budget -= len(text)

        try:
            # Third-party source is full of invalid escape sequences and other
            # SyntaxWarnings. We are judging behaviour, not style -- and a 15k
            # package run would otherwise bury real output in warning spam.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            # Python 2 leftovers and deliberately mangled files both land here.
            n_syntax_errors += 1
            continue
        before = _counters(acc)
        try:
            _Visitor(acc).visit(tree)
        except RecursionError:
            # Deeply nested code (generated parsers, deliberate nesting bombs)
            # can outrun the recursive walk. Count it like a parse failure
            # rather than losing the whole package.
            n_syntax_errors += 1
            continue
        n_parsed += 1
        delta = {k: v - before[k] for k, v in _counters(acc).items()}
        file_cats = [cat for cat, fields in FILE_CATEGORIES.items() if any(delta[f] for f in fields)]
        worst_categories = max(worst_categories, len(file_cats))
        _add_behaviour(beh, file_cats, tree, path, root)
        file_loc = max(text.count("\n") + 1, MIN_DENSITY_LOC)
        worst_density = max(worst_density,
                            sum(delta[f] for f in DENSITY_FIELDS) * 1000 / file_loc)

        if path.name == "setup.py":
            has_setup_py = 1
            for k, v in analyse_setup(tree).items():
                setup_feats[k] = max(setup_feats[k], v) if k.endswith(
                    ("cmdclass", "subclass")) else setup_feats[k] + v

    # Behaviour only, for Python files beyond the MAX_PY_FILES cap. A payload
    # injected into one module of a 5,000-file library (telnyx 4.87.1) sits
    # past any depth-first cap; a cheap token pre-filter keeps this fast.
    _scan_beyond_cap(beh, root, set(py_files))

    all_files = [p for p in root.rglob("*")
                 if p.is_file() and not is_non_source_file(p)]
    names_lower = {p.name.lower() for p in all_files}

    feats: dict[str, float] = {
        # -- install-time (highest signal) --
        **{k: float(v) for k, v in setup_feats.items()},
        "install_has_setup_py": float(has_setup_py),

        # -- dangerous calls --
        "call_exec": float(acc.exec_calls),
        "call_process": float(acc.process_calls),
        "call_network": float(acc.network_calls),
        "call_deserialize": float(acc.deserialize_calls),
        "call_dynamic_attr": float(acc.dynamic_attr),
        "call_dunder_access": float(acc.dunder_access),
        "call_chmod": float(acc.chmod_calls),
        "call_file_write": float(acc.file_writes),
        "call_env_read": float(acc.env_reads),
        "call_suspicious_import": float(acc.suspicious_imports),
        "call_import_in_function": float(acc.import_inside_function),
        "call_try_except_pass": float(acc.try_except_pass),
        "call_lambda": float(acc.lambda_count),
        "call_computed_import": float(acc.computed_import),
        "call_obfuscated_import": float(acc.obfuscated_import),
        "obf_char_code_build": float(acc.char_code_build),

        # -- worst single file: one bad file cannot be averaged away --
        "worst_file_categories": float(worst_categories),
        "worst_file_suspicious_per_kloc": worst_density,

        # -- decode / unpack --
        "decode_calls": float(acc.decode_calls),
        "decode_then_exec": float(acc.decode_then_exec),

        # -- package shape --
        "pkg_n_files": float(len(all_files)),
        "pkg_n_py_files": float(len(py_files)),
        "pkg_loc": float(total_loc),
        "pkg_loc_per_file": total_loc / max(n_parsed, 1),
        "pkg_parse_failures": float(n_syntax_errors),
        "pkg_has_readme": float(any(n in DOC_NAMES for n in names_lower)),
        "pkg_has_license": float(any(n in LICENSE_NAMES for n in names_lower)),
        "pkg_has_tests": float(any("test" in p.name.lower() for p in all_files)),
        "pkg_n_binaries": float(sum(
            1 for p in all_files if p.suffix.lower() in BINARY_SUFFIXES
        )),
        "pkg_py_file_ratio": len(py_files) / max(len(all_files), 1),
    }

    feats.update(text_signals(blobs))
    feats.update(string_signals(acc.string_literals))
    beh.update(behaviour.scan_other_files(all_files, root))
    feats.update({k: float(v) for k, v in beh.items()})

    # Normalised variants: raw counts scale with package size, so a 50k-line
    # project with 3 subprocess calls should not outrank a 40-line dropper with
    # the same 3. The model sees both the raw count and the density.
    kloc = max(total_loc, 1) / 1000.0
    for key in ("call_exec", "call_process", "call_network", "call_deserialize",
                "decode_calls", "exfil_total"):
        feats[f"{key}_per_kloc"] = feats[key] / kloc

    # -- identifier shape --
    idents = acc.identifiers
    if idents:
        lengths = [len(i) for i in idents]
        feats["obf_ident_len_mean"] = sum(lengths) / len(lengths)
        feats["obf_ident_single_char_ratio"] = sum(1 for n in lengths if n == 1) / len(lengths)
        feats["obf_ident_unique_ratio"] = len(set(idents)) / len(idents)
    else:
        feats["obf_ident_len_mean"] = 0.0
        feats["obf_ident_single_char_ratio"] = 0.0
        feats["obf_ident_unique_ratio"] = 0.0

    # -- typosquatting --
    if package_name and popular_names:
        feats["pkg_typosquat_distance"] = float(typosquat_distance(package_name, popular_names))
    else:
        feats["pkg_typosquat_distance"] = 4.0

    return feats


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "install_has_cmdclass": "setup.py registers a custom cmdclass, overriding what pip runs during install",
    "install_hook_subclass": "a class subclasses setuptools' install/develop/egg_info command to hook installation",
    "install_toplevel_stmts": "statements at setup.py module level that execute as soon as the file is read",
    "install_toplevel_call": "function calls at setup.py module level, outside any function",
    "install_exec_at_toplevel": "eval/exec/compile called at setup.py module level",
    "install_net_at_toplevel": "network calls at setup.py module level -- code that phones out during install",
    "install_proc_at_toplevel": "subprocess or shell execution at setup.py module level",
    "install_has_setup_py": "the package ships a setup.py at all",
    "call_exec": "calls to eval, exec, compile or __import__",
    "call_process": "subprocess/os.system/shell execution",
    "call_network": "outbound network calls (sockets, requests, urllib)",
    "call_deserialize": "unsafe deserialisation (pickle, marshal, yaml.load)",
    "call_dynamic_attr": "getattr/setattr used to reach attributes by computed name",
    "call_dunder_access": "access to dunder attributes, often used to escape sandboxes",
    "call_chmod": "chmod calls, typically to mark a dropped file executable",
    "call_file_write": "files opened for writing",
    "call_env_read": "environment variables read, a common way to harvest secrets",
    "call_suspicious_import": "imports of subprocess, socket, ctypes, marshal and similar",
    "call_import_in_function": "imports hidden inside function bodies rather than at module level",
    "call_try_except_pass": "'except: pass' blocks that silently swallow failures",
    "call_lambda": "lambda expressions, often used to inline obfuscated logic",
    "call_computed_import": "__import__/import_module called with a computed name, hiding which module is loaded",
    "call_obfuscated_import": "__import__/import_module of a name built from character codes or decoded data -- the module is deliberately disguised",
    "obf_char_code_build": "strings spelled out as character codes (chr(111), map(chr, [...])) instead of written",
    "worst_file_categories": "most distinct kinds of suspicious operation (exec, decoding, hidden imports, home-directory access, file walking, deleting, ...) found together in one file",
    "worst_file_suspicious_per_kloc": "suspicious calls per 1000 lines in the single worst file",
    "decode_calls": "base64/hex/zlib decoding calls",
    "decode_then_exec": "a decode call nested directly inside exec/eval -- a packed payload",
    "pkg_n_files": "total files in the package",
    "pkg_n_py_files": "Python source files",
    "pkg_loc": "total lines of Python",
    "pkg_loc_per_file": "average lines per Python file",
    "pkg_parse_failures": "files that failed to parse as Python",
    "pkg_has_readme": "the package ships a README",
    "pkg_has_license": "the package ships a LICENSE",
    "pkg_has_tests": "the package ships anything test-shaped",
    "pkg_n_binaries": "bundled binaries (.so/.dll/.exe/.pyd)",
    "pkg_py_file_ratio": "share of files that are Python source",
    "pkg_typosquat_distance": "edit distance to the nearest popular package name (1-2 means likely typosquat)",
    "exfil_ssh": "references to SSH keys and known_hosts",
    "exfil_cloud": "references to cloud credential files (AWS, kube, docker, netrc)",
    "exfil_env": "references to .env, .pypirc, .npmrc and git credentials",
    "exfil_browser": "references to browser cookie and saved-password stores",
    "exfil_wallet": "references to cryptocurrency wallet files",
    "exfil_token": "references to Discord/Telegram/Steam session data",
    "exfil_total": "all credential- and secret-harvesting references combined",
    "net_suspicious_host": "Discord webhooks, Telegram bot API, paste sites, tunnels and OAST hosts",
    "net_suspicious_tld": "URLs on TLDs heavily favoured by throwaway infrastructure",
    "net_hardcoded_ip": "hardcoded routable IP addresses",
    "net_url_count": "URLs in the source",
    "obf_max_line_len": "longest line -- very long lines usually mean packed payloads",
    "obf_long_line_count": "lines over 500 characters",
    "obf_non_ascii_ratio": "share of non-ASCII characters",
    "obf_str_entropy_mean": "mean Shannon entropy of string literals",
    "obf_str_entropy_max": "highest string-literal entropy -- encoded blobs sit near 6 bits/char",
    "obf_str_len_max": "longest string literal",
    "obf_str_len_mean": "mean string-literal length",
    "obf_hexish_count": "long pure-hex strings",
    "obf_b64ish_count": "long base64-shaped strings",
    "obf_high_entropy_count": "long, high-entropy strings -- the signature of an embedded payload",
    "obf_ident_len_mean": "mean identifier length",
    "obf_ident_single_char_ratio": "share of single-character identifiers, a minifier/obfuscator tell",
    "obf_ident_unique_ratio": "identifier reuse rate",
}


def feature_names() -> list[str]:
    """Stable, sorted feature order. Training and inference both use this."""
    probe = extract_features(Path("/nonexistent"), "", set())
    return sorted(probe)


def describe(name: str) -> str:
    """Human-readable description of a feature, for the LLM explanation prompt."""
    from behaviour import BEHAVIOUR_DESCRIPTIONS
    if name in BEHAVIOUR_DESCRIPTIONS:
        return BEHAVIOUR_DESCRIPTIONS[name]
    base = name.removesuffix("_per_kloc")
    desc = FEATURE_DESCRIPTIONS.get(name) or FEATURE_DESCRIPTIONS.get(base, name)
    if name.endswith("_per_kloc") and base in FEATURE_DESCRIPTIONS:
        return f"{desc} (per 1000 lines of code)"
    return desc
