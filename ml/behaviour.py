"""Behaviour features: *what runs automatically*, and *where data flows*.

The first feature set (features.py) counts risky calls across the whole
package. That generalises poorly to malware families the model has not seen:

  * a payload in one file of a 30-file package is diluted by the benign code
    around it -- recall on unseen families fell from 94% (<= 3 files) to 9%
    (30+ files);
  * it only treats top-level setup.py code as "runs by itself", so payloads
    started from __init__.py, a thread target or an install command's run()
    look like any library function;
  * `exec(b64decode(x))` counts, `code = b64decode(x); exec(code)` does not;
  * payloads in compiled modules, .pth files and shell scripts are invisible.

This module answers those directly, per file, so nothing is averaged away:

  autorun_*   operations in code that executes on `pip install` (setup.py
              module level, functions it calls, install-command classes) or
              on `import` (module level of any library module, functions it
              calls, threads/atexit hooks it starts). Library code that merely
              *defines* a network call is not counted -- only code that runs.
  flow_*      light, flow-insensitive taint tracking within one scope:
              decoded / downloaded / char-code-built values reaching exec,
              a process, or a file that is then made executable.
  proc_*      process-launch tradecraft: hidden windows and detached sessions,
              shell downloaders and reverse shells, inline `python -c`,
              executing from a temp directory.
  bin_*/pth_* non-Python payloads: strings inside compiled binaries, compiled
              modules with no source, .pth lines that run at every start-up,
              downloader shell scripts.

Nothing is executed; everything is `ast` and regex over bytes.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# Imported lazily-free: features.py imports this module at the bottom of its
# own definitions, after these names exist.
from features import (  # noqa: E402
    DECODE_CALLS, EXEC_CALLS, EXFIL_PATTERNS, IP_RE, NETWORK_CALLS, PROCESS_CALLS,
    SUSPICIOUS_HOSTS, SUSPICIOUS_TLDS, URL_RE, _is_char_code_build, _tail_matches, dotted_name,
)

NET_CALLS = NETWORK_CALLS | {
    "urllib.request.urlretrieve", "urlretrieve", "urllib.request.Request",
    "socket.gethostbyname", "socket.gethostbyname_ex", "socket.getaddrinfo",
    "requests.request", "requests.Session", "httpx.Client", "httpx.AsyncClient",
    "aiohttp.ClientSession", "urllib3.PoolManager", "urllib3.request",
    "dns.resolver.resolve", "dns.resolver.query", "paramiko.SSHClient",
}
PROC_CALLS = PROCESS_CALLS | {
    "os.startfile", "os.spawnl", "os.spawnlp", "os.execl", "os.execlp", "os.execvp",
    "os.posix_spawn", "subprocess.getstatusoutput", "asyncio.create_subprocess_shell",
    "asyncio.create_subprocess_exec", "ctypes.windll.shell32.ShellExecuteW",
}
NATIVE_LOAD_CALLS = {"ctypes.CDLL", "ctypes.cdll.LoadLibrary", "ctypes.WinDLL", "ctypes.PyDLL",
                     "ctypes.OleDLL", "ctypes.windll.LoadLibrary", "cdll.LoadLibrary", "CDLL",
                     "WinDLL", "LoadLibrary", "cffi.FFI.dlopen", "dlopen"}
# Crypto secrets: a wallet library has no business sending these anywhere.
SECRET_WORDS = re.compile(r"mnemonic|seed[_ ]?phrase|private[_ ]?key|privkey|keystore|wallet[_ ]?(key|seed)",
                          re.IGNORECASE)
SPAWN_CALLS = {"threading.Thread", "threading.Timer", "multiprocessing.Process",
               "atexit.register", "Thread", "Timer", "Process"}
DECODE_SOURCES = {c for c in DECODE_CALLS if "encode" not in c} | {"bytes.fromhex", "fromhex"}
DESER_SINKS = {"marshal.loads", "pickle.loads", "dill.loads", "types.FunctionType",
               "types.CodeType"}
WRITE_METHODS = {"write", "write_bytes", "write_text"}
INSTALL_HOOK_BASES = {"install", "develop", "egg_info", "build_py", "sdist", "bdist_wheel",
                      "build_ext", "build", "easy_install", "Command"}

# Command strings that fetch-and-run, open a reverse shell or decode inline.
SHELL_DOWNLOADER = re.compile(
    r"\b(curl|wget)\b[^\n]{0,200}(\||&&|;|-o\b|-O\b)|Invoke-WebRequest|\biwr\b|"
    r"DownloadString|DownloadFile|certutil\b[^\n]{0,40}-urlcache|bitsadmin|mshta\b|"
    r"/dev/tcp/|\bnc\b[^\n]{0,20}-e\b|base64\s+(-d|--decode)|powershell[^\n]{0,40}-e(nc)?\b",
    re.IGNORECASE)
TEMP_PATH = re.compile(r"/tmp/|/var/tmp/|%TEMP%|%APPDATA%|AppData[/\\\\]|\\\\Temp\\\\", re.IGNORECASE)
HIDING_FLAGS = {"CREATE_NO_WINDOW", "DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP",
                "STARTF_USESHOWWINDOW", "SW_HIDE"}

SCRIPT_SUFFIXES = {".sh", ".bash", ".bat", ".cmd", ".ps1", ".vbs", ".js"}
BIN_SUFFIXES = {".so", ".pyd", ".dll", ".exe", ".dylib", ".bin", ".node", ".msi", ".scr"}
SOURCE_STEM_SUFFIXES = {".py", ".pyx", ".pxd", ".c", ".cc", ".cpp", ".cxx", ".h", ".rs",
                        ".f", ".f90", ".go", ".zig", ".pyi"}
TEST_PATH = re.compile(r"(^|/)(tests?|testing|docs?|examples?|benchmarks?|tools|ci|\.github|\.circleci)(/|$)"
                       r"|(^|/)(test_[^/]*|[^/]*_test|conftest)\.py$")
# Only the builtins: `re.compile`, `regex.compile` and friends are everywhere.
EXEC_SINKS = {"exec", "eval", "compile", "builtins.exec", "builtins.eval", "builtins.compile",
              "__builtins__.exec", "__import__"}

MAX_BIN_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BIN_BYTES = 24 * 1024 * 1024
MAX_REACHED_FUNCS = 60
_PRINTABLE = re.compile(rb"[\x20-\x7e]{6,}")
_HOST_RE = re.compile("|".join(SUSPICIOUS_HOSTS), re.IGNORECASE)
_TLD_RE = re.compile("|".join(SUSPICIOUS_TLDS), re.IGNORECASE)
_EXFIL_RE = re.compile("|".join(p for ps in EXFIL_PATTERNS.values() for p in ps), re.IGNORECASE)
# Path-shaped only, for "touches secrets" in code that runs by itself: the
# broad words above ("credentials", "Cookies") are in every cloud SDK.
_SECRET_PATH_RE = re.compile("|".join([
    r"\.ssh[/\\]", r"id_rsa", r"id_ed25519", r"\.aws[/\\]credentials", r"\.kube[/\\]config",
    r"\.docker[/\\]config", r"\.pypirc", r"\.npmrc", r"\.git-credentials", r"\.netrc",
    r"Login Data", r"cookies\.sqlite", r"key4\.db", r"logins\.json", r"Local State",
    r"Chrome[/\\]User Data", r"wallet\.dat", r"Ethereum[/\\]keystore", r"Telegram Desktop",
    r"Exodus[/\\]", r"Electrum[/\\]", r"/etc/passwd", r"/etc/shadow",
]), re.IGNORECASE)

OPS = ("network", "process", "exec", "decode", "write", "env", "secret")

BEHAVIOUR_FEATURES = [
    *(f"autorun_install_{o}" for o in OPS),
    *(f"autorun_import_{o}" for o in OPS),
    "autorun_spawn", "autorun_categories", "autorun_native_load",
    "flow_decode_exec", "flow_net_exec", "flow_charcode_exec", "flow_tainted_process",
    "flow_download_execute", "flow_net_to_file", "flow_hidden_endpoint", "flow_secret_send",
    "proc_hidden", "proc_shell_downloader", "proc_python_inline", "proc_temp_exec",
    "bin_orphan_modules", "bin_compiled_init", "bin_net_indicators", "bin_url_count",
    "bin_exfil_refs", "bin_shell_cmd", "pth_code_lines", "script_downloader",
    "worst_file_danger", "n_danger_files",
]

BEHAVIOUR_DESCRIPTIONS = {
    "autorun_install_network": "network calls in code that runs during pip install",
    "autorun_install_process": "commands launched by code that runs during pip install",
    "autorun_install_exec": "eval/exec in code that runs during pip install",
    "autorun_install_decode": "decoding (base64/zlib/hex) in code that runs during pip install",
    "autorun_install_write": "files written or made executable during pip install",
    "autorun_install_env": "environment variables read during pip install",
    "autorun_install_secret": "credential/wallet/browser paths touched during pip install",
    "autorun_import_network": "network calls that fire as soon as the package is imported",
    "autorun_import_process": "commands launched as soon as the package is imported",
    "autorun_import_exec": "eval/exec that runs as soon as the package is imported",
    "autorun_import_decode": "decoding that runs as soon as the package is imported",
    "autorun_import_write": "files written or made executable on import",
    "autorun_import_env": "environment variables read on import",
    "autorun_import_secret": "credential/wallet/browser paths touched on import",
    "autorun_spawn": "background threads, processes or exit hooks started at import/install time",
    "autorun_categories": "distinct kinds of risky operation in code that runs by itself (install or import)",
    "autorun_native_load": "a bundled native library (.so/.dll) is loaded by code that runs on import or install",
    "flow_hidden_endpoint": "a network call whose destination is decoded or built from character codes -- the address is hidden",
    "flow_secret_send": "wallet mnemonics, seed phrases or private keys are sent over the network",
    "flow_decode_exec": "decoded data flows into eval/exec/unmarshal -- a packed payload, even across lines",
    "flow_net_exec": "data fetched from the network flows into eval/exec -- remote code execution",
    "flow_charcode_exec": "a string built from character codes flows into eval/exec",
    "flow_tainted_process": "downloaded or decoded data is passed to a shell/process",
    "flow_download_execute": "a file is downloaded (or unpacked from an embedded blob) and then made executable or run in the same code",
    "flow_net_to_file": "network data written straight to a file",
    "proc_hidden": "processes launched hidden: no window, detached session or silenced output",
    "proc_shell_downloader": "shell commands that download-and-run, open a reverse shell or decode inline",
    "proc_python_inline": "a new Python interpreter started with inline code (python -c)",
    "proc_temp_exec": "programs launched from a temp directory",
    "bin_orphan_modules": "compiled modules (.so/.pyd) shipped without any source they could be built from",
    "bin_compiled_init": "the package's __init__ itself is a compiled binary -- its import code cannot be read",
    "bin_net_indicators": "webhook/paste/tunnel hosts, throwaway TLDs or hardcoded IPs inside compiled binaries",
    "bin_url_count": "URLs embedded in compiled binaries",
    "bin_exfil_refs": "credential, wallet or browser-store paths embedded in compiled binaries",
    "bin_shell_cmd": "download-and-run or reverse-shell commands embedded in compiled binaries",
    "pth_code_lines": ".pth files with code lines -- they run at every Python start-up",
    "script_downloader": "shell/batch/PowerShell/JS files that download and run something",
    "worst_file_danger": "weighted count of distinct risky operation kinds in the single worst file",
    "n_danger_files": "files that combine three or more kinds of risky operation",
}


# ---- per-file AST analysis ---------------------------------------------------

def _short(node: ast.Call) -> tuple[str, str]:
    name = dotted_name(node.func)
    return name, name.split(".")[-1]


def _op_kinds(call: ast.Call) -> set[str]:
    """Which risky operation kinds a single call is."""
    name, short = _short(call)
    kinds = set()
    if _tail_matches(name, NET_CALLS):
        kinds.add("network")
    if _tail_matches(name, PROC_CALLS):
        kinds.add("process")
    if _is_exec(name) or _tail_matches(name, DESER_SINKS):
        kinds.add("exec")
    if _tail_matches(name, DECODE_SOURCES):
        kinds.add("decode")
    if short in {"chmod", "urlretrieve"} or (short == "open" and _mode_writes(call)) \
            or short in {"write_bytes", "write_text", "copyfile", "copy2"}:
        kinds.add("write")
    if name.endswith(("environ.get", "os.getenv")) or short == "getenv":
        kinds.add("env")
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _SECRET_PATH_RE.search(arg.value):
            kinds.add("secret")
    return kinds


def _is_exec(name: str) -> bool:
    return name in EXEC_SINKS


def _mode_writes(call: ast.Call) -> bool:
    modes = [a for a in call.args[1:2]] + [k.value for k in call.keywords if k.arg == "mode"]
    return any(isinstance(m, ast.Constant) and isinstance(m.value, str)
               and any(c in m.value for c in "wax") for m in modes)


def _is_main_guard(node: ast.stmt) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and dotted_name(node.test.left) == "__name__")


def _auto_nodes(tree: ast.Module, install: bool) -> list[ast.AST]:
    """Statements that execute by themselves: module level (minus function
    bodies and `if __name__ == "__main__"`), the module's own functions they
    call, thread/atexit targets, and -- in setup.py -- install-command classes."""
    funcs: dict[str, ast.AST] = {}
    classes: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node

    roots: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
            continue
        if _is_main_guard(node):
            continue
        if isinstance(node, ast.ClassDef):
            # Class bodies run at definition; methods only when called.
            roots += [s for s in node.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
            hooked = install and any(dotted_name(b).split(".")[-1] in INSTALL_HOOK_BASES
                                     for b in node.bases)
            if hooked:
                roots += [s for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
            continue
        roots.append(node)

    reached: list[ast.AST] = list(roots)
    seen: set[str] = set()
    queue = list(roots)
    while queue and len(seen) < MAX_REACHED_FUNCS:
        node = queue.pop()
        for sub in ast.walk(node):
            targets: list[str] = []
            if isinstance(sub, ast.Call):
                name, short = _short(sub)
                if isinstance(sub.func, ast.Name):
                    targets.append(sub.func.id)
                if _tail_matches(name, SPAWN_CALLS) or short in {"register", "Thread", "Timer", "Process"}:
                    targets += [a.id for a in sub.args if isinstance(a, ast.Name)]
                    targets += [k.value.id for k in sub.keywords
                                if k.arg == "target" and isinstance(k.value, ast.Name)]
                # cmdclass={"install": Evil}
                for k in sub.keywords:
                    if k.arg == "cmdclass" and isinstance(k.value, ast.Dict):
                        targets += [v.id for v in k.value.values if isinstance(v, ast.Name)]
            for t in targets:
                if t in seen:
                    continue
                if t in funcs:
                    seen.add(t)
                    reached.append(funcs[t])
                    queue.append(funcs[t])
                elif t in classes and install:
                    seen.add(t)
                    reached.append(classes[t])
                    queue.append(classes[t])
    return reached


def _spawn_count(nodes: list[ast.AST]) -> int:
    n = 0
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                name, short = _short(sub)
                if _tail_matches(name, SPAWN_CALLS) or name in {"atexit.register"}:
                    n += 1
    return n


def _native_loads(nodes: list[ast.AST]) -> int:
    """ctypes/cffi loads of a *bundled* library (path built from __file__ or a
    .so/.dll/.dylib literal), not `ctypes.CDLL("libc.so.6")`-style system ones."""
    n = 0
    for node in nodes:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            name, short = _short(sub)
            if not (_tail_matches(name, NATIVE_LOAD_CALLS) or short in {"CDLL", "WinDLL", "LoadLibrary", "dlopen"}):
                continue
            for a in sub.args[:1]:
                names = {dotted_name(x) for x in ast.walk(a) if isinstance(x, (ast.Name, ast.Attribute))}
                lits = [x.value for x in ast.walk(a) if isinstance(x, ast.Constant) and isinstance(x.value, str)]
                if "__file__" in names or any(x.endswith(("dirname", "abspath")) for x in names) or any(
                        l.endswith((".so", ".dll", ".dylib", ".pyd", ".bin")) and not l.startswith("lib")
                        for l in lits):
                    n += 1
    return n


def _count_ops(nodes: list[ast.AST]) -> dict[str, int]:
    out = dict.fromkeys(OPS, 0)
    seen_ids: set[int] = set()
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and id(sub) not in seen_ids:
                seen_ids.add(id(sub))
                for k in _op_kinds(sub):
                    out[k] += 1
    return out


class _Taint:
    """Flow-insensitive taint within one scope (a module body or a function)."""

    def __init__(self) -> None:
        self.vars: dict[str, set[str]] = {}

    def of(self, expr: ast.AST) -> set[str]:
        kinds: set[str] = set()
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name) and sub.id in self.vars:
                kinds |= self.vars[sub.id]
            elif isinstance(sub, ast.Call):
                name, short = _short(sub)
                if _tail_matches(name, DECODE_SOURCES):
                    kinds.add("decode")
                if _tail_matches(name, NET_CALLS) and short not in {"gethostbyname", "getaddrinfo"}:
                    kinds.add("net")
                if _is_char_code_build(sub) or (short == "join" and any(
                        isinstance(a, ast.Call) and _is_char_code_build(a) for a in sub.args)):
                    kinds.add("charcode")
        return kinds

    def assign(self, targets: list[ast.AST], kinds: set[str]) -> None:
        if not kinds:
            return
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name):
                    self.vars.setdefault(sub.id, set()).update(kinds)


def _scopes(tree: ast.Module) -> list[list[ast.stmt]]:
    out = [[s for s in tree.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(node.body)
    return out


def _is_sink(call: ast.Call) -> bool:
    name, short = _short(call)
    return (_is_exec(name) or _tail_matches(name, NET_CALLS) or _tail_matches(name, DESER_SINKS) or _tail_matches(name, PROC_CALLS)
            or short in {"startfile", "chmod", "urlretrieve"} or short in WRITE_METHODS)


def _flow(tree: ast.Module) -> dict[str, int]:
    out = dict.fromkeys(("flow_decode_exec", "flow_net_exec", "flow_charcode_exec",
                         "flow_tainted_process", "flow_download_execute", "flow_net_to_file",
                         "proc_hidden", "proc_shell_downloader", "proc_python_inline",
                         "proc_temp_exec", "flow_hidden_endpoint", "flow_secret_send"), 0)
    for body in _scopes(tree):
        stmts = [n for s in body for n in ast.walk(s)
                 if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not any(isinstance(n, ast.Call) and _is_sink(n) for n in stmts):
            continue
        taint = _Taint()
        # Two passes reach a fixpoint for the short chains malware uses
        # (resp -> data -> code), whatever order they are written in.
        for _ in range(2):
            for n in stmts:
                if isinstance(n, ast.Assign):
                    taint.assign(n.targets, taint.of(n.value))
                elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and n.value is not None:
                    taint.assign([n.target], taint.of(n.value))
                elif isinstance(n, (ast.With, ast.AsyncWith)):
                    for item in n.items:
                        if item.optional_vars is not None:
                            taint.assign([item.optional_vars], taint.of(item.context_expr))

        downloaded = executed = False
        for n in stmts:
            if not isinstance(n, ast.Call):
                continue
            name, short = _short(n)
            arg_taint: set[str] = set()
            for a in [*n.args, *(k.value for k in n.keywords)]:
                arg_taint |= taint.of(a)
            if _is_exec(name) or _tail_matches(name, DESER_SINKS):
                out["flow_decode_exec"] += "decode" in arg_taint
                out["flow_net_exec"] += "net" in arg_taint
                out["flow_charcode_exec"] += "charcode" in arg_taint
                executed = True
            if _tail_matches(name, NET_CALLS):
                first = n.args[:1] + [k.value for k in n.keywords if k.arg in {"url", "host"}]
                if any(taint.of(a) & {"decode", "charcode"} for a in first):
                    out["flow_hidden_endpoint"] += 1
                sent = [x for a in [*n.args[1:], *(k.value for k in n.keywords if k.arg in
                                                  {"json", "data", "params", "files"})]
                        for x in ast.walk(a)]
                if any((isinstance(x, ast.Constant) and isinstance(x.value, str) and SECRET_WORDS.search(x.value))
                       or (isinstance(x, ast.Name) and SECRET_WORDS.search(x.id)) for x in sent):
                    out["flow_secret_send"] += 1
            is_proc = _tail_matches(name, PROC_CALLS)
            if is_proc:
                executed = True
                out["flow_tainted_process"] += bool(arg_taint & {"net", "decode"})
                strings = [s.value for a in [*n.args, *(k.value for k in n.keywords)]
                           for s in ast.walk(a) if isinstance(s, ast.Constant) and isinstance(s.value, str)]
                text = " ".join(strings)
                out["proc_shell_downloader"] += bool(SHELL_DOWNLOADER.search(text))
                out["proc_temp_exec"] += bool(TEMP_PATH.search(text)) or any(
                    dotted_name(s.func).endswith(("gettempdir", "mkdtemp", "mkstemp"))
                    for a in n.args for s in ast.walk(a) if isinstance(s, ast.Call))
                inline = any(s in {"-c", "-m"} for s in strings) and any(
                    dotted_name(s) == "sys.executable" or (isinstance(s, ast.Constant) and
                                                          str(s.value).startswith("python"))
                    for a in n.args for s in ast.walk(a))
                out["proc_python_inline"] += inline
                hidden = False
                for k in n.keywords:
                    v = k.value
                    if k.arg == "start_new_session" and isinstance(v, ast.Constant) and v.value is True:
                        hidden = True
                    if k.arg in {"creationflags", "startupinfo"} and any(
                            isinstance(s, (ast.Name, ast.Attribute)) and dotted_name(s).split(".")[-1] in HIDING_FLAGS
                            for s in ast.walk(v)):
                        hidden = True
                    if k.arg == "creationflags" and isinstance(v, ast.Constant) and v.value:
                        hidden = True
                devnull = sum(1 for k in n.keywords if k.arg in {"stdout", "stderr"}
                              and dotted_name(k.value).endswith("DEVNULL"))
                out["proc_hidden"] += hidden or devnull >= 2
            if short in {"startfile", "chmod"}:
                executed = True
            if short == "urlretrieve":
                downloaded = True
            if short in WRITE_METHODS and "net" in arg_taint:
                out["flow_net_to_file"] += 1
                downloaded = True
            elif short in WRITE_METHODS and "decode" in arg_taint:
                downloaded = True   # dropped from an embedded blob rather than fetched
        out["flow_download_execute"] += downloaded and executed
    return out


def analyse_file(tree: ast.Module, rel_path: str, is_setup: bool) -> dict[str, int]:
    """Behaviour counters for one parsed file."""
    out: dict[str, int] = {}
    in_tests = bool(TEST_PATH.search(rel_path))
    auto = _auto_nodes(tree, install=is_setup)
    ops = _count_ops(auto)
    ctx = "install" if is_setup else "import"
    for o in OPS:
        out[f"autorun_install_{o}"] = ops[o] if ctx == "install" else 0
        # Tests and examples only run when someone runs them.
        out[f"autorun_import_{o}"] = ops[o] if ctx == "import" and not in_tests else 0
    out["autorun_spawn"] = 0 if in_tests else _spawn_count(auto)
    out["autorun_native_load"] = 0 if in_tests else _native_loads(auto)
    out["_auto_categories"] = 0 if in_tests and not is_setup else sum(1 for o in OPS if ops[o])
    # Test suites legitimately spawn interpreters and shell out; they only
    # run when someone runs the tests.
    if not in_tests or is_setup:
        out.update(_flow(tree))
    return out


# ---- non-Python files --------------------------------------------------------

def scan_other_files(all_files: list[Path], root: Path | None = None) -> dict[str, float]:
    out = dict.fromkeys(("bin_orphan_modules", "bin_compiled_init", "bin_net_indicators",
                         "bin_url_count", "bin_exfil_refs", "bin_shell_cmd", "pth_code_lines",
                         "script_downloader"), 0.0)
    stems = {p.name.split(".")[0] for p in all_files if p.suffix.lower() in SOURCE_STEM_SUFFIXES}
    budget = MAX_TOTAL_BIN_BYTES
    for p in all_files:
        suffix = p.suffix.lower()
        try:
            if suffix in BIN_SUFFIXES:
                stem = p.name.split(".")[0]
                if suffix in {".so", ".pyd"} and stem not in stems:
                    out["bin_orphan_modules"] += 1
                    out["bin_compiled_init"] += stem == "__init__"
                if budget <= 0:
                    continue
                data = p.read_bytes()[:MAX_BIN_BYTES]
                budget -= len(data)
                text = "\n".join(m.decode("ascii") for m in _PRINTABLE.findall(data))
                out["bin_net_indicators"] += len(_HOST_RE.findall(text)) + len(_TLD_RE.findall(
                    " ".join(URL_RE.findall(text)))) + sum(
                    1 for ip in IP_RE.findall(text)
                    if not ip.startswith(("127.", "0.", "10.", "192.168.", "255.", "1.", "2."))
                    and all(0 <= int(x) <= 255 for x in ip.split(".")))
                out["bin_url_count"] += len(URL_RE.findall(text))
                out["bin_exfil_refs"] += len(_EXFIL_RE.findall(text))
                out["bin_shell_cmd"] += len(SHELL_DOWNLOADER.findall(text))
            elif suffix == ".pth":
                lines = p.read_text(errors="replace").splitlines()
                out["pth_code_lines"] += sum(1 for ln in lines
                                             if ln.startswith(("import ", "import\t")) or "exec(" in ln)
            elif suffix in SCRIPT_SUFFIXES and p.stat().st_size < 2_000_000 and not (
                    root is not None and TEST_PATH.search(str(p.relative_to(root)))):
                out["script_downloader"] += bool(SHELL_DOWNLOADER.search(p.read_text(errors="replace")))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
    return out


# ---- shared by training, experiments and the server -------------------------

# Features where "more" can only mean "more suspicious": trained with a
# monotone constraint, so a large benign-looking package cannot cancel them.
MONOTONE_PREFIXES = ("autorun_", "flow_", "proc_", "bin_", "pth_", "script_", "exfil_",
                     "decode_then_exec", "net_suspicious_host", "obf_char_code_build",
                     "call_obfuscated_import", "install_exec_at_toplevel",
                     "install_net_at_toplevel", "install_proc_at_toplevel", "install_hook_subclass")

# Concrete malicious capability: something that runs by itself, a data flow
# into exec or a process, launch tradecraft, a non-Python payload, credential
# paths or exfiltration hosts. A package the model flags on shape alone (tiny,
# no README) but that can do none of these is not called malicious.
CAPABILITY = ["autorun_install_network", "autorun_install_process", "autorun_install_exec",
              "autorun_install_decode", "autorun_install_write", "autorun_install_secret",
              "autorun_import_network", "autorun_import_process", "autorun_import_exec",
              "autorun_import_decode", "autorun_import_write", "autorun_import_secret",
              "autorun_spawn", "autorun_native_load", "flow_hidden_endpoint", "flow_secret_send",
              "flow_decode_exec", "flow_net_exec", "flow_charcode_exec",
              "flow_tainted_process", "flow_download_execute", "proc_hidden",
              "proc_shell_downloader", "proc_python_inline", "proc_temp_exec",
              "bin_compiled_init", "bin_net_indicators", "bin_shell_cmd", "pth_code_lines",
              "script_downloader", "decode_then_exec", "net_suspicious_host", "exfil_total",
              "install_exec_at_toplevel", "install_net_at_toplevel", "install_proc_at_toplevel",
              "call_obfuscated_import", "obf_char_code_build"]


def capabilities(feats: dict[str, float]) -> list[str]:
    """Capability features present in one package's feature dict."""
    return [c for c in CAPABILITY if feats.get(c, 0) > 0]
