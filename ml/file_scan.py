"""Per-file scan: which files hold the suspicious code, and on which lines.

The classifier answers "is this package malicious?". This answers the question
a reviewer asks next: "where do I look?". It walks every Python file with the
same AST visitor the features use, so what it points at is exactly what the
model counted, then adds line-level text matches (credential paths, exfil
hosts, packed blobs) that the features only count package-wide.

Files are ranked by how many *different kinds* of suspicious operation they
hold, weighted by how rarely each kind appears in legitimate code. That mirrors
the `worst_file_*` features: benign libraries spread these operations across
modules, while malware bolted onto a real library packs them into one place.

It is guidance for review, not a second verdict. Nothing here executes code.
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from features import IP_RE, SUSPICIOUS_HOSTS, _Acc, _Visitor, iter_python_files, read_text

# How strongly each kind of operation points at malware when present in a file.
WEIGHTS = {
    "hidden_name": 4, "secrets": 3, "exfil_host": 4, "packed_blob": 3,
    "exec": 3, "process": 3, "network": 2, "deserialize": 2, "decode": 2,
    "chmod": 2, "home": 1, "fs_walk": 1, "delete": 1, "env": 1, "silent_except": 1,
    "hardcoded_ip": 2, "dynamic_import": 1,
}

LABELS = {
    "hidden_name": "hidden import / name built from char codes",
    "secrets": "reads credential or wallet paths",
    "exfil_host": "exfiltration host (webhook, paste, tunnel)",
    "packed_blob": "very long line (packed payload?)",
    "exec": "eval / exec / dynamic import",
    "process": "runs shell commands",
    "network": "network calls",
    "deserialize": "unsafe deserialisation",
    "decode": "base64 / zlib decoding",
    "chmod": "makes files executable",
    "home": "home-directory lookup",
    "fs_walk": "walks the filesystem",
    "delete": "deletes files",
    "env": "reads environment variables",
    "silent_except": "swallows errors silently",
    "hardcoded_ip": "hardcoded IP address",
    "dynamic_import": "imports a module by computed name",
}

# Path-shaped subset of features.EXFIL_PATTERNS. The features can afford broad
# words like "credentials" (the model weighs them); pointing a reviewer at a
# line cannot, or every HTTP library lights up.
_SECRET_RE = re.compile("|".join([
    r"\.ssh[/\\]", r"id_rsa", r"id_ed25519", r"\.aws[/\\]credentials", r"\.kube[/\\]config",
    r"\.docker[/\\]config", r"\.pypirc", r"\.npmrc", r"\.git-credentials",
    r"Login Data", r"cookies\.sqlite", r"key4\.db", r"logins\.json", r"Local State",
    r"Chrome[/\\]User Data", r"wallet\.dat", r"Ethereum[/\\]keystore", r"Telegram Desktop",
    r"discord[^\n]{0,40}token", r"Exodus[/\\]", r"Electrum[/\\]",
]), re.IGNORECASE)
TEST_PATH = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")
_HOST_RE = re.compile("|".join(SUSPICIOUS_HOSTS), re.IGNORECASE)
PACKED_LINE = 500
MAX_CODE = 180


@dataclass
class FileReport:
    path: str
    loc: int
    runs_at_install: bool
    is_test: bool = False
    hits: list[tuple[str, int, str]] = field(default_factory=list)

    @property
    def categories(self) -> list[str]:
        seen = {c for c, _, _ in self.hits}
        return sorted(seen, key=lambda c: (-WEIGHTS[c], c))

    @property
    def score(self) -> int:
        return sum(WEIGHTS[c] for c in self.categories)


def _text_hits(text: str) -> list[tuple[str, int, str]]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if len(line) > PACKED_LINE:
            hits.append(("packed_blob", i, f"{len(line)} chars"))
            continue  # a blob matches everything; one hit is enough
        if m := _SECRET_RE.search(line):
            hits.append(("secrets", i, m.group(0)))
        if m := _HOST_RE.search(line):
            hits.append(("exfil_host", i, m.group(0)))
        for ip in IP_RE.findall(line):
            if not ip.startswith(("127.", "0.", "10.", "192.168.", "255.")):
                hits.append(("hardcoded_ip", i, ip))
                break
    return hits


def scan_file(path: Path, text: str, root: Path) -> FileReport:
    rel = str(path.relative_to(root))
    report = FileReport(rel, text.count("\n") + 1, path.name == "setup.py",
                        bool(TEST_PATH.search(rel)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
        _Visitor(_Acc(), hits=report.hits).visit(tree)
    except (SyntaxError, ValueError, RecursionError):
        pass  # text matches below still apply
    report.hits += _text_hits(text)
    return report


def scan_package(root: Path, max_files: int = 6, max_hits: int = 14) -> dict:
    """Rank a package's files by concentration of suspicious operations."""
    reports = []
    n_scanned = 0
    for path in iter_python_files(root):
        text = read_text(path)
        if text is None:
            continue
        n_scanned += 1
        r = scan_file(path, text, root)
        if r.hits:
            reports.append((r, text))

    # Test files only run when someone runs the tests; real modules and
    # setup.py run on import or install, so they are reviewed first.
    reports.sort(key=lambda rt: (rt[0].is_test, -rt[0].score, rt[0].path))
    files = []
    for r, text in reports[:max_files]:
        lines = text.splitlines()
        # One hit per (category, line), strongest categories first.
        uniq = {}
        for cat, line, name in r.hits:
            uniq.setdefault((cat, line), name)
        ordered = sorted(uniq.items(), key=lambda kv: (-WEIGHTS[kv[0][0]], kv[0][1]))[:max_hits]
        files.append({
            "path": r.path,
            "loc": r.loc,
            "score": r.score,
            "runs_at_install": r.runs_at_install,
            "is_test": r.is_test,
            "categories": [{"id": c, "label": LABELS[c]} for c in r.categories],
            "hits": [
                {
                    "line": line,
                    "category": cat,
                    "label": LABELS[cat],
                    "call": name,
                    "code": (lines[line - 1].strip()[:MAX_CODE] if 0 < line <= len(lines) else ""),
                }
                for (cat, line), name in sorted(ordered, key=lambda kv: kv[0][1])
            ],
        })
    return {"n_files_scanned": n_scanned, "n_files_flagged": len(reports), "files": files}
