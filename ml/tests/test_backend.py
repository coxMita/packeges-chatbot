"""Backend tests that do not require a trained model or network access."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

import llm  # noqa: E402
from fetch import PackageNotFound, parse_spec  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("requests", ("requests", None)),
    ("  requests  ", ("requests", None)),
    ("requests==2.31.0", ("requests", "2.31.0")),
    ("requests == 2.31.0", ("requests", "2.31.0")),
    ("requests@2.31.0", ("requests", "2.31.0")),
    ("requests 2.31.0", ("requests", "2.31.0")),
    ("requests==v2.31.0", ("requests", "2.31.0")),
    ("python-dateutil", ("python-dateutil", None)),
    ("zope.interface==5.4.0", ("zope.interface", "5.4.0")),
])
def test_parse_spec(text, expected):
    assert parse_spec(text) == expected


@pytest.mark.parametrize("text", [
    "", "   ", "../../etc/passwd", "requests; rm -rf /", "<script>", "-leading-dash",
])
def test_parse_spec_rejects_junk(text):
    """Chat input is untrusted; anything that is not a package name is refused."""
    with pytest.raises(PackageNotFound):
        parse_spec(text)


def _verdict(**over):
    base = {
        "verdict": "malicious",
        "confidence": 92.4,
        "malicious_probability": 0.97,
        "threshold": 0.41,
        "evidence": [{
            "feature": "install_net_at_toplevel",
            "value": 2.0,
            "contribution": 1.85,
            "direction": "malicious",
            "description": "network calls at setup.py module level",
        }],
    }
    base.update(over)
    return base


def test_prompt_contains_only_verdict_and_evidence():
    """The LLM must receive the decision and the evidence -- and nothing else."""
    prompt = llm.build_prompt("evilpkg", "1.0", _verdict(), {"summary": "a utility"})

    assert "MALICIOUS" in prompt
    assert "network calls at setup.py module level" in prompt
    assert "+1.850" in prompt
    assert "evilpkg" in prompt


def test_prompt_handles_empty_evidence():
    """A verdict with no strong feature must still produce a usable prompt."""
    prompt = llm.build_prompt("quiet", "1.0", _verdict(evidence=[]), {})
    assert "no individual feature" in prompt.lower()


def test_system_prompt_forbids_reclassification():
    """The separation of concerns is load-bearing; assert it stays in the prompt."""
    s = llm.SYSTEM_PROMPT.lower()
    assert "never re-classify" in s or "never dispute" in s
    assert "do not invent" in s


def test_benign_verdict_renders():
    prompt = llm.build_prompt("tidy", "2.0", _verdict(verdict="benign"), {})
    assert "BENIGN" in prompt


def test_prompt_spells_out_absent_features():
    """'the package ships a README' + 'value: 0.0' was read by the LLM as a
    README being present. Absence must be stated in words."""
    ev = [{"feature": "pkg_has_readme", "value": 0.0, "contribution": 1.86,
           "direction": "malicious", "description": "the package ships a README"},
          {"feature": "call_suspicious_import", "value": 0.0, "contribution": -0.7,
           "direction": "benign", "description": "imports of subprocess, socket"}]
    prompt = llm.build_prompt("tiny", "1.0", _verdict(evidence=ev), {})

    assert "observed: NOT present" in prompt
    assert "observed: 0 (none found)" in prompt


def test_prompt_flags_close_calls_only():
    near = llm.build_prompt("kerwin", "0.2", _verdict(
        verdict="benign", malicious_probability=0.817, threshold=0.85), {})
    far = llm.build_prompt("evilpkg", "1.0", _verdict(), {})

    assert "CLOSE CALL" in near and "just below" in near
    assert "CLOSE CALL" not in far
