"""Local LLM explanation layer (Ollama).

The model decides; the LLM writes. That separation is the whole design, and it
is enforced structurally rather than by asking nicely:

  * The prompt contains **only** the verdict, the confidence, and the ranked
    SHAP evidence. The package source is never sent, so the LLM has nothing to
    form an independent opinion from.
  * The verdict is stated to it as settled fact, not as a question.
  * Feature descriptions come from `FEATURE_DESCRIPTIONS`, so the prose stays
    anchored to what the classifier actually measured.

Anything the LLM says beyond the supplied evidence is unsupported by definition,
and the system prompt tells it so.
"""

from __future__ import annotations

import json
import os
import re
from typing import AsyncIterator

import httpx

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")

GENERATION_TIMEOUT = 180.0

SYSTEM_PROMPT = """\
You explain the output of a machine-learning classifier that detects malicious \
PyPI packages. You are the presentation layer, not the decision maker.

Rules:
1. The verdict and confidence are already decided by the model. Never dispute \
them, never re-classify, never say "I think this is actually safe/unsafe".
2. You are shown the ranked feature evidence the model used. Explain ONLY \
those features. Do not invent behaviours, filenames, URLs or CVEs.
3. If the evidence is weak or thin, say so plainly -- that is useful to the \
reader, and it is not the same as disagreeing with the verdict.
4. Write for a developer deciding whether to install this package. Be concrete \
and specific about what each cited feature means in practice.
5. Read each "observed" line literally. "NOT present" and "none found" mean \
the thing is absent -- never describe an absent feature as present.
6. If the verdict is marked a close call, say so, and do not invent reasons \
the evidence does not give.
7. A per-file scan, if given, says WHERE the suspect code is: name the file \
with the most kinds of suspicious operation, the calls it makes and their line \
numbers, and what those operations are used for in malware. It is not a verdict.
8. If a code-similarity section is given, it is a separate second opinion: \
say which known packages the code most resembles and what share of them are \
malware. If it disagrees with the model's verdict, say so plainly rather than \
reconciling the two. Similarity is likeness to known samples, not proof.
9. If an overall assessment is given, lead with it. SUSPICIOUS means a human \
should read the code; the "Why this tier" line says whether one method flagged \
it or the classifier score is elevated but below its alarm threshold -- say \
which. Quote the calibrated chance as given; do not invent others.
10. Evidence lines may carry "in training data": how common the observed \
value was among the malicious and the benign packages the model learned from. \
Use it to say WHY a feature pushed the score, e.g. "39% of the malware the \
model was trained on did this, against almost no benign packages". Quote the \
shares as given; do not invent others.
11. Structure: first the overall assessment, then the suspect code (file, \
calls, lines), then how the strongest evidence compares with the training \
data. Plain prose, 3-4 short paragraphs. No preamble, no bullet lists, no \
markdown headers. Do not restate the confidence number -- the UI shows it.
"""


# Close-call band: within this distance of the threshold, the verdict could
# plausibly have gone the other way and the reader should be told so.
CLOSE_CALL_MARGIN = 0.15


def describe_observation(feature: str, value: float) -> str:
    """State a measured value as a plain fact the LLM cannot misread.

    Descriptions are phrased positively ("the package ships a README"), and a
    small model pairing that with "value: 0.0" tends to read the description
    as true. Absence has to be spelled out.
    """
    if feature.startswith(("pkg_has_", "install_has_")) or feature.endswith(
            ("_cmdclass", "_subclass")):
        return "present" if value else "NOT present"
    if value == 0:
        return "0 (none found)"
    return f"{value:.4g}"


def describe_training(feature: str, value: float, t: dict) -> str:
    """One sentence comparing a value with the training data, for the LLM."""
    binary = feature.startswith(("pkg_has_", "install_has_")) or feature.endswith(
        ("_cmdclass", "_subclass"))
    if binary:
        cond = "had this" if value else "lacked this"
    elif value == 0 and t["tail"] == "le":
        cond = "had none"
    else:
        cond = f"had {'at most' if t['tail'] == 'le' else 'at least'} {value:.4g}"
    m, b, r = t["malicious_share"], t["benign_share"], t["ratio"]
    lean = (f"{r:g}x more common among malware" if r >= 1.05 else
            f"{1 / max(r, 1e-3):.1f}x more common among benign packages" if r <= 0.95 else
            "about equally common in both")
    return (f"{m:.0%} of the {t['n_malicious']:,} malicious and {b:.0%} of the "
            f"{t['n_benign']:,} benign training packages {cond} ({lean})")


def _ident(text: str) -> str:
    """Keep a scanner-extracted call name to identifier characters only; it
    comes from untrusted code and must not carry instructions into the prompt."""
    return re.sub(r"[^\w./:-]", "", text or "")[:60]


def build_prompt(package: str, version: str, verdict: dict, metadata: dict) -> str:
    """Render the evidence payload the LLM is allowed to talk about."""
    lines = evidence_lines(package, version, verdict, metadata)
    if verdict.get("assessment"):
        lines += ["", f"Explain to a developer why the overall assessment is "
                      f"{verdict['assessment']['tier'].upper()}, quoting the calibrated "
                      "chance, citing only the evidence above."]
    else:
        lines += ["", f"Explain to a developer why the model reached the "
                      f"{verdict['verdict'].upper()} verdict for this package, "
                      "citing only the evidence above."]
    return "\n".join(lines)


def evidence_lines(package: str, version: str, verdict: dict, metadata: dict) -> list[str]:
    """The facts of one analysis, shared by the explanation and follow-up chat."""
    lines = [
        f"Package: {package} {version}",
        f"Classifier verdict: {verdict['verdict'].upper()}",
        f"Confidence: {verdict['confidence']}%",
        f"Malicious probability: {verdict['malicious_probability']} "
        f"(decision threshold {verdict['threshold']})",
    ]

    ass = verdict.get("assessment")
    if ass:
        lines[1:1] = [f"OVERALL ASSESSMENT (lead with this): {ass['tier'].upper()} "
                      f"(classifier flags it: {'yes' if ass['classifier_flags'] else 'no'}; "
                      f"code similarity flags it: "
                      f"{'n/a' if ass['similarity_flags'] is None else 'yes' if ass['similarity_flags'] else 'no'})",
                  f"Why this tier: {ass['basis']}",
                  f"Calibrated chance it is really malware: {ass['chance_low']}% if picked "
                  f"at random from PyPI, {ass['chance_high']}% if already suspected."]

    if metadata.get("summary"):
        lines.append(f"Stated purpose: {metadata['summary']}")
    if metadata.get("n_releases"):
        lines.append(f"Releases published: {metadata['n_releases']}")

    margin = verdict["malicious_probability"] - verdict["threshold"]
    if abs(margin) < CLOSE_CALL_MARGIN:
        side = "just below" if margin < 0 else "just above"
        lines.append(f"CLOSE CALL: the probability is {side} the threshold. "
                     "Say that this verdict is borderline.")

    lines += ["", "Ranked evidence from the classifier "
                  "(contribution > 0 pushed toward MALICIOUS, < 0 toward BENIGN):"]

    for i, e in enumerate(verdict["evidence"], 1):
        lines.append(
            f"{i}. {e['description']}\n"
            f"   observed: {describe_observation(e['feature'], e['value'])}   "
            f"contribution: {e['contribution']:+.3f} ({e['direction']})"
        )
        if e.get("training"):
            lines.append("   in training data: "
                         + describe_training(e["feature"], e["value"], e["training"]))

    if not verdict["evidence"]:
        lines.append("(no individual feature had a meaningful contribution)")

    scan = verdict.get("file_scan")
    if scan and scan.get("files"):
        lines += ["", f"Per-file scan ({scan['n_files_flagged']} of {scan['n_files_scanned']} "
                      "Python files contain at least one suspicious operation; most "
                      "concentrated first):"]
        for f in scan["files"][:3]:
            where = " (runs at install)" if f.get("runs_at_install") else \
                    " (test file)" if f.get("is_test") else ""
            kinds = ", ".join(c["label"] for c in f["categories"])
            lines.append(f"- {_ident_path(f['path'])}{where}, {f['loc']} lines: {kinds}")
            calls = [f"line {h['line']}: {_ident(h.get('call', ''))} [{h.get('label', '')}]"
                     for h in f.get("hits", [])
                     if _ident(h.get("call", "")) and h.get("category") != "packed_blob"][:6]
            if calls:
                lines.append(f"  suspicious calls: {', '.join(calls)}")

    sim = verdict.get("similarity")
    if sim:
        lines += ["", "Code similarity to the known training packages "
                      "(second opinion, independent of the model above):",
                  f"{sim['malicious_percent']}% malicious-weighted: "
                  f"{sim['n_malicious']} of the {sim['k']} closest known packages "
                  "by code are malware."]
        for n in sim["neighbours"][:5]:
            lines.append(f"- {n['package']} {n['version']}: {n['label']}, "
                         f"similarity {n['similarity']:.3f}")
        for m in sim.get("matches", []):
            lines.append(f"Most similar file to known {m['known_label']} code: "
                         f"{m['query_file']} resembles {m['known_file']} from "
                         f"{m['known_package']} (similarity {m['similarity']:.3f})")

    return lines


def _ident_path(path: str) -> str:
    return re.sub(r"[^\w./-]", "", path or "")[:120]


async def is_available() -> bool:
    """True if an Ollama server is reachable and has the configured model."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
            names = {m["name"] for m in r.json().get("models", [])}
            return any(n == OLLAMA_MODEL or n.startswith(OLLAMA_MODEL.split(":")[0])
                       for n in names)
    except (httpx.HTTPError, ValueError, KeyError):
        return False


async def stream_explanation(package: str, version: str, verdict: dict,
                             metadata: dict) -> AsyncIterator[str]:
    """Yield explanation text chunk by chunk as the local model generates it."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": build_prompt(package, version, verdict, metadata),
        "system": SYSTEM_PROMPT,
        "stream": True,
        "think": False,   # ask for prose, not a reasoning trace
        "options": {
            "temperature": 0.3,   # explanation, not creative writing
            "top_p": 0.9,
            # ~7 tok/s for a 4B model on CPU, so this caps an explanation at
            # roughly 1.5 minutes. The verdict card is already on screen by then.
            "num_predict": 650,
        },
    }
    async for chunk in _stream("/api/generate", payload,
                               lambda c: c.get("response", "")):
        yield chunk


# ---- follow-up questions -----------------------------------------------------

CHAT_SYSTEM_PROMPT = """\
You answer a developer's follow-up questions in a chat where a machine-learning \
classifier has analysed PyPI packages for malware. You are the presentation \
layer, not the decision maker.

About the detector, in case you are asked how it works: packages are downloaded \
and parsed as text, never installed or run. A LightGBM model scores 64 static \
features (install-time code, eval/exec, decoding, obfuscation, credential \
access, network indicators, package shape, typosquatting) and SHAP ranks which \
features drove the score. A second opinion compares the code's embeddings with \
~9,600 known packages. Both are combined into a tier: MALICIOUS (both flag it), \
SUSPICIOUS (one flags it, or together they lean malicious -- a human should \
read the code), or CLEAN. It only recognises malware resembling its training \
data; brand-new malware styles are often missed.

Rules:
1. Answer from the analyses below and the conversation so far. The verdicts \
are settled: never dispute, re-classify or give your own safety opinion.
2. You have not seen any package's source code. Never claim to know what the \
code does beyond what the evidence states. Do not invent behaviours, \
filenames, URLs or CVEs.
3. Read each "observed" line literally: "NOT present" and "none found" mean \
absent. "in training data" lines say how common that value was among the \
malware and the benign packages the model learned from -- use them to explain \
why a feature counted, quoting the shares as given.
4. If the question is about a package that has not been analysed, say so and \
tell the user to type its name to analyse it.
5. If the evidence cannot answer the question, say that plainly.
6. Questions are about the MOST RECENT analysis unless the user names another \
analysed package. "it", "this", "the package" always mean the most recent one.
7. Plain prose, short: 1-3 paragraphs. No markdown headers.
"""

# The 4B model's context is small and it runs on CPU: keep the newest few.
MAX_CHAT_ANALYSES = 3
MAX_CHAT_TURNS = 8


def build_chat_messages(question: str, analyses: list[dict],
                        history: list[dict]) -> list[dict]:
    """Assemble an Ollama /api/chat conversation grounded in past analyses."""
    recent = analyses[-MAX_CHAT_ANALYSES:]
    if recent:
        blocks = []
        for i, a in enumerate(recent):
            tag = " (MOST RECENT)" if i == len(recent) - 1 else ""
            blocks.append(f"=== Analysis{tag} ===\n" + "\n".join(evidence_lines(
                a["package"], a.get("version", ""), a, a.get("metadata", {}))))
        context = "Analyses in this conversation:\n\n" + "\n\n".join(blocks)
    else:
        context = ("No packages have been analysed yet in this conversation. "
                   "The user can type a package name to analyse one.")

    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT + "\n" + context}]
    for turn in history[-MAX_CHAT_TURNS:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    return messages


async def stream_chat(question: str, analyses: list[dict],
                      history: list[dict]) -> AsyncIterator[str]:
    """Yield the answer to a follow-up question chunk by chunk."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": build_chat_messages(question, analyses, history),
        "stream": True,
        "think": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 400,
            # Three evidence blocks plus history overflow Ollama's default window.
            "num_ctx": 8192,
        },
    }
    async for chunk in _stream("/api/chat", payload,
                               lambda c: c.get("message", {}).get("content", "")):
        yield chunk


async def _stream(endpoint: str, payload: dict, extract) -> AsyncIterator[str]:
    """POST to Ollama and yield each streamed text chunk that `extract` pulls out."""
    try:
        async with httpx.AsyncClient(timeout=GENERATION_TIMEOUT) as client:
            async with client.stream("POST", f"{OLLAMA_HOST}{endpoint}",
                                     json=payload) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    yield (f"[The local model returned HTTP {resp.status_code}. "
                           f"Is `ollama serve` running with {OLLAMA_MODEL} pulled?]")
                    return

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        continue
                    text = extract(chunk)
                    if text:
                        yield text
                    if chunk.get("done"):
                        break
    except httpx.HTTPError as exc:
        yield (f"[Could not reach the local model at {OLLAMA_HOST}: {exc}. "
               f"Start it with `ollama serve` and `ollama pull {OLLAMA_MODEL}`. "
               "The verdict and evidence above are unaffected -- they come from "
               "the classifier, not the LLM.]")
