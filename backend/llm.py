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
5. Plain prose, 2-4 short paragraphs. No preamble, no bullet lists, no \
markdown headers. Do not restate the confidence number -- the UI shows it.
"""


def build_prompt(package: str, version: str, verdict: dict, metadata: dict) -> str:
    """Render the evidence payload the LLM is allowed to talk about."""
    lines = [
        f"Package: {package} {version}",
        f"Model verdict: {verdict['verdict'].upper()}",
        f"Confidence: {verdict['confidence']}%",
        f"Malicious probability: {verdict['malicious_probability']} "
        f"(decision threshold {verdict['threshold']})",
    ]

    if metadata.get("summary"):
        lines.append(f"Stated purpose: {metadata['summary']}")
    if metadata.get("n_releases"):
        lines.append(f"Releases published: {metadata['n_releases']}")

    lines += ["", "Ranked evidence from the classifier "
                  "(contribution > 0 pushed toward MALICIOUS, < 0 toward BENIGN):"]

    for i, e in enumerate(verdict["evidence"], 1):
        lines.append(
            f"{i}. {e['description']}\n"
            f"   measured value: {e['value']}   "
            f"contribution: {e['contribution']:+.3f} ({e['direction']})"
        )

    if not verdict["evidence"]:
        lines.append("(no individual feature had a meaningful contribution)")

    lines += ["", f"Explain to a developer why the model reached the "
                  f"{verdict['verdict'].upper()} verdict for this package, "
                  "citing only the evidence above."]
    return "\n".join(lines)


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
            # roughly a minute. The verdict card is already on screen by then.
            "num_predict": 400,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=GENERATION_TIMEOUT) as client:
            async with client.stream("POST", f"{OLLAMA_HOST}/api/generate",
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
                    if chunk.get("response"):
                        yield chunk["response"]
                    if chunk.get("done"):
                        break
    except httpx.HTTPError as exc:
        yield (f"[Could not reach the local model at {OLLAMA_HOST}: {exc}. "
               f"Start it with `ollama serve` and `ollama pull {OLLAMA_MODEL}`. "
               "The verdict and evidence above are unaffected -- they come from "
               "the classifier, not the LLM.]")
