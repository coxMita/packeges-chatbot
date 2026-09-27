"""FastAPI server for the package analyser.

    POST /api/analyze   fetch a package, extract features, return the verdict
                        plus the ranked evidence behind it, the closest
                        known packages by code similarity, and a per-file
                        scan of where the suspicious code sits
    POST /api/explain   stream a plain-English explanation of a verdict (SSE)
    POST /api/chat      stream an answer to a follow-up question, grounded in
                        the analyses already shown in the conversation (SSE)
    GET  /api/health    readiness of the model and the local LLM

The two are split on purpose: /analyze is fast and deterministic, so the UI can
paint the verdict card immediately, while /explain streams token by token from a
4B model on CPU and takes a few seconds.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

import llm  # noqa: E402
from assess import Assessor  # noqa: E402
from fetch import PackageNotFound, fetch_package  # noqa: E402
from file_scan import scan_package  # noqa: E402
from predict import Scorer  # noqa: E402
from similarity import SimilarityIndex  # noqa: E402

scorer = Scorer()
similarity = SimilarityIndex()
assessor = Assessor()
_popular_names: set[str] = set()


def _load_popular() -> None:
    """Load the typosquat reference list once, if it has been cached."""
    global _popular_names
    try:
        from acquire_benign import fetch_top_packages, normalise
        _popular_names = {normalise(p) for p in fetch_top_packages()[:2000]}
    except Exception as exc:  # offline, or the cache has not been built yet
        print(f"[startup] typosquat reference list unavailable ({exc}); "
              "the pkg_typosquat_distance feature will default to 'far'")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_popular()
    yield


app = FastAPI(title="Malicious PyPI Package Detector", version="0.2.0", lifespan=lifespan)

# Only needed when the UI is served from another origin (the Vite dev server).
# In Docker, nginx serves the UI and proxies /api, so every call is same-origin.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # /api/analyze payloads from this conversation, oldest first, code stripped.
    analyses: list[dict] = Field(default_factory=list, max_length=20)
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)


class ExplainRequest(BaseModel):
    package: str = Field(..., min_length=1, max_length=200)
    version: str = Field("", max_length=100)
    # The /api/analyze payload being explained, code stripped.
    verdict: dict


class AnalyzeRequest(BaseModel):
    package: str = Field(..., min_length=1, max_length=200,
                         description="e.g. 'requests' or 'requests==2.31.0'")


@app.get("/api/health")
async def health() -> dict:
    return {
        "model_loaded": scorer.ready,
        "llm_available": await llm.is_available(),
        "llm_model": llm.OLLAMA_MODEL,
        "n_features": len(scorer.feature_names),
        "similarity_ready": similarity.ready,
        "calibration_ready": assessor.ready,
    }


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest) -> dict:
    if not scorer.ready:
        raise HTTPException(
            503, "No trained model is loaded. Run `python ml/train_gbdt.py` first."
        )

    try:
        pkg = await fetch_package(req.package)
    except PackageNotFound as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch from PyPI: {exc}")

    try:
        verdict = scorer.score(pkg.root, pkg.name, _popular_names)
        files = await run_in_threadpool(scan_package, pkg.root)

        # Encoding is CPU-bound for a second or two; keep the event loop free.
        similar = None
        if similarity.ready:
            try:
                found = await run_in_threadpool(similarity.search, pkg.root)
                similar = found.to_dict() if found else None
            except Exception as exc:  # a second opinion must never sink the first
                print(f"[similarity] search failed for {pkg.name}: {exc}")

        return {
            "package": pkg.name,
            "version": pkg.version,
            "metadata": {
                "summary": pkg.summary,
                "author": pkg.author,
                "home_page": pkg.home_page,
                "n_releases": pkg.n_releases,
                "sdist_bytes": pkg.sdist_bytes,
                "upload_time": pkg.upload_time,
            },
            **verdict.to_dict(),
            "similarity": similar,
            "file_scan": files,
            "assessment": assessor.assess(
                verdict.malicious_probability,
                similar["malicious_percent"] / 100 if similar else None,
            ),
        }
    finally:
        # The package source is untrusted and has served its purpose.
        pkg.cleanup()


@app.post("/api/explain")
async def explain(req: ExplainRequest) -> StreamingResponse:
    """Stream the LLM's explanation of an already-computed verdict.

    The client posts back the /api/analyze payload rather than the server
    re-analysing, so the explanation is guaranteed to describe the exact
    verdict the user is looking at. (It is a POST because that payload, with
    per-feature training context and the file scan, outgrows a URL.)
    """
    payload = req.verdict
    for key in ("verdict", "confidence", "malicious_probability", "threshold", "evidence"):
        if key not in payload:
            raise HTTPException(400, f"verdict payload is missing '{key}'")

    metadata = payload.get("metadata", {})
    return _sse(llm.stream_explanation(req.package, req.version, payload, metadata))


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream an answer to a follow-up question about earlier analyses.

    Like /explain, the client sends back what it was shown, so the answer is
    grounded in exactly those verdicts; the server keeps no conversation state.
    """
    for a in req.analyses:
        for key in ("package", "verdict", "confidence", "malicious_probability",
                    "threshold", "evidence"):
            if key not in a:
                raise HTTPException(400, f"an analysis is missing '{key}'")

    history = [t.model_dump() for t in req.history]
    return _sse(llm.stream_chat(req.question, req.analyses, history))


def _sse(chunks) -> StreamingResponse:
    """Wrap an async text iterator as a server-sent event stream."""
    async def event_stream():
        try:
            async for chunk in chunks:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as exc:  # never leave the client hanging on an open stream
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
