"""FastAPI app: /api/ask, /api/search and a small static UI."""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import guardrails, llm
from .retrieval import MODES, Retriever

ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = ROOT / "data" / "index"
MANIFEST = ROOT / "data" / "manifest.json"

app = FastAPI(title="RAG demo: FastAPI docs assistant")
_state: dict = {}
_hits_log: dict[str, deque] = defaultdict(deque)
RATE_LIMIT, WINDOW = 15, 60  # requests per IP per minute; protects a free API key if deployed


def retriever() -> Retriever:
    if "r" not in _state:
        _state["r"] = Retriever(INDEX_DIR)
    return _state["r"]


def source_url(doc: str) -> str | None:
    if "manifest" not in _state:
        _state["manifest"] = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    return _state["manifest"].get(doc, {}).get("url")


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    q = _hits_log[ip]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests; please wait a minute.")
    q.append(now)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = "hybrid"


def _sources(hits) -> list[dict]:
    return [{**h.to_dict(), "url": source_url(h.chunk.doc)} for h in hits]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm": llm.llm_enabled(), "modes": MODES}


@app.get("/api/search")
def search(q: str, mode: str = "hybrid", k: int = 5, request: Request = None) -> dict:
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    rate_limit(request)
    return {"query": q, "mode": mode, "results": _sources(retriever().search(q, mode, min(k, 10)))}


@app.post("/api/ask")
def ask(body: Ask, request: Request) -> dict:
    if body.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    rate_limit(request)
    if guardrails.looks_like_injection(body.question):
        return {"answer": guardrails.INJECTION_REFUSAL, "refused": "injection", "sources": []}
    hits = retriever().search(body.question, body.mode, k=4)
    if not guardrails.is_relevant(hits):
        return {"answer": guardrails.REFUSAL, "refused": "off_topic", "sources": _sources(hits)}
    if not llm.llm_enabled():
        return {
            "answer": "LLM answering is not configured on this server (no GROQ_API_KEY), so here are the most relevant passages.",
            "refused": None,
            "llm": False,
            "sources": _sources(hits),
        }
    try:
        answer = llm.generate(body.question, hits)
    except Exception as exc:  # network / quota errors should not 500 the UI
        raise HTTPException(502, f"LLM call failed: {type(exc).__name__}") from exc
    return {"answer": answer, "refused": None, "llm": True, "sources": _sources(hits)}


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")
