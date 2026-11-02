# FastAPI Docs Assistant (RAG demo)

A retrieval-augmented generation (RAG) app that answers questions about the [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/), cites the passages it used, and refuses when the docs don't cover the question.

**Status:** working code with tests and a measured retrieval evaluation. It is **not deployed yet** (no public demo URL). Answer generation uses Groq and has only been tested with a mocked LLM; see [Limitations](#limitations).

## How it works

```
question ──► injection check ──► retrieve ──► relevance guardrail ──► Groq LLM ──► answer + [n] citations
                                    │
              BM25 (keywords) ──┐   │
                                ├─► Reciprocal Rank Fusion (k=60) ──► (optional cross-encoder rerank)
        dense bge-small ────────┘
```

- **Corpus:** 40 tutorial pages from the FastAPI docs (MIT), split by markdown heading into 434 chunks (~180 words max). Heading path is kept with each chunk.
- **Retrieval:** BM25 (`rank-bm25`) plus dense embeddings (`BAAI/bge-small-en-v1.5` via ONNX/fastembed, brute-force cosine in NumPy), fused with Reciprocal Rank Fusion. Four selectable modes: `bm25`, `dense`, `hybrid`, `hybrid_rerank`.
- **Guardrails:** refuses when the best chunk similarity is below a calibrated threshold (0.67); a regex check blocks obvious "ignore previous instructions" prompts; retrieved text is fenced in the prompt and labelled untrusted. These are baseline protections, not a complete defence.
- **Generation:** Groq chat model (default `llama-3.3-70b-versatile`, override with `GROQ_MODEL`), temperature 0.1, instructed to cite sources and say when it can't find the answer.
- **API/UI:** FastAPI (`/api/ask`, `/api/search`, `/api/health`), per-IP rate limit, and a small dependency-free chat UI.

## Evaluation (retrieval only, no API key needed)

30 hand-written questions, each mapped to the page(s) that answer it ([`eval/questions.json`](eval/questions.json)). A hit means a correct page is among the top-k chunks.

| Mode | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| BM25 | 0.70 | 0.87 | 0.90 | 0.77 |
| Dense | 0.83 | 0.90 | 0.93 | 0.87 |
| **Hybrid (default)** | 0.80 | 0.97 | **1.00** | **0.89** |
| Hybrid + rerank | 0.73 | 0.93 | 0.93 | 0.82 |

- Hybrid retrieval is best overall here: keyword and semantic search miss different questions. The off-the-shelf MS MARCO cross-encoder did **not** help on this set, so it is not the default.
- **Guardrail:** best-chunk similarity ranged 0.694-0.854 for the 30 in-domain questions and 0.440-0.646 for 8 off-topic ones, so the 0.67 threshold wrongly refused none of the in-domain questions and refused all 8 off-topic ones.

Reproduce: `python eval/run_eval.py` (writes `eval/results.json`).

## Limitations

- The question set is small and written by the author, so treat the numbers as indicative, not a benchmark. The guardrail threshold was tuned on the same set and may not transfer.
- Questions about FastAPI topics that are **not** in the 40 indexed pages (for example OAuth2/JWT or `Depends`) can still pass the guardrail because they are semantically close to indexed pages.
- Answer quality and citation accuracy have not been measured. Generation has been tested only with a mocked LLM, not against the live Groq API in this repo's tests.
- No answer-faithfulness evaluation (for example RAGAS), no conversation memory, and an in-memory NumPy index, which is fine for hundreds of chunks but not millions.

## Run it

Requires Python 3.12 (3.14 lacks some ML wheels at the time of writing).

```bash
pip install -r requirements.txt
python scripts/fetch_corpus.py      # downloads the docs (optional; the built index is committed)
python scripts/build_index.py       # only if you changed the corpus; takes a few minutes on CPU
export GROQ_API_KEY=...             # optional; without it the app shows retrieved passages only
uvicorn ragdemo.api:app --app-dir src --port 7860
pytest                              # 17 tests
```

If the Hugging Face model download hangs, try `HF_HUB_DISABLE_XET=1`.

## Deploy

A `Dockerfile` is included (listens on port 7860, which Hugging Face Spaces expects). Set `GROQ_API_KEY` as a secret on your host, never in the repo. The built-in rate limit (15 requests/min/IP) helps protect a free-tier key.

## License and attribution

Code: MIT. Indexed text comes from the FastAPI documentation (MIT); see [NOTICE.md](NOTICE.md).
