"""Retrieval latency of the rag-demo, per mode, on the 30 evaluation questions (CPU only, no LLM call)."""
import json, sys, time, statistics, platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ragdemo.retrieval import MODES, Retriever  # noqa: E402

qs = [q["q"] for q in json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))["in_domain"]]
t0 = time.perf_counter()
r = Retriever(ROOT / "data" / "index")
load_s = time.perf_counter() - t0
# Warm up each path once (model graphs, caches), then time every question.
for mode in MODES:
    r.search(qs[0], mode, k=5)

out = {"machine": platform.processor() or platform.machine(), "python": platform.python_version(),
       "chunks": len(r.chunks), "docs": len({c.doc for c in r.chunks}), "load_seconds": round(load_s, 1), "modes": {}}
for mode in MODES:
    times = []
    for q in qs:
        t = time.perf_counter(); r.search(q, mode, k=5); times.append((time.perf_counter() - t) * 1000)
    times.sort()
    out["modes"][mode] = {"p50_ms": round(statistics.median(times)), "p95_ms": round(times[int(0.95 * len(times)) - 1]), "max_ms": round(times[-1])}
    print(mode, out["modes"][mode], flush=True)
print(json.dumps({k: v for k, v in out.items() if k != "modes"}))
json.dump(out, open(sys.argv[1], "w"), indent=2)
