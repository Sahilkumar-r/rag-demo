"""Retrieval evaluation (no API key needed): python eval/run_eval.py

Reports hit@k and MRR at the document level for each retrieval mode, plus a sweep of the
relevance-guardrail threshold (false refusals on in-domain vs. refusals on off-topic questions).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragdemo.retrieval import MODES, Retriever  # noqa: E402

QS = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
K = 5


def main() -> None:
    r = Retriever(ROOT / "data" / "index")
    results: dict = {"questions": len(QS["in_domain"]), "k": K, "modes": {}}
    for mode in MODES:
        ranks = []
        for item in QS["in_domain"]:
            hits = r.search(item["q"], mode, k=K)
            docs = [h.chunk.doc for h in hits]
            ranks.append(next((i + 1 for i, d in enumerate(docs) if d in item["expected"]), None))
        n = len(ranks)
        results["modes"][mode] = {
            "hit@1": sum(1 for x in ranks if x == 1) / n,
            "hit@3": sum(1 for x in ranks if x and x <= 3) / n,
            "hit@5": sum(1 for x in ranks if x and x <= 5) / n,
            "mrr": sum(1 / x for x in ranks if x) / n,
            "misses": [QS["in_domain"][i]["q"] for i, x in enumerate(ranks) if x is None],
        }

    # Guardrail calibration uses the dense similarity of the best hybrid hit.
    in_scores = [max(h.dense for h in r.search(i["q"], "hybrid", k=4)) for i in QS["in_domain"]]
    off_scores = [max(h.dense for h in r.search(q, "hybrid", k=4)) for q in QS["off_topic"]]
    sweep = []
    for t in [x / 100 for x in range(40, 76, 2)]:
        sweep.append({
            "threshold": t,
            "in_domain_refused": sum(s < t for s in in_scores) / len(in_scores),
            "off_topic_refused": sum(s < t for s in off_scores) / len(off_scores),
        })
    results["guardrail_sweep"] = sweep
    results["in_domain_min_max_dense"] = [round(min(in_scores), 3), round(max(in_scores), 3)]
    results["off_topic_min_max_dense"] = [round(min(off_scores), 3), round(max(off_scores), 3)]

    (ROOT / "eval" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{'mode':<15}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>7}")
    for mode, m in results["modes"].items():
        print(f"{mode:<15}{m['hit@1']:>7.2f}{m['hit@3']:>7.2f}{m['hit@5']:>7.2f}{m['mrr']:>7.2f}")
    print("\nguardrail sweep (threshold: in-domain refused / off-topic refused)")
    for s in sweep:
        print(f"  {s['threshold']:.2f}: {s['in_domain_refused']:.2f} / {s['off_topic_refused']:.2f}")
    print("in-domain best-dense range", results["in_domain_min_max_dense"], "off-topic", results["off_topic_min_max_dense"])


if __name__ == "__main__":
    main()
