"""Evaluate the RAG pipeline on eval/eval_set.json.

  python eval/run_eval.py --retrieval-only   # free: no LLM key needed (retrieval + guardrail only)
  python eval/run_eval.py                    # full: also checks answer keywords and refusals (needs GROQ_API_KEY)

Metrics: Hit@k and MRR of the expected source, keyword accuracy of answers, refusal rate on out-of-scope
and restricted questions (should be escalated, never answered), average latency. Writes eval/results.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402
from app.rag.retriever import HybridRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    if not config.CHUNKS_FILE.exists():
        from app.rag import indexer
        print("Building index first…", indexer.build_index())
    retriever = HybridRetriever()
    cases = json.loads((ROOT / "eval_set.json").read_text(encoding="utf-8"))

    hits, rr, kw_scores, refusals_ok, refusal_total, latencies, rows = [], [], [], 0, 0, [], []
    for case in cases:
        user = {"id": 0, "role": case["role"]}
        level = config.ROLE_LEVEL[case["role"]]
        started = time.time()
        if args.retrieval_only:
            query = case.get("question_en", case["question"])
            found = retriever.search(query, level=level)
            refused = (not found) or found[0]["rerank_score"] < config.MIN_RERANK_SCORE
            sources = [] if refused else [c["source"] for c in found]
            answer = ""
        else:
            from app.rag import pipeline
            res = pipeline.ask(user, case["question"], [], retriever, create_ticket=False)
            refused = res["route"] == "escalated"
            sources = [s["source"] for s in res["sources"]]
            answer = res["answer"].lower()
        latencies.append((time.time() - started) * 1000)

        if case["type"] == "in_scope":
            rank = sources.index(case["expected_source"]) + 1 if case["expected_source"] in sources else None
            hits.append(rank is not None)
            rr.append(1 / rank if rank else 0)
            ok = ""
            if not args.retrieval_only and case["keywords"]:
                score = sum(k.lower() in answer for k in case["keywords"]) / len(case["keywords"])
                kw_scores.append(score)
                ok = f"{score:.0%}"
            rows.append((case["id"], case["type"], case["question"][:55], "✅" if rank else "❌", ok))
        else:
            refusal_total += 1
            refusals_ok += refused
            rows.append((case["id"], case["type"], case["question"][:55], "✅ refused" if refused else "❌ answered", ""))

    def pct(x):
        return f"{x:.1%}"
    summary = [
        ("In-scope questions", sum(c["type"] == "in_scope" for c in cases)),
        ("Hit@k (expected source retrieved)", pct(sum(hits) / len(hits))),
        ("MRR", f"{sum(rr) / len(rr):.3f}"),
        ("Refusal rate (out-of-scope + restricted)", pct(refusals_ok / refusal_total) if refusal_total else "n/a"),
        ("Answer keyword accuracy", pct(sum(kw_scores) / len(kw_scores)) if kw_scores else "n/a (retrieval-only)"),
        ("Avg latency per question", f"{sum(latencies) / len(latencies):.0f} ms"),
    ]
    mode = "retrieval-only" if args.retrieval_only else "full pipeline"
    md = [f"# Evaluation results ({mode})", "", "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary]
    md += ["", "| ID | Type | Question | Result | Keywords |", "|---|---|---|---|---|"]
    md += [f"| {' | '.join(map(str, r))} |" for r in rows]
    (ROOT / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:12]))
    print("\nFull table written to eval/results.md")


if __name__ == "__main__":
    main()
