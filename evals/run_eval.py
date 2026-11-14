"""Benchmark the agent on evals/questions.jsonl (needs ANTHROPIC_API_KEY).

    python evals/run_eval.py                 # all questions
    python evals/run_eval.py --limit 3       # quick check

Scoring:
  number  -> some number in the answer is within 1% of the expected value
  label   -> the expected word appears in the answer (case-insensitive)
  warning -> the run raised that trap warning (from check_findings or the automatic check)
Results go to evals/results/<timestamp>.json; paste the summary table into the README.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst_agent.agent.llm import AnthropicLLM  # noqa: E402
from analyst_agent.agent.loop import Agent  # noqa: E402
from analyst_agent.analyst.verify import numbers_in  # noqa: E402
from analyst_agent.data.store import Session  # noqa: E402
from analyst_agent.glossary import Glossary  # noqa: E402


def score(item: dict, run) -> bool:
    exp = item["expected"]
    if exp["type"] == "number":
        return any(abs(v - exp["value"]) <= 0.01 * abs(exp["value"]) or
                   (s == "%" and abs(v - exp["value"]) <= 0.05)
                   for _, v, _, s in numbers_in(run.answer))
    if exp["type"] == "label":
        return exp["value"].lower() in run.answer.lower()
    if exp["type"] == "warning":
        return any(w["type"] == exp["value"] for w in run.warnings)
    raise ValueError(exp["type"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model")
    args = ap.parse_args()

    items = [json.loads(line) for line in (ROOT / "evals" / "questions.jsonl").read_text().splitlines() if line]
    items = items[: args.limit] if args.limit else items
    llm = AnthropicLLM(model=args.model) if args.model else AnthropicLLM()
    glossary = Glossary(ROOT / "evals" / ".eval_glossary.sqlite")

    rows = []
    for item in items:
        session = Session()                     # fresh session per question: no leakage
        session.load_file(ROOT / "sample_data" / "retail_orders.csv")
        t = time.time()
        try:
            run = Agent(session, llm, glossary).ask(item["question"])
            ok = score(item, run)
            rows.append({"id": item["id"], "category": item["category"], "correct": ok,
                         "steps": len(run.steps), "errors": sum(1 for s in run.steps if s.error),
                         "verified": run.verification.get("verified"), "tokens_in": run.tokens_in,
                         "tokens_out": run.tokens_out, "latency_s": round(time.time() - t, 1),
                         "answer": run.answer})
        except Exception as e:
            rows.append({"id": item["id"], "category": item["category"], "correct": False, "crash": str(e)})
        print(f"{'PASS' if rows[-1]['correct'] else 'FAIL'}  {item['id']}  {item['question']}")

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["correct"])
    print("\n| category | correct | total | accuracy |\n|---|---|---|---|")
    for cat, vals in sorted(by_cat.items()):
        print(f"| {cat} | {sum(vals)} | {len(vals)} | {sum(vals) / len(vals):.0%} |")
    total = [r["correct"] for r in rows]
    print(f"| **all** | {sum(total)} | {len(total)} | {sum(total) / len(total):.0%} |")

    out = ROOT / "evals" / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
