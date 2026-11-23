"""Ablation: does the repo map in the prompt cut model rounds?

    python -m evals.ablate [--repo URL] [--repeats N]

Each repo is set up once (clone + map + overview). Every question then runs in a
fresh conversation twice, once with the map in the system prompt and once without
(the overview stays in both, so only the map varies). Arms are interleaved so API
latency drift hits both equally. Raw rows go to evals/results/.
"""

import argparse
import asyncio
import json
import shutil
import statistics as st
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
# Haiku 5.5 list prices, $/MTok, prompts under 100K tokens
PRICE = {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write": 0.125}


def cost(r: dict) -> float:
    return (r["input_tokens"] * PRICE["in"] + r["output_tokens"] * PRICE["out"]
            + r["cache_read"] * PRICE["cache_read"] + r["cache_write"] * PRICE["cache_write"]) / 1e6


async def run_one(s, question: str, use_map: bool, full_map: str) -> dict:
    from server.agent import answer
    s.history, s.focus = [], {}
    s.repo_map = full_map if use_map else ""

    async def noop(_):
        pass

    try:
        stats = await answer(s, question, noop, noop)
    except Exception as e:  # keep going; record the failure
        return {"error": repr(e)}
    return {
        "rounds": stats.rounds, "tool_calls": len(stats.tool_calls),
        "tools": [n for n, _ in stats.tool_calls], "shown": stats.shown,
        "first_text_s": stats.first_text_s, "total_s": stats.total_s,
        "input_tokens": stats.input_tokens, "output_tokens": stats.output_tokens,
        "cache_read": stats.cache_read, "cache_write": stats.cache_write,
        "stop_reasons": stats.stop_reasons, "answer": stats.text.strip(),
    }


def summarize(rows: list[dict], arm: str) -> dict:
    rs = [r for r in rows if r["arm"] == arm and "error" not in r]
    if not rs:
        return {}
    ft = [r["first_text_s"] for r in rs if r["first_text_s"] is not None]
    return {
        "n": len(rs),
        "rounds_mean": st.mean(r["rounds"] for r in rs),
        "rounds_median": st.median(r["rounds"] for r in rs),
        "tools_mean": st.mean(r["tool_calls"] for r in rs),
        "first_text_p50": st.median(ft) if ft else None,
        "total_p50": st.median(r["total_s"] for r in rs),
        "cost_per_q": st.mean(cost(r) for r in rs),
        "errors": sum(1 for r in rows if r["arm"] == arm and "error" in r),
    }


def print_table(title: str, rows: list[dict]):
    print(f"\n{title}")
    print(f"  {'arm':<7} {'n':>3} {'rounds':>7} {'med':>4} {'tools':>6} {'1st text':>9} {'total':>7} {'$/q':>9}")
    for arm in ("map", "no_map"):
        m = summarize(rows, arm)
        if not m:
            continue
        ft = f"{m['first_text_p50']:.2f}s" if m["first_text_p50"] is not None else "-"
        print(f"  {arm:<7} {m['n']:>3} {m['rounds_mean']:>7.2f} {m['rounds_median']:>4.0f} "
              f"{m['tools_mean']:>6.2f} {ft:>9} {m['total_p50']:>6.2f}s {m['cost_per_q']:>9.5f}"
              + (f"  ({m['errors']} errors)" if m["errors"] else ""))


async def main():
    load_dotenv()
    from server.session import Session
    from server.setup import setup_repo

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="only run questions for this repo URL")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    args = ap.parse_args()

    qs = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    qs = [q for q in qs if q.get("bucket") != "follow-up"]   # single-turn only here
    if args.repo:
        qs = [q for q in qs if q["repo"].rstrip("/") == args.repo.rstrip("/")]
    repos = list(dict.fromkeys(q["repo"] for q in qs))

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"ablation-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    all_rows = []
    with out_path.open("w") as out:
        for repo in repos:
            s = Session(repo_url=repo)
            t = time.monotonic()
            await setup_repo(s)
            print(f"\n{s.repo_name} @ {s.commit[:10]}: setup {time.monotonic() - t:.1f}s, "
                  f"map ~{len(s.repo_map) // 4} tokens")
            full_map = s.repo_map
            rows = []
            try:
                for q in (q for q in qs if q["repo"] == repo):
                    for rep in range(args.repeats):
                        arms = ("map", "no_map") if rep % 2 == 0 else ("no_map", "map")
                        for arm in arms:
                            r = await run_one(s, q["question"], arm == "map", full_map)
                            r.update(repo=repo, commit=s.commit, question=q["question"],
                                     bucket=q.get("bucket"), arm=arm, rep=rep)
                            rows.append(r)
                            out.write(json.dumps(r) + "\n")
                            out.flush()
                            status = r.get("error") or f"rounds={r['rounds']} tools={r['tool_calls']} total={r['total_s']:.1f}s"
                            print(f"  [{arm:<6}] {q['question'][:55]:<55} {status}")
            finally:
                shutil.rmtree(s.repo_path, ignore_errors=True)
            print_table(f"{s.repo_name}", rows)
            all_rows += rows
    if len(repos) > 1:
        print_table("ALL REPOS", all_rows)
    print(f"\nraw rows: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
