"""Offline model-side benchmark — base vs adapter, no inference server.

Run 011 reproduction path. vLLM could not serve Gemma-4-E4B on the available
stack (see log.md → Run 11 → Incidents), so generation runs locally through
Unsloth instead of an OpenAI-compatible endpoint. Scenarios are independent, so
they are batched across the benchmark turn-by-turn for ~6x speedup over single
streams. Judging reuses the benchmark's own judge + rubric, so the numbers are
method-comparable to the served `parrhesia eval` path.

Usage (on a CUDA box, in the training venv):
    python scripts/offline_model_side_eval.py \
        --base google/gemma-4-E4B-it \
        --adapter adapters/parrhesia-gemma4-media \
        --n-per-cat 5 --out-prefix results/run-011-B-gemma4-e4b/model-side

ANTHROPIC_API_KEY must be set (the judge is Claude).
"""
import argparse
import json
import random
from collections import defaultdict

import torch
from unsloth import FastLanguageModel

from parrhesia.benchmark.evaluate import (
    load_scenarios, _build_conversation_turns, judge_responses,
    compute_summary, _strip_thinking_from_responses, DIMENSIONS,
)

JUDGE = "claude-sonnet-4-5-20250929"


def _set_left_pad(tok):
    try:
        tok.padding_side = "left"
    except Exception:
        pass
    if hasattr(tok, "tokenizer"):
        tok.tokenizer.padding_side = "left"


def gen_batched(model, tok, scenarios, batch, max_tok):
    """Generate all scenarios turn-by-turn, batching across scenarios per turn."""
    FastLanguageModel.for_inference(model)
    _set_left_pad(tok)
    turnlists = [_build_conversation_turns(sc) for sc in scenarios]
    max_turns = max(len(t) for t in turnlists)
    convs = [[] for _ in scenarios]
    rts = [[] for _ in scenarios]
    for ti in range(max_turns):
        active = [i for i, tl in enumerate(turnlists) if ti < len(tl)]
        for i in active:
            convs[i].append({"role": "user",
                             "content": [{"type": "text", "text": turnlists[i][ti]["user_message"]}]})
        for b in range(0, len(active), batch):
            idxs = active[b:b + batch]
            inp = tok.apply_chat_template(
                [convs[i] for i in idxs], tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, padding=True,
            ).to("cuda")
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=max_tok, do_sample=False)
            inlen = inp["input_ids"].shape[1]  # uniform: left-padded
            for j, i in enumerate(idxs):
                r = tok.decode(g[j][inlen:], skip_special_tokens=True).strip()
                convs[i].append({"role": "assistant", "content": [{"type": "text", "text": r}]})
                rts[i].append({"turn": turnlists[i][ti]["turn_number"],
                               "user_message": turnlists[i][ti]["user_message"],
                               "assistant_response": r})
        print(f"  turn {ti + 1}/{max_turns}: {len(active)} scenarios", flush=True)
    return [{"scenario_id": scenarios[i]["id"], "category": scenarios[i]["category"],
             "turns": rts[i], "ground_truth": scenarios[i].get("ground_truth")}
            for i in range(len(scenarios))]


def run_model(path, name, scenarios, batch, max_tok, out_prefix):
    print(f"=== {name}: {path} ===", flush=True)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=4096, dtype=None, load_in_4bit=True)
    resp = gen_batched(model, tok, scenarios, batch, max_tok)
    del model
    torch.cuda.empty_cache()
    resp = _strip_thinking_from_responses(resp)
    judged = judge_responses(resp, JUDGE, DIMENSIONS, 8)
    summary = compute_summary(judged, DIMENSIONS)
    json.dump({"model": name, "n": len(scenarios), "summary": summary, "detailed_results": judged},
              open(f"{out_prefix}-{name}.json", "w"), indent=2)
    means = [summary[d]["mean"] for d in DIMENSIONS if summary[d].get("mean") is not None]
    avg = sum(means) / len(means)
    print(f"{name} AVG {avg:.3f} | " +
          " ".join(f"{d.split('_')[0]}={summary[d]['mean']:.2f}" for d in DIMENSIONS), flush=True)
    return avg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter", default="adapters/parrhesia-gemma4-media",
                    help="Local adapter dir (from the training step) or HF repo id")
    ap.add_argument("--scenarios", default=None, help="Scenario JSONL (default: full benchmark set)")
    ap.add_argument("--n-per-cat", type=int, default=5, help="Scenarios sampled per category (0 = all)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tok", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="results/run-011-B-gemma4-e4b/model-side")
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.n_per_cat:
        bycat = defaultdict(list)
        for s in scenarios:
            bycat[s["category"]].append(s)
        rng = random.Random(args.seed)
        subset = []
        for items in bycat.values():
            rng.shuffle(items)
            subset.extend(items[:args.n_per_cat])
        scenarios = subset
    print(f"subset: {len(scenarios)} scenarios", flush=True)

    avgs = {}
    for path, name in [(args.base, "base"), (args.adapter, "media")]:
        avgs[name] = run_model(path, name, scenarios, args.batch, args.max_tok, args.out_prefix)
    print(f"\nDELTA (media - base): {avgs['media'] - avgs['base']:+.3f} "
          f"(base {avgs['base']:.3f} -> media {avgs['media']:.3f})", flush=True)


if __name__ == "__main__":
    main()
