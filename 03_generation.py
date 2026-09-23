"""Generation slice: ask the questions open-ended instead of multiple choice.

    python 03_generation.py     ->  results/generation.jsonl

Why this exists
---------------
This answers the last part of the supervisor's fourth point: whether a
generation-based variant shows something the multiple-choice setting cannot.

MC1 measures RECOGNITION - the model picks the true answer from a list it is
shown. Generation measures PRODUCTION - what the model says when nothing is
offered. A model can recognise the true option and still produce the
misconception when asked openly, so the multiple-choice result plausibly
UNDERSTATES imitative falsehood. That is the gap this slice probes.

It is deliberately small: every answer has to be read and judged by a human,
so this is a qualitative check on the main result, not a second statistical
test. Do not report a p-value from 50 questions as if it settled anything.

Scoring is NOT done here. 04_score_generation.py builds a blind sheet you
score by hand: no judge model (which would reintroduce exactly the dependency
the MC design avoids) and no BLEURT or cosine similarity (which measure
resemblance to a reference string, not truthfulness).
"""
import json
import os
import time

from config import (DATA_DIR, RESULTS_DIR, TEMPERATURE, SEED, GEN_N, GEN_MODELS,
                    GEN_CONDITIONS, GEN_MAX_TOKENS, GEN_TEMPLATE)
from persona import PERSONAS
from backend import chat

OUT_NAME = "generation.jsonl"


def load_finished(out_file):
    """Keys already collected, so a stopped run can continue."""
    finished = set()
    if not os.path.exists(out_file):
        return finished
    with open(out_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            finished.add((r["model"], r["condition"], r["item_id"]))
    return finished


def main():
    items = json.load(open(f"{DATA_DIR}/truthfulqa_items.json", encoding="utf-8"))[:GEN_N]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, OUT_NAME)
    finished = load_finished(out_file)

    total = len(GEN_MODELS) * len(GEN_CONDITIONS) * len(items)
    print("generation slice: %d model(s) x %d conditions x %d questions = %d calls"
          % (len(GEN_MODELS), len(GEN_CONDITIONS), len(items), total))
    print("already collected: %d" % len(finished))

    n = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for model in GEN_MODELS:
            for condition in GEN_CONDITIONS:
                system = PERSONAS[condition]
                for item in items:
                    n += 1
                    if (model, condition, item["id"]) in finished:
                        continue

                    prompt = GEN_TEMPLATE.format(question=item["question"])
                    try:
                        raw = chat(model, system, prompt, TEMPERATURE, SEED,
                                   GEN_MAX_TOKENS)
                    except Exception as error:
                        print("ERROR", model, condition, item["id"], error)
                        time.sleep(5)
                        continue

                    answers = item.get("gen_answers") or {}
                    out.write(json.dumps({
                        "model": model,
                        "condition": condition,
                        "item_id": item["id"],
                        "category": item["category"],
                        "question": item["question"],
                        "temperature": TEMPERATURE,
                        "seed": SEED,
                        "max_new_tokens": GEN_MAX_TOKENS,
                        "template": "gen",
                        "raw": raw,
                        "generation": raw.strip(),
                        # References for the human scorer, never for auto-scoring.
                        "best_answer": answers.get("best"),
                        "correct_answers": answers.get("correct"),
                        "incorrect_answers": answers.get("incorrect"),
                    }) + "\n")
                    out.flush()

                    if n % 25 == 0:
                        print(" %d/%d" % (n, total), flush=True)
                print("done:", model, condition, flush=True)

    rows = sum(1 for _ in open(out_file, encoding="utf-8"))
    print("\n%s now holds %d rows" % (out_file, rows))
    print("Next: python 04_score_generation.py sheet")


if __name__ == "__main__":
    main()
