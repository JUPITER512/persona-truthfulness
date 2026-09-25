# Step 4: generation slice. Mistral answers the first 50 TruthfulQA questions
# in its own words (no answer options), under all five conditions.
#
#   python 03_generation.py   ->  results/generation.jsonl  (250 answers)
#
# Multiple choice tests whether the model RECOGNISES the true answer.
# This checks what the model SAYS when no options are given.
# The answers are scored by hand afterwards: see 04_score_generation.py.

import os
import time

from backend import chat
from config import (GEN_CONDITIONS, GEN_MAX_TOKENS, GEN_MODELS, GEN_N, GEN_TEMPLATE, RESULTS_DIR,
                    SEED, TEMPERATURE)
from persona import PERSONAS
from utils import append_jsonl, load_items, read_jsonl

OUT_PATH = os.path.join(RESULTS_DIR, "generation.jsonl")


def main():
    items = load_items("truthfulqa_items.json")[:GEN_N]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resume: skip answers that are already saved
    done = set()
    for row in read_jsonl(OUT_PATH):
        done.add((row["model"], row["condition"], row["item_id"]))

    total = len(GEN_MODELS) * len(GEN_CONDITIONS) * len(items)
    print(f"generation slice: {total} answers ({len(done)} already saved)")

    count = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for model in GEN_MODELS:
            for condition in GEN_CONDITIONS:
                system_prompt = PERSONAS[condition]
                for item in items:
                    count += 1
                    if (model, condition, item["id"]) in done:
                        continue

                    prompt = GEN_TEMPLATE.format(question=item["question"])
                    try:
                        raw = chat(model, system_prompt, prompt, TEMPERATURE, SEED, GEN_MAX_TOKENS)
                    except Exception as error:
                        print("ERROR", model, condition, item["id"], error)
                        time.sleep(5)
                        continue

                    # TruthfulQA's reference answers are saved for the human scorer only
                    answers = item.get("gen_answers") or {}
                    append_jsonl(out, {
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
                        "best_answer": answers.get("best"),
                        "correct_answers": answers.get("correct"),
                        "incorrect_answers": answers.get("incorrect"),
                    })
                    if count % 25 == 0:
                        print(f" {count}/{total}", flush=True)
                print("done:", model, condition, flush=True)

    print("saved in", OUT_PATH)
    print("next: python 04_score_generation.py sheet")


if __name__ == "__main__":
    main()
