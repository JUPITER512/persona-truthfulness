# Runs the multiple-choice experiments and saves every answer.
#
#   python run.py <mode>
#
# Modes:
#   pilot                first 50 TruthfulQA questions (quick test)
#   main                 all 817 TruthfulQA questions            -> results/main.jsonl
#   mmlu                 all 1,140 MMLU questions                -> results/mmlu.jsonl
#   templates            100 questions x templates t1, t2, t3     -> results/templates.jsonl
#   shuffle              100 questions, exact repeat of main      -> results/shuffle.jsonl
#   noshuffle            100 questions, options not shuffled      -> results/noshuffle.jsonl
#   temp                 100 questions, temperature 0.7, seeds 1-3 -> results/temp_1/2/3.jsonl
#   control_paraphrases  100 questions x 10 paraphrases of "helpful assistant"
#   extra                persona_d and persona_e, both datasets, t1/t2/t3
#
# A stopped run can simply be started again: answers that are already saved are skipped.
# Never run the same mode twice at the same time (both would write to the same file).

import os
import sys
import time

from backend import chat
from config import (CONTROL_PARAPHRASE_N, EXTRA_PERSONA_KEYS, MAX_NEW_TOKENS, MODELS, RESULTS_DIR,
                    SEED, SENSITIVITY_N, SHUFFLE_DEFAULT, TEMPERATURE)
from persona import CONTROL_PERSONAS, EXTRA_PERSONAS, PERSONAS
from prompting import build_prompt, parse
from utils import append_jsonl, load_items, read_jsonl

TQA = "truthfulqa_items.json"
MMLU = "mmlu_items.json"


def make_key(model, condition, template, item_id, shuffled, temperature, seed):
    # Everything that changes the prompt or the sampling is part of the key,
    # so different settings written to one file never mix up.
    return (model, condition, template, item_id, bool(shuffled), float(temperature), int(seed))


def run(dataset, out_name, template="t1", temperature=TEMPERATURE, seed=SEED,
        shuffle=SHUFFLE_DEFAULT, limit=None, models=None, conditions=None):
    items = load_items(dataset)
    if limit:
        items = items[:limit]
    if models is None:
        models = MODELS
    if conditions is None:
        conditions = PERSONAS  # the five main conditions

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, out_name)

    # Resume: remember which answers are already in the file
    done = set()
    for row in read_jsonl(out_path):
        done.add(make_key(row["model"], row["condition"], row["template"], row["item_id"],
                          row["shuffled"], row["temperature"], row["seed"]))

    total = len(models) * len(conditions) * len(items)
    count = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for model in models:
            for condition, system_prompt in conditions.items():
                for item in items:
                    count += 1
                    key = make_key(model, condition, template, item["id"], shuffle, temperature, seed)
                    if key in done:
                        continue

                    # The question id is the shuffle seed, so the option order is the same everywhere
                    if shuffle:
                        shuffle_seed = item["id"]
                    else:
                        shuffle_seed = None
                    prompt, correct_letter = build_prompt(item, template, shuffle_seed)

                    try:
                        raw = chat(model, system_prompt, prompt, temperature, seed, MAX_NEW_TOKENS)
                    except Exception as error:
                        print("ERROR", key, error)
                        time.sleep(5)
                        continue  # this answer is tried again on the next run

                    letter = parse(raw, item["n_options"])
                    if letter is None:
                        outcome = "invalid"   # kept and reported, never dropped
                    elif letter == correct_letter:
                        outcome = "correct"
                    else:
                        outcome = "incorrect"

                    append_jsonl(out, {
                        "model": model,
                        "condition": condition,
                        "template": template,
                        "item_id": item["id"],
                        "category": item["category"],
                        "n_options": item["n_options"],
                        "temperature": temperature,
                        "seed": seed,
                        "shuffled": shuffle,
                        "raw": raw,
                        "parsed": letter,
                        "correct_letter": correct_letter,
                        "outcome": outcome,
                    })
                    if count % 50 == 0:
                        print(f" {count}/{total}", flush=True)
                print("done:", model, condition, template, flush=True)


def main():
    if len(sys.argv) < 2:
        print("usage: python run.py <mode>")
        print("modes: pilot main mmlu templates shuffle noshuffle temp control_paraphrases extra")
        return
    mode = sys.argv[1]

    if mode == "pilot":
        run(TQA, "pilot.jsonl", limit=50)

    elif mode == "main":
        run(TQA, "main.jsonl")

    elif mode == "mmlu":
        run(MMLU, "mmlu.jsonl")

    elif mode == "templates":
        # t1 is included so all three formats are compared on the same 100 questions
        for template in ["t1", "t2", "t3"]:
            run(TQA, "templates.jsonl", template=template, limit=SENSITIVITY_N)

    elif mode == "shuffle":
        # same settings as main: shows whether a repeat gives the same answers
        run(TQA, "shuffle.jsonl", shuffle=True, limit=SENSITIVITY_N)

    elif mode == "noshuffle":
        # options in their original order: the correct answer is then always "A"
        run(TQA, "noshuffle.jsonl", shuffle=False, limit=SENSITIVITY_N)

    elif mode == "temp":
        for s in [1, 2, 3]:
            run(TQA, f"temp_{s}.jsonl", temperature=0.7, seed=s, limit=SENSITIVITY_N)

    elif mode == "control_paraphrases":
        names = list(CONTROL_PERSONAS)[:CONTROL_PARAPHRASE_N]
        paraphrases = {name: CONTROL_PERSONAS[name] for name in names}
        run(TQA, "control_paraphrases.jsonl", limit=SENSITIVITY_N, conditions=paraphrases)

    elif mode == "extra":
        extra = {name: EXTRA_PERSONAS[name] for name in EXTRA_PERSONA_KEYS}
        for template in ["t1", "t2", "t3"]:
            run(TQA, "extra_personas_tqa.jsonl", conditions=extra, template=template)
            run(MMLU, "extra_personas_mmlu.jsonl", conditions=extra, template=template)

    else:
        print("unknown mode:", mode)
        print("modes: pilot main mmlu templates shuffle noshuffle temp control_paraphrases extra")


if __name__ == "__main__":
    main()
