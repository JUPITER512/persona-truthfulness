# Step 2: take a fixed sample of MMLU and save it as data/mmlu_items.json
#
#   python 02_MMLU_data.py
#
# 20 questions from each of the 57 subjects = 1,140 questions.
#
# The sample is drawn in two passes. The study first used 5 questions per subject
# (ids 0-284) and answers were already collected for them. Drawing 20 at once with
# the same seed would give different questions, so the first 5 are kept and each
# subject is topped up with 15 more in a second pass (ids 285-1139).

import json
import os
import random
from collections import Counter

from datasets import load_dataset

from config import DATA_DIR, MMLU_PER_SUBJECT, MMLU_PER_SUBJECT_LARGE, MMLU_SEED

OUT_PATH = os.path.join(DATA_DIR, "mmlu_items.json")


def make_item(row, item_id, subject):
    return {
        "id": item_id,
        "question": row["question"],
        "choices": row["choices"],
        "correct_index": row["answer"],
        "n_options": len(row["choices"]),
        "category": subject,
    }


def build_sample(rows_by_subject):
    items = []

    # Pass 1: 5 questions per subject (ids 0-284)
    rng = random.Random(MMLU_SEED)
    chosen_questions = {}
    for subject, rows in sorted(rows_by_subject.items()):
        chosen = rng.sample(rows, min(MMLU_PER_SUBJECT, len(rows)))
        chosen_questions[subject] = {row["question"] for row in chosen}
        for row in chosen:
            items.append(make_item(row, len(items), subject))
    print(f"pass 1: {len(items)} items, {MMLU_PER_SUBJECT} per subject")

    # Pass 2: top every subject up to 20 questions, from the questions not chosen yet
    rng2 = random.Random(MMLU_SEED + 1)
    need = MMLU_PER_SUBJECT_LARGE - MMLU_PER_SUBJECT
    for subject, rows in sorted(rows_by_subject.items()):
        leftover = [row for row in rows if row["question"] not in chosen_questions[subject]]
        for row in rng2.sample(leftover, min(need, len(leftover))):
            items.append(make_item(row, len(items), subject))
    print(f"pass 2: {len(items)} items in total, {MMLU_PER_SUBJECT_LARGE} per subject")

    return items


def check_ids_unchanged(items):
    """Stop if an existing item id would point to a different question.
    The stored answers use these ids, so a changed id would make them wrong."""
    if not os.path.exists(OUT_PATH):
        return
    with open(OUT_PATH, encoding="utf-8") as f:
        old_items = json.load(f)
    changed = 0
    for old in old_items:
        if old["id"] >= len(items):
            changed += 1
            continue
        new = items[old["id"]]
        if (old["question"] != new["question"] or old["choices"] != new["choices"]
                or old["correct_index"] != new["correct_index"]):
            changed += 1
    if changed:
        raise SystemExit(f"STOPPED: {changed} existing ids would change meaning. Nothing was written.")
    print("existing ids unchanged")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("downloading MMLU ... (about 170 MB the first time)")
    mmlu = load_dataset("cais/mmlu", "all")["test"]

    rows_by_subject = {}
    for row in mmlu:
        rows_by_subject.setdefault(row["subject"], []).append(row)
    print(f"subjects: {len(rows_by_subject)}")

    items = build_sample(rows_by_subject)
    check_ids_unchanged(items)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    per_subject = Counter(item["category"] for item in items)
    chance = sum(1 / item["n_options"] for item in items) / len(items)
    print("wrote", OUT_PATH)
    print(f"items: {len(items)} | subjects: {len(per_subject)} | chance: {chance:.4f}")


if __name__ == "__main__":
    main()
