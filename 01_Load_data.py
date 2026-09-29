# Step 1: download TruthfulQA and save it as data/truthfulqa_items.json
#
#   python 01_Load_data.py
#
# Uses the MC1 version: every question has exactly one correct option (2 to 13 options).

import json
import os
from collections import Counter

from datasets import load_dataset

from config import DATA_DIR


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("downloading TruthfulQA ...")
    generation = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    multiple_choice = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]

    # The multiple-choice version has no categories and no reference answers,
    # so both are taken from the generation version (matched by question text).
    category_of = {}
    answers_of = {}
    for row in generation:
        question = row["question"].strip()
        category_of[question] = row["category"]
        answers_of[question] = {
            "best": row["best_answer"],
            "correct": row["correct_answers"],
            "incorrect": row["incorrect_answers"],
        }

    items = []
    for row in multiple_choice:
        question = row["question"].strip()
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]  # 1 = correct, 0 = wrong
        items.append({
            "id": len(items),
            "question": question,
            "choices": choices,
            "correct_index": labels.index(1),
            "n_options": len(choices),
            "category": category_of.get(question, "UNKNOWN"),
            "gen_answers": answers_of.get(question),
        })

    path = os.path.join(DATA_DIR, "truthfulqa_items.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    unknown = 0
    for item in items:
        if item["category"] == "UNKNOWN":
            unknown += 1
    chance = sum(1 / item["n_options"] for item in items) / len(items)

    print("wrote", path)
    print("items:", len(items))
    print("unknown category:", unknown)
    print(f"chance accuracy: {chance:.4f}")
    print("\nquestions per category:")
    for category, n in Counter(item["category"] for item in items).most_common():
        print(f"  {n:4d}  {category}")


if __name__ == "__main__":
    main()
