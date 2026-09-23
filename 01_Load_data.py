from datasets import load_dataset
from collections import Counter
import json, os
from config import DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

print("downloading TruthfulQA ...")

gen = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]

mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]

category_of = {r["question"].strip(): r["category"] for r in gen}

answers_of = { r["question"].strip(): 
              { 
                  "best": r["best_answer"],
                  "correct": r["correct_answers"],
                  "incorrect": r["incorrect_answers"]} for r in gen}


items = []

for row in mc:
    q = row["question"].strip()
    choices = row["mc1_targets"]["choices"]
    labels = row["mc1_targets"]["labels"]
    items.append({
                    "id": len(items),
                    "question": q,
                    "choices": choices,
                    "correct_index": labels.index(1),
                    "n_options": len(choices),
                    "category": category_of.get(q, "UNKNOWN"),
                    "gen_answers": answers_of.get(q),
                    })

with open(f"{DATA_DIR}/truthfulqa_items.json", "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2)

print("items:", len(items))
print("unknown category:", sum(1 for i in items if i["category"] == "UNKNOWN"))

chance = sum(1 / i["n_options"] for i in items) / len(items)

print(f"mean chance baseline:{chance}")
print("\ncategory counts")

for cat, n in Counter(i["category"] for i in items).most_common():
    print(" ", n, cat)