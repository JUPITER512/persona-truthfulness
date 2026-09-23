"""Build the MMLU sample: MMLU_PER_SUBJECT_LARGE questions from each subject.

    python 02_MMLU_data.py        ->  data/mmlu_items.json   (57 x 20 = 1,140 items)

Why the sample is drawn in two halves
-------------------------------------
MMLU's test split holds about 14,000 questions, far too many to run five
conditions against on a laptop, so a fixed number is taken from each of the 57
subjects. Sampling per subject rather than at random keeps small subjects
represented instead of letting large ones dominate.

The draw happens in two passes, and this is deliberate rather than
complicated. The study first ran with 5 questions per subject (285 items), and
17,100 responses were collected against those item ids. Python's
`random.sample` is not nested: drawing 20 per subject with the same seed does
NOT give you the first 5 plus 15 more, it gives a different set entirely - so
simply raising the number would have renumbered every item, and id 7 would
have come to mean a different question than the collected rows assumed.

So the sample is defined as:

    ids 0-284     MMLU_PER_SUBJECT       per subject, Random(MMLU_SEED)
    ids 285-1139  the remainder up to    per subject, Random(MMLU_SEED + 1),
                  MMLU_PER_SUBJECT_LARGE drawn from that subject's leftovers

Both halves are fixed by seed alone, so this file reproduces the identical
dataset on any machine, and running it twice changes nothing. Ids assigned in
the first pass keep their meaning forever, which is what lets already
collected responses stay valid.

Ids are therefore not contiguous per subject: each subject owns one low block
and one high block. Nothing depends on id ranges - grouping is by the
`category` field - but it is worth knowing when reading the raw file.
"""
import json
import os
import random
import collections

from datasets import load_dataset

from config import DATA_DIR, MMLU_PER_SUBJECT, MMLU_PER_SUBJECT_LARGE, MMLU_SEED

OUT = os.path.join(DATA_DIR, "mmlu_items.json")


def as_item(row, item_id, subject):
    return {
        "id": item_id,
        "question": row["question"],
        "choices": row["choices"],
        "correct_index": row["answer"],
        "n_options": len(row["choices"]),
        "category": subject,
    }


def build(by_subject):
    """Return the full item list, first pass then second pass."""
    items = []

    # --- pass 1: the original sample -----------------------------------------
    rng = random.Random(MMLU_SEED)
    first = {}
    for subject, rows in sorted(by_subject.items()):
        chosen = rng.sample(rows, min(MMLU_PER_SUBJECT, len(rows)))
        first[subject] = {r["question"] for r in chosen}
        for r in chosen:
            items.append(as_item(r, len(items), subject))
    n_first = len(items)

    # --- pass 2: top each subject up to the larger size ----------------------
    rng2 = random.Random(MMLU_SEED + 1)
    need = MMLU_PER_SUBJECT_LARGE - MMLU_PER_SUBJECT
    if need > 0:
        for subject, rows in sorted(by_subject.items()):
            leftover = [r for r in rows if r["question"] not in first[subject]]
            for r in rng2.sample(leftover, min(need, len(leftover))):
                items.append(as_item(r, len(items), subject))

    return items, n_first


def check_ids_unchanged(items):
    """Refuse to write if an existing id would come to mean a different question.

    Collected responses are keyed by item_id. If this script ever produced a
    different question for an id already on disk, every row referring to that
    id would silently become wrong. Aborting is always better than that.
    """
    if not os.path.exists(OUT):
        print("no existing dataset to compare against")
        return
    old = json.load(open(OUT, encoding="utf-8"))
    changed = 0
    for o in old:
        if o["id"] >= len(items):
            changed += 1
            continue
        n = items[o["id"]]
        if (o["question"] != n["question"] or o["choices"] != n["choices"]
                or o["correct_index"] != n["correct_index"]):
            changed += 1
    print("existing ids that would change meaning: %d" % changed)
    if changed:
        raise SystemExit(
            "ABORTED: this would invalidate the collected responses in "
            "results/mmlu.jsonl, which are keyed by item id. Nothing written.")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("downloading MMLU ... (about 170 MB the first time, then cached)")
    mmlu = load_dataset("cais/mmlu", "all")["test"]

    by_subject = {}
    for row in mmlu:
        by_subject.setdefault(row["subject"], []).append(row)
    print("subjects: %d | questions available: %d"
          % (len(by_subject), sum(len(v) for v in by_subject.values())))

    items, n_first = build(by_subject)
    print("pass 1: %d items (ids 0-%d), %d per subject"
          % (n_first, n_first - 1, MMLU_PER_SUBJECT))
    print("pass 2: %d more (ids %d-%d), up to %d per subject"
          % (len(items) - n_first, n_first, len(items) - 1, MMLU_PER_SUBJECT_LARGE))

    check_ids_unchanged(items)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    per = collections.Counter(i["category"] for i in items)
    chance = sum(1.0 / i["n_options"] for i in items) / len(items)
    print("\nwrote %s" % OUT)
    print("  items    : %d" % len(items))
    print("  subjects : %d (min %d, max %d per subject)"
          % (len(per), min(per.values()), max(per.values())))
    print("  chance   : %.4f" % chance)
    print("\nNext: python run.py mmlu")


if __name__ == "__main__":
    main()
