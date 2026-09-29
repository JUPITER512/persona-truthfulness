# Section 21: the generation slice (open answers, scored by hand).

import csv
import os
import re

from statsmodels.stats.proportion import proportion_confint

from analysis.common import CONDITIONS, PERSONAS, section, write_csv
from analysis.stats import mcnemar_test
from config import RESULTS_DIR
from utils import read_jsonl

GEN_FILE = os.path.join(RESULTS_DIR, "generation.jsonl")
SHEET = os.path.join(RESULTS_DIR, "generation_sheet.csv")
KEY = os.path.join(RESULTS_DIR, "generation_key.csv")

# First-person sentences where the model speaks as the persona ("I work part-time in retail").
# Only counted, never used as a truth score.
IN_ROLE = re.compile(
    r"\b(I am|I'm) an? \d+-year-old|\bI (work|attend|live|grew up)\b|"
    r"\bmy (church|faith|job|work|household|community|background|experience as)\b|"
    r"\bas an? (nurse|software developer|developer|woman|man|mother|conservative|"
    r"liberal|republican|democrat|christian|churchgoer)\b", re.I)


def condition_order(model_and_condition):
    model, condition = model_and_condition
    if condition in CONDITIONS:
        return (model, CONDITIONS.index(condition))
    return (model, 99)


def read_hand_scores():
    """{row_id: "1" / "0" / "x"} from the scoring sheet (empty dict if nothing is scored)."""
    scores = {}
    if os.path.exists(SHEET) and os.path.exists(KEY):
        with open(SHEET, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                value = (row.get("score") or "").strip().lower()
                if value in ("1", "0", "x"):
                    scores[int(row["row_id"])] = value
    return scores


def report_generation_full():
    section("21. GENERATION SLICE - HAND SCORES")
    if not os.path.exists(GEN_FILE):
        print("   generation.jsonl not collected")
        return None
    rows = read_jsonl(GEN_FILE)
    table = {}  # (model, condition) -> one output row (description and hand scores together)

    print("   A. answers in which the model speaks as the persona (description only)")
    for model in sorted(set(row["model"] for row in rows)):
        for c in CONDITIONS:
            sub = [row for row in rows if row["model"] == model and row["condition"] == c]
            if not sub:
                continue
            in_role = sum(1 for row in sub if IN_ROLE.search(row["generation"]))
            print(f"   {model:<14}{c:<11} n={len(sub):<4} speaks as the persona {in_role}")
            table[(model, c)] = [model, c, len(sub), in_role, "", "", "", "", ""]

    scores = read_hand_scores()
    results = None
    if not scores:
        print(f"\n   B. hand scores: none yet. Fill the 'score' column of {SHEET}")
        print("      (1 true, 0 false, x unjudgeable) and run this script again.")
    else:
        with open(KEY, encoding="utf-8-sig") as f:
            key = {int(row["row_id"]): row for row in csv.DictReader(f)}
        scores_by_group = {}  # (model, condition) -> {item_id: score}
        for row_id, value in scores.items():
            k = key[row_id]
            scores_by_group.setdefault((k["model"], k["condition"]), {})[int(k["item_id"])] = value

        print(f"\n   B. hand scores: {len(scores)} of {len(key)} rows scored")
        results = {}
        for group in sorted(scores_by_group, key=condition_order):
            model, c = group
            values = scores_by_group[group]
            n_true = sum(1 for v in values.values() if v == "1")
            n_false = sum(1 for v in values.values() if v == "0")
            n_x = sum(1 for v in values.values() if v == "x")
            if n_true + n_false:
                low, high = proportion_confint(n_true, n_true + n_false, alpha=0.05, method="wilson")
                rate = n_true / (n_true + n_false)
            else:
                low = high = rate = float("nan")
            results[group] = (rate, low, high)
            print(f"   {model:<14}{c:<11} true {n_true:3d}  false {n_false:3d}  x {n_x:3d}"
                  f"   truthful {100 * rate:5.1f}%  [{100 * low:5.1f}, {100 * high:5.1f}]")
            row = table.setdefault(group, [model, c, len(values), "", "", "", "", "", ""])
            row[4:9] = [n_true, n_false, n_x, f"{rate:.4f}", f"[{low:.4f}, {high:.4f}]"]

        print("\n   C. persona vs none on the same questions (McNemar, only descriptive at n = 50)")
        for model in sorted(set(group[0] for group in scores_by_group)):
            base = scores_by_group.get((model, "none"), {})
            for other_condition in ["control"] + PERSONAS:
                other = scores_by_group.get((model, other_condition), {})
                ids = sorted(i for i in set(base) & set(other) if base[i] != "x" and other[i] != "x")
                if not ids:
                    continue
                b, c, p, method = mcnemar_test([int(base[i]) for i in ids], [int(other[i]) for i in ids])
                print(f"      {model:<14} none vs {other_condition:<10} n={len(ids):<3} b={b:<3} c={c:<3} p={p:.4f}")
    write_csv("generation_summary.csv",
              ["model", "condition", "n", "mentions_persona", "true", "false", "unjudgeable",
               "truthful_rate", "ci95"],
              [table[group] for group in sorted(table, key=condition_order)])
    return results
