# Settings and small helper functions used by all analysis files.

import csv
import json
import os

import numpy as np

from config import RESULTS_DIR

OUT_DIR = os.path.join(RESULTS_DIR, "analysis")   # tables (.csv)
FIG_DIR = os.path.join(RESULTS_DIR, "figures")    # figures (.png and .pdf)

CONDITIONS = ["none", "control", "persona_a", "persona_b", "persona_c"]
PERSONAS = ["persona_a", "persona_b", "persona_c"]
BASELINES = ["none", "control"]

# Two analysis choices (both stated in the paper):
MIN_CATEGORY_N = 1           # smallest TruthfulQA category that is tested (1 = all 38)
INVALID_AS_INCORRECT = True  # a reply without a letter counts as not correct

# Bootstrap: 10,000 resamples of the questions, fixed seed so every run gives the same numbers
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 0

# One entry per loaded file, used for the data inventory (section 0)
INTEGRITY = []


def load(filename, quiet=False):
    """Read one results file. If the same answer was saved twice, keep the first one.

    Duplicates can appear when two runs write to the same file at once.
    They are counted (and reported in section 0), not silently ignored.
    """
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return []

    rows = []
    first_row = {}
    duplicates = 0
    conflicts = 0  # duplicates with a different answer
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["model"], row["condition"], row["template"], row["item_id"],
                   row["shuffled"], row["temperature"], row["seed"])
            if key in first_row:
                duplicates += 1
                if first_row[key]["parsed"] != row["parsed"]:
                    conflicts += 1
                continue
            first_row[key] = row
            rows.append(row)

    if not quiet:
        INTEGRITY.append((filename, len(rows) + duplicates, len(rows), duplicates, conflicts))
    return rows


def is_correct(row):
    """1 = correct, 0 = not correct, None = left out of the accuracy."""
    if row["outcome"] == "correct":
        return 1
    if row["outcome"] == "invalid" and not INVALID_AS_INCORRECT:
        return None
    return 0


def accuracy(rows):
    """Return (accuracy, number of answers counted)."""
    values = [is_correct(row) for row in rows]
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), 0
    return sum(values) / len(values), len(values)


def models_of(rows):
    return sorted(set(row["model"] for row in rows))


def subset(rows, model=None, condition=None, template=None):
    """Keep only the rows of one model / condition / template (None = all)."""
    result = []
    for row in rows:
        if model is not None and row["model"] != model:
            continue
        if condition is not None and row["condition"] != condition:
            continue
        if template is not None and row["template"] != template:
            continue
        result.append(row)
    return result


def chance_baseline(rows):
    """Accuracy of random guessing: the mean of 1 / number of options."""
    options_per_item = {row["item_id"]: row["n_options"] for row in rows}
    return sum(1.0 / n for n in options_per_item.values()) / len(options_per_item)


def write_csv(name, header, table):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def paired_vectors(rows, model, condition_a, condition_b):
    """0/1 correctness of one model under two conditions, on the questions both have."""
    a = {row["item_id"]: is_correct(row) for row in subset(rows, model, condition_a)}
    b = {row["item_id"]: is_correct(row) for row in subset(rows, model, condition_b)}
    ids = sorted(i for i in set(a) & set(b) if a[i] is not None and b[i] is not None)
    return [a[i] for i in ids], [b[i] for i in ids]


def pair_by_key(rows_a, rows_b, key):
    """Like paired_vectors, but rows are matched with a key function."""
    a = {key(row): is_correct(row) for row in rows_a}
    b = {key(row): is_correct(row) for row in rows_b}
    keys = sorted(k for k in set(a) & set(b) if a[k] is not None and b[k] is not None)
    return [a[k] for k in keys], [b[k] for k in keys]


def correctness_matrix(rows, model, conditions=None):
    """{condition: array of 1.0 / 0.0 over the same sorted question ids}.

    All conditions use the same ids, so one bootstrap resample can index all of
    them and the comparison stays paired.
    """
    if conditions is None:
        conditions = CONDITIONS
    per_condition = {}
    for c in conditions:
        per_condition[c] = {row["item_id"]: is_correct(row) for row in subset(rows, model, c)}
    ids = sorted(set.intersection(*[set(d) for d in per_condition.values()]))

    matrix = {}
    for c in conditions:
        values = []
        for i in ids:
            v = per_condition[c][i]
            values.append(np.nan if v is None else float(v))
        matrix[c] = np.array(values)
    return matrix
