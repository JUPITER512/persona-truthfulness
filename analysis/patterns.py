# Sections 15-16: kinds of invalid MMLU replies (and invalid rate by domain), parser check.

import re
from collections import Counter

import numpy as np

from analysis.common import CONDITIONS, PERSONAS, accuracy, load, models_of, section, subset, write_csv
from analysis.mmlu_domains import DOMAIN_OF, MMLU_DOMAINS
from prompting import LETTERS, parse_corrected

REFUSAL = re.compile(r"\b(I (cannot|can't|can not|won't|am unable|am not able)|"
                     r"I'm (sorry|afraid|not able|unable)|as an AI|I apologi[sz]e)", re.I)


def classify_invalid(raw, n_options):
    """What kind of invalid reply is this? Only descriptive, it changes no score."""
    text = raw.strip()
    if not text:
        return "empty"
    if REFUSAL.search(text):
        return "refusal"
    letters = set(re.findall(r"(?<![A-Za-z])([A-M])(?![A-Za-z])", text)) & set(LETTERS[:n_options])
    if len(letters) >= 2:
        return "several letters"
    return "prose, no letter reached"


def report_invalid_kinds(mmlu):
    section("15. KINDS OF INVALID MMLU REPLIES, AND INVALID RATE BY DOMAIN")
    missing = sorted(set(row["category"] for row in mmlu) - set(DOMAIN_OF))
    print(f"   subjects without a domain: {missing or 'none'}")
    print("\n   invalid rate none -> personas, per domain (all models together)")
    for domain in MMLU_DOMAINS:
        rows_d = [row for row in mmlu if DOMAIN_OF.get(row["category"]) == domain]
        n_items = len(set(row["item_id"] for row in rows_d))
        invalid_none = np.mean([row["outcome"] == "invalid" for row in rows_d if row["condition"] == "none"])
        invalid_personas = np.mean([row["outcome"] == "invalid" for row in rows_d if row["condition"] in PERSONAS])
        print(f"   {domain:<16}{n_items:>7}   invalid {100 * invalid_none:.2f}% -> {100 * invalid_personas:.2f}%")

    print("\n   invalid MMLU replies by kind (all models together)")
    kinds = ["prose, no letter reached", "refusal", "several letters", "empty"]
    rows_out = []
    for group, conditions in [("none+control", ["none", "control"]), ("personas", PERSONAS)]:
        invalid = [row for row in mmlu if row["condition"] in conditions and row["outcome"] == "invalid"]
        counts = Counter(classify_invalid(row["raw"], row["n_options"]) for row in invalid)
        stem = sum(1 for row in invalid if DOMAIN_OF.get(row["category"]) == "STEM")
        print(f"   {group:<13} {len(invalid):4d} invalid: "
              + ", ".join(f"{k} {counts[k]}" for k in kinds) + f" | STEM questions: {stem}")
        rows_out.append([group, len(invalid)] + [counts[k] for k in kinds] + [stem])
    write_csv("invalid_kinds_mmlu.csv", ["conditions", "invalid"] + kinds + ["stem"], rows_out)


def rescore(rows):
    """Copy of the rows, re-scored with the corrected parser."""
    result = []
    for row in rows:
        letter = parse_corrected(row["raw"], row["n_options"])
        if letter is None:
            outcome = "invalid"
        elif letter == row["correct_letter"]:
            outcome = "correct"
        else:
            outcome = "incorrect"
        result.append(dict(row, outcome=outcome))
    return result


def report_parser_sensitivity(tqa, mmlu):
    """Would the results change if the parser's two known bugs were fixed?"""
    section("16. PARSER SENSITIVITY (the stored scores are never changed)")
    tqa_fixed = rescore(tqa)
    mmlu_fixed = rescore(mmlu)
    changed = sum(1 for a, b in zip(tqa + mmlu, tqa_fixed + mmlu_fixed) if a["outcome"] != b["outcome"])
    print(f"   main + mmlu: {changed} of {len(tqa) + len(mmlu)} outcomes change")

    table = []
    largest_cell = 0.0
    shifts = []
    sign_flips = 0
    for model in models_of(tqa):
        for c in CONDITIONS:
            for stored, fixed in [(tqa, tqa_fixed), (mmlu, mmlu_fixed)]:
                diff = abs(accuracy(subset(stored, model, c))[0] - accuracy(subset(fixed, model, c))[0])
                largest_cell = max(largest_cell, diff)
        for persona in PERSONAS:
            d = []
            for t_rows, m_rows in [(tqa, mmlu), (tqa_fixed, mmlu_fixed)]:
                t_effect = accuracy(subset(t_rows, model, persona))[0] - accuracy(subset(t_rows, model, "none"))[0]
                m_effect = accuracy(subset(m_rows, model, persona))[0] - accuracy(subset(m_rows, model, "none"))[0]
                d.append(100 * (t_effect - m_effect))
            shifts.append(abs(d[1] - d[0]))
            if (d[0] > 0) != (d[1] > 0):
                sign_flips += 1
            table.append([model, persona, f"{d[0]:.2f}", f"{d[1]:.2f}"])
    print(f"   largest accuracy change in any cell: {100 * largest_cell:.2f} pp")
    print(f"   largest change of a TQA-MMLU difference: {max(shifts):.2f} pp | sign flips: {sign_flips} of {len(shifts)}")

    for name in ["extra_personas_tqa.jsonl", "extra_personas_mmlu.jsonl"]:
        rows = load(name, quiet=True)
        if rows:
            n_changed = sum(1 for a, b in zip(rows, rescore(rows)) if a["outcome"] != b["outcome"])
            print(f"   {name:<28} {n_changed} of {len(rows)} outcomes change")
    write_csv("parser_sensitivity.csv",
              ["model", "persona", "difference_pp_stored", "difference_pp_corrected"], table)
