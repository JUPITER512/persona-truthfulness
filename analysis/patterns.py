# Sections 13-16: rankings, answer changes (churn), MMLU domains, invalid replies, parser check.

import re
from collections import Counter

import numpy as np
from scipy.stats import kendalltau

from analysis.common import (CONDITIONS, PERSONAS, accuracy, load, models_of, paired_vectors,
                             section, subset, write_csv)
from analysis.mmlu_domains import DOMAIN_OF, MMLU_DOMAINS
from prompting import LETTERS, parse_corrected

REFUSAL = re.compile(r"\b(I (cannot|can't|can not|won't|am unable|am not able)|"
                     r"I'm (sorry|afraid|not able|unable)|as an AI|I apologi[sz]e)", re.I)


def report_kendall(tqa, mmlu):
    """Do the five conditions come in the same order on both tasks, and for all models?"""
    section("13. KENDALL'S TAU - DO THE CONDITION RANKINGS AGREE?")
    print("   With only five conditions tau moves in steps of 0.2: read it as a description.")
    table = []

    print("\n   TruthfulQA ranking vs MMLU ranking, per model")
    for model in models_of(tqa):
        tqa_accs = [accuracy(subset(tqa, model, c))[0] for c in CONDITIONS]
        mmlu_accs = [accuracy(subset(mmlu, model, c))[0] for c in CONDITIONS]
        tau, p = kendalltau(tqa_accs, mmlu_accs)
        print(f"      {model:<14} tau = {tau:+.2f}   (p = {p:.3f})")
        table.append(["across_tasks", model, "", f"{tau:.4f}", f"{p:.4f}"])

    for label, rows in [("TruthfulQA", tqa), ("MMLU", mmlu)]:
        models = models_of(rows)
        accs = {m: [accuracy(subset(rows, m, c))[0] for c in CONDITIONS] for m in models}
        taus = []
        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                tau, p = kendalltau(accs[model_a], accs[model_b])
                taus.append(tau)
                table.append(["across_models_" + label, model_a, model_b, f"{tau:.4f}", f"{p:.4f}"])
        print(f"   between models, {label:<10} mean tau = {np.mean(taus):+.2f}"
              f"  (from {min(taus):+.2f} to {max(taus):+.2f})")
    write_csv("kendall_tau.csv", ["comparison", "model_a", "model_b", "tau", "p"], table)


def report_churn(tqa, mmlu):
    """How many answers change between persona and none, compared with the net change."""
    section("14. CHURN - ANSWERS THAT FLIP vs NET CHANGE (persona vs none)")
    print("   flips = questions right under one condition and wrong under the other (b + c)")
    print("   net   = change in the number of correct answers (c - b)")
    table = []
    for label, rows in [("TruthfulQA", tqa), ("MMLU", mmlu)]:
        print(f"\n   {label}")
        for model in models_of(rows):
            for persona in PERSONAS:
                x, y = paired_vectors(rows, model, "none", persona)
                b = sum(1 for i, j in zip(x, y) if i == 1 and j == 0)
                c = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)
                if c != b:
                    ratio = f"{(b + c) / abs(c - b):.2f}"
                else:
                    ratio = "inf"
                print(f"   {model:<14}{persona:<11} flips {b + c:4d} ({100.0 * (b + c) / len(x):4.1f}%)"
                      f"   net {c - b:+4d}   ratio {ratio}")
                table.append([label, model, persona, b + c, len(x), c - b, ratio])
    write_csv("churn.csv", ["dataset", "model", "persona", "flips", "n_items", "net", "flip_to_net_ratio"],
              table)


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


def report_mmlu_domains(mmlu):
    section("15. MMLU BY DOMAIN, AND WHAT THE INVALID REPLIES ARE")
    missing = sorted(set(row["category"] for row in mmlu) - set(DOMAIN_OF))
    print(f"   subjects without a domain: {missing or 'none'}")
    print("\n   mean persona effect vs none (pp) per domain, and invalid rate none -> personas")
    table = []
    for domain in MMLU_DOMAINS:
        rows_d = [row for row in mmlu if DOMAIN_OF.get(row["category"]) == domain]
        n_items = len(set(row["item_id"] for row in rows_d))
        line = f"   {domain:<16}{n_items:>7}"
        for model in models_of(mmlu):
            base = accuracy(subset(rows_d, model, "none"))[0]
            effect = np.mean([accuracy(subset(rows_d, model, p))[0] for p in PERSONAS]) - base
            line += f"{100 * effect:>+10.1f}"
            table.append([domain, model, f"{100 * effect:.2f}"])
        invalid_none = np.mean([row["outcome"] == "invalid" for row in rows_d if row["condition"] == "none"])
        invalid_personas = np.mean([row["outcome"] == "invalid" for row in rows_d if row["condition"] in PERSONAS])
        line += f"   invalid {100 * invalid_none:.2f}% -> {100 * invalid_personas:.2f}%"
        print(line)
    write_csv("mmlu_domains.csv", ["domain", "model", "mean_persona_delta_pp"], table)

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
