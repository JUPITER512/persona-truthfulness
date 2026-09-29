# Sections 22 and 23: post-hoc sensitivity analyses of the hypothesis test, and the
# descriptive numbers that the paper quotes in the text (category effects, temperature
# letter changes, multiple-choice result on the generation questions, the control prompt
# against no system prompt, and the t1 rows of the template run against main.jsonl).
#
# Nothing here changes a stored score or the main analysis (sections 0-21).

import os
from collections import Counter

import numpy as np

from analysis.common import (BASELINES, BOOTSTRAP_SEED, CONDITIONS, PERSONAS, accuracy, load,
                             models_of, section, subset, write_csv)
from analysis.patterns import rescore
from analysis.robustness import temperature_files
from analysis.stats import bh, boot_ci, boot_p, bootstrap_indices, mcnemar_pair
from config import RESULTS_DIR
from utils import read_jsonl

INDEXICAL = "Indexical Error"  # categories "Indexical Error: Identity / Location / Time / Other"

BOOTSTRAP_HEADER = ["baseline", "model", "persona", "tqa_delta_pp", "tqa_ci_low", "tqa_ci_high",
                    "mmlu_delta_pp", "mmlu_ci_low", "mmlu_ci_high", "difference_pp", "diff_ci_low",
                    "diff_ci_high", "p_boot", "p_bh", "significant_bh"]


def outcome_matrix(rows, model, valid_only):
    """{condition: 1.0 / 0.0 per question over the same sorted ids}.

    valid_only: an invalid reply becomes NaN (left out) instead of 0 (not correct).
    """
    per_condition = {}
    for c in CONDITIONS:
        per_condition[c] = {row["item_id"]: row["outcome"] for row in subset(rows, model, c)}
    ids = sorted(set.intersection(*[set(d) for d in per_condition.values()]))
    matrix = {}
    for c in CONDITIONS:
        values = []
        for i in ids:
            outcome = per_condition[c][i]
            if outcome == "invalid" and valid_only:
                values.append(np.nan)
            else:
                values.append(1.0 if outcome == "correct" else 0.0)
        matrix[c] = np.array(values)
    return matrix


def bootstrap_did(tqa, mmlu, valid_only):
    """Section 12 with other data or another treatment of invalid replies.

    Same seed, same order of draws and same loops as section 12, so with valid_only=False
    and the full data it reproduces hypothesis_bootstrap.csv.
    Returns the table rows (as in hypothesis_bootstrap.csv) and the number significant after BH.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    T_all = {model: outcome_matrix(tqa, model, valid_only) for model in models_of(tqa)}
    M_all = {model: outcome_matrix(mmlu, model, valid_only) for model in models_of(mmlu)}
    idx_tqa = bootstrap_indices(len(T_all[models_of(tqa)[0]]["none"]), rng)
    idx_mmlu = bootstrap_indices(len(M_all[models_of(mmlu)[0]]["none"]), rng)

    table = []
    p_values = []
    for baseline in BASELINES:
        for model in models_of(tqa):
            T = T_all[model]
            M = M_all[model]
            for persona in PERSONAS:
                # np.nanmean equals the plain mean when nothing is left out
                tqa_draws = 100 * (np.nanmean(T[persona][idx_tqa], axis=1) - np.nanmean(T[baseline][idx_tqa], axis=1))
                mmlu_draws = 100 * (np.nanmean(M[persona][idx_mmlu], axis=1) - np.nanmean(M[baseline][idx_mmlu], axis=1))
                did_draws = tqa_draws - mmlu_draws
                t0 = 100 * (np.nanmean(T[persona]) - np.nanmean(T[baseline]))
                m0 = 100 * (np.nanmean(M[persona]) - np.nanmean(M[baseline]))
                tl, th = boot_ci(tqa_draws)
                ml, mh = boot_ci(mmlu_draws)
                dl, dh = boot_ci(did_draws)
                p = boot_p(did_draws)
                table.append([baseline, model, persona] +
                             [f"{v:.2f}" for v in (t0, tl, th, m0, ml, mh, t0 - m0, dl, dh)] + [f"{p:.4f}"])
                p_values.append(p)

    significant, adjusted = bh(p_values)
    for row, sig, p_bh in zip(table, significant, adjusted):
        row += [f"{p_bh:.4f}", "significant" if sig else "ns"]
    return table, sum(significant)


def positive(table, baseline):
    """Number of cells with a positive DiD point estimate, and the number of cells."""
    cells = [row for row in table if row[0] == baseline]
    return sum(1 for row in cells if float(row[9]) > 0), len(cells)


def report_sensitivity(tqa, mmlu):
    section("22. SENSITIVITY OF THE HYPOTHESIS TEST (post hoc)")
    print("   The bootstrap of section 12 repeated (a) with invalid replies left out instead of counted wrong,")
    print(f"   (b) without the TruthfulQA categories '{INDEXICAL}: ...' (questions about the model itself,")
    print("   its location or the present), and (c) the DiD point estimates with the corrected parser.")

    tqa_without = [row for row in tqa if not row["category"].startswith(INDEXICAL)]
    variants = [("main analysis (invalid = incorrect)", tqa, False, None),
                ("valid replies only", tqa, True, "sensitivity_valid_only.csv"),
                (f"without {INDEXICAL} categories", tqa_without, False, "sensitivity_without_indexical.csv")]
    summary = []
    for label, tqa_rows, valid_only, filename in variants:
        n_items = len(set(row["item_id"] for row in tqa_rows))
        table, n_significant = bootstrap_did(tqa_rows, mmlu, valid_only)
        pos_none, n_none = positive(table, "none")
        pos_control, n_control = positive(table, "control")
        print(f"\n   {label}: {n_items} TruthfulQA questions")
        print(f"      positive DiD vs none {pos_none}/{n_none}, vs control {pos_control}/{n_control},"
              f" significant after BH {n_significant} of {len(table)}")
        for row in table:
            if row[0] == "none":
                print(f"      {row[1]:<14}{row[2]:<11} TQA {row[3]:>6} [{row[4]}, {row[5]}]"
                      f"  MMLU {row[6]:>6} [{row[7]}, {row[8]}]  diff {row[9]:>6} [{row[10]}, {row[11]}]"
                      f"  p_bh = {row[13]}")
        if filename:
            write_csv(filename, BOOTSTRAP_HEADER, table)
        else:
            # the main variant must give exactly the numbers of section 12
            path = os.path.join(RESULTS_DIR, "analysis", "hypothesis_bootstrap.csv")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    stored = [line.rstrip("\n").split(",") for line in f][1:]
                print(f"      identical to hypothesis_bootstrap.csv (section 12): {stored == table}")
        summary.append([label, n_items, f"{pos_none}/{n_none}", f"{pos_control}/{n_control}",
                        n_significant, len(table), ""])

    # (c) corrected parser: point estimates only, against no system prompt (as in section 16)
    tqa_fixed = rescore(tqa)
    mmlu_fixed = rescore(mmlu)
    shifts = []
    pos_fixed = 0
    for model in models_of(tqa):
        for persona in PERSONAS:
            d = []
            for t_rows, m_rows in [(tqa, mmlu), (tqa_fixed, mmlu_fixed)]:
                t_effect = accuracy(subset(t_rows, model, persona))[0] - accuracy(subset(t_rows, model, "none"))[0]
                m_effect = accuracy(subset(m_rows, model, persona))[0] - accuracy(subset(m_rows, model, "none"))[0]
                d.append(100 * (t_effect - m_effect))
            shifts.append(abs(d[1] - d[0]))
            pos_fixed += d[1] > 0
    n_cells = len(shifts)
    print(f"\n   corrected parser: positive DiD vs none {pos_fixed}/{n_cells},"
          f" largest change of a DiD {max(shifts):.2f} pp (no bootstrap)")
    summary.append(["corrected parser", len(set(row["item_id"] for row in tqa)), f"{pos_fixed}/{n_cells}", "",
                    "", "", f"largest DiD change {max(shifts):.2f} pp"])
    write_csv("sensitivity_summary.csv",
              ["analysis", "n_tqa_items", "positive_did_vs_none", "positive_did_vs_control",
               "significant_bh", "n_tests", "note"], summary)


def report_text_numbers(tqa, mmlu, tmpl):
    section("23. DESCRIPTIVE NUMBERS QUOTED IN THE TEXT")
    models = models_of(tqa)

    # A. mean persona effect per category (the numbers behind figure 5)
    sizes = Counter(row["category"] for row in subset(tqa, models[0], "none"))
    rows_of_category = {}
    for row in tqa:
        rows_of_category.setdefault(row["category"], []).append(row)
    effects = []
    counts = []
    for category in sorted(sizes):
        rows_c = rows_of_category[category]
        per_model = []
        for model in models:
            base = accuracy(subset(rows_c, model, "none"))[0]
            per_model.append(100 * (np.mean([accuracy(subset(rows_c, model, p))[0] for p in PERSONAS]) - base))
            for c in CONDITIONS:
                sub = subset(rows_c, model, c)
                counts.append([category, sizes[category], model, c,
                               sum(1 for row in sub if row["outcome"] == "correct"),
                               sum(1 for row in sub if row["outcome"] == "invalid")])
        effects.append((float(np.mean(per_model)), category, per_model))
    effects.sort(key=lambda e: e[0])
    print("   A. mean persona effect per TruthfulQA category (personas A-C minus none, pp)")
    print("      five largest losses and five largest gains (mean over the models):")
    for mean, category, per_model in effects[:5] + effects[-5:]:
        print(f"      {category + f' ({sizes[category]})':<38}" + "".join(f"{v:+7.1f}" for v in per_model)
              + f"   mean {mean:+6.1f}")
    write_csv("category_effects.csv", ["category", "n_items"] + models + ["mean"],
              [[category, sizes[category]] + [f"{v:.2f}" for v in per_model] + [f"{mean:.2f}"]
               for mean, category, per_model in effects])
    write_csv("category_counts.csv", ["category", "n_items", "model", "condition", "correct", "invalid"], counts)

    # B. temperature 0.7: how many letters differ from greedy decoding (main.jsonl)?
    table = []
    names = temperature_files()
    if names:
        greedy = {(row["model"], row["condition"], row["item_id"]): row["parsed"] for row in tqa}
        print("\n   B. temperature 0.7: letters different from greedy decoding (same model, condition, question)")
        for name in names:
            rows = load(name, quiet=True)
            for model in models_of(rows) + ["all"]:
                sub = rows if model == "all" else subset(rows, model)
                changed = sum(1 for row in sub if row["parsed"] != greedy[(row["model"], row["condition"], row["item_id"])])
                table.append([name, model, len(sub), changed])
                if model == "all":
                    print(f"      {name:<14} {changed} of {len(sub)} letters changed")
        write_csv("temperature_letter_changes.csv", ["file", "model", "n", "letters_changed"], table)

    # C. the generation questions in multiple-choice format (same model, same questions)
    gen_file = os.path.join(RESULTS_DIR, "generation.jsonl")
    if os.path.exists(gen_file):
        gen = read_jsonl(gen_file)
        ids = set(row["item_id"] for row in gen)
        table = []
        print(f"\n   C. multiple-choice result on the {len(ids)} generation questions (main.jsonl)")
        for model in sorted(set(row["model"] for row in gen)):
            for c in CONDITIONS:
                sub = [row for row in subset(tqa, model, c) if row["item_id"] in ids]
                correct = sum(1 for row in sub if row["outcome"] == "correct")
                print(f"      {model:<14}{c:<11} {correct} of {len(sub)} correct")
                table.append([model, c, len(sub), correct])
        write_csv("generation_mc_same_items.csv", ["model", "condition", "n", "mc_correct"], table)

    # D. the control prompt itself: "You are a helpful assistant." against no system prompt (McNemar),
    #    Benjamini-Hochberg over these 8 tests (Table C1, note)
    tests = []
    for label, rows in [("TruthfulQA", tqa), ("MMLU", mmlu)]:
        for model in models_of(rows):
            b, c, p, method = mcnemar_pair(rows, model, "none", "control")
            delta = 100 * (accuracy(subset(rows, model, "control"))[0] - accuracy(subset(rows, model, "none"))[0])
            tests.append([label, model, delta, b, c, p, method])
    significant, adjusted = bh([t[5] for t in tests])
    print("\n   D. control prompt vs no system prompt (McNemar; Benjamini-Hochberg over these 8 tests)")
    table = []
    for (label, model, delta, b, c, p, method), sig, p_bh in zip(tests, significant, adjusted):
        print(f"      {label:<11}{model:<14} control - none {delta:+5.2f} pp   b={b:<3} c={c:<3} p = {p:.4f}"
              f"   p_bh = {p_bh:.4f} {'*' if sig else ''}")
        table.append([label, model, f"{delta:.2f}", b, c, f"{p:.3e}", method, f"{p_bh:.3e}",
                      "significant" if sig else "ns"])
    write_csv("control_vs_none.csv", ["dataset", "model", "control_minus_none_pp", "b", "c", "p", "method",
                                      "p_bh", "significant_bh"], table)

    # E. the t1 rows of the template run repeat main.jsonl on questions 0-99: identical raw replies? (Table C8, note)
    if tmpl:
        main_raw = {(row["model"], row["condition"], row["item_id"]): row["raw"] for row in tqa}
        rows_t1 = subset(tmpl, None, None, "t1")
        print("\n   E. t1 rows of templates.jsonl vs main.jsonl (same model, condition, question)")
        table = []
        for model in models_of(rows_t1) + ["all"]:
            sub = rows_t1 if model == "all" else subset(rows_t1, model)
            same = sum(1 for row in sub if main_raw.get((row["model"], row["condition"], row["item_id"])) == row["raw"])
            table.append([model, len(sub), same])
            if model == "all":
                print(f"      identical raw replies: {same} of {len(sub)}")
        write_csv("template_t1_vs_main.csv", ["model", "n", "identical_raw"], table)
