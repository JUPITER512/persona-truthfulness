# Sections 9 and 17: the extra personas D and E (other religions).
# They were run on both full datasets under all three templates; only template t1 has a
# matching no-persona baseline on the full datasets, so only t1 enters the analysis.

import numpy as np

from analysis.common import (BASELINES, BOOTSTRAP_SEED, accuracy, correctness_matrix, load, models_of,
                             section, subset, write_csv)
from analysis.stats import bh, bootstrap_difference, bootstrap_indices


def report_extra_personas(tqa, mmlu):
    section("9. EXTRA PERSONAS (accuracy, template t1)")
    for label, filename, main_rows in [("TruthfulQA", "extra_personas_tqa.jsonl", tqa),
                                       ("MMLU", "extra_personas_mmlu.jsonl", mmlu)]:
        rows = [row for row in load(filename, quiet=True) if row["template"] == "t1"]
        if not rows:
            print(f"   {label}: not collected")
            continue
        extras = sorted(set(row["condition"] for row in rows))
        ids = set(row["item_id"] for row in rows)
        print(f"\n   {label} - t1, {len(ids)} questions")
        for model in models_of(rows):
            line = f"   {model:<14}"
            for c in ["none", "control"]:
                sub = [row for row in main_rows if row["model"] == model and row["condition"] == c and row["item_id"] in ids]
                line += f"  {c} {100 * accuracy(sub)[0]:5.1f}%"
            for c in extras:
                line += f"  {c} {100 * accuracy(subset(rows, model, c))[0]:5.1f}%"
            print(line)
    print("\n   Only t1 rows are used here: the files also hold t2 and t3.")


def report_extra_full(tqa, mmlu):
    section("17. EXTRA PERSONAS - BOOTSTRAP TEST OF THE HYPOTHESIS (template t1)")
    extra_tqa = load("extra_personas_tqa.jsonl", quiet=True)
    extra_mmlu = load("extra_personas_mmlu.jsonl", quiet=True)
    if not extra_tqa or not extra_mmlu:
        print("   extra_personas_*.jsonl not collected")
        return None
    extras = sorted(set(row["condition"] for row in extra_tqa))
    tqa_t1 = tqa + [row for row in extra_tqa if row["template"] == "t1"]
    mmlu_t1 = mmlu + [row for row in extra_mmlu if row["template"] == "t1"]

    # template t1: bootstrap of TQA delta - MMLU delta (same draws as section 12)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    conditions = BASELINES + extras
    tqa_matrices = {model: correctness_matrix(tqa_t1, model, conditions) for model in models_of(tqa_t1)}
    mmlu_matrices = {model: correctness_matrix(mmlu_t1, model, conditions) for model in models_of(mmlu_t1)}
    first_model = models_of(tqa_t1)[0]
    idx_tqa = bootstrap_indices(len(tqa_matrices[first_model]["none"]), rng)
    idx_mmlu = bootstrap_indices(len(mmlu_matrices[models_of(mmlu_t1)[0]]["none"]), rng)

    print("   t1 - bootstrap of TQA delta - MMLU delta (pp)")
    table = []
    p_values = []
    results = {}
    for baseline in BASELINES:
        for model in models_of(tqa_t1):
            for persona in extras:
                r = bootstrap_difference(tqa_matrices[model], mmlu_matrices[model], idx_tqa, idx_mmlu,
                                         baseline, persona)
                results[(baseline, model, persona)] = r
                t0, tl, th = r["tqa"]
                m0, ml, mh = r["mmlu"]
                d0, dl, dh = r["did"]
                print(f"   {baseline:<9}{model:<14}{persona:<11} diff {d0:+6.2f} [{dl:+5.1f},{dh:+5.1f}]  p = {r['p']:.4f}")
                table.append([baseline, model, persona] +
                             [f"{v:.2f}" for v in (t0, tl, th, m0, ml, mh, d0, dl, dh)] +
                             [f"{r['p']:.4f}"])
                p_values.append(r["p"])
    significant, adjusted = bh(p_values)
    for row, sig, p_bh in zip(table, significant, adjusted):
        row += [f"{p_bh:.4f}", "significant" if sig else "ns"]
    print(f"   Benjamini-Hochberg over {len(p_values)} tests: {sum(significant)} significant")
    write_csv("extra_hypothesis_bootstrap.csv",
              ["baseline", "model", "persona", "tqa_delta_pp", "tqa_ci_low", "tqa_ci_high",
               "mmlu_delta_pp", "mmlu_ci_low", "mmlu_ci_high", "difference_pp", "diff_ci_low",
               "diff_ci_high", "p_boot", "p_bh", "significant_bh"], table)

    return results
