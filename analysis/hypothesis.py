# Sections 3, 4, 5 and 12: the hypothesis test and the significance tests.
#
# H1: a persona lowers accuracy on TruthfulQA but not on MMLU,
#     so (TruthfulQA change) - (MMLU change) should be negative.

from collections import Counter

import numpy as np

from analysis.common import (BASELINES, BOOTSTRAP_SEED, MIN_CATEGORY_N, PERSONAS, accuracy,
                             correctness_matrix, models_of, section, subset, write_csv)
from analysis.stats import bh, bootstrap_difference, bootstrap_indices, cochran_q, mcnemar_pair


def report_hypothesis(tqa, mmlu):
    section("3. TQA DELTA MINUS MMLU DELTA (the hypothesis, point estimates)")
    print("   delta = persona accuracy - baseline accuracy, in percentage points")
    table = []
    for baseline in BASELINES:
        print(f"\n   baseline = {baseline}")
        print(f"   {'model':<14}{'persona':<11}{'TQA d':>9}{'MMLU d':>9}{'diff':>9}")
        for model in models_of(tqa):
            for p in PERSONAS:
                t = (accuracy(subset(tqa, model, p))[0] - accuracy(subset(tqa, model, baseline))[0]) * 100
                u = (accuracy(subset(mmlu, model, p))[0] - accuracy(subset(mmlu, model, baseline))[0]) * 100
                print(f"   {model:<14}{p:<11}{t:>+9.2f}{u:>+9.2f}{t - u:>+9.2f}")
                table.append([baseline, model, p, f"{t:.2f}", f"{u:.2f}", f"{t - u:.2f}"])
    write_csv("hypothesis_deltas.csv",
              ["baseline", "model", "persona", "tqa_delta_pp", "mmlu_delta_pp", "difference_pp"], table)


def report_significance(tqa, mmlu):
    section("4. SIGNIFICANCE TESTS (Cochran's Q and McNemar)")
    table = []
    family = []  # the 48 McNemar tests, corrected together
    for label, rows in [("TruthfulQA", tqa), ("MMLU", mmlu)]:
        print(f"\n   {label} - Cochran's Q: do the five conditions differ at all?")
        for model in models_of(rows):
            q, p, n = cochran_q(rows, model)
            print(f"   {model:<14} Q = {q:8.3f}   p = {p:.2e}   ({n} questions)")
            table.append([label, model, "cochran_q", "", f"{q:.4f}", f"{p:.3e}", n])

        print(f"\n   {label} - McNemar: baseline vs persona (b = baseline right & persona wrong, c = the reverse)")
        for model in models_of(rows):
            for baseline in BASELINES:
                for persona in PERSONAS:
                    b, c, p, method = mcnemar_pair(rows, model, baseline, persona)
                    print(f"   {model:<14}{baseline:<10}{persona:<11}{b:6d}{c:6d}{p:11.4f}  {method}")
                    table.append([label, model, "mcnemar", f"{baseline} vs {persona}",
                                  f"b={b} c={c}", f"{p:.3e}", method])
                    family.append((label, model, baseline, persona, p))

    significant, adjusted = bh([f[4] for f in family])
    raw_count = sum(1 for f in family if f[4] < 0.05)
    print(f"\n   Benjamini-Hochberg over all {len(family)} McNemar tests:"
          f" raw p < .05: {raw_count} | still significant: {sum(significant)}")
    for (label, model, baseline, persona, p), sig, p_bh in zip(family, significant, adjusted):
        if sig:
            print(f"      {label:<11} {model:<14} {baseline:<8} {persona:<10} p_bh = {p_bh:.5f}")
        table.append([label, model, "mcnemar_bh", f"{baseline} vs {persona}", "",
                      f"{p_bh:.3e}", "significant" if sig else "ns"])
    write_csv("significance.csv",
              ["dataset", "model", "test", "comparison", "statistic", "p_value", "note"], table)


def report_categories(tqa):
    section("5. TRUTHFULQA CATEGORIES (McNemar per category, Benjamini-Hochberg)")
    first_model = models_of(tqa)[0]
    sizes = Counter(row["category"] for row in subset(tqa, first_model, "none"))
    tested = sorted(c for c, n in sizes.items() if n >= MIN_CATEGORY_N)
    print(f"   {len(sizes)} categories, {len(tested)} tested (MIN_CATEGORY_N = {MIN_CATEGORY_N})")

    rows_of_category = {}
    for row in tqa:
        rows_of_category.setdefault(row["category"], []).append(row)

    table = []
    for model in models_of(tqa):
        for baseline in BASELINES:
            for persona in PERSONAS:
                results = []
                for category in tested:
                    b, c, p, method = mcnemar_pair(rows_of_category[category], model, baseline, persona)
                    results.append((category, b, c, p))
                significant, adjusted = bh([r[3] for r in results])
                hits = []
                for (category, b, c, p), sig, p_bh in zip(results, significant, adjusted):
                    table.append([model, baseline, persona, category, sizes[category], b, c,
                                  f"{p:.5f}", f"{p_bh:.5f}", bool(sig)])
                    if sig:
                        hits.append(category)
                print(f"   {model:<14}{baseline:<9}{persona:<11} significant: {len(hits)}/{len(results)}"
                      + (("  -> " + ", ".join(hits)) if hits else ""))
    write_csv("category_tests.csv",
              ["model", "baseline", "persona", "category", "n_items", "b", "c", "p_raw", "p_bh",
               "significant_bh"], table)


def report_bootstrap_did(tqa, mmlu):
    """The hypothesis test with confidence intervals (paired bootstrap of the questions)."""
    section("12. BOOTSTRAP TEST OF THE HYPOTHESIS (TQA delta - MMLU delta)")
    print("   10,000 resamples of the questions (seed 0). 95% CI = 2.5th and 97.5th percentile.")
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    matrices = {}
    for label, rows in [("TruthfulQA", tqa), ("MMLU", mmlu)]:
        for model in models_of(rows):
            matrices[(label, model)] = correctness_matrix(rows, model)
    n_tqa = len(matrices[("TruthfulQA", models_of(tqa)[0])]["none"])
    n_mmlu = len(matrices[("MMLU", models_of(mmlu)[0])]["none"])
    # The same resamples are used for every model and persona
    idx_tqa = bootstrap_indices(n_tqa, rng)
    idx_mmlu = bootstrap_indices(n_mmlu, rng)

    table = []
    p_values = []
    results = {}  # (baseline, model, persona) -> estimates, used by figures 2 and 9
    for baseline in BASELINES:
        print(f"\n   baseline = {baseline}   (percentage points)")
        for model in models_of(tqa):
            T = matrices[("TruthfulQA", model)]
            M = matrices[("MMLU", model)]
            for persona in PERSONAS:
                r = bootstrap_difference(T, M, idx_tqa, idx_mmlu, baseline, persona)
                results[(baseline, model, persona)] = r
                t0, tl, th = r["tqa"]
                m0, ml, mh = r["mmlu"]
                d0, dl, dh = r["did"]
                print(f"   {model:<14}{persona:<11} TQA {t0:+6.2f} [{tl:+5.1f},{th:+5.1f}]"
                      f"  MMLU {m0:+6.2f} [{ml:+5.1f},{mh:+5.1f}]"
                      f"  diff {d0:+6.2f} [{dl:+5.1f},{dh:+5.1f}]  p = {r['p']:.4f}")
                table.append([baseline, model, persona] +
                             [f"{v:.2f}" for v in (t0, tl, th, m0, ml, mh, d0, dl, dh)] +
                             [f"{r['p']:.4f}"])
                p_values.append(r["p"])

    significant, adjusted = bh(p_values)
    for row, sig, p_bh in zip(table, significant, adjusted):
        row += [f"{p_bh:.4f}", "significant" if sig else "ns"]
    above_zero = sum(1 for key, r in results.items() if key[0] == "none" and r["did"][1] > 0)
    print(f"\n   Benjamini-Hochberg over {len(table)} tests: {sum(significant)} significant")
    print(f"   baseline none: CI completely above zero in {above_zero} of {len(PERSONAS) * len(models_of(tqa))} cells")
    write_csv("hypothesis_bootstrap.csv",
              ["baseline", "model", "persona", "tqa_delta_pp", "tqa_ci_low", "tqa_ci_high",
               "mmlu_delta_pp", "mmlu_ci_low", "mmlu_ci_high", "difference_pp", "diff_ci_low",
               "diff_ci_high", "p_boot", "p_bh", "significant_bh"], table)
    return results
