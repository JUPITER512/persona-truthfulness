# Sections 9 and 17: the extra personas D and E (other religions).
# They were run on both full datasets under all three templates.

import numpy as np

from analysis.common import (BASELINES, BOOTSTRAP_SEED, accuracy, correctness_matrix, load, models_of,
                             pair_by_key, section, subset, write_csv)
from analysis.stats import (bh, bootstrap_difference, bootstrap_indices, mcnemar_pair, mcnemar_test,
                            wilson)


def by_item(row):
    return row["item_id"]


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
    section("17. EXTRA PERSONAS - FULL TESTS (all three templates)")
    extra_tqa = load("extra_personas_tqa.jsonl", quiet=True)
    extra_mmlu = load("extra_personas_mmlu.jsonl", quiet=True)
    if not extra_tqa or not extra_mmlu:
        print("   extra_personas_*.jsonl not collected")
        return None
    extras = sorted(set(row["condition"] for row in extra_tqa))
    templates = sorted(set(row["template"] for row in extra_tqa))

    # A. accuracy with 95% Wilson CI for every template
    print("   A. accuracy with 95% Wilson CI, every template")
    table = []
    for label, rows in [("TruthfulQA", extra_tqa), ("MMLU", extra_mmlu)]:
        for t in templates:
            for model in models_of(rows):
                for c in extras:
                    acc, low, high, n = wilson(subset(rows, model, c, t))
                    print(f"   {label:<11}{t:<5}{model:<14}{c:<11}{100 * acc:6.1f}%   [{100 * low:5.1f}, {100 * high:5.1f}]")
                    table.append([label, t, model, c, n, f"{acc:.4f}", f"{low:.4f}", f"{high:.4f}"])
    write_csv("extra_accuracy_ci.csv",
              ["dataset", "template", "model", "persona", "n", "accuracy", "ci_low", "ci_high"], table)

    # B. template t1: McNemar against none and control (same design as section 4)
    tqa_t1 = tqa + [row for row in extra_tqa if row["template"] == "t1"]
    mmlu_t1 = mmlu + [row for row in extra_mmlu if row["template"] == "t1"]
    tests = []
    for label, rows in [("TruthfulQA", tqa_t1), ("MMLU", mmlu_t1)]:
        for model in models_of(rows):
            for baseline in BASELINES:
                for persona in extras:
                    b, c, p, method = mcnemar_pair(rows, model, baseline, persona)
                    tests.append([label, model, baseline, persona, b, c, p, method])
    significant, adjusted = bh([t[6] for t in tests])
    print("\n   B. t1 - McNemar against the baselines")
    table = []
    for test, sig, p_bh in zip(tests, significant, adjusted):
        label, model, baseline, persona, b, c, p, method = test
        print(f"   {label:<11}{model:<14}{baseline:<9}{persona:<11} b={b:<4} c={c:<4} p_bh = {p_bh:.4f} {'*' if sig else ''}")
        table.append([label, model, baseline, persona, b, c, f"{p:.3e}", method, f"{p_bh:.3e}",
                      "significant" if sig else "ns"])
    print(f"   Benjamini-Hochberg over {len(tests)} tests: {sum(significant)} significant")
    write_csv("extra_significance.csv",
              ["dataset", "model", "baseline", "persona", "b", "c", "p", "method", "p_bh", "significant_bh"],
              table)

    # C. template t1: bootstrap of TQA delta - MMLU delta (same draws as section 12)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    conditions = BASELINES + extras
    tqa_matrices = {model: correctness_matrix(tqa_t1, model, conditions) for model in models_of(tqa_t1)}
    mmlu_matrices = {model: correctness_matrix(mmlu_t1, model, conditions) for model in models_of(mmlu_t1)}
    first_model = models_of(tqa_t1)[0]
    idx_tqa = bootstrap_indices(len(tqa_matrices[first_model]["none"]), rng)
    idx_mmlu = bootstrap_indices(len(mmlu_matrices[models_of(mmlu_t1)[0]]["none"]), rng)

    print("\n   C. t1 - bootstrap of TQA delta - MMLU delta (pp)")
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

    # D. templates t2 and t3. The only "none" rows for t2/t3 are in templates.jsonl
    #    (TruthfulQA 0-99), so there is no MMLU baseline; persona_d vs persona_e is also tested.
    tmpl = load("templates.jsonl", quiet=True)
    tests = []
    for t in templates:
        if t == "t1":
            continue
        for model in models_of(extra_tqa):
            base = subset(tmpl, model, "none", t)
            for persona in extras:
                x, y = pair_by_key(base, subset(extra_tqa, model, persona, t), by_item)
                if not x:
                    continue
                b, c, p, method = mcnemar_test(x, y)
                tests.append(["TruthfulQA 0-99", t, model, "none vs " + persona, len(x),
                              100.0 * (np.mean(y) - np.mean(x)), b, c, p])
    for label, rows in [("TruthfulQA", extra_tqa), ("MMLU", extra_mmlu)]:
        for t in templates:
            for model in models_of(rows):
                x, y = pair_by_key(subset(rows, model, extras[0], t), subset(rows, model, extras[-1], t), by_item)
                b, c, p, method = mcnemar_test(x, y)
                tests.append([label, t, model, f"{extras[0]} vs {extras[-1]}", len(x),
                              100.0 * (np.mean(y) - np.mean(x)), b, c, p])
    significant, adjusted = bh([t[8] for t in tests])
    print("\n   D. templates t2 and t3 (McNemar)")
    table = []
    for test, sig, p_bh in zip(tests, significant, adjusted):
        print(f"   {test[0]:<16}{test[1]:<5}{test[2]:<14}{test[3]:<24}{test[4]:6d}{test[5]:+8.1f}   p_bh = {p_bh:.4f} {'*' if sig else ''}")
        table.append(test[:5] + [f"{test[5]:.2f}", test[6], test[7], f"{test[8]:.3e}", f"{p_bh:.3e}", bool(sig)])
    write_csv("extra_templates_tests.csv",
              ["data", "template", "model", "comparison", "n", "delta_pp", "b", "c", "p", "p_bh",
               "significant_bh"], table)

    # E. accuracy gap TruthfulQA - MMLU per template (constant if the format did not matter)
    print("\n   E. accuracy gap TruthfulQA - MMLU for each extra persona, by template (pp)")
    for model in models_of(extra_tqa):
        parts = []
        for persona in extras:
            gaps = []
            for t in templates:
                gap = 100 * (accuracy(subset(extra_tqa, model, persona, t))[0]
                             - accuracy(subset(extra_mmlu, model, persona, t))[0])
                gaps.append(f"{t} {gap:+5.1f}")
            parts.append(f"{persona[-1]}: " + " ".join(gaps))
        print(f"   {model:<14} " + "   |   ".join(parts))
    return results
