# Robustness checks on TruthfulQA questions 0-99:
#   sections 6 and 18: prompt format (templates t1, t2, t3)
#   sections 7 and 19: 10 paraphrases of "You are a helpful assistant."
#   sections 8 and 20: answer order, temperature 0.7, repeat run, pilot

import os

import numpy as np

from analysis.common import (CONDITIONS, PERSONAS, accuracy, is_correct, load, models_of, pair_by_key,
                             section, subset, write_csv)
from analysis.stats import (bh, cochran_matrix, cochran_test, mcnemar_pair, mcnemar_test, wilson)
from config import RESULTS_DIR


def by_condition_and_item(row):
    return (row["condition"], row["item_id"])


def temperature_files():
    names = []
    for name in os.listdir(RESULTS_DIR):
        if name.startswith("temp_") and name.endswith(".jsonl"):
            names.append(name)
    return sorted(names)


def agreement(rows_a, rows_b):
    """Share of (model, condition, question) cells with the same letter in both files."""
    a = {(r["model"], r["condition"], r["item_id"]): r for r in rows_a}
    b = {(r["model"], r["condition"], r["item_id"]): r for r in rows_b}
    keys = set(a) & set(b)
    if not keys:
        return None, 0
    same = sum(1 for k in keys if a[k]["parsed"] == b[k]["parsed"])
    return same / len(keys), len(keys)


# ---------------------------------------------------------------------------
# Prompt format

def report_templates(tmpl):
    section("6. PROMPT FORMAT (templates.jsonl)")
    if not tmpl:
        print("   templates.jsonl not found")
        return
    templates = sorted(set(row["template"] for row in tmpl))
    table = []

    print("   accuracy by template")
    for model in models_of(tmpl):
        accs = []
        for t in templates:
            sub = subset(tmpl, model, None, t)
            if sub:
                accs.append(accuracy(sub)[0] * 100)
            else:
                accs.append(float("nan"))
        print(f"   {model:<14}" + "".join(f"{a:8.1f}%" for a in accs) + f"   spread {max(accs) - min(accs):.1f} pp")
        table.append([model, "accuracy"] + [f"{a:.2f}" for a in accs])

    print("\n   invalid replies by template")
    for model in models_of(tmpl):
        rates = []
        for t in templates:
            sub = subset(tmpl, model, None, t)
            rates.append(100.0 * sum(1 for row in sub if row["outcome"] == "invalid") / len(sub))
        print(f"   {model:<14}" + "".join(f"{r:8.1f}%" for r in rates))
        table.append([model, "invalid"] + [f"{r:.2f}" for r in rates])

    print("\n   same question and condition, different template: share of identical letters")
    for model in models_of(tmpl):
        cells = []
        for i, t_a in enumerate(templates):
            for t_b in templates[i + 1:]:
                share, n = agreement(subset(tmpl, model, None, t_a), subset(tmpl, model, None, t_b))
                cells.append(f"{t_a} vs {t_b}: {100 * share:.1f}%")
        print(f"   {model:<14}" + "   ".join(cells))
    write_csv("templates.csv", ["model", "measure"] + templates, table)


def report_template_tests(tmpl):
    section("18. PROMPT FORMAT - SIGNIFICANCE TESTS (templates.jsonl)")
    if not tmpl:
        print("   templates.jsonl not found")
        return
    templates = sorted(set(row["template"] for row in tmpl))
    table = []

    # A. Cochran's Q over the three templates, for every model and condition
    tests = []
    for model in models_of(tmpl):
        for c in CONDITIONS:
            per_template = {}
            for t in templates:
                per_template[t] = {row["item_id"]: is_correct(row) for row in subset(tmpl, model, c, t)}
            ids = sorted(set.intersection(*[set(d) for d in per_template.values()]))
            ids = [i for i in ids if all(per_template[t][i] is not None for t in templates)]
            matrix = np.array([[per_template[t][i] for t in templates] for i in ids])
            q, p = cochran_test(matrix)
            tests.append([model, c, "cochran_q " + "/".join(templates), len(ids), f"{q:.3f}", p])
    significant, adjusted = bh([t[5] for t in tests])
    print("   A. Cochran's Q over the templates (per model and condition)")
    for test, sig, p_bh in zip(tests, significant, adjusted):
        print(f"   {test[0]:<14}{test[1]:<11} Q = {test[4]:>7}   p = {test[5]:.4f}   p_bh = {p_bh:.4f}")
        table.append(test[:5] + [f"{test[5]:.3e}", f"{p_bh:.3e}", bool(sig)])
    print(f"   Benjamini-Hochberg over {len(tests)} tests: {sum(significant)} significant")

    # B. persona vs none within each template, and does the effect keep its sign?
    tests = []
    signs = {}
    for t in templates:
        rows_t = subset(tmpl, None, None, t)
        for model in models_of(tmpl):
            for persona in PERSONAS:
                b, c, p, method = mcnemar_pair(rows_t, model, "none", persona)
                effect = 100 * (accuracy(subset(rows_t, model, persona))[0]
                                - accuracy(subset(rows_t, model, "none"))[0])
                signs.setdefault((model, persona), []).append(np.sign(effect))
                tests.append([model, t, persona, b, c, effect, p])
    significant, adjusted = bh([t[6] for t in tests])
    print("\n   B. persona vs none within each template (McNemar)")
    for test, sig, p_bh in zip(tests, significant, adjusted):
        model, t, persona, b, c, effect, p = test
        print(f"   {model:<14}{t:<5}{persona:<11} b={b:<4} c={c:<4} effect {effect:+5.1f}   p_bh = {p_bh:.4f}")
        table.append([model, f"{t} {persona} vs none", "mcnemar", f"b={b} c={c}",
                      f"{effect:.2f}", f"{p:.3e}", f"{p_bh:.3e}", bool(sig)])
    same_sign = sum(1 for values in signs.values() if len(set(values)) == 1)
    print(f"   Benjamini-Hochberg over {len(tests)} tests: {sum(significant)} significant")
    print(f"   the persona effect keeps its sign under all templates in {same_sign} of {len(signs)} cells")
    write_csv("template_tests.csv",
              ["model", "comparison", "test", "n_or_counts", "statistic_or_delta", "p", "p_bh",
               "significant_bh"], table)


# ---------------------------------------------------------------------------
# Control paraphrases

def report_control_spread(ctrl, tqa):
    section("7. CONTROL PARAPHRASES vs PERSONAS (prompt sensitivity)")
    if not ctrl:
        print("   control_paraphrases.jsonl not found")
        return
    paraphrases = sorted(set(row["condition"] for row in ctrl))
    ids = set(row["item_id"] for row in ctrl)
    print(f"   {len(paraphrases)} paraphrases, questions {min(ids)}-{max(ids)}."
          " The five conditions are taken from main.jsonl on the same questions.")
    print("   If the paraphrase spread is as large as the persona spread, a persona")
    print("   difference of that size is not evidence of a persona effect.\n")

    table = []
    for model in models_of(ctrl):
        paraphrase_accs = []
        for c in paraphrases:
            sub = subset(ctrl, model, c)
            if len(sub) >= len(ids):  # only complete paraphrases
                paraphrase_accs.append((accuracy(sub)[0] * 100, c))
        condition_accs = []
        for c in CONDITIONS:
            sub = [row for row in tqa if row["model"] == model and row["condition"] == c and row["item_id"] in ids]
            if sub:
                condition_accs.append((accuracy(sub)[0] * 100, c))
        low = min(paraphrase_accs)[0]
        high = max(paraphrase_accs)[0]
        persona_spread = max(condition_accs)[0] - min(condition_accs)[0]
        print(f"   {model:<14} paraphrases {low:.1f}-{high:.1f}% (spread {high - low:.1f} pp)"
              f"   conditions spread {persona_spread:.1f} pp")
        table.append([model, f"{low:.2f}", f"{high:.2f}", f"{high - low:.2f}", f"{persona_spread:.2f}",
                      len(paraphrase_accs)])
    write_csv("control_spread.csv",
              ["model", "control_min", "control_max", "control_spread_pp", "persona_spread_pp",
               "n_complete_paraphrases"], table)


def report_control_tests(ctrl, tqa):
    section("19. CONTROL PARAPHRASES - INTERVALS AND TESTS")
    if not ctrl:
        print("   control_paraphrases.jsonl not found")
        return
    paraphrases = sorted(set(row["condition"] for row in ctrl))
    ids = set(row["item_id"] for row in ctrl)
    table = []

    print(f"   A. Cochran's Q over the {len(paraphrases)} paraphrases: do prompts with the same meaning differ?")
    tests = []
    for model in models_of(ctrl):
        matrix = cochran_matrix(ctrl, model, paraphrases)
        q, p = cochran_test(matrix)
        tests.append((model, q, p, len(matrix)))
    significant, adjusted = bh([t[2] for t in tests])
    for (model, q, p, n), sig, p_bh in zip(tests, significant, adjusted):
        print(f"      {model:<14} Q = {q:6.2f}   p = {p:.4f}   p_bh = {p_bh:.4f} {'*' if sig else ' '} ({n} questions)")
        table.append([model, "cochran_q_paraphrases", n, f"{q:.3f}",
                      f"p={p:.3e} p_bh={p_bh:.3e} {'significant' if sig else 'ns'}"])

    print("\n   B. each condition on the same questions (95% Wilson CI): inside the paraphrase range?")
    for model in models_of(ctrl):
        accs = [accuracy(subset(ctrl, model, c))[0] for c in paraphrases]
        range_low = min(accs)
        range_high = max(accs)
        print(f"      {model}   paraphrase range {100 * range_low:.0f}-{100 * range_high:.0f}%")
        for c in CONDITIONS:
            rows = [row for row in tqa if row["model"] == model and row["condition"] == c and row["item_id"] in ids]
            acc, low, high, n = wilson(rows)
            if acc > range_high:
                where = "above range"
            elif acc < range_low:
                where = "below range"
            else:
                where = "inside range"
            print(f"        {c:<11} {100 * acc:5.1f}%  [{100 * low:5.1f}, {100 * high:5.1f}]  {where}")
            table.append([model, c, n, f"{acc:.4f}", f"{acc:.4f} [{low:.4f}, {high:.4f}] {where}"])
    write_csv("control_paraphrase_tests.csv", ["model", "measure", "n", "value", "detail"], table)


# ---------------------------------------------------------------------------
# Answer order, temperature, repeat run and pilot

def report_robustness(tqa):
    section("8. REPEAT RUN, ANSWER ORDER AND TEMPERATURE (agreement)")
    for name, label in [("shuffle.jsonl", "repeat run with the same settings"),
                        ("noshuffle.jsonl", "shuffled vs unshuffled options")]:
        rows = load(name, quiet=True)
        if rows:
            share, n = agreement(tqa, rows)
            print(f"   {label:<42} {100 * share:.1f}% identical letters ({n} rows)")

    names = temperature_files()
    if names:
        files = {name: load(name, quiet=True) for name in names}
        print("\n   temperature 0.7: agreement between the seeds")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                share, n = agreement(files[a], files[b])
                print(f"      {a} vs {b}: {100 * share:.1f}% ({n})")
        print("\n   accuracy at temperature 0.7 vs 0.0 (same questions)")
        ids = set(row["item_id"] for row in files[names[0]])
        greedy = [row for row in tqa if row["item_id"] in ids]
        for model in models_of(greedy):
            line = f"      {model:<14} greedy {100 * accuracy(subset(greedy, model))[0]:5.1f}%"
            for name in names:
                line += f"   {name[:-6]} {100 * accuracy(subset(files[name], model))[0]:5.1f}%"
            print(line)


def report_robustness_tests(tqa):
    section("20. ANSWER ORDER, TEMPERATURE AND PILOT - TESTS")
    table = []

    unshuffled = load("noshuffle.jsonl", quiet=True)
    if unshuffled:
        print("   A. shuffled (main.jsonl) vs unshuffled options, questions 0-99, McNemar")
        print("      Unshuffled, the correct answer is always A.")
        last_id = max(row["item_id"] for row in unshuffled)
        for model in models_of(unshuffled):
            shuffled = [row for row in tqa if row["model"] == model and row["item_id"] <= last_id]
            unshuffled_m = subset(unshuffled, model)
            x, y = pair_by_key(shuffled, unshuffled_m, by_condition_and_item)
            b, c, p, method = mcnemar_test(x, y)
            a_share_shuffled = np.mean([row["parsed"] == "A" for row in shuffled if row["parsed"]])
            a_share_unshuffled = np.mean([row["parsed"] == "A" for row in unshuffled_m if row["parsed"]])
            print(f"   {model:<14} acc {100 * np.mean(x):.1f}% -> {100 * np.mean(y):.1f}%"
                  f"   answers 'A' {100 * a_share_shuffled:.1f}% -> {100 * a_share_unshuffled:.1f}%   p = {p:.4f}")
            table.append(["shuffled_vs_unshuffled", model, len(x), f"{np.mean(x):.4f}", f"{np.mean(y):.4f}",
                          b, c, f"{p:.3e}", f"A share {a_share_shuffled:.3f} -> {a_share_unshuffled:.3f}"])

        ids = set(row["item_id"] for row in unshuffled)
        print(f"\n      persona effect (persona - none, pp) under each order, questions 0-{max(ids)}")
        for model in models_of(unshuffled):
            shuffled_m = [row for row in tqa if row["model"] == model and row["item_id"] in ids]
            for persona in PERSONAS:
                effect_shuffled = 100 * (accuracy(subset(shuffled_m, model, persona))[0]
                                         - accuracy(subset(shuffled_m, model, "none"))[0])
                effect_unshuffled = 100 * (accuracy(subset(unshuffled, model, persona))[0]
                                           - accuracy(subset(unshuffled, model, "none"))[0])
                print(f"   {model:<14}{persona:<11}{effect_shuffled:+10.1f}{effect_unshuffled:+13.1f}")
                table.append(["persona_effect_by_ordering", model, len(ids), f"{effect_shuffled:.2f}",
                              f"{effect_unshuffled:.2f}", "", "", "",
                              f"{persona}: shuffled vs unshuffled delta (pp)"])

    names = temperature_files()
    if names:
        print("\n   B. temperature 0.7 vs greedy (main.jsonl), same questions, McNemar")
        for name in names:
            rows = load(name, quiet=True)
            last_id = max(row["item_id"] for row in rows)
            for model in models_of(rows):
                greedy = [row for row in tqa if row["model"] == model and row["item_id"] <= last_id]
                x, y = pair_by_key(greedy, subset(rows, model), by_condition_and_item)
                b, c, p, method = mcnemar_test(x, y)
                print(f"   {model:<14}{name:<14} {100 * np.mean(x):.1f}% vs {100 * np.mean(y):.1f}%   b/c {b}/{c}   p = {p:.4f}")
                table.append(["greedy_vs_" + name, model, len(x), f"{np.mean(x):.4f}", f"{np.mean(y):.4f}",
                              b, c, f"{p:.3e}", ""])

    repeat = load("shuffle.jsonl", quiet=True)
    if repeat:
        print("\n   C. repeat run (shuffle.jsonl) vs main.jsonl, same settings")
        main_by_key = {(row["model"], row["condition"], row["item_id"]): row for row in tqa}
        for model in models_of(repeat):
            x, y = pair_by_key(subset(tqa, model), subset(repeat, model), by_condition_and_item)
            b, c, p, method = mcnemar_test(x, y)
            raw_same = 0
            for row in subset(repeat, model):
                original = main_by_key.get((model, row["condition"], row["item_id"]))
                if original and original["raw"] == row["raw"]:
                    raw_same += 1
            print(f"      {model:<14} outcomes differ on {b + c} of {len(x)} | identical raw text: {raw_same}")
            table.append(["repeat_run", model, len(x), f"{np.mean(x):.4f}", f"{np.mean(y):.4f}",
                          b, c, f"{p:.3e}", f"raw identical {raw_same}"])

    pilot = load("pilot.jsonl", quiet=True)
    if pilot:
        print(f"\n   D. pilot (pilot.jsonl) vs main.jsonl, questions 0-{max(row['item_id'] for row in pilot)}")
        main_by_key = {(row["model"], row["condition"], row["item_id"]): row for row in tqa}
        for model in models_of(pilot):
            sub = subset(pilot, model)
            same_letter = sum(1 for row in sub if main_by_key[(model, row["condition"], row["item_id"])]["parsed"] == row["parsed"])
            same_raw = sum(1 for row in sub if main_by_key[(model, row["condition"], row["item_id"])]["raw"] == row["raw"])
            print(f"      {model:<14} same letter {same_letter}/{len(sub)} | same raw text {same_raw}/{len(sub)}")
            table.append(["pilot_vs_main", model, len(sub), "", "", "", "", "",
                          f"answers {same_letter} raw {same_raw}"])

    # One Benjamini-Hochberg correction for all accuracy tests of this section
    tested = [i for i, row in enumerate(table) if row[7]]
    significant, adjusted = bh([float(table[i][7]) for i in tested])
    for row in table:
        row += ["", ""]
    for i, sig, p_bh in zip(tested, significant, adjusted):
        table[i][-2:] = [f"{p_bh:.3e}", "significant" if sig else "ns"]
    print(f"\n   Benjamini-Hochberg over {len(tested)} tests: {sum(significant)} significant")
    write_csv("robustness_tests.csv",
              ["comparison", "model", "n", "acc_a", "acc_b", "b", "c", "p", "note", "p_bh",
               "significant_bh"], table)
