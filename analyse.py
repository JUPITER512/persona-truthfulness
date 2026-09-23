"""Statistics for the persona-conditioning study.

Prints tables and writes CSV copies into results/analysis/ for the appendix.

This script COMPUTES; it does not interpret. Every number needs your reading.

Two choices are yours to make and to justify in the Method section. Both are
printed in the output, so they end up on the record:
  MIN_CATEGORY_N       - the smallest category worth significance-testing
  INVALID_AS_INCORRECT - how a refusal or unparseable reply is scored
"""
import json, os, csv, collections
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats.multitest import multipletests

RESULTS_DIR = "results"
OUT_DIR = os.path.join(RESULTS_DIR, "analysis")
CONDITIONS = ["none", "control", "persona_a", "persona_b", "persona_c"]
PERSONAS = ["persona_a", "persona_b", "persona_c"]
BASELINES = ["none", "control"]

# --- the two choices ---------------------------------------------------------
# Smallest category that gets a significance test. 1 = test all 38. The
# trade-off: the smallest categories hold 4-6 items, where McNemar has almost
# no power, and testing all 38 enlarges the Benjamini-Hochberg family, making
# the correction stricter for every category including the large ones.
MIN_CATEGORY_N = 1

# An invalid row is a refusal or an unparseable reply. True = count it as not
# correct, which keeps every item in the denominator. Invalid rates are always
# reported separately, so nothing is silently dropped either way.
INVALID_AS_INCORRECT = True
# -----------------------------------------------------------------------------

# Filled in by load(); reported in section 0.
INTEGRITY = []


def load(filename, quiet=False):
    """Read one JSONL file, keeping the first row for each measurement key.

    Duplicate keys happen when two runs write to the same file at once, since
    resume has no file lock. A duplicate whose answer DIFFERS is worth knowing
    about: it means the model was not deterministic for that prompt. Both
    counts are recorded and reported rather than raising, so a partly
    duplicated file can still be analysed.
    """
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return []
    rows, seen, dup, conflict = [], {}, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["model"], r["condition"], r["template"], r["item_id"],
                   r["shuffled"], r["temperature"], r["seed"])
            if key in seen:
                dup += 1
                if seen[key]["parsed"] != r["parsed"]:
                    conflict += 1
                continue
            seen[key] = r
            rows.append(r)
    if not quiet:
        INTEGRITY.append((filename, len(rows) + dup, len(rows), dup, conflict))
    return rows


def is_correct(row):
    """1 = correct, 0 = not correct, None = excluded from accuracy."""
    if row["outcome"] == "correct":
        return 1
    if row["outcome"] == "invalid" and not INVALID_AS_INCORRECT:
        return None
    return 0


def accuracy(rows):
    vals = [v for v in (is_correct(r) for r in rows) if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def models_of(rows):
    return sorted(set(r["model"] for r in rows))


def subset(rows, model=None, condition=None, template=None):
    return [r for r in rows
            if (model is None or r["model"] == model)
            and (condition is None or r["condition"] == condition)
            and (template is None or r["template"] == template)]


def chance_baseline(rows):
    """Mean of 1/n_options. MC1 items have 2-13 options, so chance is not 25%."""
    per_item = {r["item_id"]: r["n_options"] for r in rows}
    return sum(1.0 / n for n in per_item.values()) / len(per_item)


def write_csv(name, header, table):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)


def paired_vectors(rows, model, cond_a, cond_b):
    """Correctness for items answered under BOTH conditions by one model."""
    a = {r["item_id"]: is_correct(r) for r in subset(rows, model, cond_a)}
    b = {r["item_id"]: is_correct(r) for r in subset(rows, model, cond_b)}
    ids = sorted(i for i in set(a) & set(b) if a[i] is not None and b[i] is not None)
    return [a[i] for i in ids], [b[i] for i in ids]


def mcnemar_pair(rows, model, baseline, persona):
    """McNemar on baseline vs persona. Returns (b, c, statistic, p, method).

    b = baseline right, persona wrong;  c = baseline wrong, persona right.
    Only these discordant pairs carry information. The exact binomial test is
    used when there are few of them, where the chi-square approximation is
    unreliable.
    """
    x, y = paired_vectors(rows, model, baseline, persona)
    both = sum(1 for i, j in zip(x, y) if i == 1 and j == 1)
    b = sum(1 for i, j in zip(x, y) if i == 1 and j == 0)
    c = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)
    neither = sum(1 for i, j in zip(x, y) if i == 0 and j == 0)
    if b + c == 0:
        # Both conditions answered every item identically: nothing to test.
        # Must be handled here or the correction step receives a NaN.
        return b, c, 0.0, 1.0, "no discordant pairs"
    exact = (b + c) < 25
    res = mcnemar([[both, b], [c, neither]], exact=exact, correction=not exact)
    return b, c, res.statistic, res.pvalue, "exact" if exact else "chi2+cc"


def cochran_q(rows, model):
    """Omnibus test: do all five conditions have the same accuracy?"""
    per_cond = {c: {r["item_id"]: is_correct(r) for r in subset(rows, model, c)}
                for c in CONDITIONS}
    ids = sorted(set.intersection(*[set(d) for d in per_cond.values()]))
    ids = [i for i in ids if all(per_cond[c][i] is not None for c in CONDITIONS)]
    matrix = np.array([[per_cond[c][i] for c in CONDITIONS] for i in ids])
    res = cochrans_q(matrix)
    return res.statistic, res.pvalue, len(ids)


def agreement(rows_a, rows_b):
    """Share of shared (model, condition, item) cells with the same answer."""
    a = {(r["model"], r["condition"], r["item_id"]): r for r in rows_a}
    b = {(r["model"], r["condition"], r["item_id"]): r for r in rows_b}
    keys = set(a) & set(b)
    if not keys:
        return None, 0
    same = sum(1 for k in keys if a[k]["parsed"] == b[k]["parsed"])
    return same / len(keys), len(keys)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #

def report_inventory():
    section("0. DATA INVENTORY AND INTEGRITY")
    print("   %-28s %8s %8s %7s %9s" % ("file", "lines", "unique", "dupes", "conflicts"))
    table = []
    for name, lines, uniq, dup, conflict in INTEGRITY:
        flag = ""
        if dup:
            flag = "  <- duplicated rows (two runs wrote at once)"
        if conflict:
            flag = "  <- %d duplicate(s) DISAGREE: model not deterministic" % conflict
        print("   %-28s %8d %8d %7d %9d%s" % (name, lines, uniq, dup, conflict, flag))
        table.append([name, lines, uniq, dup, conflict])
    write_csv("inventory.csv", ["file", "lines", "unique_rows", "duplicates", "conflicts"], table)


def report_descriptives(tqa, mmlu):
    section("1. ACCURACY BY MODEL AND CONDITION")
    print("   invalid counted as incorrect: %s" % INVALID_AS_INCORRECT)
    print("   TruthfulQA chance baseline: %.4f    MMLU chance baseline: %.4f"
          % (chance_baseline(tqa), chance_baseline(mmlu)))
    table = []
    for label, rows in (("TruthfulQA", tqa), ("MMLU", mmlu)):
        print("\n   " + label)
        print("   " + "%-14s" % "model" + "".join("%12s" % c for c in CONDITIONS))
        for m in models_of(rows):
            accs = [accuracy(subset(rows, m, c))[0] for c in CONDITIONS]
            print("   " + "%-14s" % m + "".join("%11.1f%%" % (a * 100) for a in accs))
            table.append([label, m] + ["%.4f" % a for a in accs])
    write_csv("accuracy.csv", ["dataset", "model"] + CONDITIONS, table)


def report_invalid(tqa, mmlu):
    section("2. INVALID RATE (refusals and unparseable replies)")
    table = []
    for label, rows in (("TruthfulQA", tqa), ("MMLU", mmlu)):
        print("\n   " + label)
        print("   " + "%-14s" % "model" + "".join("%13s" % c for c in CONDITIONS))
        for m in models_of(rows):
            line, cells = "", []
            for c in CONDITIONS:
                sub = subset(rows, m, c)
                n = sum(1 for r in sub if r["outcome"] == "invalid")
                line += "%7d%5.1f%%" % (n, 100.0 * n / len(sub))
                cells.append(str(n))
            print("   " + "%-14s" % m + line)
            table.append([label, m] + cells)
    write_csv("invalid_counts.csv", ["dataset", "model"] + CONDITIONS, table)


def report_hypothesis(tqa, mmlu):
    """The directional prediction: persona hurts TruthfulQA more than MMLU."""
    section("3. TQA DELTA MINUS MMLU DELTA  (the hypothesis test)")
    print("   delta = persona accuracy - baseline accuracy, in percentage points.")
    print("   A negative TQA delta with a near-zero MMLU delta is the prediction.")
    table = []
    for baseline in BASELINES:
        print("\n   baseline = " + baseline)
        print("   " + "%-14s%-11s%9s%9s%9s"
              % ("model", "persona", "TQA d", "MMLU d", "diff"))
        for m in models_of(tqa):
            for p in PERSONAS:
                t = (accuracy(subset(tqa, m, p))[0]
                     - accuracy(subset(tqa, m, baseline))[0]) * 100
                u = (accuracy(subset(mmlu, m, p))[0]
                     - accuracy(subset(mmlu, m, baseline))[0]) * 100
                print("   " + "%-14s%-11s%+9.2f%+9.2f%+9.2f" % (m, p, t, u, t - u))
                table.append([baseline, m, p, "%.2f" % t, "%.2f" % u, "%.2f" % (t - u)])
    write_csv("hypothesis_deltas.csv",
              ["baseline", "model", "persona", "tqa_delta_pp",
               "mmlu_delta_pp", "difference_pp"], table)


def report_significance(tqa, mmlu):
    section("4. SIGNIFICANCE TESTS")
    table, family = [], []
    for label, rows in (("TruthfulQA", tqa), ("MMLU", mmlu)):
        print("\n   %s - Cochran's Q across all five conditions" % label)
        print("   " + "%-14s%10s%12s%8s" % ("model", "Q", "p", "items"))
        for m in models_of(rows):
            q, p, n = cochran_q(rows, m)
            print("   " + "%-14s%10.3f%12.2e%8d" % (m, q, p, n))
            table.append([label, m, "cochran_q", "", "%.4f" % q, "%.3e" % p, n])

        print("\n   %s - McNemar, baseline vs persona (paired)" % label)
        print("   " + "%-14s%-10s%-11s%6s%6s%11s  %s"
              % ("model", "baseline", "persona", "b", "c", "p", "method"))
        for m in models_of(rows):
            for baseline in BASELINES:
                for p_name in PERSONAS:
                    b, c, stat, pv, how = mcnemar_pair(rows, m, baseline, p_name)
                    print("   " + "%-14s%-10s%-11s%6d%6d%11.4f  %s"
                          % (m, baseline, p_name, b, c, pv, how))
                    table.append([label, m, "mcnemar", "%s vs %s" % (baseline, p_name),
                                  "b=%d c=%d" % (b, c), "%.3e" % pv, how])
                    family.append((label, m, baseline, p_name, pv))
    print("\n   b = baseline correct & persona wrong;  c = baseline wrong & persona correct")

    # These model-level tests are a family too, not just the category tests.
    print("\n   Benjamini-Hochberg across all %d model-level McNemar tests" % len(family))
    rej, adj, _, _ = multipletests([f[4] for f in family], alpha=0.05, method="fdr_bh")
    kept = [(f, a) for f, r, a in zip(family, rej, adj) if r]
    print("   raw p < .05: %d | survive BH: %d"
          % (sum(1 for f in family if f[4] < 0.05), len(kept)))
    for (label, m, baseline, p_name, pv), a in kept:
        print("      KEEP %-11s %-14s %-8s %-10s p=%.5f  p_bh=%.5f"
              % (label, m, baseline, p_name, pv, a))
    for (label, m, baseline, p_name, pv), r_, a_ in zip(family, rej, adj):
        table.append([label, m, "mcnemar_bh", "%s vs %s" % (baseline, p_name),
                      "", "%.3e" % a_, "significant" if r_ else "ns"])
    write_csv("significance.csv",
              ["dataset", "model", "test", "comparison",
               "statistic", "p_value", "note"], table)


def report_categories(tqa):
    section("5. PER-CATEGORY TESTS WITH BENJAMINI-HOCHBERG CORRECTION")
    sizes = collections.Counter()
    for r in subset(tqa, models_of(tqa)[0], "none"):
        sizes[r["category"]] += 1
    tested = sorted(c for c, n in sizes.items() if n >= MIN_CATEGORY_N)
    skipped = sorted(c for c, n in sizes.items() if n < MIN_CATEGORY_N)
    print("   MIN_CATEGORY_N = %d" % MIN_CATEGORY_N)
    print("   %d categories: %d tested, %d too small"
          % (len(sizes), len(tested), len(skipped)))
    if skipped:
        print("   not tested: " + ", ".join("%s(%d)" % (c, sizes[c]) for c in skipped))

    table = []
    for m in models_of(tqa):
        for baseline in BASELINES:
            for p_name in PERSONAS:
                raw = []
                for cat in tested:
                    sub = [r for r in tqa if r["category"] == cat]
                    b, c, stat, pv, how = mcnemar_pair(sub, m, baseline, p_name)
                    raw.append((cat, b, c, pv))
                if not raw:
                    continue
                rej, adj, _, _ = multipletests([x[3] for x in raw],
                                               alpha=0.05, method="fdr_bh")
                for (cat, b, c, pv), r_, a_ in zip(raw, rej, adj):
                    table.append([m, baseline, p_name, cat, sizes[cat], b, c,
                                  "%.5f" % pv, "%.5f" % a_, bool(r_)])
                hits = [x[0] for x, r_ in zip(raw, rej) if r_]
                print("   %-14s%-9s%-11s significant after BH: %d/%d%s"
                      % (m, baseline, p_name, len(hits), len(raw),
                         ("  -> " + ", ".join(hits)) if hits else ""))
    write_csv("category_tests.csv",
              ["model", "baseline", "persona", "category", "n_items",
               "b", "c", "p_raw", "p_bh", "significant_bh"], table)


def report_templates(tmpl, tqa):
    """Hau's point 4: does the prompt format change the result?"""
    section("6. PROMPT FORMAT SENSITIVITY (templates.jsonl)")
    if not tmpl:
        print("   templates.jsonl not found")
        return
    present = sorted(set(r["template"] for r in tmpl))
    table = []

    print("   Accuracy by template")
    print("   " + "%-14s" % "model" + "".join("%9s" % t for t in present) + "%10s" % "spread")
    for m in models_of(tmpl):
        accs = []
        for t in present:
            sub = subset(tmpl, m, None, t)
            accs.append(accuracy(sub)[0] * 100 if sub else float("nan"))
        print("   " + "%-14s" % m + "".join("%8.1f%%" % a for a in accs)
              + "%9.1f pp" % (max(accs) - min(accs)))
        table.append([m, "accuracy"] + ["%.2f" % a for a in accs])

    print("\n   Invalid rate by template")
    print("   " + "%-14s" % "model" + "".join("%9s" % t for t in present))
    for m in models_of(tmpl):
        cells = []
        for t in present:
            sub = subset(tmpl, m, None, t)
            cells.append(100.0 * sum(r["outcome"] == "invalid" for r in sub) / len(sub))
        print("   " + "%-14s" % m + "".join("%8.1f%%" % c for c in cells))
        table.append([m, "invalid"] + ["%.2f" % c for c in cells])

    print("\n   Same question, same persona, different format: share of identical answers")
    idx = {}
    for r in tmpl:
        idx.setdefault(r["template"], {})[(r["model"], r["condition"], r["item_id"])] = r
    pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1:]]
    print("   " + "%-14s" % "model" + "".join("%14s" % ("%s vs %s" % p) for p in pairs))
    for m in models_of(tmpl):
        cells = ""
        for a, b in pairs:
            ks = [k for k in idx[a] if k[0] == m and k in idx[b]]
            ag = sum(idx[a][k]["parsed"] == idx[b][k]["parsed"] for k in ks)
            cells += "%13.1f%%" % (100.0 * ag / len(ks)) if ks else "%14s" % "-"
        print("   " + "%-14s" % m + cells)

    print("\n   Condition ranking per template (best -> worst)")
    for m in models_of(tmpl):
        print("   " + m)
        for t in present:
            accs = sorted(((accuracy(subset(tmpl, m, c, t))[0] * 100, c)
                           for c in CONDITIONS if subset(tmpl, m, c, t)), reverse=True)
            print("     %-4s %s" % (t, "  ".join("%s(%.0f%%)" % (c, a) for a, c in accs)))
    write_csv("templates.csv", ["model", "measure"] + present, table)


def report_control_spread(ctrl, tqa):
    """Is a persona effect bigger than plain prompt sensitivity?"""
    section("7. CONTROL PARAPHRASES vs PERSONAS (prompt sensitivity)")
    if not ctrl:
        print("   control_paraphrases.jsonl not found")
        return
    conds = sorted(set(r["condition"] for r in ctrl))
    items = sorted(set(r["item_id"] for r in ctrl))
    print("   %d control paraphrases, questions %d-%d"
          % (len(conds), min(items), max(items)))
    print("   Personas are read from main.jsonl restricted to the same questions,")
    print("   so the two spreads are measured on identical items.\n")

    ids = set(items)
    table = []
    print("   " + "%-14s%11s%11s%15s%16s"
          % ("model", "ctrl min", "ctrl max", "ctrl spread", "persona spread"))
    for m in models_of(ctrl):
        accs = []
        incomplete = []
        for c in conds:
            sub = subset(ctrl, m, c)
            if len(sub) < len(ids):
                incomplete.append("%s(%d/%d)" % (c, len(sub), len(ids)))
                continue
            accs.append((accuracy(sub)[0] * 100, c))
        if not accs:
            print("   %-14s no complete conditions yet" % m)
            continue
        pers = []
        for p in CONDITIONS:
            sub = [r for r in tqa if r["model"] == m and r["condition"] == p
                   and r["item_id"] in ids]
            if sub:
                pers.append((accuracy(sub)[0] * 100, p))
        cmin, cmax = min(accs)[0], max(accs)[0]
        pspread = max(pers)[0] - min(pers)[0] if pers else float("nan")
        print("   %-14s%10.1f%%%10.1f%%%12.1f pp%13.1f pp"
              % (m, cmin, cmax, cmax - cmin, pspread))
        table.append([m, "%.2f" % cmin, "%.2f" % cmax,
                      "%.2f" % (cmax - cmin), "%.2f" % pspread, len(accs)])
        if incomplete:
            print("      incomplete, excluded: %s" % ", ".join(incomplete))
    print("\n   Read it as: if the control spread is as large as the persona spread,")
    print("   a persona difference of that size is not evidence of a persona effect.")
    write_csv("control_spread.csv",
              ["model", "control_min", "control_max", "control_spread_pp",
               "persona_spread_pp", "n_complete_paraphrases"], table)


def report_robustness(tqa):
    section("8. REPRODUCIBILITY, ORDER AND TEMPERATURE")
    main_rows = tqa

    for other, label in (("shuffle.jsonl", "run-to-run, same settings (noise floor)"),
                         ("noshuffle.jsonl", "shuffled vs unshuffled options")):
        rows = load(other)
        if not rows:
            continue
        ag, n = agreement(main_rows, rows)
        if ag is not None:
            print("   %-42s %.1f%% identical (%d rows)" % (label, ag * 100, n))

    temps = sorted(f for f in os.listdir(RESULTS_DIR)
                   if f.startswith("temp_") and f.endswith(".jsonl"))
    if temps:
        print("\n   Temperature 0.7, agreement between seeds")
        loaded = {f: load(f) for f in temps}
        for i, a in enumerate(temps):
            for b in temps[i + 1:]:
                ag, n = agreement(loaded[a], loaded[b])
                if ag is not None:
                    flag = "   <- identical: the seeds are not independent" if ag > 0.99 else ""
                    print("      %-22s vs %-22s %.1f%% (%d)%s" % (a, b, ag * 100, n, flag))

        print("\n   Accuracy at temperature 0.7 vs 0.0 (same questions)")
        ids = set(r["item_id"] for r in next(iter(loaded.values())))
        base = [r for r in main_rows if r["item_id"] in ids]
        print("      %-14s%12s%12s%12s%12s"
              % ("model", "temp 0.0", "seed 1", "seed 2", "seed 3"))
        for m in models_of(base):
            cells = "%11.1f%%" % (accuracy(subset(base, m))[0] * 100)
            for f in temps:
                sub = subset(loaded[f], m)
                cells += "%11.1f%%" % (accuracy(sub)[0] * 100) if sub else "%12s" % "-"
            print("      %-14s%s" % (m, cells))


def report_extra_personas(tqa, mmlu):
    """persona_d / persona_e, compared with the frozen conditions."""
    section("9. EXTRA PERSONAS (non-Christian religions)")
    for label, fname, base in (("TruthfulQA", "extra_personas_tqa.jsonl", tqa),
                               ("MMLU", "extra_personas_mmlu.jsonl", mmlu)):
        rows = [r for r in load(fname) if r["template"] == "t1"]
        if not rows:
            print("   %s: not collected" % label)
            continue
        extra_conds = sorted(set(r["condition"] for r in rows))
        items = set(r["item_id"] for r in rows)
        print("\n   %s - t1 only, %d questions, conditions %s"
              % (label, len(items), extra_conds))
        cols = ["none", "control"] + extra_conds
        print("   " + "%-14s" % "model" + "".join("%12s" % c for c in cols))
        for m in models_of(rows):
            cells = ""
            for c in ["none", "control"]:
                sub = [r for r in base if r["model"] == m and r["condition"] == c
                       and r["item_id"] in items]
                cells += "%11.1f%%" % (accuracy(sub)[0] * 100) if sub else "%12s" % "-"
            for c in extra_conds:
                sub = subset(rows, m, c)
                cells += "%11.1f%%" % (accuracy(sub)[0] * 100) if sub else "%12s" % "-"
            print("   " + "%-14s" % m + cells)
    print("\n   Only t1 rows are used: these files also hold t2 and t3, and")
    print("   aggregating without filtering would count every question three times.")


def report_generation():
    """The open-ended slice. Scores come from the blind hand-scored sheet."""
    section("10. GENERATION SLICE")
    path = os.path.join(RESULTS_DIR, "generation.jsonl")
    if not os.path.exists(path):
        print("   not collected -- run: python 04_generation.py")
        return
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    lens = [len(r["generation"]) for r in rows]
    print("   %d generations | models %s | %d questions | %d conditions"
          % (len(rows), sorted(set(r["model"] for r in rows)),
             len(set(r["item_id"] for r in rows)),
             len(set(r["condition"] for r in rows))))
    print("   answer length: mean %.0f chars (min %d, max %d) | empty: %d"
          % (sum(lens) / len(lens), min(lens), max(lens),
             sum(1 for r in rows if not r["generation"].strip())))

    sheet = os.path.join(RESULTS_DIR, "generation_sheet.csv")
    if not os.path.exists(sheet):
        print("   no scoring sheet yet -- run: python 05_score_generation.py sheet")
        return
    with open(sheet, encoding="utf-8") as f:
        scored = [r for r in csv.DictReader(f) if (r.get("score") or "").strip()]
    if not scored:
        print("   sheet built but not scored yet -- %d rows awaiting hand scoring" % len(rows))
        print("   (this is the only step in the study that cannot be automated)")
        return
    print("   %d of %d scored -- see 05_score_generation.py merge for the breakdown"
          % (len(scored), len(rows)))


def main():
    tqa, mmlu = load("main.jsonl"), load("mmlu.jsonl")
    tmpl, ctrl = load("templates.jsonl"), load("control_paraphrases.jsonl")
    report_inventory()
    if not tqa or not mmlu:
        raise SystemExit("main.jsonl and mmlu.jsonl are required")
    report_descriptives(tqa, mmlu)
    report_invalid(tqa, mmlu)
    report_hypothesis(tqa, mmlu)
    report_significance(tqa, mmlu)
    report_categories(tqa)
    report_templates(tmpl, tqa)
    report_control_spread(ctrl, tqa)
    report_robustness(tqa)
    report_extra_personas(tqa, mmlu)
    report_generation()
    print("\nCSV copies written to %s/" % OUT_DIR)


if __name__ == "__main__":
    main()
