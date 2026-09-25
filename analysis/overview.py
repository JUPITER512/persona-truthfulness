# Sections 0, 1, 2 and 11: data inventory, accuracy, invalid replies, confidence intervals.

from statsmodels.stats.proportion import proportion_confint

from analysis.common import (CONDITIONS, INTEGRITY, INVALID_AS_INCORRECT, accuracy, chance_baseline,
                             is_correct, models_of, section, subset, write_csv)


def report_inventory():
    section("0. DATA INVENTORY AND INTEGRITY")
    print(f"   {'file':<28} {'lines':>8} {'unique':>8} {'dupes':>7} {'conflicts':>9}")
    table = []
    for name, lines, unique, duplicates, conflicts in INTEGRITY:
        note = ""
        if duplicates:
            note = "  <- duplicated rows (two runs wrote at once)"
        if conflicts:
            note = f"  <- {conflicts} duplicate(s) with a different answer"
        print(f"   {name:<28} {lines:>8} {unique:>8} {duplicates:>7} {conflicts:>9}{note}")
        table.append([name, lines, unique, duplicates, conflicts])
    write_csv("inventory.csv", ["file", "lines", "unique_rows", "duplicates", "conflicts"], table)


def report_descriptives(tqa, mmlu):
    section("1. ACCURACY BY MODEL AND CONDITION")
    print(f"   invalid counted as incorrect: {INVALID_AS_INCORRECT}")
    print(f"   chance: TruthfulQA {chance_baseline(tqa):.4f}   MMLU {chance_baseline(mmlu):.4f}")
    table = []
    for label, rows in [("TruthfulQA", tqa), ("MMLU", mmlu)]:
        print(f"\n   {label}")
        print("   " + f"{'model':<14}" + "".join(f"{c:>12}" for c in CONDITIONS))
        for model in models_of(rows):
            accs = [accuracy(subset(rows, model, c))[0] for c in CONDITIONS]
            print("   " + f"{model:<14}" + "".join(f"{a * 100:>11.1f}%" for a in accs))
            table.append([label, model] + [f"{a:.4f}" for a in accs])
    write_csv("accuracy.csv", ["dataset", "model"] + CONDITIONS, table)


def report_invalid(tqa, mmlu):
    section("2. INVALID REPLIES (no letter could be read)")
    table = []
    for label, rows in [("TruthfulQA", tqa), ("MMLU", mmlu)]:
        print(f"\n   {label}  (count and percent)")
        print("   " + f"{'model':<14}" + "".join(f"{c:>13}" for c in CONDITIONS))
        for model in models_of(rows):
            line = ""
            counts = []
            for c in CONDITIONS:
                sub = subset(rows, model, c)
                n_invalid = sum(1 for row in sub if row["outcome"] == "invalid")
                line += f"{n_invalid:>7d}{100.0 * n_invalid / len(sub):>5.1f}%"
                counts.append(str(n_invalid))
            print("   " + f"{model:<14}" + line)
            table.append([label, model] + counts)
    write_csv("invalid_counts.csv", ["dataset", "model"] + CONDITIONS, table)


def report_accuracy_ci(tqa, mmlu):
    """Every accuracy of section 1 with a 95% Wilson interval."""
    section("11. ACCURACY WITH 95% WILSON CONFIDENCE INTERVALS")
    table = []
    results = {}  # (dataset, model, condition) -> (accuracy, low, high), used by figure 1
    for label, rows in [("TruthfulQA", tqa), ("MMLU", mmlu)]:
        print(f"\n   {label}")
        print(f"   {'model':<14}{'condition':<11}{'acc':>9}{'95% CI':>18}{'n':>7}")
        for model in models_of(rows):
            for c in CONDITIONS:
                values = [is_correct(row) for row in subset(rows, model, c)]
                values = [v for v in values if v is not None]
                k = sum(values)
                n = len(values)
                low, high = proportion_confint(k, n, alpha=0.05, method="wilson")
                results[(label, model, c)] = (k / n, low, high)
                print(f"   {model:<14}{c:<11}{100.0 * k / n:>8.1f}%   [{100 * low:5.1f}, {100 * high:5.1f}]{n:>7d}")
                table.append([label, model, c, k, n, f"{k / n:.4f}", f"{low:.4f}", f"{high:.4f}"])
    write_csv("accuracy_ci.csv",
              ["dataset", "model", "condition", "correct", "n", "accuracy", "ci_low", "ci_high"], table)
    return results
