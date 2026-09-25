# Step 6: the complete analysis.
#
#   python analyse.py               # all tables and all figures
#   python analyse.py --no-figures  # tables only
#
# Reads every file in results/, writes the tables to results/analysis/ and the
# figures to results/figures/. It only computes; it never asks a model anything.
#
# Sections of the printed report:
#    0  data inventory                    11  accuracy with 95% Wilson CI
#    1  accuracy                          12  bootstrap test of the hypothesis
#    2  invalid replies                   13  Kendall's tau (rankings)
#    3  TQA delta minus MMLU delta        14  churn (answers that flip)
#    4  Cochran's Q and McNemar           15  MMLU domains, kinds of invalid replies
#    5  TruthfulQA categories             16  parser sensitivity
#    6  prompt templates                  17  extra personas: all tests
#    7  control paraphrases               18  prompt templates: tests
#    8  repeat run, order, temperature    19  control paraphrases: tests
#    9  extra personas                    20  order, temperature, pilot: tests
#   10  generation slice                  21  generation slice: hand scores

import os
import sys

from analysis import extra_personas, figures, generation, hypothesis, overview, patterns, robustness
from analysis.common import load
from config import RESULTS_DIR


def main():
    tqa = load("main.jsonl")
    mmlu = load("mmlu.jsonl")
    tmpl = load("templates.jsonl")
    ctrl = load("control_paraphrases.jsonl")

    # Load the other files once as well, so the inventory (section 0) lists every file
    other_files = ["shuffle.jsonl", "noshuffle.jsonl"]
    other_files += sorted(name for name in os.listdir(RESULTS_DIR)
                          if name.startswith("temp_") and name.endswith(".jsonl"))
    other_files += ["extra_personas_tqa.jsonl", "extra_personas_mmlu.jsonl"]
    for name in other_files:
        load(name)

    overview.report_inventory()                              # 0
    if not tqa or not mmlu:
        raise SystemExit("main.jsonl and mmlu.jsonl are needed")

    overview.report_descriptives(tqa, mmlu)                  # 1
    overview.report_invalid(tqa, mmlu)                       # 2
    hypothesis.report_hypothesis(tqa, mmlu)                  # 3
    hypothesis.report_significance(tqa, mmlu)                # 4
    hypothesis.report_categories(tqa)                        # 5
    robustness.report_templates(tmpl)                        # 6
    robustness.report_control_spread(ctrl, tqa)              # 7
    robustness.report_robustness(tqa)                        # 8
    extra_personas.report_extra_personas(tqa, mmlu)          # 9
    generation.report_generation()                           # 10
    ci = overview.report_accuracy_ci(tqa, mmlu)              # 11
    boot = hypothesis.report_bootstrap_did(tqa, mmlu)        # 12
    patterns.report_kendall(tqa, mmlu)                       # 13
    patterns.report_churn(tqa, mmlu)                         # 14
    patterns.report_mmlu_domains(mmlu)                       # 15
    patterns.report_parser_sensitivity(tqa, mmlu)            # 16
    boot_extra = extra_personas.report_extra_full(tqa, mmlu) # 17
    robustness.report_template_tests(tmpl)                   # 18
    robustness.report_control_tests(ctrl, tqa)               # 19
    robustness.report_robustness_tests(tqa)                  # 20
    gen = generation.report_generation_full()                # 21
    print(f"\ntables written to {os.path.join(RESULTS_DIR, 'analysis')}")

    if "--no-figures" not in sys.argv:
        figures.make_figures(tqa, mmlu, tmpl, ctrl, ci, boot, boot_extra, gen)


if __name__ == "__main__":
    main()
