# Step 6: the complete analysis.
#
#   python analyse.py               # all tables and all figures
#   python analyse.py --no-figures  # tables only
#
# Reads every file in results/, writes the tables to results/analysis/ and the
# figures to results/figures/. It only computes; it never asks a model anything.
#
# This version computes exactly the statistics reported in the term paper. Analyses that the
# paper does not report were removed (Kendall's tau, churn, persona effect by MMLU domain,
# Wilson intervals for the accuracy tables, and several extra tests); the section numbers of
# the remaining sections are unchanged.
#
# Sections of the printed report:
#    0  data inventory                    15  kinds of invalid MMLU replies, invalid rate by domain
#    1  accuracy                          16  parser sensitivity
#    2  invalid replies                   17  extra personas: bootstrap test of the hypothesis
#    4  Cochran's Q and McNemar           18  prompt templates: persona effect per template
#    5  TruthfulQA categories             19  control paraphrases: Cochran's Q
#    6  prompt templates                  20  order, temperature, repeat run, pilot: tests
#    7  control paraphrases               21  generation slice: hand scores
#    9  extra personas                    22  sensitivity of the hypothesis test (post hoc)
#   12  bootstrap test of the hypothesis  23  descriptive numbers quoted in the text

import os
import sys

from analysis import (extra_personas, figures, generation, hypothesis, overview, patterns, robustness,
                      sensitivity)
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
    hypothesis.report_significance(tqa, mmlu)                # 4
    hypothesis.report_categories(tqa)                        # 5
    robustness.report_templates(tmpl)                        # 6
    robustness.report_control_spread(ctrl, tqa)              # 7
    extra_personas.report_extra_personas(tqa, mmlu)          # 9
    boot = hypothesis.report_bootstrap_did(tqa, mmlu)        # 12
    patterns.report_invalid_kinds(mmlu)                      # 15
    patterns.report_parser_sensitivity(tqa, mmlu)            # 16
    boot_extra = extra_personas.report_extra_full(tqa, mmlu) # 17
    robustness.report_template_tests(tmpl)                   # 18
    robustness.report_control_tests(ctrl, tqa)               # 19
    robustness.report_robustness_tests(tqa)                  # 20
    generation.report_generation_full()                      # 21
    sensitivity.report_sensitivity(tqa, mmlu)                # 22
    sensitivity.report_text_numbers(tqa, mmlu, tmpl)         # 23
    print(f"\ntables written to {os.path.join(RESULTS_DIR, 'analysis')}")

    if "--no-figures" not in sys.argv:
        figures.make_figures(tqa, ctrl, boot, boot_extra)


if __name__ == "__main__":
    main()
