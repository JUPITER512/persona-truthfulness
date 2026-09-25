# All figures of the study. Saved as PNG (for Word) and PDF in results/figures/.

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # draw into files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from analysis.common import (BOOTSTRAP_N, CONDITIONS, FIG_DIR, PERSONAS, accuracy, chance_baseline,
                             models_of, paired_vectors, section, subset)
from analysis.mmlu_domains import DOMAIN_OF, MMLU_DOMAINS

COLOR = {"none": "#4D4D4D", "control": "#A6A6A6",
         "persona_a": "#E69F00", "persona_b": "#56B4E9", "persona_c": "#009E73",
         "persona_d": "#CC79A7", "persona_e": "#0072B2"}
LABEL = {"none": "no system prompt", "control": "helpful assistant",
         "persona_a": "persona A", "persona_b": "persona B", "persona_c": "persona C",
         "persona_d": "persona D", "persona_e": "persona E"}


def setup_style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "figure.dpi": 100})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{extension}"))
    plt.close(fig)
    print(f"   {name}.png / .pdf")


def short(model):
    return model.split(":")[0]  # "qwen2.5:7b" -> "qwen2.5"


def fig_accuracy(ci, tqa, mmlu):
    """Figure 1: accuracy of every model and condition, with 95% Wilson intervals."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, (label, rows) in zip(axes, [("TruthfulQA", tqa), ("MMLU", mmlu)]):
        models = models_of(rows)
        width = 0.16
        for j, c in enumerate(CONDITIONS):
            xs = np.arange(len(models)) + (j - 2) * width
            acc = [100 * ci[(label, m, c)][0] for m in models]
            below = [100 * (ci[(label, m, c)][0] - ci[(label, m, c)][1]) for m in models]
            above = [100 * (ci[(label, m, c)][2] - ci[(label, m, c)][0]) for m in models]
            ax.bar(xs, acc, width, color=COLOR[c], label=LABEL[c], yerr=[below, above],
                   error_kw=dict(lw=0.6, capsize=1.5))
        ax.axhline(100 * chance_baseline(rows), color="black", lw=0.7, ls=":")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([short(m) for m in models])
        ax.set_title(label)
        ax.set_ylim(0, 85)
    axes[0].set_ylabel("accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, -0.09, "Error bars: 95% Wilson CI. Dotted line: chance.", ha="center", fontsize=7)
    save(fig, "fig1_accuracy")


def fig_hypothesis(boot):
    """Figure 2 (the main result): TruthfulQA effect, MMLU effect and their difference."""
    setup_style()
    keys = [k for k in boot if k[0] == "none"]
    models = sorted(set(k[1] for k in keys))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.4), sharey=True)
    y_labels = []
    for i, m in enumerate(models):
        for j, p in enumerate(PERSONAS):
            y_labels.append((i * 4 + j, f"{short(m)}  {p.replace('persona_', '')}"))
    for ax, part, title in zip(axes, ["tqa", "mmlu", "did"],
                               ["TruthfulQA delta", "MMLU delta", "Difference (TQA - MMLU)"]):
        for i, m in enumerate(models):
            for j, p in enumerate(PERSONAS):
                estimate, low, high = boot[("none", m, p)][part]
                y = i * 4 + j
                ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]], fmt="o", ms=3.5,
                            color=COLOR[p], lw=1)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("percentage points")
    axes[0].set_yticks([y for y, _ in y_labels])
    axes[0].set_yticklabels([text for _, text in y_labels], fontsize=7)
    axes[0].invert_yaxis()
    fig.text(0.5, -0.04, "Persona minus no-system-prompt baseline. Bars: 95% bootstrap CI, "
             f"{BOOTSTRAP_N} item resamples.", ha="center", fontsize=7)
    save(fig, "fig2_hypothesis_difference")


def fig_control_spread(ctrl, tqa):
    """Figure 3: spread of the 10 paraphrases next to the five conditions."""
    if not ctrl:
        return
    setup_style()
    ids = set(row["item_id"] for row in ctrl)
    models = models_of(ctrl)
    paraphrases = sorted(set(row["condition"] for row in ctrl))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, m in enumerate(models):
        control_accs = [100 * accuracy(subset(ctrl, m, c))[0]
                        for c in paraphrases if len(subset(ctrl, m, c)) >= len(ids)]
        jitter = np.random.default_rng(i).uniform(-0.07, 0.07, len(control_accs))
        ax.scatter(i - 0.2 + jitter, control_accs, s=12, color="#A6A6A6",
                   label="10 control paraphrases" if i == 0 else None, zorder=2)
        ax.vlines(i - 0.2, min(control_accs), max(control_accs), color="#A6A6A6", lw=10, alpha=0.25)
        for c in CONDITIONS:
            sub = [row for row in tqa if row["model"] == m and row["condition"] == c and row["item_id"] in ids]
            ax.scatter(i + 0.15, 100 * accuracy(sub)[0], s=26, color=COLOR[c],
                       marker="D" if c == "control" else "o",
                       label=LABEL[c] if i == 0 else None, zorder=3,
                       edgecolor="black" if c == "control" else "white", lw=0.6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([short(m) for m in models])
    ax.set_ylabel(f"TruthfulQA accuracy (%), items 0-{max(ids)}")
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title("Persona spread vs paraphrase spread (same items)", fontsize=9)
    save(fig, "fig3_control_spread")


def fig_invalid(tqa, mmlu):
    """Figure 4: share of invalid replies."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, (label, rows) in zip(axes, [("TruthfulQA", tqa), ("MMLU", mmlu)]):
        models = models_of(rows)
        width = 0.16
        for j, c in enumerate(CONDITIONS):
            xs = np.arange(len(models)) + (j - 2) * width
            values = [100.0 * np.mean([row["outcome"] == "invalid" for row in subset(rows, m, c)])
                      for m in models]
            ax.bar(xs, values, width, color=COLOR[c], label=LABEL[c])
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([short(m) for m in models])
        ax.set_title(label)
    axes[0].set_ylabel("invalid replies (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig4_invalid_rate")


def fig_categories(tqa):
    """Figure 5: mean persona effect per TruthfulQA category (heat map)."""
    setup_style()
    models = models_of(tqa)
    sizes = Counter(row["category"] for row in subset(tqa, models[0], "none"))
    categories = [c for c, n in sizes.items()]
    grid = np.zeros((len(categories), len(models)))
    for i, category in enumerate(categories):
        rows_c = [row for row in tqa if row["category"] == category]
        for j, m in enumerate(models):
            base = accuracy(subset(rows_c, m, "none"))[0]
            grid[i, j] = 100 * (np.mean([accuracy(subset(rows_c, m, p))[0] for p in PERSONAS]) - base)
    order = np.argsort(-grid.mean(axis=1))
    grid = grid[order]
    categories = [categories[i] for i in order]
    fig, ax = plt.subplots(figsize=(4.6, 8.4))
    limit = 25  # small categories reach +-66 pp and would wash out the colours of the rest
    image = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([short(m) for m in models], fontsize=7)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([f"{c} ({sizes[c]})" for c in categories], fontsize=6.5)
    ax.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, extend="both")
    colorbar.set_label("mean persona delta vs none (pp)", fontsize=7)
    ax.set_title("TruthfulQA categories (n items)", fontsize=9)
    save(fig, "fig5_truthfulqa_categories")


def fig_templates(tmpl):
    """Figure 6: accuracy under each prompt template."""
    if not tmpl:
        return
    setup_style()
    templates = sorted(set(row["template"] for row in tmpl))
    fig, axes = plt.subplots(1, len(models_of(tmpl)), figsize=(7.4, 2.6), sharey=True)
    for ax, m in zip(axes, models_of(tmpl)):
        for c in CONDITIONS:
            ax.plot(templates, [100 * accuracy(subset(tmpl, m, c, t))[0] for t in templates],
                    marker="o", ms=3, color=COLOR[c], label=LABEL[c], lw=1)
        ax.set_title(short(m), fontsize=9)
        ax.set_xlabel("template")
    axes[0].set_ylabel("TruthfulQA accuracy (%)")
    axes[-1].legend(frameon=False, fontsize=6.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "fig6_templates")


def fig_domains(mmlu):
    """Figure 7: mean persona effect on MMLU per domain."""
    setup_style()
    models = models_of(mmlu)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    width = 0.2
    colors = ["#332288", "#88CCEE", "#CC6677", "#DDCC77"]
    for j, m in enumerate(models):
        values = []
        for domain in MMLU_DOMAINS:
            rows_d = [row for row in mmlu if DOMAIN_OF.get(row["category"]) == domain]
            base = accuracy(subset(rows_d, m, "none"))[0]
            values.append(100 * (np.mean([accuracy(subset(rows_d, m, p))[0] for p in PERSONAS]) - base))
        ax.bar(np.arange(len(MMLU_DOMAINS)) + (j - 1.5) * width, values, width,
               label=short(m), color=colors[j % 4])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(len(MMLU_DOMAINS)))
    ax.set_xticklabels(list(MMLU_DOMAINS))
    ax.set_ylabel("mean persona delta vs none (pp)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("MMLU by domain", fontsize=9)
    save(fig, "fig7_mmlu_domains")


def fig_churn(tqa, mmlu):
    """Figure 8: answers that flip vs the net change."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, (label, rows) in zip(axes, [("TruthfulQA", tqa), ("MMLU", mmlu)]):
        names = []
        flips = []
        nets = []
        for m in models_of(rows):
            for p in PERSONAS:
                x, y = paired_vectors(rows, m, "none", p)
                b = sum(1 for i, j in zip(x, y) if i == 1 and j == 0)
                c = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)
                names.append(f"{short(m)} {p[-1].upper()}")
                flips.append(100.0 * (b + c) / len(x))
                nets.append(100.0 * (c - b) / len(x))
        xs = np.arange(len(names))
        ax.bar(xs, flips, 0.7, color="#D9D9D9", label="answers that flip")
        ax.bar(xs, nets, 0.4, color=["#009E73" if v > 0 else "#D55E00" for v in nets], label="net change")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=90, fontsize=6.5)
        ax.set_title(label)
    axes[0].set_ylabel("% of items (persona vs none)")
    fig.legend(handles=[Patch(color="#D9D9D9", label="answers that flip"),
                        Patch(color="#009E73", label="net gain"),
                        Patch(color="#D55E00", label="net loss")],
               frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    save(fig, "fig8_churn")


def fig_all_personas(boot, boot_extra):
    """Figure 9: like figure 2, with the extra personas D and E."""
    if not boot_extra:
        return
    setup_style()
    both = dict(boot)
    both.update(boot_extra)
    models = sorted(set(k[1] for k in both if k[0] == "none"))
    personas = sorted(set(k[2] for k in both if k[0] == "none"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 4.6), sharey=True)
    step = len(personas) + 1
    for ax, part, title in zip(axes, ["tqa", "mmlu", "did"],
                               ["TruthfulQA delta", "MMLU delta", "Difference (TQA - MMLU)"]):
        for i, m in enumerate(models):
            for j, p in enumerate(personas):
                estimate, low, high = both[("none", m, p)][part]
                ax.errorbar(estimate, i * step + j, xerr=[[estimate - low], [high - estimate]], fmt="o",
                            ms=3.2, color=COLOR[p], lw=1,
                            label=LABEL[p] if (i == 0 and part == "tqa") else None)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("percentage points")
    axes[0].set_yticks([i * step + (len(personas) - 1) / 2 for i in range(len(models))])
    axes[0].set_yticklabels([short(m) for m in models])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=len(personas),
               loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, -0.07, "t1, persona minus no-system-prompt baseline. Bars: 95% bootstrap CI, "
             f"{BOOTSTRAP_N} item resamples. D and E are the extra personas.", ha="center", fontsize=7)
    save(fig, "fig9_all_personas")


def fig_template_effects(tmpl):
    """Figure 10: persona minus none under each template (TruthfulQA questions 0-99)."""
    if not tmpl:
        return
    setup_style()
    templates = sorted(set(row["template"] for row in tmpl))
    models = models_of(tmpl)
    fig, axes = plt.subplots(1, len(models), figsize=(7.4, 2.6), sharey=True)
    for ax, m in zip(axes, models):
        for p in PERSONAS:
            effects = [100 * (accuracy(subset(tmpl, m, p, t))[0] - accuracy(subset(tmpl, m, "none", t))[0])
                       for t in templates]
            ax.plot(templates, effects, marker="o", ms=3, color=COLOR[p], label=LABEL[p], lw=1)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(short(m), fontsize=9)
        ax.set_xlabel("template")
    axes[0].set_ylabel("persona - none (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.text(0.5, -0.25, "TruthfulQA items 0-99 (templates.jsonl): each point is 100 questions, so "
             "differences of a few points are within noise. No change survives BH (section 18).",
             ha="center", fontsize=7)
    save(fig, "fig10_template_effects")


def fig_generation(gen):
    """Figure 11: hand-scored truthful rate per condition (only when scores exist)."""
    if not gen:
        return
    setup_style()
    keys = sorted(gen, key=lambda k: (k[0], CONDITIONS.index(k[1]) if k[1] in CONDITIONS else 99))
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for i, (m, c) in enumerate(keys):
        rate, low, high = gen[(m, c)]
        ax.bar(i, 100 * rate, 0.7, color=COLOR.get(c, "#888888"),
               yerr=[[100 * (rate - low)], [100 * (high - rate)]], error_kw=dict(lw=0.6, capsize=2))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([LABEL.get(c, c) for _, c in keys], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("truthful (%), hand-scored")
    ax.set_ylim(0, 100)
    models = ", ".join(sorted(set(short(m) for m, _ in keys)))
    ax.set_title(f"Generation slice ({models}), 95% Wilson CI", fontsize=9)
    save(fig, "fig11_generation")


def make_figures(tqa, mmlu, tmpl, ctrl, ci, boot, boot_extra=None, gen=None):
    section(f"FIGURES -> {FIG_DIR}/")
    fig_accuracy(ci, tqa, mmlu)
    fig_hypothesis(boot)
    fig_control_spread(ctrl, tqa)
    fig_invalid(tqa, mmlu)
    fig_categories(tqa)
    fig_templates(tmpl)
    fig_domains(mmlu)
    fig_churn(tqa, mmlu)
    fig_all_personas(boot, boot_extra)
    fig_template_effects(tmpl)
    fig_generation(gen)
    if not gen:
        print("   fig11_generation skipped: no hand scores yet")
