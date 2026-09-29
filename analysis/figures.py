# The figures of the term paper (Appendix Figures D1-D4). Saved as PNG and PDF in results/figures/.
#   fig2_hypothesis_difference  -> Figure D1
#   fig3_control_spread         -> Figure D2
#   fig5_truthfulqa_categories  -> Figure D3
#   fig9_all_personas           -> Figure D4

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # draw into files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from analysis.common import BOOTSTRAP_N, CONDITIONS, FIG_DIR, PERSONAS, accuracy, models_of, section, subset

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
        # no creation date inside the PDF, so a re-run writes identical files
        metadata = {"CreationDate": None} if extension == "pdf" else None
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{extension}"), metadata=metadata)
    plt.close(fig)
    print(f"   {name}.png / .pdf")


def short(model):
    return model.split(":")[0]  # "qwen2.5:7b" -> "qwen2.5"


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


def make_figures(tqa, ctrl, boot, boot_extra=None):
    section(f"FIGURES -> {FIG_DIR}/")
    fig_hypothesis(boot)
    fig_control_spread(ctrl, tqa)
    fig_categories(tqa)
    fig_all_personas(boot, boot_extra)
