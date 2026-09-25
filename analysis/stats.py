# The statistical tests used in the analysis.

import numpy as np
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from analysis.common import (BOOTSTRAP_N, CONDITIONS, INVALID_AS_INCORRECT, is_correct,
                             paired_vectors, subset)


def wilson(rows):
    """Accuracy with a 95% Wilson confidence interval: (accuracy, low, high, n)."""
    values = [is_correct(row) for row in rows]
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    low, high = proportion_confint(sum(values), len(values), alpha=0.05, method="wilson")
    return sum(values) / len(values), low, high, len(values)


def mcnemar_test(x, y):
    """McNemar test for two paired 0/1 lists. Returns (b, c, p, method).

    b = right in x but wrong in y,  c = wrong in x but right in y.
    Exact test when b + c < 25, otherwise chi-square with continuity correction.
    """
    both = 0
    b = 0
    c = 0
    neither = 0
    for i, j in zip(x, y):
        if i == 1 and j == 1:
            both += 1
        elif i == 1 and j == 0:
            b += 1
        elif i == 0 and j == 1:
            c += 1
        elif i == 0 and j == 0:
            neither += 1
    if b + c == 0:
        return b, c, 1.0, "no discordant pairs"
    exact = (b + c) < 25
    result = mcnemar([[both, b], [c, neither]], exact=exact, correction=not exact)
    if exact:
        return b, c, result.pvalue, "exact"
    return b, c, result.pvalue, "chi2+cc"


def mcnemar_pair(rows, model, condition_a, condition_b):
    """McNemar test of one model: condition_a (e.g. none) against condition_b (e.g. a persona)."""
    x, y = paired_vectors(rows, model, condition_a, condition_b)
    return mcnemar_test(x, y)


def cochran_test(matrix):
    """Cochran's Q for a 0/1 matrix (rows = questions, columns = conditions)."""
    if matrix.size == 0 or (matrix == matrix[:, :1]).all():
        return 0.0, 1.0  # every question answered the same way everywhere
    result = cochrans_q(matrix)
    return result.statistic, result.pvalue


def cochran_matrix(rows, model, conditions):
    """Build the question x condition matrix for cochran_test."""
    per_condition = {}
    for c in conditions:
        per_condition[c] = {row["item_id"]: is_correct(row) for row in subset(rows, model, c)}
    ids = sorted(set.intersection(*[set(d) for d in per_condition.values()]))
    ids = [i for i in ids if all(per_condition[c][i] is not None for c in conditions)]
    return np.array([[per_condition[c][i] for c in conditions] for i in ids])


def cochran_q(rows, model, conditions=CONDITIONS):
    """Do the conditions differ at all? Returns (Q, p, number of questions)."""
    matrix = cochran_matrix(rows, model, conditions)
    q, p = cochran_test(matrix)
    return q, p, len(matrix)


def bh(p_values):
    """Benjamini-Hochberg correction. Returns (significant?, adjusted p) lists."""
    if not p_values:
        return [], []
    reject, adjusted, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    return list(reject), list(adjusted)


# --- bootstrap -------------------------------------------------------------

def bootstrap_indices(n_items, rng):
    """BOOTSTRAP_N resamples of the question positions, drawn with replacement."""
    return rng.integers(0, n_items, size=(BOOTSTRAP_N, n_items), dtype=np.int32)


def boot_mean(values, indices):
    """Accuracy in every resample."""
    draws = values[indices]
    if INVALID_AS_INCORRECT:
        return draws.mean(axis=1)
    return np.nanmean(draws, axis=1)


def point_mean(values):
    if INVALID_AS_INCORRECT:
        return values.mean()
    return np.nanmean(values)


def boot_p(draws):
    """Two-sided bootstrap p-value for 'the true value is 0'."""
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return min(1.0, max(p, 1.0 / BOOTSTRAP_N))


def boot_ci(draws):
    """95% percentile interval."""
    return np.percentile(draws, [2.5, 97.5])


def bootstrap_difference(T, M, idx_t, idx_m, baseline, persona):
    """Persona effect on TruthfulQA (T), on MMLU (M), and their difference, with 95% CIs.

    Returns a dict with tqa, mmlu and did = (estimate, ci_low, ci_high), and p.
    All values in percentage points.
    """
    t_base = boot_mean(T[baseline], idx_t)
    m_base = boot_mean(M[baseline], idx_m)
    tqa_draws = 100 * (boot_mean(T[persona], idx_t) - t_base)
    mmlu_draws = 100 * (boot_mean(M[persona], idx_m) - m_base)
    did_draws = tqa_draws - mmlu_draws

    tqa_est = 100 * (point_mean(T[persona]) - point_mean(T[baseline]))
    mmlu_est = 100 * (point_mean(M[persona]) - point_mean(M[baseline]))
    tqa_low, tqa_high = boot_ci(tqa_draws)
    mmlu_low, mmlu_high = boot_ci(mmlu_draws)
    did_low, did_high = boot_ci(did_draws)
    return {
        "tqa": (tqa_est, tqa_low, tqa_high),
        "mmlu": (mmlu_est, mmlu_low, mmlu_high),
        "did": (tqa_est - mmlu_est, did_low, did_high),
        "p": boot_p(did_draws),
    }
