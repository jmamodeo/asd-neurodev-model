"""Statistics, tables and figures of the paper (used by reproduce.py).

The statistical plan is fixed in docs/paper/analysis_plan.md (pre-specified before the runs): two-way ANOVA (Wnt, CCL5,
Wnt x CCL5) on the area under the training-loss curve (AUC) and on epochs-to-threshold, Welch one-tailed contrasts
against the control with Holm correction, a within-seed permutation test of the interaction, and held-out accuracy
compared with chance (25%).
"""

import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend has to be chosen before pyplot is imported)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

# Figures are drawn at the printed width of the paper (6.5 in) so that all text is at least 8 pt on the page.
PAGE_WIDTH = 6.5
plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "figure.titlesize": 9,
        "lines.linewidth": 1.4,
    }
)
FIGURE_HEIGHTS = {  # inches
    "loss_curves": 4.2,
    "sessions_to_criterion": 3.4,
    "observed_vs_additive": 6.0,
    "summary": 4.4,
}
# supplementary (learning rate 0.1) figures that differ in height from the main text
SUPPLEMENT_FIGURE_HEIGHTS = {"loss_curves": 4.8}

CONDITION_ORDER = ["control", "ccl5", "wnt", "both"]
CONDITION_LABELS = {"control": "Control", "ccl5": "CCL5", "wnt": "Wnt", "both": "Wnt + CCL5"}
CONDITION_COLORS = {"control": "#444444", "ccl5": "#1f77b4", "wnt": "#d62728", "both": "#9467bd"}
IMAGE_SET_LABELS = {"original": "Authors' images", "cifar3": "CIFAR-10 subset"}
LN4 = float(np.log(4))  # cross-entropy of a uniform guess over 4 classes
CHANCE = 25.0  # held-out accuracy (%) expected by chance with four classes

FINAL_WINDOW_SESSIONS = 10  # "final" loss = mean over the last 10 sessions
N_PERMUTATIONS = 10_000  # within-seed permutations for the interaction test
N_BOOTSTRAP_EFFECTS = 10_000  # bootstrap resamples of seeds for observed-vs-additive intervals
N_BOOTSTRAP_COHENS_D = 2_000  # bootstrap resamples for the Cohen's d intervals
PERMUTATION_TOLERANCE = 1e-12  # float tolerance when comparing permuted and observed contrasts


def _confidence_interval(values):
    """Return the mean and the half-width of its 95% t confidence interval (NaN for fewer than two values)."""
    values = np.asarray(values, float)
    n = len(values)
    if n < 2:
        return float(values.mean()), float("nan")
    return float(values.mean()), float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def _cohens_d(a, b):
    """Return Cohen's d of ``b`` relative to ``a`` (pooled standard deviation), NaN if that is zero."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / pooled_sd if pooled_sd > 0 else np.nan


def _holm_correction(p_values):
    """Return the Holm-adjusted p-values (same order as the input) as a list."""
    p_values = np.array(p_values)
    order = np.argsort(p_values)
    n_tests = len(p_values)
    adjusted = np.empty(n_tests)
    running_max = 0
    for rank, index in enumerate(order):
        running_max = max(running_max, (n_tests - rank) * p_values[index])
        adjusted[index] = min(1, running_max)
    return adjusted.tolist()


def _first_session(reached):
    """Return the first session (1-based) where ``reached`` is True; one past the last session if it never is."""
    sessions = np.where(reached)[0]
    return int(sessions[0]) + 1 if len(sessions) else len(reached) + 1  # right-censored one past the last session


def per_run_table(runs):
    """Reduce each run to the per-network measures used by the statistics (AUC, final loss, sessions to threshold, ...).

    Parameters
    ----------
    runs : list of dict
        Runs as written by experiment.run_experiment (the entries of ``reproduce_data/runs_lr*.json.gz``).

    Returns
    -------
    pandas.DataFrame
        One row per run, sorted by image set, condition and seed.
    """
    rows = []
    for run in runs:
        train_loss = np.array(run["train_loss"])
        val_loss = np.array(run["val_loss"])
        val_acc = np.array(run["val_acc"])
        halved = train_loss <= 0.5 * train_loss[0]
        below_ln4 = train_loss <= LN4
        rows.append(
            dict(
                data=run["data"],
                cond=run["cond"],
                seed=run["seed"],
                wnt=int(run["cond"] in ("wnt", "both")),
                ccl5=int(run["cond"] in ("ccl5", "both")),
                initial_loss=train_loss[0],
                auc=train_loss.mean(),
                final_loss=train_loss[-FINAL_WINDOW_SESSIONS:].mean(),
                ept_rel=_first_session(halved),
                reached_rel=int(halved.any()),
                ept_abs=_first_session(below_ln4),
                reached_abs=int(below_ln4.any()),
                val_auc=val_loss.mean(),
                final_val_loss=val_loss[-FINAL_WINDOW_SESSIONS:].mean(),
                final_val_acc=float(val_acc[-1]),
                final_train_acc=run["final_train_acc"],
                test_acc_fixed=run["final_test_acc_fixed_weights"],
                min_train_loss=train_loss.min(),
            )
        )
    return pd.DataFrame(rows).sort_values(["data", "cond", "seed"]).reset_index(drop=True)


def _anova(per_run, measure):
    """Two-way ANOVA (type II) of one measure on Wnt, CCL5 and their interaction, with partial eta squared."""
    model = smf.ols(f"{measure} ~ C(wnt)*C(ccl5)", per_run).fit()
    table = sm.stats.anova_lm(model, typ=2)
    effects = {}
    for key, name in (("C(wnt)", "Wnt"), ("C(ccl5)", "CCL5"), ("C(wnt):C(ccl5)", "Wnt x CCL5")):
        sum_sq = table.loc[key, "sum_sq"]
        effects[name] = dict(
            F=float(table.loc[key, "F"]),
            p=float(table.loc[key, "PR(>F)"]),
            partial_eta2=float(sum_sq / (sum_sq + table.loc["Residual", "sum_sq"])),
            df=[int(table.loc[key, "df"]), int(table.loc["Residual", "df"])],
        )
    return effects


def _interaction_contrast(wide_values):
    """Mean over seeds of both - wnt - ccl5 + control (columns in CONDITION_ORDER)."""
    return (wide_values[:, 3] - wide_values[:, 2] - wide_values[:, 1] + wide_values[:, 0]).mean()


def _permutation_interaction(per_run, measure, rng, n_permutations=N_PERMUTATIONS):
    """Permutation test of the interaction: shuffle the four conditions within each seed.

    Returns the observed contrast and the two-sided permutation p-value.
    """
    wide = per_run.pivot(index="seed", columns="cond", values=measure)[CONDITION_ORDER].dropna()
    wide_values = wide.values
    observed = _interaction_contrast(wide_values)
    count = 0
    for _ in range(n_permutations):
        permuted = np.array([row[rng.permutation(4)] for row in wide_values])
        count += abs(_interaction_contrast(permuted)) >= abs(observed) - PERMUTATION_TOLERANCE
    return float(observed), float((count + 1) / (n_permutations + 1))


def _mixed_model(per_run, measure):
    """Mixed model with a random intercept per seed; returns the coefficients and p-values (or the error text)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(f"{measure} ~ wnt*ccl5", per_run, groups=per_run["seed"]).fit(reml=True)
        return {
            key: dict(coef=float(model.params[key]), p=float(model.pvalues[key])) for key in ("wnt", "ccl5", "wnt:ccl5")
        }
    except Exception as error:  # pragma: no cover
        return str(error)


def compute_statistics(per_run, rng=None):
    """Compute every statistic of the paper for each image set in the per-run table.

    Parameters
    ----------
    per_run : pandas.DataFrame
        Output of ``per_run_table``.
    rng : numpy.random.Generator, optional
        Random generator for the bootstrap and permutation tests (default: seed 0).

    Returns
    -------
    dict
        For each image set: group sizes, summaries with 95% CIs, ANOVAs, permutation and mixed-model results,
        contrasts against the control (Holm-corrected) and held-out accuracy against chance.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    def bootstrap_d_interval(a, b, n=N_BOOTSTRAP_COHENS_D):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d_values = [_cohens_d(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n)]
        return np.nanpercentile(d_values, [2.5, 97.5]).tolist()

    summary_measures = (
        "initial_loss",
        "auc",
        "final_loss",
        "ept_rel",
        "ept_abs",
        "val_auc",
        "final_val_loss",
        "final_val_acc",
        "final_train_acc",
        "test_acc_fixed",
    )
    contrast_conditions = ("ccl5", "wnt", "both")

    results = {}
    for image_set in sorted(per_run.data.unique()):
        subset = per_run[per_run.data == image_set]
        stats_for_set = {
            "n_per_cond": {condition: int((subset.cond == condition).sum()) for condition in CONDITION_ORDER},
            "summary": {},
            "anova": {},
            "contrasts": {},
            "accuracy": {},
        }
        for condition in CONDITION_ORDER:
            condition_runs = subset[subset.cond == condition]
            condition_summary = {}
            for measure in summary_measures:
                mean, half_width = _confidence_interval(condition_runs[measure])
                condition_summary[measure] = dict(mean=mean, ci95=half_width)
            condition_summary["frac_reached_rel"] = float(condition_runs.reached_rel.mean())
            condition_summary["frac_reached_abs"] = float(condition_runs.reached_abs.mean())
            stats_for_set["summary"][condition] = condition_summary
        for measure in ("auc", "ept_rel", "ept_abs", "final_loss", "final_val_acc"):
            stats_for_set["anova"][measure] = _anova(subset, measure)
        with_log_auc = subset.copy()
        with_log_auc["log_auc"] = np.log(with_log_auc.auc)
        stats_for_set["anova"]["log_auc"] = _anova(with_log_auc, "log_auc")
        stats_for_set["perm_interaction_auc"] = dict(
            zip(("observed_contrast", "p"), _permutation_interaction(subset, "auc", rng), strict=True)
        )
        stats_for_set["perm_interaction_ept_rel"] = dict(
            zip(("observed_contrast", "p"), _permutation_interaction(subset, "ept_rel", rng), strict=True)
        )
        stats_for_set["mixed_auc"] = _mixed_model(subset, "auc")
        stats_for_set["mixed_ept_rel"] = _mixed_model(subset, "ept_rel")
        for measure in ("auc", "final_loss", "ept_rel"):
            control_values = subset[subset.cond == "control"][measure].values
            p_values = []
            contrasts = {}
            for condition in contrast_conditions:
                values = subset[subset.cond == condition][measure].values
                test = stats.ttest_ind(control_values, values, equal_var=False, alternative="less")
                contrasts[condition] = dict(
                    diff_mean=float(values.mean() - control_values.mean()),
                    t=float(test.statistic),
                    p_one_tailed=float(test.pvalue),
                    cohen_d=float(_cohens_d(control_values, values)),
                    d_ci95=bootstrap_d_interval(control_values, values),
                )
                p_values.append(test.pvalue)
            for condition, p_holm in zip(contrast_conditions, _holm_correction(p_values), strict=True):
                contrasts[condition]["p_holm"] = p_holm
            stats_for_set["contrasts"][measure] = contrasts
        for condition in CONDITION_ORDER:
            accuracy = subset[subset.cond == condition].final_val_acc.values
            if accuracy.std() > 0:
                test = stats.ttest_1samp(accuracy, CHANCE)
                t_statistic, p_value = float(test.statistic), float(test.pvalue)
            else:
                t_statistic = p_value = None
            stats_for_set["accuracy"][condition] = dict(
                mean=float(accuracy.mean()),
                ci95=_confidence_interval(accuracy)[1],
                t_vs_chance25=t_statistic,
                p_two_sided=p_value,
                frac_seeds_above_25=float((accuracy > CHANCE).mean()),
            )
        results[image_set] = stats_for_set
    return results


def _group_by_key(runs):
    """Group runs by (image set, condition)."""
    groups = {}
    for run in runs:
        groups.setdefault((run["data"], run["cond"]), []).append(run)
    return groups


def figure_loss_curves(runs, path, title, height=None):
    """Plot the mean training and held-out loss per session (with 95% CIs) for every condition and image set."""
    groups = _group_by_key(runs)
    image_sets = sorted({run["data"] for run in runs})
    fig, axes = plt.subplots(
        2,
        len(image_sets),
        figsize=(PAGE_WIDTH, height or FIGURE_HEIGHTS["loss_curves"]),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for column, image_set in enumerate(image_sets):
        for row, (key, name) in enumerate((("train_loss", "training loss"), ("val_loss", "held-out loss"))):
            ax = axes[row, column]
            for condition in CONDITION_ORDER:
                curves = np.array([run[key] for run in groups[(image_set, condition)]])
                n_runs = len(curves)
                mean = curves.mean(0)
                half_width = stats.t.ppf(0.975, n_runs - 1) * curves.std(0, ddof=1) / np.sqrt(n_runs)
                sessions = np.arange(1, curves.shape[1] + 1)
                ax.plot(sessions, mean, color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition])
                ax.fill_between(
                    sessions, mean - half_width, mean + half_width, color=CONDITION_COLORS[condition], alpha=0.18
                )
            ax.axhline(LN4, color="k", lw=0.6, ls=":")
            if column == 0:
                ax.set_ylabel(f"{name} (cross-entropy)")
            if row == 0:
                ax.set_title(IMAGE_SET_LABELS.get(image_set, image_set))
            if row == 0 and column == 0:
                ax.legend(loc="upper right")
            if row == 1:
                ax.set_xlabel("training session")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_sessions_to_criterion(per_run, sessions, path, title, height=None):
    """Cumulative fraction of networks whose training loss has reached the chance-level loss (ln 4) by each session."""
    image_sets = sorted(per_run.data.unique())
    fig, axes = plt.subplots(
        1,
        len(image_sets),
        figsize=(PAGE_WIDTH, height or FIGURE_HEIGHTS["sessions_to_criterion"]),
        squeeze=False,
        sharey=True,
    )
    for column, image_set in enumerate(image_sets):
        ax = axes[0, column]
        for condition in CONDITION_ORDER:
            # 1-based session; one past the end = never reached
            first_sessions = per_run[(per_run.data == image_set) & (per_run.cond == condition)].ept_abs.values
            session_axis = np.arange(0, sessions + 1)
            fraction_reached = np.array([(first_sessions <= session).mean() for session in session_axis])
            median = np.median(first_sessions)
            label = (
                f"{CONDITION_LABELS[condition]}: {median:g}"
                if median <= sessions
                else f"{CONDITION_LABELS[condition]}: not reached"
            )
            ax.step(session_axis, fraction_reached, where="post", color=CONDITION_COLORS[condition], lw=2, label=label)
        ax.set_xlabel("training session")
        ax.set_title(IMAGE_SET_LABELS.get(image_set, image_set))
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", title="median session")
        if column == 0:
            ax.set_ylabel("fraction of networks that\nreached chance-level loss")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _matched_arrays(per_run, measure):
    """Per-condition arrays of a measure over the seeds present in all four conditions (seed-matched)."""
    seeds = sorted(set.intersection(*[set(per_run[per_run.cond == condition].seed) for condition in CONDITION_ORDER]))
    return {
        condition: per_run[per_run.cond == condition].set_index("seed").loc[seeds, measure].values.astype(float)
        for condition in CONDITION_ORDER
    }


def _effects(by_condition):
    """The four condition values, the additive prediction CCL5 + Wnt - control, and the interaction contrast."""
    additive = by_condition["ccl5"] + by_condition["wnt"] - by_condition["control"]
    return {
        "control": by_condition["control"],
        "ccl5": by_condition["ccl5"],
        "wnt": by_condition["wnt"],
        "both": by_condition["both"],
        "additive": additive,
        "interaction": by_condition["both"] - additive,
    }


def observed_vs_additive(per_run, measure, n_bootstrap=N_BOOTSTRAP_EFFECTS, seed=0):
    """Compare the observed effects with the additive prediction.

    Means with 95% bootstrap CIs (resampling seeds, the same seeds in all four conditions) for the four conditions,
    the additive prediction CCL5 + Wnt - control, and the interaction contrast both - additive.

    Parameters
    ----------
    per_run : pandas.DataFrame
        Per-run table (one image set) from ``per_run_table``.
    measure : str
        Column to analyze, e.g. ``"auc"``.
    n_bootstrap : int
        Number of bootstrap resamples of the seeds.
    seed : int
        Seed of the bootstrap random generator.

    Returns
    -------
    dict
        Name (condition, ``additive`` or ``interaction``) mapped to (estimate, lower 95% bound, upper 95% bound).
    """
    matched = _matched_arrays(per_run, measure)
    n_seeds = len(matched["control"])
    rng = np.random.default_rng(seed)
    resampled_seeds = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    point_estimates = {name: float(values.mean()) for name, values in _effects(matched).items()}
    # arrays of shape (n_bootstrap, n_seeds)
    bootstrapped = _effects({condition: matched[condition][resampled_seeds] for condition in CONDITION_ORDER})
    estimates = {}
    for name, values in bootstrapped.items():
        means = values.mean(1)
        estimates[name] = (point_estimates[name], float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    return estimates


def figure_observed_vs_additive(per_run, statistics, path, title, height=None):
    """Plot observed means and the additive prediction (with bootstrap CIs) for AUC and final loss."""
    image_sets = sorted(per_run.data.unique())
    outcomes = (("auc", "AUC (mean training loss)"), ("final_loss", "final training loss"))
    fig, axes = plt.subplots(
        len(outcomes),
        len(image_sets),
        figsize=(PAGE_WIDTH, height or FIGURE_HEIGHTS["observed_vs_additive"]),
        squeeze=False,
    )
    categories = ["control", "ccl5", "wnt", "both", "additive"]
    names = ["Control", "CCL5", "Wnt", "Both (observed)", "Additive prediction"]
    for row, (measure, y_label) in enumerate(outcomes):
        for column, image_set in enumerate(image_sets):
            ax = axes[row, column]
            effects = observed_vs_additive(per_run[per_run.data == image_set], measure)
            for position, category in enumerate(categories):
                mean, lower, upper = effects[category]
                if category == "additive":
                    ax.errorbar(
                        position,
                        mean,
                        yerr=[[mean - lower], [upper - mean]],
                        fmt="D",
                        mfc="white",
                        mec="k",
                        color="k",
                        capsize=4,
                        ms=8,
                        lw=1.5,
                    )
                else:
                    ax.errorbar(
                        position,
                        mean,
                        yerr=[[mean - lower], [upper - mean]],
                        fmt="o",
                        color=CONDITION_COLORS[category],
                        capsize=4,
                        ms=8,
                        lw=1.5,
                    )
            ax.axhline(effects["additive"][0], color="gray", lw=0.8, ls=":")
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_ylabel(y_label)
            ax.set_xlim(-0.5, len(categories) - 0.5)
            y_low, y_high = ax.get_ylim()
            ax.set_ylim(y_low, y_high + 0.3 * (y_high - y_low))
            interaction, interaction_low, interaction_high = effects["interaction"]
            p_interaction = statistics[image_set]["anova"][measure]["Wnt x CCL5"]["p"]
            ax.set_title(f"{IMAGE_SET_LABELS.get(image_set, image_set)}")
            ax.text(
                0.03,
                0.97,
                f"observed − additive = {interaction:+.2f} [{interaction_low:+.2f}, {interaction_high:+.2f}]\n"
                f"Wnt × CCL5 (ANOVA) p = {p_interaction:.2g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_summary(per_run, path, title, height=None):
    """Plot each network's final loss, sessions to chance-level loss and held-out accuracy by condition."""
    image_sets = sorted(per_run.data.unique())
    rng = np.random.default_rng(1)
    measures = (
        ("final_loss", "final training loss", None),
        ("ept_abs", "sessions to reach ln 4\n(101 = not reached)", None),
        ("final_val_acc", "held-out accuracy (%)", CHANCE),
    )
    fig, axes = plt.subplots(
        len(image_sets), 3, figsize=(PAGE_WIDTH, height or FIGURE_HEIGHTS["summary"]), squeeze=False
    )
    for row, image_set in enumerate(image_sets):
        image_set_runs = per_run[per_run.data == image_set]
        for column, (measure, name, reference) in enumerate(measures):
            ax = axes[row, column]
            for position, condition in enumerate(CONDITION_ORDER):
                values = image_set_runs[image_set_runs.cond == condition][measure].values
                mean, half_width = _confidence_interval(values)
                ax.scatter(
                    position + rng.uniform(-0.18, 0.18, len(values)),
                    values,
                    s=10,
                    color=CONDITION_COLORS[condition],
                    alpha=0.35,
                )
                ax.errorbar(position, mean, yerr=half_width, color="k", marker="D", capsize=4, ms=5)
            if reference is not None:
                ax.axhline(reference, color="k", ls=":", lw=0.8)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["Control", "CCL5", "Wnt", "Both"], rotation=35, ha="right")
            ax.set_title(f"{IMAGE_SET_LABELS.get(image_set, image_set)}")
            ax.set_ylabel(name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def markdown_tables(statistics):
    """Format the summary statistics of every image set as Markdown tables (mean ± 95% CI)."""
    lines = []
    for image_set, image_set_stats in statistics.items():
        lines += [
            f"\n### {IMAGE_SET_LABELS.get(image_set, image_set)}",
            "| condition | n | session-1 loss | AUC | final loss | sessions to 50% of initial (frac reached) | "
            "sessions to ln 4 (frac reached) | held-out acc % | held-out loss |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for condition in CONDITION_ORDER:
            summary = image_set_stats["summary"][condition]

            def mean_ci(measure, places=2, summary=summary):
                return f"{summary[measure]['mean']:.{places}f} ± {summary[measure]['ci95']:.{places}f}"

            lines.append(
                f"| {CONDITION_LABELS[condition]} | {image_set_stats['n_per_cond'][condition]} | "
                f"{mean_ci('initial_loss')} | {mean_ci('auc')} | {mean_ci('final_loss')} | "
                f"{mean_ci('ept_rel', 1)} ({summary['frac_reached_rel']:.2f}) | "
                f"{mean_ci('ept_abs', 1)} ({summary['frac_reached_abs']:.2f}) | "
                f"{mean_ci('final_val_acc', 1)} | {mean_ci('final_val_loss')} |"
            )
    return "\n".join(lines) + "\n"


def make_figure(kind, runs, tag, figure_dir, title, height=None):
    """Write one figure to ``figure_dir/<kind>_<tag>.png`` and return its path.

    Parameters
    ----------
    kind : str
        ``loss_curves``, ``sessions_to_criterion``, ``observed_vs_additive`` or ``summary``.
    runs : list of dict
        Runs of one learning-rate configuration.
    tag : str
        Configuration tag used in the file name, e.g. ``lr0.03``.
    figure_dir : str
        Output directory (created if missing).
    title : str
        Figure title.
    height : float, optional
        Figure height in inches (default: ``FIGURE_HEIGHTS[kind]``).
    """
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f"{kind}_{tag}.png")
    if kind == "loss_curves":
        figure_loss_curves(runs, path, title, height)
    elif kind == "sessions_to_criterion":
        figure_sessions_to_criterion(per_run_table(runs), len(runs[0]["train_loss"]), path, title, height)
    elif kind == "observed_vs_additive":
        per_run = per_run_table(runs)
        figure_observed_vs_additive(per_run, compute_statistics(per_run), path, title, height)
    elif kind == "summary":
        figure_summary(per_run_table(runs), path, title, height)
    else:
        raise ValueError(kind)
    return path


def write_tables(runs, tag, table_dir):
    """Write per-run values, the statistics (JSON) and the summary table (Markdown); return the statistics dict."""
    os.makedirs(table_dir, exist_ok=True)
    per_run = per_run_table(runs)
    statistics = compute_statistics(per_run)
    per_run.to_csv(os.path.join(table_dir, f"per_run_{tag}.csv"), index=False)
    with open(os.path.join(table_dir, f"statistics_{tag}.json"), "w") as f:
        json.dump(dict(tag=tag, datasets=statistics), f, indent=1, default=float)
    with open(os.path.join(table_dir, f"summary_{tag}.md"), "w") as f:
        f.write(markdown_tables(statistics))
    return statistics
