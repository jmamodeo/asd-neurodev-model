"""Statistics, tables and figures (used by reproduce.py).

The statistical plan is fixed in docs/paper/analysis_plan.md (pre-specified before the runs): two-way ANOVA (Wnt, CCL5,
Wnt x CCL5) on the area under the training-loss curve (AUC) and on epochs-to-threshold, Welch one-tailed contrasts
against the control with Holm correction, a within-seed permutation test of the interaction, and held-out accuracy
compared with chance (25%).
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDS = ["control", "ccl5", "wnt", "both"]
LABEL = {"control": "Control", "ccl5": "CCL5", "wnt": "Wnt", "both": "Wnt + CCL5"}
COLOR = {"control": "#444444", "ccl5": "#1f77b4", "wnt": "#d62728", "both": "#9467bd"}
DATA_LABEL = {"original": "Authors' images", "cifar3": "CIFAR-10 subset"}
LN4 = float(np.log(4))   # cross-entropy of a uniform guess over 4 classes
CHANCE = 25.0


def _ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def _cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / sp if sp > 0 else np.nan


def _holm(p):
    p = np.array(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj.tolist()


def per_run_table(runs):
    rows = []
    for d in runs:
        tl, vl, va = np.array(d["train_loss"]), np.array(d["val_loss"]), np.array(d["val_acc"])

        def first(mask):
            i = np.where(mask)[0]
            return int(i[0]) + 1 if len(i) else len(tl) + 1      # right-censored one past the last session
        rows.append(dict(
            data=d["data"], cond=d["cond"], seed=d["seed"],
            wnt=int(d["cond"] in ("wnt", "both")), ccl5=int(d["cond"] in ("ccl5", "both")),
            initial_loss=tl[0], auc=tl.mean(), final_loss=tl[-10:].mean(),
            ept_rel=first(tl <= 0.5 * tl[0]), reached_rel=int((tl <= 0.5 * tl[0]).any()),
            ept_abs=first(tl <= LN4), reached_abs=int((tl <= LN4).any()),
            val_auc=vl.mean(), final_val_loss=vl[-10:].mean(), final_val_acc=float(va[-1]),
            final_train_acc=d["final_train_acc"], test_acc_fixed=d["final_test_acc_fixed_weights"],
            min_train_loss=tl.min()))
    return pd.DataFrame(rows).sort_values(["data", "cond", "seed"]).reset_index(drop=True)


def _anova(d, y):
    model = smf.ols(f"{y} ~ C(wnt)*C(ccl5)", d).fit()
    table = sm.stats.anova_lm(model, typ=2); out = {}
    for key, name in (("C(wnt)", "Wnt"), ("C(ccl5)", "CCL5"), ("C(wnt):C(ccl5)", "Wnt x CCL5")):
        ss = table.loc[key, "sum_sq"]
        out[name] = dict(F=float(table.loc[key, "F"]), p=float(table.loc[key, "PR(>F)"]),
                         partial_eta2=float(ss / (ss + table.loc["Residual", "sum_sq"])),
                         df=[int(table.loc[key, "df"]), int(table.loc["Residual", "df"])])
    return out


def _permutation_interaction(d, y, rng, n=10000):
    wide = d.pivot(index="seed", columns="cond", values=y)[CONDS].dropna(); X = wide.values
    stat = lambda M: (M[:, 3] - M[:, 2] - M[:, 1] + M[:, 0]).mean()
    observed = stat(X); count = 0
    for _ in range(n):
        P = np.array([row[rng.permutation(4)] for row in X]); count += abs(stat(P)) >= abs(observed) - 1e-12
    return float(observed), float((count + 1) / (n + 1))


def _mixed(d, y):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(f"{y} ~ wnt*ccl5", d, groups=d["seed"]).fit(reml=True)
        return {k: dict(coef=float(m.params[k]), p=float(m.pvalues[k])) for k in ("wnt", "ccl5", "wnt:ccl5")}
    except Exception as e:           # pragma: no cover
        return str(e)


def compute_statistics(df, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)

    def boot_d(a, b, n=2000):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ds = [_cohen_d(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n)]
        return np.nanpercentile(ds, [2.5, 97.5]).tolist()

    res = {}
    for data in sorted(df.data.unique()):
        d = df[df.data == data]
        R = {"n_per_cond": {c: int((d.cond == c).sum()) for c in CONDS}, "summary": {}, "anova": {}, "contrasts": {}, "accuracy": {}}
        for c in CONDS:
            x = d[d.cond == c]; s = {}
            for k in ("initial_loss", "auc", "final_loss", "ept_rel", "ept_abs", "val_auc", "final_val_loss",
                      "final_val_acc", "final_train_acc", "test_acc_fixed"):
                m, h = _ci(x[k]); s[k] = dict(mean=m, ci95=h)
            s["frac_reached_rel"] = float(x.reached_rel.mean()); s["frac_reached_abs"] = float(x.reached_abs.mean())
            R["summary"][c] = s
        for y in ("auc", "ept_rel", "ept_abs", "final_loss", "final_val_acc"):
            R["anova"][y] = _anova(d, y)
        d2 = d.copy(); d2["log_auc"] = np.log(d2.auc); R["anova"]["log_auc"] = _anova(d2, "log_auc")
        R["perm_interaction_auc"] = dict(zip(("observed_contrast", "p"), _permutation_interaction(d, "auc", rng)))
        R["perm_interaction_ept_rel"] = dict(zip(("observed_contrast", "p"), _permutation_interaction(d, "ept_rel", rng)))
        R["mixed_auc"] = _mixed(d, "auc"); R["mixed_ept_rel"] = _mixed(d, "ept_rel")
        for y in ("auc", "final_loss", "ept_rel"):
            ctrl = d[d.cond == "control"][y].values; ps = []; tmp = {}
            for c in ("ccl5", "wnt", "both"):
                x = d[d.cond == c][y].values
                t = stats.ttest_ind(ctrl, x, equal_var=False, alternative="less")
                tmp[c] = dict(diff_mean=float(x.mean() - ctrl.mean()), t=float(t.statistic), p_one_tailed=float(t.pvalue),
                              cohen_d=float(_cohen_d(ctrl, x)), d_ci95=boot_d(ctrl, x)); ps.append(t.pvalue)
            for c, ph in zip(("ccl5", "wnt", "both"), _holm(ps)):
                tmp[c]["p_holm"] = ph
            R["contrasts"][y] = tmp
        for c in CONDS:
            x = d[d.cond == c].final_val_acc.values
            if x.std() > 0:
                t = stats.ttest_1samp(x, CHANCE); tt, pp = float(t.statistic), float(t.pvalue)
            else:
                tt = pp = None
            R["accuracy"][c] = dict(mean=float(x.mean()), ci95=_ci(x)[1], t_vs_chance25=tt, p_two_sided=pp,
                                    frac_seeds_above_25=float((x > CHANCE).mean()))
        res[data] = R
    return res


def _by_key(runs):
    out = {}
    for d in runs:
        out.setdefault((d["data"], d["cond"]), []).append(d)
    return out


def figure_loss_curves(runs, path, title):
    by = _by_key(runs); datas = sorted({d["data"] for d in runs})
    fig, axs = plt.subplots(2, len(datas), figsize=(4.8 * len(datas), 6.4), squeeze=False, sharex=True)
    for j, data in enumerate(datas):
        for i, (key, name) in enumerate((("train_loss", "training loss (mean per example)"), ("val_loss", "held-out loss"))):
            ax = axs[i, j]
            for c in CONDS:
                A = np.array([r[key] for r in by[(data, c)]]); n = len(A)
                m = A.mean(0); h = stats.t.ppf(.975, n - 1) * A.std(0, ddof=1) / np.sqrt(n); x = np.arange(1, A.shape[1] + 1)
                ax.plot(x, m, color=COLOR[c], label=f"{LABEL[c]} (n={n})"); ax.fill_between(x, m - h, m + h, color=COLOR[c], alpha=.18)
            ax.axhline(LN4, color="k", lw=.6, ls=":"); ax.set_ylabel("cross-entropy"); ax.set_title(f"{DATA_LABEL.get(data, data)}: {name}")
            ax.legend(fontsize=8)
            if i == 1:
                ax.set_xlabel("training session")
    fig.suptitle(f"{title}\nmean ± 95% CI over seeds (dotted line = chance-level loss ln 4)"); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def figure_auc_interaction(df, stat, path, title):
    datas = sorted(df.data.unique())
    fig, axs = plt.subplots(1, len(datas), figsize=(4.4 * len(datas), 3.9), squeeze=False)
    for j, data in enumerate(datas):
        ax = axs[0, j]; d = df[df.data == data]
        for ccl, ls in ((0, "-"), (1, "--")):
            ms, hs = zip(*[_ci(d[(d.wnt == w) & (d.ccl5 == ccl)].auc) for w in (0, 1)])
            ax.errorbar([0, 1], ms, yerr=hs, marker="o", ls=ls, capsize=4, label="CCL5 on" if ccl else "CCL5 off")
        m = [_ci(d[(d.wnt == w) & (d.ccl5 == c)].auc)[0] for w, c in ((0, 0), (1, 0), (0, 1), (1, 1))]
        ax.plot([0, 1], [m[2], m[2] + (m[1] - m[0])], color="gray", lw=.8, ls=":", label="additive prediction")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Wnt off", "Wnt on"]); ax.set_ylabel("AUC (mean training loss)")
        p = stat[data]["anova"]["auc"]["Wnt x CCL5"]["p"]
        ax.set_title(f"{DATA_LABEL.get(data, data)}\nWnt x CCL5 interaction p = {p:.2g}"); ax.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def figure_sessions_to_criterion(df, sessions, path, title):
    """Cumulative fraction of networks whose training loss has reached the chance-level loss (ln 4) by each session."""
    datas = sorted(df.data.unique())
    fig, axs = plt.subplots(1, len(datas), figsize=(4.5 * len(datas), 4.0), squeeze=False, sharey=True)
    for j, data in enumerate(datas):
        ax = axs[0, j]
        for c in CONDS:
            t = df[(df.data == data) & (df.cond == c)].ept_abs.values            # 1-based session; one past the end = never reached
            x = np.arange(0, sessions + 1)
            y = np.array([(t <= k).mean() for k in x])
            med = np.median(t)
            lab = f"{LABEL[c]} (median {med:g})" if med <= sessions else f"{LABEL[c]} (median not reached)"
            ax.step(x, y, where="post", color=COLOR[c], lw=2, label=lab)
        ax.set_xlabel("training session"); ax.set_title(DATA_LABEL.get(data, data))
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, loc="lower right", title="median sessions to reach", title_fontsize=8)
        if j == 0:
            ax.set_ylabel("fraction of networks with training loss below ln 4")
    n = int(df.groupby(["data", "cond"]).size().min())
    fig.suptitle(f"{title}\nChance-level loss (ln 4 = {LN4:.2f}) reached, n = {n} networks per condition")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _matched_arrays(d, y):
    """Per-condition arrays of outcome y over the seeds present in all four conditions (seed-matched)."""
    seeds = sorted(set.intersection(*[set(d[d.cond == c].seed) for c in CONDS]))
    return {c: d[d.cond == c].set_index("seed").loc[seeds, y].values.astype(float) for c in CONDS}


def observed_vs_additive(d, y, n_boot=10000, seed=0):
    """Means with 95% bootstrap CIs (resampling seeds, the same seeds in all four conditions) for the four conditions,
    the additive prediction CCL5 + Wnt - control, and the interaction contrast both - additive."""
    A = _matched_arrays(d, y); n = len(A["control"])
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, size=(n_boot, n))
    def est(a):
        return {"control": a["control"], "ccl5": a["ccl5"], "wnt": a["wnt"], "both": a["both"],
                "additive": a["ccl5"] + a["wnt"] - a["control"], "interaction": a["both"] - (a["ccl5"] + a["wnt"] - a["control"])}
    point = {k: float(v.mean()) for k, v in est(A).items()}
    B = est({c: A[c][idx] for c in CONDS})           # arrays of shape (n_boot, n)
    out = {}
    for k, v in B.items():
        m = v.mean(1); out[k] = (point[k], float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))
    return out


def figure_observed_vs_additive(df, stat, path, title):
    datas = sorted(df.data.unique())
    outcomes = (("auc", "AUC (mean training loss)"), ("final_loss", "final training loss"))
    fig, axs = plt.subplots(len(outcomes), len(datas), figsize=(4.5 * len(datas), 3.5 * len(outcomes)), squeeze=False)
    cats = ["control", "ccl5", "wnt", "both", "additive"]
    names = ["Control", "CCL5", "Wnt", "Wnt + CCL5\n(observed)", "Additive\nprediction"]
    for i, (y, yl) in enumerate(outcomes):
        for j, data in enumerate(datas):
            ax = axs[i, j]; E = observed_vs_additive(df[df.data == data], y)
            for k, c in enumerate(cats):
                m, lo, hi = E[c]
                if c == "additive":
                    ax.errorbar(k, m, yerr=[[m - lo], [hi - m]], fmt="D", mfc="white", mec="k", color="k", capsize=4, ms=8, lw=1.5)
                else:
                    ax.errorbar(k, m, yerr=[[m - lo], [hi - m]], fmt="o", color=COLOR[c], capsize=4, ms=8, lw=1.5)
            ax.axhline(E["additive"][0], color="gray", lw=.8, ls=":")
            ax.set_xticks(range(len(cats))); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylabel(yl)
            ax.set_xlim(-0.5, len(cats) - 0.5)
            lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.3 * (hi_ - lo_))
            ia, il, ih = E["interaction"]; p = stat[data]["anova"][y]["Wnt x CCL5"]["p"]
            ax.set_title(f"{DATA_LABEL.get(data, data)}", fontsize=10)
            ax.text(0.03, 0.97, f"observed − additive = {ia:+.2f} [{il:+.2f}, {ih:+.2f}]\nWnt × CCL5 interaction (ANOVA) p = {p:.2g}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    fig.suptitle(f"{title}\nMeans with 95% bootstrap CIs over seeds (seed-matched); additive prediction = CCL5 + Wnt − control")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def figure_summary(df, path, title):
    datas = sorted(df.data.unique()); rng = np.random.default_rng(1)
    metrics = (("final_loss", "final training loss (last 10 sessions)", None),
               ("ept_abs", "sessions to reach chance-level loss (ln 4)\n(101 = never reached)", None),
               ("final_val_acc", "held-out accuracy (%)", CHANCE))
    fig, axs = plt.subplots(len(datas), 3, figsize=(10.5, 3.5 * len(datas)), squeeze=False)
    for i, data in enumerate(datas):
        d = df[df.data == data]
        for j, (k, name, ref) in enumerate(metrics):
            ax = axs[i, j]
            for xi, c in enumerate(CONDS):
                v = d[d.cond == c][k].values; m, h = _ci(v)
                ax.scatter(xi + rng.uniform(-.18, .18, len(v)), v, s=10, color=COLOR[c], alpha=.35)
                ax.errorbar(xi, m, yerr=h, color="k", marker="D", capsize=4, ms=5)
            if ref is not None:
                ax.axhline(ref, color="k", ls=":", lw=.8); ax.text(-0.45, ref, "chance", va="bottom", ha="left", fontsize=8)
            ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[c] for c in CONDS], fontsize=8)
            ax.set_title(f"{DATA_LABEL.get(data, data)}", fontsize=9); ax.set_ylabel(name, fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def markdown_tables(stat):
    lines = []
    for data, R in stat.items():
        lines += [f"\n### {DATA_LABEL.get(data, data)}",
                  "| condition | n | session-1 loss | AUC | final loss | sessions to 50% of initial (frac reached) | sessions to ln 4 (frac reached) | held-out acc % | held-out loss |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in CONDS:
            s = R["summary"][c]; f = lambda k, p=2: f"{s[k]['mean']:.{p}f} ± {s[k]['ci95']:.{p}f}"
            lines.append(f"| {LABEL[c]} | {R['n_per_cond'][c]} | {f('initial_loss')} | {f('auc')} | {f('final_loss')} | "
                         f"{f('ept_rel', 1)} ({s['frac_reached_rel']:.2f}) | {f('ept_abs', 1)} ({s['frac_reached_abs']:.2f}) | "
                         f"{f('final_val_acc', 1)} | {f('final_val_loss')} |")
    return "\n".join(lines) + "\n"


FIGURE_KINDS = ("loss_curves", "sessions_to_criterion", "auc_interaction", "observed_vs_additive", "summary")


def make_figure(kind, runs, tag, fig_dir, title):
    """Write one figure (kind in FIGURE_KINDS) to fig_dir/<kind>_<tag>.png and return its path."""
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, f"{kind}_{tag}.png")
    if kind == "loss_curves":
        figure_loss_curves(runs, path, title)
    elif kind == "auc_interaction":
        df = per_run_table(runs)
        figure_auc_interaction(df, compute_statistics(df), path, title)
    elif kind == "sessions_to_criterion":
        figure_sessions_to_criterion(per_run_table(runs), len(runs[0]["train_loss"]), path, title)
    elif kind == "observed_vs_additive":
        df = per_run_table(runs)
        figure_observed_vs_additive(df, compute_statistics(df), path, title)
    elif kind == "summary":
        figure_summary(per_run_table(runs), path, title)
    else:
        raise ValueError(kind)
    return path


def write_tables(runs, tag, table_dir):
    """Write per-run values, the full statistics (JSON) and the summary table (Markdown); returns the statistics dict."""
    os.makedirs(table_dir, exist_ok=True)
    df = per_run_table(runs); stat = compute_statistics(df)
    df.to_csv(os.path.join(table_dir, f"per_run_{tag}.csv"), index=False)
    with open(os.path.join(table_dir, f"statistics_{tag}.json"), "w") as f:
        json.dump(dict(tag=tag, datasets=stat), f, indent=1, default=float)
    with open(os.path.join(table_dir, f"summary_{tag}.md"), "w") as f:
        f.write(markdown_tables(stat))
    return stat


def analyze(runs, tag, fig_dir, table_dir, title):
    """Tables plus all figures for one configuration."""
    df = per_run_table(runs); stat = write_tables(runs, tag, table_dir)
    os.makedirs(fig_dir, exist_ok=True)
    figure_loss_curves(runs, os.path.join(fig_dir, f"loss_curves_{tag}.png"), title)
    figure_auc_interaction(df, stat, os.path.join(fig_dir, f"auc_interaction_{tag}.png"), title)
    figure_sessions_to_criterion(df, len(runs[0]["train_loss"]), os.path.join(fig_dir, f"sessions_to_criterion_{tag}.png"), title)
    figure_observed_vs_additive(df, stat, os.path.join(fig_dir, f"observed_vs_additive_{tag}.png"), title)
    figure_summary(df, os.path.join(fig_dir, f"summary_{tag}.png"), title)
    return stat
