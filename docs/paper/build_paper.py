#!/usr/bin/env python
"""Build the revised paper PDF from paper.md (+ references.md) with numbers filled in from tables/statistics_*.json.

    pip install -e .[paper]
    python reproduce.py            # regenerates figures/ and tables/ (fast: uses the distributed run cache)
    python docs/paper/build_paper.py

Placeholders in paper.md:  {{S:data:cond:metric[@tag]}} mean ± 95% CI,  {{M:...}} mean,
{{F:data:outcome:effect[@tag]}} ANOVA,  {{D:data:outcome:cond[@tag]}} contrast vs control,
{{PCT:data:cond[@tag]}} % decrease in mean loss session 1 -> end,  {{ACC:data:cond[@tag]}} held-out accuracy vs chance,
{{TABLE:summary|anova|contrast[@tag]}}.  Default tag is lr0.03.
"""

import json
import os
import re

import markdown
from weasyprint import HTML

from asd_model.paper_analysis import CONDITION_LABELS, CONDITION_ORDER, IMAGE_SET_LABELS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_TAG = "lr0.03"
_statistics_cache = {}


def load_statistics(tag):
    """Return the statistics of every image set for a configuration tag (read once from the tables folder)."""
    if tag not in _statistics_cache:
        with open(os.path.join(ROOT, "tables", f"statistics_{tag}.json")) as f:
            _statistics_cache[tag] = json.load(f)["datasets"]
    return _statistics_cache[tag]


def format_p_value(p):
    """Format a p-value the way the paper prints it ("p < 0.0001", "p = 0.012", ...); "n/a" if there is none."""
    if p is None:
        return "n/a"
    if p < 1e-4:
        return "p < 0.0001"
    if p < 0.001:
        return f"p = {p:.4f}"
    if p < 0.1:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def decimals_for(metric):
    """Number of decimals printed for a metric (one for session counts and accuracies, two otherwise)."""
    return 1 if metric.startswith("ept") or metric.startswith("final_val_acc") or metric.endswith("_acc") else 2


def resolve_placeholder(match):
    """Return the text for one ``{{KIND:...}}`` placeholder of paper.md, read from the statistics tables."""
    body = match.group(1)
    tag = DEFAULT_TAG
    if "@" in body:
        body, learning_rate = body.split("@")
        tag = "lr" + learning_rate
    kind, *fields = body.split(":")
    statistics = load_statistics(tag)
    if kind in ("S", "M"):
        image_set, condition, metric = fields
        summary = statistics[image_set]["summary"][condition][metric]
        places = decimals_for(metric)
        return f"{summary['mean']:.{places}f}" + (f" ± {summary['ci95']:.{places}f}" if kind == "S" else "")
    if kind == "F":
        image_set, outcome, effect = fields
        anova = statistics[image_set]["anova"][outcome][effect]
        return (
            f"F({anova['df'][0]}, {anova['df'][1]}) = {anova['F']:.1f}, {format_p_value(anova['p'])}, "
            f"partial η² = {anova['partial_eta2']:.2f}"
        )
    if kind == "D":
        image_set, outcome, condition = fields
        contrast = statistics[image_set]["contrasts"][outcome][condition]
        return (
            f"{contrast['diff_mean']:+.2f}, Holm-adjusted {format_p_value(contrast['p_holm'])}, "
            f"d = {contrast['cohen_d']:.1f} [{contrast['d_ci95'][0]:.1f}, {contrast['d_ci95'][1]:.1f}]"
        )
    if kind == "PCT":
        image_set, condition = fields
        summary = statistics[image_set]["summary"][condition]
        return f"{100 * (1 - summary['final_loss']['mean'] / summary['initial_loss']['mean']):.1f}%"
    if kind == "ACC":
        image_set, condition = fields
        accuracy = statistics[image_set]["accuracy"][condition]
        if accuracy["p_two_sided"] is None:
            return f"{accuracy['mean']:.1f}%"
        return f"{accuracy['mean']:.1f}% ({format_p_value(accuracy['p_two_sided'])} vs 25%)"
    if kind == "TABLE":
        return render_table(fields[0], statistics)
    raise ValueError(match.group(0))


def render_table(name, statistics):
    """Render one of the paper's HTML tables (``summary``, ``anova`` or ``contrast``) for every image set."""
    out = []
    for image_set in [key for key in ("original", "cifar3") if key in statistics]:
        image_set_stats = statistics[image_set]
        if name == "summary":
            out += [
                f"<p class='tcap'><b>{IMAGE_SET_LABELS[image_set]}</b> "
                f"(n = {image_set_stats['n_per_cond']['control']} seeds per condition)</p>",
                "<table><tr><th>Condition</th><th>Session-1 loss</th><th>Final loss (sessions 91–100)</th>"
                "<th>AUC (mean loss over sessions)</th>"
                "<th>Sessions to reach chance-level loss (ln 4)</th><th>Final training accuracy (%)</th>"
                "<th>Held-out accuracy (%)</th><th>Held-out loss</th></tr>",
            ]
            for condition in CONDITION_ORDER:
                summary = image_set_stats["summary"][condition]

                def mean_ci(measure, places=2, summary=summary):
                    return f"{summary[measure]['mean']:.{places}f} ± {summary[measure]['ci95']:.{places}f}"

                out.append(
                    f"<tr><td>{CONDITION_LABELS[condition]}</td><td>{mean_ci('initial_loss')}</td>"
                    f"<td>{mean_ci('final_loss')}</td><td>{mean_ci('auc')}</td><td>{mean_ci('ept_abs', 1)}</td>"
                    f"<td>{mean_ci('final_train_acc', 1)}</td><td>{mean_ci('final_val_acc', 1)}</td>"
                    f"<td>{mean_ci('final_val_loss')}</td></tr>"
                )
            out.append("</table>")
        elif name == "anova":
            out += [
                f"<p class='tcap'><b>{IMAGE_SET_LABELS[image_set]}</b></p>",
                "<table><tr><th>Outcome</th><th>Effect</th><th>F (df)</th><th>p</th><th>Partial η²</th></tr>",
            ]
            outcomes = (
                ("auc", "AUC"),
                ("final_loss", "Final loss"),
                ("ept_abs", "Sessions to ln 4"),
                ("ept_rel", "Sessions to 50% of initial loss"),
            )
            for outcome, outcome_label in outcomes:
                for effect in ("Wnt", "CCL5", "Wnt x CCL5"):
                    anova = image_set_stats["anova"][outcome][effect]
                    p_text = format_p_value(anova["p"]).replace("p ", "").replace("= ", "")
                    out.append(
                        f"<tr><td>{outcome_label}</td><td>{effect.replace(' x ', ' × ')}</td>"
                        f"<td>{anova['F']:.1f} ({anova['df'][0]}, {anova['df'][1]})</td>"
                        f"<td>{p_text}</td><td>{anova['partial_eta2']:.2f}</td></tr>"
                    )
            out.append("</table>")
        elif name == "contrast":
            out += [
                f"<p class='tcap'><b>{IMAGE_SET_LABELS[image_set]}</b></p>",
                "<table><tr><th>Outcome</th><th>Condition vs control</th><th>Difference</th>"
                "<th>Holm-adjusted p (one-tailed)</th><th>Cohen's d [95% bootstrap CI]</th></tr>",
            ]
            outcomes = (
                ("auc", "AUC"),
                ("final_loss", "Final loss"),
                ("ept_rel", "Sessions to 50% of initial loss"),
            )
            for outcome, outcome_label in outcomes:
                for condition in ("ccl5", "wnt", "both"):
                    contrast = image_set_stats["contrasts"][outcome][condition]
                    p_text = format_p_value(contrast["p_holm"]).replace("p ", "").replace("= ", "")
                    out.append(
                        f"<tr><td>{outcome_label}</td><td>{CONDITION_LABELS[condition]}</td>"
                        f"<td>{contrast['diff_mean']:+.2f}</td><td>{p_text}</td>"
                        f"<td>{contrast['cohen_d']:.1f} "
                        f"[{contrast['d_ci95'][0]:.1f}, {contrast['d_ci95'][1]:.1f}]</td></tr>"
                    )
            out.append("</table>")
    return "\n".join(out)


def build_pdf(markdown_text, out, references=None, extra_css=""):
    """Render markdown text (with ``{{...}}`` placeholders) to a PDF at ``out``."""
    if references is None:
        with open(os.path.join(HERE, "references.md")) as f:
            references = f.read()
    markdown_text = markdown_text.replace("{{REFERENCES}}", references)
    markdown_text = re.sub(r"\{\{([^}]+)\}\}", resolve_placeholder, markdown_text)
    html = markdown.markdown(markdown_text, extensions=["extra", "sane_lists", "md_in_html"])
    with open(os.path.join(HERE, "style.css")) as f:
        css = f.read() + extra_css
    document = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>"
    HTML(string=document, base_url=HERE + "/").write_pdf(out)


def main():
    """Build ``paper.pdf`` next to ``paper.md``."""
    with open(os.path.join(HERE, "paper.md")) as f:
        markdown_text = f.read()
    out = os.path.join(HERE, "paper.pdf")
    build_pdf(markdown_text, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
