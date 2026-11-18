#!/usr/bin/env python
"""Build the revised paper PDF from paper.md (+ references.md) with numbers filled in from tables/statistics_*.json.

    pip install -r docs/paper/requirements.txt
    python reproduce.py            # regenerates figures/ and tables/ (fast: uses the distributed run cache)
    python docs/paper/build_paper.py

Placeholders in paper.md:  {{S:data:cond:metric[@tag]}} mean ± 95% CI,  {{M:...}} mean,  {{F:data:outcome:effect[@tag]}} ANOVA,
{{D:data:outcome:cond[@tag]}} contrast vs control,  {{PCT:data:cond[@tag]}} % decrease in mean loss session 1 -> end,
{{ACC:data:cond[@tag]}} held-out accuracy vs chance,  {{TABLE:summary|anova|contrast[@tag]}}.  Default tag is lr0.03.
"""
import json
import os
import re
import sys

import markdown
from weasyprint import HTML

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CONDS = ["control", "ccl5", "wnt", "both"]
LABEL = {"control": "Control", "ccl5": "CCL5", "wnt": "Wnt", "both": "Wnt + CCL5"}
DATA = {"original": "Original images", "cifar3": "CIFAR-10 subset"}
_cache = {}


def stats(tag):
    if tag not in _cache:
        with open(os.path.join(ROOT, "tables", f"statistics_{tag}.json")) as f:
            _cache[tag] = json.load(f)["datasets"]
    return _cache[tag]


def fp(p):
    if p is None:
        return "n/a"
    return "p < 0.0001" if p < 1e-4 else (f"p = {p:.4f}" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}")


def dec(metric):
    return 1 if metric.startswith("ept") or metric.startswith("final_val_acc") or metric.endswith("_acc") else 2


def resolve(m):
    body = m.group(1); tag = "lr0.03"
    if "@" in body:
        body, t = body.split("@"); tag = "lr" + t
    kind, *a = body.split(":")
    S = stats(tag)
    if kind in ("S", "M"):
        data, cond, metric = a; s = S[data]["summary"][cond][metric]; k = dec(metric)
        return f"{s['mean']:.{k}f}" + (f" ± {s['ci95']:.{k}f}" if kind == "S" else "")
    if kind == "F":
        data, y, eff = a; r = S[data]["anova"][y][eff]
        return f"F({r['df'][0]}, {r['df'][1]}) = {r['F']:.1f}, {fp(r['p'])}, partial η² = {r['partial_eta2']:.2f}"
    if kind == "D":
        data, y, cond = a; r = S[data]["contrasts"][y][cond]
        return f"{r['diff_mean']:+.2f}, Holm-adjusted {fp(r['p_holm'])}, d = {r['cohen_d']:.1f} [{r['d_ci95'][0]:.1f}, {r['d_ci95'][1]:.1f}]"
    if kind == "PCT":
        data, cond = a; s = S[data]["summary"][cond]
        return f"{100 * (1 - s['final_loss']['mean'] / s['initial_loss']['mean']):.1f}%"
    if kind == "ACC":
        data, cond = a; r = S[data]["accuracy"][cond]
        return f"{r['mean']:.1f}% ({fp(r['p_two_sided'])} vs 25%)" if r["p_two_sided"] is not None else f"{r['mean']:.1f}%"
    if kind == "TABLE":
        return table(a[0], S)
    raise ValueError(m.group(0))


def table(name, S):
    out = []
    for data, R in S.items():
        if name == "summary":
            out += [f"<p class='tcap'><b>{DATA[data]}</b> (n = {R['n_per_cond']['control']} seeds per condition)</p>",
                    "<table><tr><th>Condition</th><th>Session-1 loss</th><th>Final loss (sessions 91–100)</th><th>AUC (mean loss over sessions)</th>"
                    "<th>Sessions to reach chance-level loss (ln 4)</th><th>Final training accuracy (%)</th><th>Held-out accuracy (%)</th><th>Held-out loss</th></tr>"]
            for c in CONDS:
                s = R["summary"][c]; f = lambda k, p=2: f"{s[k]['mean']:.{p}f} ± {s[k]['ci95']:.{p}f}"
                out.append(f"<tr><td>{LABEL[c]}</td><td>{f('initial_loss')}</td><td>{f('final_loss')}</td><td>{f('auc')}</td><td>{f('ept_abs', 1)}</td>"
                           f"<td>{f('final_train_acc', 1)}</td><td>{f('final_val_acc', 1)}</td><td>{f('final_val_loss')}</td></tr>")
            out.append("</table>")
        elif name == "anova":
            out += [f"<p class='tcap'><b>{DATA[data]}</b></p>", "<table><tr><th>Outcome</th><th>Effect</th><th>F (df)</th><th>p</th><th>Partial η²</th></tr>"]
            for y, yl in (("auc", "AUC"), ("final_loss", "Final loss"), ("ept_abs", "Sessions to ln 4"), ("ept_rel", "Sessions to 50% of initial loss")):
                for e in ("Wnt", "CCL5", "Wnt x CCL5"):
                    r = R["anova"][y][e]
                    out.append(f"<tr><td>{yl}</td><td>{e.replace(' x ', ' × ')}</td><td>{r['F']:.1f} ({r['df'][0]}, {r['df'][1]})</td><td>{fp(r['p']).replace('p ', '').replace('= ', '')}</td><td>{r['partial_eta2']:.2f}</td></tr>")
            out.append("</table>")
        elif name == "contrast":
            out += [f"<p class='tcap'><b>{DATA[data]}</b></p>", "<table><tr><th>Outcome</th><th>Condition vs control</th><th>Difference</th><th>Holm-adjusted p (one-tailed)</th><th>Cohen's d [95% bootstrap CI]</th></tr>"]
            for y, yl in (("auc", "AUC"), ("final_loss", "Final loss"), ("ept_rel", "Sessions to 50% of initial loss")):
                for c in ("ccl5", "wnt", "both"):
                    r = R["contrasts"][y][c]
                    out.append(f"<tr><td>{yl}</td><td>{LABEL[c]}</td><td>{r['diff_mean']:+.2f}</td><td>{fp(r['p_holm']).replace('p ', '').replace('= ', '')}</td>"
                               f"<td>{r['cohen_d']:.1f} [{r['d_ci95'][0]:.1f}, {r['d_ci95'][1]:.1f}]</td></tr>")
            out.append("</table>")
    return "\n".join(out)


def main():
    with open(os.path.join(HERE, "paper.md")) as f:
        md = f.read()
    with open(os.path.join(HERE, "references.md")) as f:
        refs = f.read()
    md = md.replace("{{REFERENCES}}", refs)
    md = re.sub(r"\{\{([^}]+)\}\}", resolve, md)
    html = markdown.markdown(md, extensions=["extra", "sane_lists", "md_in_html"])
    with open(os.path.join(HERE, "style.css")) as f:
        css = f.read()
    doc = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>"
    out = os.path.join(HERE, "paper.pdf")
    HTML(string=doc, base_url=HERE + "/").write_pdf(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
