"""Mechanism report for the reviewers (no training; reads finished runs).

For every experiment tag and method, at the first active refresh (before
CTAL-specific updates can reshape the representation) and at the last refresh:

  A  AUROC for noisy-label detection: loss, co-tail, context, activation, risk
  B  Inside the loss tail only: noisy vs clean-hard AUROC of loss and of risk.
     For loss_tail / base this equals the loss; risk > loss means sample weights
     are NOT a function of loss rank alone (Phbj Q2, BK6m Q2, AC).
  C  Mean weights: clean, noisy, clean-hard, noisy-in-tail; spread of the
     amplification r / x_L inside the tail (exactly 0 for loss-rank-only weights)
  D  Assumption 1: mean excess concentration s_i - E_pi[s_i^pi] for noisy / clean
  E  Weights by analysis subgroup (controlled: rare clean subgroup and
     contamination cluster; long-tail: per clean class)
  F  Figure: weight vs observed-class loss coordinate, noisy vs clean

    python analyze_mechanism.py runs/ --out report/
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import numpy as np


def ms(xs, scale=1.0, nd=3):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{scale * xs[0]:.{nd}f}"
    return f"{scale * statistics.mean(xs):.{nd}f} ± {scale * statistics.stdev(xs):.{nd}f}"


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def table(title, header, rows, out):
    out.append(f"\n#### {title}\n")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")


def report_tag(tag, runs, which, out):
    """runs: method -> seed -> summary.  which: 'first' or 'last' refresh."""
    methods = sorted(runs)

    def diags(m):
        res = []
        for s in sorted(runs[m]):
            refs = runs[m][s]["history"]["refreshes"]
            if refs:
                res.append(refs[0] if which == "first" else refs[-1])
        return res

    rowsA, rowsB, rowsC, rowsD = [], [], [], []
    for m in methods:
        D = diags(m)
        if not D or get(D[0], "auroc") is None:
            continue
        rowsA.append([m] + [ms([get(d, "auroc", k) for d in D]) for k in
                            ("loss", "cotail", "context", "activation", "risk")])
        rowsB.append([m, ms([get(d, "tail_auroc", "loss") for d in D]),
                      ms([get(d, "tail_auroc", "risk") for d in D]),
                      ms([get(d, "tail_auroc", "corroboration") for d in D]),
                      ms([get(d, "tail_auroc", "activation") for d in D]),
                      ms([get(d, "tail_auroc", "n_noisy") for d in D], nd=0),
                      ms([get(d, "tail_auroc", "n_clean_hard") for d in D], nd=0)])
        rowsC.append([m] + [ms([get(d, k, "mean") for d in D]) for k in
                            ("w_clean", "w_noisy", "w_clean_hard", "w_noisy_tail")]
                     + [ms([get(d, "tail_amplification_std") for d in D])])
        rowsD.append([m, ms([get(d, "assumption1", "noisy_mean_excess") for d in D], nd=4),
                      ms([get(d, "assumption1", "clean_mean_excess") for d in D], nd=4)])
    lab = f"{tag} — {which} active refresh"
    table(f"A. Noisy-label AUROC ({lab})", ["method", "loss", "co-tail", "context", "activation", "risk"],
          rowsA, out)
    table(f"B. Inside the loss tail: noisy vs clean-hard AUROC ({lab})",
          ["method", "loss", "risk", "corroboration", "activation", "#noisy", "#clean-hard"], rowsB, out)
    table(f"C. Mean weights ({lab})",
          ["method", "clean", "noisy", "clean-hard", "noisy in tail",
           "std of r/x_L in tail (0 = loss rank only)"], rowsC, out)
    table(f"D. Assumption 1: mean excess local concentration ({lab})",
          ["method", "noisy", "clean"], rowsD, out)

    # E. groups
    groups = set()
    for m in methods:
        for d in diags(m):
            groups |= set((get(d, "w_by_group") or {}).keys())
    if groups:
        groups = sorted(groups, key=lambda g: (len(g), g))
        rowsE = []
        for m in methods:
            D = diags(m)
            rowsE.append([m] + [ms([get(d, "w_by_group", g, "mean") for d in D]) for g in groups])
        table(f"E. Mean weight by analysis subgroup ({lab})", ["method"] + list(groups), rowsE, out)
        if which == "first":
            d0 = diags(methods[0])[0]
            rowsN = [[g, ms([get(d0, "w_by_group", g, "noise_rate")]),
                      ms([get(d0, "w_by_group", g, "tail_frac")])] for g in groups]
            table(f"E'. Subgroup noise rate and loss-tail fraction ({tag})",
                  ["group", "noise rate", "in loss tail"], rowsN, out)


def figure(tag_dir, tag, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for method in ("loss_tail", "pointwise", "full"):
        files = sorted(glob.glob(os.path.join(tag_dir, f"*-{method}__s*_states", "refresh_001_*.npz")))
        if files:
            panels.append((method, np.load(files[0])))
    if not panels or "y_clean" not in panels[0][1]:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), sharey=True, squeeze=False)
    for ax, (method, z) in zip(axes[0], panels):
        noisy = z["y_obs"] != z["y_clean"]
        tail = z["x_L"] > 0
        for mask, color, label in ((~noisy & tail, "#2a7ab9", "clean"), (noisy & tail, "#d1495b", "noisy")):
            ax.scatter(z["U_L"][mask], z["w"][mask], s=3, alpha=0.35, color=color, label=label,
                       rasterized=True)
        ax.set_title(method)
        ax.set_xlabel("observed-class loss coordinate $U_L$")
    axes[0][0].set_ylabel("stored weight $w$")
    axes[0][0].legend(markerscale=4, frameon=False)
    fig.suptitle(f"{tag}: loss-tail examples at the first active refresh", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}_weight_vs_loss.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out", default="report")
    p.add_argument("--prefix", default="")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    by_tag = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(os.path.join(a.run_dir, "**", "*_summary.json"), recursive=True)):
        with open(path) as f:
            s = json.load(f)
        tag = s.get("tag", "default")
        if a.prefix and not tag.startswith(a.prefix):
            continue
        by_tag[tag][s.get("method", s["config"]["variant"])][s["config"]["seed"]] = s

    out = ["# CTAL mechanism report"]
    for tag in sorted(by_tag):
        out.append(f"\n## {tag}")
        for which in ("first", "last"):
            report_tag(tag, by_tag[tag], which, out)
        fig = figure(os.path.join(a.run_dir, tag), tag, a.out)
        if fig:
            out.append(f"\n![weights vs loss]({os.path.basename(fig)})")
    path = os.path.join(a.out, "mechanism_report.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
