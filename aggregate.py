"""Aggregate run summaries into result tables (Sec. 6 'Statistical reporting').

For every experiment tag: each seed, mean ± sample std across seeds, and the
paired per-seed difference against the same base loss without reweighting.
Subgroup accuracies (controlled / long-tail runs) and timing are reported too.

    python aggregate.py runs/                 # all tags
    python aggregate.py runs/ --prefix E2_    # one experiment
    python aggregate.py runs/ --markdown > results.md
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def mean_std(xs, scale=100.0):
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{scale * xs[0]:.2f}"
    return f"{scale * statistics.mean(xs):.2f} ± {scale * statistics.stdev(xs):.2f}"


def load(run_dir, prefix):
    res = defaultdict(dict)   # (tag, method) -> seed -> summary
    for path in sorted(glob.glob(os.path.join(run_dir, "**", "*_summary.json"), recursive=True)):
        with open(path) as f:
            s = json.load(f)
        tag = s.get("tag", "default")
        if prefix and not tag.startswith(prefix):
            continue
        res[(tag, s.get("method", s["config"]["variant"]))][s["config"]["seed"]] = s
    return res


def baseline_method(method):
    return method.split("-")[0] + "-base"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--prefix", default="")
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args()
    res = load(a.run_dir, a.prefix)
    if not res:
        print("no finished runs found")
        return

    for tag in sorted({k[0] for k in res}):
        rows = sorted(m for (t, m) in res if t == tag)
        any_groups = any("final_group_acc" in s for m in rows for s in res[(tag, m)].values())
        print(f"\n### {tag}\n")
        hdr = ["method", "seeds", "final acc", "last-10 mean", "paired Δ final vs base", "train h", "refresh h"]
        if any_groups:
            gnames = sorted({g for m in rows for s in res[(tag, m)].values()
                             for g in s.get("final_group_acc", {})})
            if len(gnames) > 6:  # long-tail: summarise head / tail classes instead
                gnames = []
            hdr += [f"acc[{g}]" for g in gnames]
        else:
            gnames = []
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for m in rows:
            runs = res[(tag, m)]
            seeds = sorted(runs)
            fin = [runs[s]["final_acc"] for s in seeds]
            l10 = [runs[s]["last10_mean_acc"] for s in seeds]
            base = res.get((tag, baseline_method(m)), {})
            common = [s for s in seeds if s in base]
            diff = [runs[s]["final_acc"] - base[s]["final_acc"] for s in common]
            dstr = mean_std(diff) if diff and m != baseline_method(m) else "-"
            tr = [runs[s].get("timing", {}).get("train_s", 0) / 3600 for s in seeds]
            rf = [runs[s].get("timing", {}).get("refresh_s", 0) / 3600 for s in seeds]
            cells = [m, ",".join(map(str, seeds)), mean_std(fin), mean_std(l10), dstr,
                     f"{statistics.mean(tr):.2f}", f"{statistics.mean(rf):.2f}"]
            for g in gnames:
                cells.append(mean_std([runs[s]["final_group_acc"][g] for s in seeds
                                       if g in runs[s].get("final_group_acc", {})]))
            print("| " + " | ".join(cells) + " |")
        if not a.markdown:
            for m in rows:
                for s in sorted(res[(tag, m)]):
                    r = res[(tag, m)][s]
                    print(f"    {m} seed {s}: final {100 * r['final_acc']:.2f}  last10 "
                          f"{100 * r['last10_mean_acc']:.2f}  (best {100 * r['best_acc_diagnostic']:.2f} "
                          f"@ {r['best_epoch_diagnostic']}, diagnostic only)")


if __name__ == "__main__":
    main()
