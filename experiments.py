"""Experiment manifest for the resubmission (CIFAR-10N / CIFAR-100N / controlled CIFAR-10).

Every experiment is tied to the reviewer question it answers (see README,
"Reviewer experiments").  Runs are resumable: a run whose summary already
exists is skipped, so the same command can be relaunched after a crash.

    python experiments.py --list                       # what will run, with counts
    python experiments.py --tier 1 --print             # print commands
    CUDA_VISIBLE_DEVICES=0 python experiments.py --tier 1 --shard 0/4 --run
    CUDA_VISIBLE_DEVICES=1 python experiments.py --tier 1 --shard 1/4 --run
    ...
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from typing import Dict, List

SEEDS = (42, 124, 2024)

CIFARN = {
    "c10n_aggre": ["--dataset", "cifar10", "--noise_type", "aggre_label", "--label_file", "{c10}"],
    "c10n_r1": ["--dataset", "cifar10", "--noise_type", "random_label1", "--label_file", "{c10}"],
    "c10n_r2": ["--dataset", "cifar10", "--noise_type", "random_label2", "--label_file", "{c10}"],
    "c10n_r3": ["--dataset", "cifar10", "--noise_type", "random_label3", "--label_file", "{c10}"],
    "c10n_worst": ["--dataset", "cifar10", "--noise_type", "worse_label", "--label_file", "{c10}"],
    "c100n": ["--dataset", "cifar100", "--noise_type", "noisy_label", "--label_file", "{c100}"],
}

# method name -> extra args
DECOMPOSITION = {
    "base": ["--variant", "base"],
    "loss_tail": ["--variant", "loss_tail"],
    "pointwise": ["--variant", "pointwise"],
    "collective": ["--variant", "collective"],
    "full": ["--variant", "full"],
    "collective+permctrl": ["--variant", "collective", "--activation_control", "permute"],
    "full+permctrl": ["--variant", "full", "--activation_control", "permute"],
}


def experiments(mech_base: str) -> List[Dict]:
    """Return a list of {exp, tier, question, tag, args} (one entry per run)."""
    E: List[Dict] = []

    def add(exp, tier, question, tag, args, seeds=SEEDS):
        for s in seeds:
            E.append({"exp": exp, "tier": tier, "question": question, "tag": tag,
                      "args": args + ["--seed", str(s), "--tag", tag]})

    b = ["--base_loss", mech_base]

    # ---------------- Tier 1: core evidence --------------------------------
    for ds in ("c10n_worst", "c100n"):
        for m, extra in DECOMPOSITION.items():
            add("E1_decomposition", 1,
                "Is the gain from sample-level co-tail identification and genuine graph alignment, "
                "not generic reweighting? (Phbj Q2, BK6m Q2, vzUn Q3, AC)",
                f"E1_{ds}_{mech_base}", CIFARN[ds] + b + extra)

    for noise in ("cluster", "pair_random", "symmetric"):
        for m in ("base", "loss_tail", "pointwise", "full", "full+permctrl"):
            add("E2_controlled", 1,
                "Do pointwise methods fail on collective noise, does CTAL help exactly when noise is "
                "collective, and does it spare a rare clean subgroup? (Phbj Q1/Q4, vzUn Q1, BK6m Q3)",
                f"E2_c10ctrl_{noise}_{mech_base}",
                ["--dataset", "cifar10_controlled", "--ctrl_noise", noise] + b + DECOMPOSITION[m])

    for base in ("gce", "sce"):
        for m in ("base", "full"):
            add("E4_aggre", 1, "Complete the CIFAR-10N table (Aggregate split missing)",
                f"E4_c10n_aggre_{base}", CIFARN["c10n_aggre"] + ["--base_loss", base] + DECOMPOSITION[m])

    # ---------------- Tier 2: specific reviewer asks -----------------------
    for ds in ("c10n_worst", "c100n"):
        for form in ("additive", "product", "min", "indicator"):
            add("E5_cotail_form", 2,
                "Is the joint-tail statistic justified vs additive / product / min / indicator? (BK6m Q1)",
                f"E5_{ds}_{mech_base}_{form}", CIFARN[ds] + b + ["--variant", "full", "--cotail_form", form])

    for name, extra in (("bs64", ["--batch_size", "64"]), ("bs256", ["--batch_size", "256"]),
                        ("classblocked", ["--batch_sampler", "class_blocked"])):
        for m in ("base", "full"):
            add("E6_batch", 2, "Sensitivity to minibatch composition and size (vzUn Q2, AC)",
                f"E6_c10n_worst_{mech_base}_{name}", CIFARN["c10n_worst"] + b + extra + DECOMPOSITION[m])

    for m in ("base", "loss_tail", "pointwise", "full"):
        add("E7_longtail", 2,
            "Are clean tail-class examples systematically suppressed? (Phbj Q4)",
            f"E7_c10n_worst_lt10_{mech_base}",
            CIFARN["c10n_worst"] + ["--imbalance_ratio", "10"] + b + DECOMPOSITION[m])

    # ---------------- Tier 3: robustness of the claims ----------------------
    sens = {
        "q0.7": ["--q", "0.7"], "q0.9": ["--q", "0.9"],
        "k16": ["--k", "16"], "k64": ["--k", "64"],
        "P10": ["--P", "10"], "P50": ["--P", "50"],
        "kappa1": ["--kappa", "1"], "kappa4": ["--kappa", "4"],
        "lam1": ["--lam", "1"], "lam4": ["--lam", "4"],
        "omega0.05": ["--omega", "0.05"], "omega0.2": ["--omega", "0.2"],
        "eta0.8-0.95": ["--eta0", "0.8", "--eta1", "0.95"],
    }
    for name, extra in sens.items():
        add("E8_sensitivity", 3, "Hyper-parameter sensitivity (Limitations section)",
            f"E8_c10n_worst_{mech_base}_{name}", CIFARN["c10n_worst"] + b + ["--variant", "full"] + extra)

    for ds in CIFARN:
        for m in ("base", "full"):
            add("E9_ce_dropin", 3, "Drop-in on a third base loss (CE) across all CIFAR-N splits (Phbj Q3)",
                f"E9_{ds}_ce", CIFARN[ds] + ["--base_loss", "ce"] + DECOMPOSITION[m])
    return E


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", type=int, nargs="*", default=[1, 2, 3])
    p.add_argument("--exp", nargs="*", default=None, help="restrict to experiment ids, e.g. E1_decomposition")
    p.add_argument("--mech_base", default="gce", choices=["ce", "gce", "sce"],
                   help="base loss used for the mechanism experiments (use your main-table base)")
    p.add_argument("--c10_labels", default="data/CIFAR-10_human.pt")
    p.add_argument("--c100_labels", default="data/CIFAR-100_human.pt")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    p.add_argument("--filter", default=None, help="regex matched against the run's argument string")
    p.add_argument("--extra", default="", help="extra args appended to every run (e.g. for smoke tests)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--print", action="store_true")
    p.add_argument("--run", action="store_true")
    a = p.parse_args()

    jobs = [j for j in experiments(a.mech_base) if j["tier"] in a.tier
            and (a.exp is None or j["exp"] in a.exp)]
    if a.filter:
        import re
        jobs = [j for j in jobs if re.search(a.filter, " ".join(j["args"]))]
    if a.list:
        from collections import OrderedDict
        counts = OrderedDict()
        for j in jobs:
            key = (j["tier"], j["exp"])
            counts.setdefault(key, [0, j["question"]])[0] += 1
        for (tier, exp), (n, q) in counts.items():
            print(f"T{tier} {exp:<18} {n:>4} runs  {q}")
        print(f"total: {len(jobs)} runs")
        return

    i, n = map(int, a.shard.split("/"))
    jobs = jobs[i::n]
    common = ["--data_root", a.data_root, "--download", "--out_dir", a.out_dir] + shlex.split(a.extra)
    for j in jobs:
        args = [x.format(c10=a.c10_labels, c100=a.c100_labels) for x in j["args"]] + common
        cmd = [sys.executable, "train.py"] + args
        if a.print or not a.run:
            print(shlex.join(cmd))
        if a.run:
            print(f"\n=== {j['exp']} | {j['tag']} ===", flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
