# Reviewer experiments (CIFAR-10N / CIFAR-100N / controlled CIFAR-10)

The NeurIPS reviews were written about the **old** CTAL, where each sample's
weight was `(1 − U_i)^γ_t`: loss rank only, with the multi-view signal reduced to
one batch-level exponent. The new CTAL changes exactly that. Each sample's
weight now depends on its own corroboration `c_i` and its own neighbourhood
activation `a_i` (Eq. 14), and everything is computed over the full training
set rather than the minibatch.

So most objections are now answered **by design**. The reviewers will still ask
for evidence that the design does what it says on real noise. The experiments
below provide that evidence, using CIFAR data only.

Commands: `python experiments.py --list`, then run tiers (see README).
`--mech_base` (default `gce`) is the base loss for the mechanism experiments.
Set it to whichever base is your headline result.

## Map from reviewer question to evidence

| # | Reviewer question | Experiment | Supports the claim if… |
|---|---|---|---|
| 1 | Weights depend only on loss rank; multi-view signals don't decide which sample is down-weighted (Phbj Q2, BK6m Q2, vzUn Q3, AC) | **E1** decomposition on Worst + CIFAR-100N, plus `analyze_mechanism.py` tables B, C and the figure | Inside the loss tail, **risk AUROC > loss AUROC** for noisy vs clean-hard (table B). The spread of `r/x_L` in the tail is > 0 for `full` and exactly 0 for `loss_tail` (table C). The figure shows a single curve for `loss_tail` and a spread for `full`, with noisy examples lower. |
| 2 | Is the gain from genuine collective structure or generic reweighting? (AC, vzUn Q1) | **E1** `collective` vs `collective+permctrl`, `full` vs `full+permctrl` | Paired Δ(`full` − `full+permctrl`) > 0 across seeds. Assumption-1 excess is positive for noisy and ≈ 0 for clean examples (table D). |
| 3 | Controlled evidence that pointwise methods fail on collective noise (Phbj Q1, vzUn Q1, BK6m Q3) | **E2**: identical images, identical transition matrix and per-class flip counts. Noise is `cluster` (collective), `pair_random` (same pairs, scattered) or `symmetric`. | Gain of `full` over `pointwise` / `loss_tail` is **larger under `cluster` than under `pair_random`**. `full+permctrl` loses that advantage. Test accuracy on contamination-cluster images improves. |
| 4 | Does CTAL suppress rare but informative clean examples? (BK6m Q3, Phbj Q4, AC) | **E2** rare clean subgroup (5% of one visual cluster per class, never relabelled). **E7** long-tailed CIFAR-10N Worst (ratio 10). | Rare-subgroup weight under `full` ≥ under `loss_tail` (table E), and rare-subgroup / tail-class test accuracy not below `base`. Note that Eq. (7) ranks within the observed class, so a tail class is not penalised for its higher average loss. |
| 5 | Joint-tail statistic vs product / min / indicator (BK6m Q1) | **E5** co-tail forms on Worst + CIFAR-100N | Geometric ≥ alternatives, or comparable with lower variance. Either outcome is reportable; the question is answered as long as the comparison exists. |
| 6 | Sensitivity to minibatch composition (vzUn Q2, AC) | **E6** batch size 64 / 256 and class-blocked minibatches (mostly single-class) | Weights are computed on the full dataset, so by construction they cannot depend on batch composition (Eq. 15). Eq. (16) only renormalises within the batch. E6 shows Δ(`full` − `base`) stays stable. |
| 7 | Drop-in "base vs base + CTAL" (Phbj Q3) | Your GCE/SCE tables + **E4** (Aggregate split, missing from both tables) + **E9** (CE base, all splits) | Three bases, all six CIFAR-N splits, paired across seeds. |
| 8 | Hyper-parameter sensitivity (Limitations section) | **E8** one-at-a-time on Worst | Accuracy flat over reasonable ranges. |
| 9 | Overhead | Every run logs train vs refresh time; `aggregate.py` prints both | Report refresh hours / train hours. |

Run counts: Tier 1 = E1, E2, E4 (99 runs); Tier 2 = E5, E6, E7 (54 runs);
Tier 3 = E8, E9 (75 runs). 228 in total, 3 seeds each.

## Order of work

1. **Pilot first (2 runs).** Run E1 `base` and `full` for seed 42 on Worst:
   `python experiments.py --exp E1_decomposition --filter "worse_label.*--variant (base|full) --seed 42 " --run`.
   Compare with your main table. If this code's numbers differ by more than
   about one standard deviation from your own code's, find out why before
   spending the GPU budget. Every reviewer-facing number must come from one
   implementation.
2. **Time the pilot.** `aggregate.py` prints train and refresh hours. Multiply
   by the run count to get the budget.
3. **Tier 1**, then `analyze_mechanism.py`. Read tables B and D and E2's paired
   Δ before running anything else. They are the core of the rebuttal.
4. Tier 2, then Tier 3 as budget allows.

## How to read a negative result

These are tests, not decorations. If one fails, the paper should say so and
narrow its claim. Reviewers punish claims the evidence doesn't support far more
than they punish a narrower scope.

- **`full` ≈ `full+permctrl` on CIFAR-N.** The neighbourhood term is not why
  CTAL helps there. Present the gain as corroborated pointwise reweighting, and
  keep the collective claim only where E2 supports it.
- **E2 gain no larger under `cluster` than `pair_random`.** The collective
  premise isn't demonstrated, even in the controlled case. Check table D first:
  if the excess for noisy examples is ≈ 0, the representation doesn't recover
  the cluster, which is Assumption 1 failing rather than the method.
- **Rare subgroup suppressed more than under `loss_tail`.** This is the
  limitation the paper already names. Report it with the numbers.

## Not covered by these experiments (text changes)

- **Waterbirds / Clothing1M / JSRT.** You asked to focus on CIFAR. If those
  results stay in the paper, they still need repeated runs (BK6m, AC). The
  Waterbirds text must also say how hyperparameters were chosen without group
  labels. Otherwise remove those claims and say the evaluation is CIFAR-N plus
  the controlled study.
- **DivideMix + CTAL** (Phbj Q3 names it explicitly). Not in this manifest:
  wiring CTAL weights into DivideMix's labelled-set loss is a design decision,
  not a mechanical change.
- **"Joint-tail" wording.** The new paper already uses "co-tail" with the loss
  gate. E5 supplies the comparison BK6m asked for.
- **Code release.** Anonymous repository of this code (vzUn Q4).
- **One protocol for the main table.** Report final-epoch accuracy over the
  prespecified seeds, as Sec. 6 says, for every row.
