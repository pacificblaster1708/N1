# CTAL: Collective Tail-Adaptive Loss — PyTorch implementation

Implementation of *CTAL: Collective Tail-Adaptive Loss for Robust Learning under
Structured Contamination* (NeurIPS 2026 submission), following Algorithm 1 and
Eqs. (1)–(16).

## Layout

| File | Contents |
|---|---|
| `ctal/core.py` | Every equation of Sec. 4 as a pure function: hardness views (4)–(6), observed-class midrank coordinates (7), exceedances (8), co-tail score (9), same-label leave-one-out cosine kNN, local means (10), class-preserving permutation null (11), pooled quantiles (12), activation (13), risk (14), weights (15), normalized objective (16), memory update (3) |
| `ctal/ema.py` | Live EMA with warm-start decay `d_u = min(δ, (1+u)/(10+u))`; floating buffers averaged, integer buffers copied; frozen scorer snapshots |
| `ctal/trainer.py` | Algorithm 1: memory init, refresh schedule, λ ramp, scoring before memory update, fixed weights between refreshes, diagnostics |
| `ctal/data.py` | CIFAR-10N / CIFAR-100N loader (verifies sample order against torchvision), synthetic structured-contamination benchmark, batched crop+flip |
| `ctal/models.py` | CIFAR ResNet-34/18 and a small CNN, all returning `(logits, penultimate features)` |
| `ctal/losses.py` | Base losses CTAL can wrap: CE, GCE (q = 0.7), SCE (official-code clamps and per-dataset α, β) |
| `ctal/metrics.py` | AUROC (midrank), within-loss-tail AUROC, Assumption 1 statistic, weights by group |
| `train.py` | CLI (resumable: finished runs are skipped) |
| `experiments.py` | Manifest of all rebuttal experiments, tiered, shardable across GPUs |
| `aggregate.py` | Accuracy tables: per seed, mean ± sample std, paired Δ vs. the same base loss, subgroup accuracy, timing |
| `analyze_mechanism.py` | Mechanism report: AUROC tables, noisy vs clean-hard inside the loss tail, Assumption 1, subgroup weights, weight-vs-loss figure |
| `REVIEWER_EXPERIMENTS.md` | Which experiment answers which reviewer question, and how to read the result |
| `tests/` | 93 tests (see Verification) |

## Usage

Download `CIFAR-10_human.pt` / `CIFAR-100_human.pt` from
<https://github.com/UCSC-REAL/cifar-10-100n> (folder `data/`) into `data/`.

```bash
pip install torch torchvision numpy pytest

# one run, paper protocol (ResNet-34, 100 epochs, defaults = Sec. 6)
python train.py --dataset cifar10 --noise_type aggre_label \
    --label_file data/CIFAR-10_human.pt --data_root data --download --variant full --seed 42

# rebuttal experiments (see REVIEWER_EXPERIMENTS.md), e.g. tier 1 on 4 GPUs
python experiments.py --list
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i nohup python experiments.py --tier 1 --shard $i/4 --run > log_$i.txt 2>&1 & done
python aggregate.py runs/ --markdown > results.md
python analyze_mechanism.py runs/ --out report/

# tests
python -m pytest -q
```

Variants (`--variant`): `base` (alias `ce`: the base loss without reweighting), `loss_tail` (r = x_L), `pointwise` (r = x_L(1+ρc)),
`collective` (r = x_L(1+κa)), `full` (Eq. 14). Negative control:
`--activation_control permute`.

Outputs per run in `runs/<tag>/`: a `.jsonl` log (every epoch and every
refresh, including AUROCs and weight summaries), a `_summary.json` with
final-epoch accuracy (primary), last-10-epoch mean, best accuracy (diagnostic
only, never used for checkpoint selection), per-class and subgroup accuracy and
timing, and a `_states/` folder with per-sample refresh states (first and last
active refresh) for `analyze_mechanism.py`. Clean labels and subgroup ids are
used only for these diagnostics, never for weights. A summary is written
atomically at the end, so relaunching a manifest skips finished runs.

## Choices the paper leaves open

These are the places where I had to pick an interpretation. Each is a single,
clearly marked spot in the code if you want it changed.

1. **λ ramp.** "Ramped from zero … reaching full strength at epoch 20" is
   implemented as λ_e = λ·(e − E_w)/R, so epoch 11 uses 0.1λ and epoch 20 uses λ
   (`trainer.lambda_at`). Starting at exactly 0 at epoch 11 would make the
   first active refresh a no-op.
2. **Permuted negative control.** "Replace a_i by an activation obtained from a
   class-preserving permutation" is implemented as a_i → a_{π(i)} with an
   independent within-class permutation per refresh. This keeps each class's
   activation distribution identical and breaks only its alignment with the
   sample.
3. **EMA step index** starts at u = 1 for the first optimizer step
   (θ̄₀ = initial student), giving d₁ = 2/11.
4. **Strong scoring view** = the training augmentation (random crop with
   4-pixel black padding + horizontal flip), drawn from a dedicated RNG stream.
   Weak view = no augmentation.
5. **ε values** are not given in the paper: `eps_a = 1e-6`, `eps_z = 1e-12`
   (both CLI flags).
6. **Classes with ≤ k examples** raise an error (Appendix C asks that this be
   stated); there is no silent fallback.
7. **"Clean-hard"** in the diagnostics = clean examples in the loss tail
   (U_L > q).
8. **Test accuracy** is the student's; the EMA's accuracy is logged as an
   extra diagnostic.

## Verification

The reviewer-experiment code adds 22 tests:
- base losses match their formulas;
- all co-tail forms keep the loss gate and bounds;
- the class-blocked sampler produces single-class batches;
- controlled noise uses identical images, groups and transition matrices across
  noise types, keeps the rare subgroup clean, and matches flip counts;
- the long-tail profile is correct;
- the Assumption 1 statistic equals s − class-mean τ;
- the r/x_L spread is exactly 0 for loss-only weighting;
- every one of the 228 manifest commands parses and maps to a unique run;
- the real CIFAR-N label files pass through the full CLI (noise rates match
  the published ones).


1. **Independent reference implementation** (`tests/reference_impl.py`):
   explicit Python loops written straight from the equations, sharing no code
   with `ctal/core.py`. The vectorized code matches it on every intermediate
   quantity (h, U, x, c, τ, s, the null s⁽ᵖ⁾, a, r, v, w) to 1e-10, on 16
   random problems including heavily tied ones. The kNN graph matches brute
   force exactly.
2. **Propositions 1–5**:
   - Prop. 1: strictly increasing transforms of each view leave everything
     bit-identical.
   - Prop. 2: bounds, mean-one weights, v_min ≤ w ≤ 1/v_min, and exact ERM
     recovery.
   - Prop. 3: exact enumeration of all n_c! permutations reproduces the stated
     mean and variance to 1e-12, plus a Monte-Carlo check of the actual sampler.
   - Prop. 4: strict monotonicity, the ratio formula, and contextual ordering.
   - Prop. 5: the Lipschitz bound, checked on 600k random and near-tail pairs
     per q.
3. **Training-loop invariants**:
   - the exact refresh schedule (11–20, then 21, 26, …, 96), λ ramp and LR
     drop after epoch 60;
   - the EMA against a hand-computed average;
   - refresh-t scores use m^{t−1}, and memory updates only after scoring;
   - augmentation is identical to slicing, with torchvision-style black
     padding;
   - stop-gradient on weights;
   - paired runs: variants are identical through warm-up, and `ce` equals
     plain unweighted SGD to 1e-5;
   - bitwise determinism across repeated runs.
4. **End to end** through the CLI on a synthetic structured-contamination
   task. At the first active refresh, the context AUROC (0.72) exceeds the loss
   AUROC (0.69), and noisy examples are down-weighted (mean w 0.93 vs 1.03 for
   clean).
5. **Full scale**: one refresh at N = 50,000, k = 32, P = 20 takes about 3 s on
   CPU. On structureless data about 10% of activations are positive, matching
   1−η₀ as the calibration intends.

**Not verified here:** I had no GPU and no CIFAR download in the build
environment, so no ResNet-34 / CIFAR-N training run was completed and no
accuracy numbers are claimed. On GPU, cuDNN nondeterminism means runs are
reproducible but not bitwise identical; set
`torch.backends.cudnn.deterministic = True` if you need that.
