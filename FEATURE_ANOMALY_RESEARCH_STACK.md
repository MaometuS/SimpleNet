# Feature-Anomaly Research Stack

This document records the implementation changes for the next research phase:
moving from SimpleNet's isotropic feature noise toward real-feature-anchored,
covariance-aware, controllable synthetic anomalies.

## Research Goal

SimpleNet trains a discriminator with synthetic anomalous features produced by
adding random Gaussian noise to normal features. The working hypothesis in this
project is:

> Better feature-level anomalies should respect the geometry of normal features:
> they should be anchored on real normal features, move in meaningful feature
> directions, have calibrated magnitude, and provide local patch supervision.

The new stack adds three experimental controls on top of the existing
`TrueSpatialLowRankGaussian` anchored generator:

1. Radius calibration.
2. Sparse patch masks.
3. PCA-subspace discriminator-guided refinement.

Defaults preserve the previous behavior unless new environment variables are set.

## Files Changed

### `low_rank_gaussian.py`

#### Device-aware sampling

The generator no longer hard-codes `cuda:0`. New helper:

```python
_device_for_sampling(device=None, anchor=None)
```

Sampling now uses the requested device, the anchor device, or the fitted tensor
device. This makes CPU smoke tests, non-zero GPU ids, and future multi-GPU
experiments less brittle.

Affected methods:

```python
generate_anomaly_at_patch(..., device=None)
generate_anomalies(..., device=None)
generate_normal_at_patch(..., device=None)
generate_normal_features(..., device=None)
```

#### Mahalanobis score helper

Added:

```python
mahal2_at_patch(p, x, device=None)
```

This computes the fitted per-patch Mahalanobis squared score:

```text
s(x) = sum_j proj_j^2 / lambda_j + ||residual||^2 / eps_p
```

It is used by anchor-aware radius calibration.

#### Radius calibration modes

Added:

```python
radius_mode
```

Supported values:

```text
threshold
patch
anchor
```

`threshold` is the old behavior:

```text
radius is None: r = sqrt(T_p) + delta * U(0, 1)
radius is set:  r = radius * U(0, 1)
```

`patch` normalizes radius by the fitted patch threshold:

```text
r = radius * sqrt(T_p / C) * U(0, 1)
```

This makes radius sweeps more comparable across patches and classes.

`anchor` calibrates the perturbation to each real anchor's distance from the
patch threshold:

```text
gap = sqrt(T_p) - sqrt(mahal2(anchor))
r = radius * max(gap, 0.05 * sqrt(T_p)) * U(0, 1)
```

This is intended to create near-boundary anchored anomalies rather than huge
out-of-distribution shifts.

### `simplenet.py`

#### New environment knobs

The SimpleNet loader now reads:

```bash
TSLRG_RADIUS_MODE=threshold|patch|anchor
TSLRG_PATCH_MASK_MODE=all|random|block
TSLRG_PATCH_MASK_RATIO=0.15
TSLRG_PATCH_MASK_BLOCK=5
TSLRG_REFINE_STEPS=0
TSLRG_REFINE_STEP_SIZE=0.1
TSLRG_REFINE_MAX_RADIUS=
```

Existing knobs still work:

```bash
TSLRG_ANOMALY_MODE=default|subspace|anchored|simplenet_noise
TSLRG_RADIUS=
```

`simplenet_noise` restores the vanilla SimpleNet-style feature baseline inside
the same training/loss path:

```python
fake = real + Normal(0, noise_std)
```

#### Sparse patch masks

Added:

```python
_sample_patch_mask(B, P, device)
```

Supported mask modes:

```text
all     - every patch is synthetic, previous behavior
random  - randomly selected patches are synthetic
block   - connected spatial blocks of patches are synthetic
```

Training now mixes fake and real features:

```python
fake_feats = where(mask, synthetic_fake, real_anchor)
```

The fake loss is computed only on masked patches:

```python
selected_fake_scores = fake_scores[patch_mask]
fake_loss = clip(selected_fake_scores + margin, min=0)
```

This prevents unmodified real patches from being labeled fake. It also gives the
discriminator more localization-like supervision, because only selected patches
carry synthetic anomaly labels.

TensorBoard now logs:

```text
fake_patch_ratio
```

#### PCA-subspace gradient refinement

Added:

```python
_project_delta_to_tslrg_subspace(delta)
_mahal_radius_from_delta(delta)
_clamp_tslrg_subspace_delta(base, candidate, max_radius)
_refine_fake_features(true_feats, fake_feats, patch_mask)
```

When `TSLRG_REFINE_STEPS > 0`, the synthetic features are refined before the main
discriminator loss:

1. Start from generated anchored/PCA fake features.
2. Compute discriminator scores on the fake features.
3. Take gradient steps that lower the discriminator's normal score.
4. Project the update into the fitted `U_p` subspace.
5. Clamp the update to the initial Mahalanobis radius, or to
   `TSLRG_REFINE_MAX_RADIUS` if set.
6. Keep unmasked patches equal to their real anchors.

This is the project's version of hard negative generation: fake features are not
just random samples, but discriminator-aware near-normal negatives constrained by
the local PCA geometry.

## Suggested Experiment Ladder

All scripts are under `experiments/`. They call `run.sh`, which now accepts
common configuration through environment variables:

```bash
CLASSNAME=screw
DATAPATH=/path/to/mvtec
GPU=0
SEED=0
META_EPOCHS=40
GAN_EPOCHS=4
BATCH_SIZE=8
RESULTS_PATH=results
```

The experiment scripts are intended to run one class at a time. `CLASSNAME`
overrides `DATASETS`; if neither is set, `screw` is used.

Run every proposed experiment:

```bash
bash experiments/run_all_proposed_experiments.sh
```

Run every proposed experiment on a different single class:

```bash
CLASSNAME=carpet bash experiments/run_all_proposed_experiments.sh
```

For quick smoke runs, override epochs:

```bash
CLASSNAME=screw META_EPOCHS=1 GAN_EPOCHS=1 bash experiments/run_01_anchored_threshold.sh
```

### Baseline 0: Vanilla SimpleNet noise

```bash
bash experiments/run_00_simplenet_noise.sh
```

Purpose: run a direct head-to-head against the geometry-aware variants while
keeping the same discriminator, backbone, loss code, and logging.

### Baseline A: Previous anchored mode

```bash
bash experiments/run_01_anchored_threshold.sh
```

Purpose: reproduce the current reported image AUROC around `0.8`.

### Baseline B: Fixed small anchored radius

```bash
bash experiments/run_02_anchored_fixed_radius.sh
```

Purpose: compare against the existing `TSLRG_RADIUS` sweep.

### Direction 1: Patch-calibrated radius

```bash
bash experiments/run_03_patch_radius_sweep.sh
```

Sweep:

```text
TSLRG_RADIUS in {0.25, 0.5, 1, 2, 5}
```

Override sweep values:

```bash
RADII="0.5 1 2" bash experiments/run_03_patch_radius_sweep.sh
```

Question: does patch-normalized magnitude improve pixel localization without
hurting image AUROC?

### Direction 2: Anchor-gap radius

```bash
bash experiments/run_04_anchor_radius_sweep.sh
```

Sweep:

```text
TSLRG_RADIUS in {0.25, 0.5, 1, 2}
```

Question: are near-boundary negatives better than fixed-radius perturbations?

### Direction 3: Sparse random patches

```bash
bash experiments/run_05_sparse_random_sweep.sh
```

Sweep:

```text
TSLRG_PATCH_MASK_RATIO in {0.02, 0.05, 0.10, 0.20}
```

Question: does sparse patch supervision improve pixel AUROC/PRO?

### Direction 4: Sparse block patches

```bash
bash experiments/run_06_sparse_block_sweep.sh
```

Sweep:

```text
TSLRG_PATCH_MASK_BLOCK in {3, 5, 7, 9}
```

Question: are spatially coherent synthetic defects better for localization than
independent random patch corruption?

### Direction 5: Gradient-guided hard negatives

```bash
bash experiments/run_07_gradient_refinement_sweep.sh
```

Sweep:

```text
TSLRG_REFINE_STEPS in {1, 2, 3}
TSLRG_REFINE_STEP_SIZE in {0.02, 0.05, 0.1}
```

Question: can discriminator-guided PCA-constrained negatives tighten the decision
boundary without collapsing into trivial fake samples?

## Metrics to Track

Primary:

```text
image AUROC
pixel AUROC
PRO
anomaly-pixel AUROC
```

Training diagnostics:

```text
p_true
p_fake
fake_patch_ratio
loss
```

Geometry diagnostics to add next:

```text
Mahalanobis radius of generated anomalies
Euclidean norm of generated shifts
PCA-subspace energy vs residual energy
discriminator score before/after refinement
```

## Presentation Framing

Slide 1:

```text
Problem: SimpleNet's random feature noise is easy but geometrically blind.
```

Slide 2:

```text
Observation: real features live in a patch-specific, low-dimensional, structured
subspace; random isotropic noise spends budget in directions that may not look
like real defects.
```

Slide 3:

```text
Our generator: real anchor + patch PCA direction + calibrated radius.
```

Slide 4:

```text
New stack:
1. radius calibration
2. sparse patch masks
3. gradient-guided PCA hard negatives
```

Slide 5:

```text
Ablation table:
vanilla SimpleNet noise
anchored PCA
anchored + patch radius
anchored + anchor radius
anchored + sparse masks
anchored + sparse masks + gradient refinement
```

Slide 6:

```text
Claim to test: geometry-aware feature anomalies improve localization because
they train the discriminator on plausible, local, near-boundary deviations.
```

## Current Verification

The changed Python files pass syntax compilation with:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/simplenet_pycache \
python3 -m py_compile simplenet.py low_rank_gaussian.py
```

A runtime smoke test was not completed in this environment because the local
Python interpreter does not have `torch` installed. The next verification should
be a short GPU run on one MVTec class, ideally `screw` or `carpet`, using
`TSLRG_REFINE_STEPS=0` first and then enabling refinement.
