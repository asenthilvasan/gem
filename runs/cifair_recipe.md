# ciFAIR-10: SupCon + Memory Wrap in the GEM framework

Reproduces the SENN-style "scratch MW vs SupCon-pretrained frozen MW" ablation
using gem's `contrastive` and `memorywrap` pipelines. Results are for a single
seed on `trainval0 / test0` with the CIFAR-adapted `mobilenet` backbone.

## Prerequisites

- CUDA-capable GPU (RTX 4090 used for these runs; ~25 GB VRAM)
- `pip install -r requirements.txt` plus `tifffile` and `entmax`
- Run all commands from the repo root (`/workspace/gem`)
- ciFAIR-10 auto-downloads on first use (~160 MB)

## Code changes required

These results assume two fixes are applied to gem's pipelines:

| File | Change |
|---|---|
| [gem/pipelines/memorywrap.py](../gem/pipelines/memorywrap.py) | AMP in train + eval; guard against `None` memory loader in `evaluate_epoch`; save/restore `train_data.transform` in `evaluate()` |
| [gem/pipelines/contrastive.py](../gem/pipelines/contrastive.py) | Removed the projection head; AMP in train + eval; parent class swapped to `ContrastiveAugmentation` so the SupCon view pipeline picks up ColorJitter + Grayscale + GaussianBlur |

The parent-class swap is the most impactful change — without strong augs, the
two SupCon views are nearly identical (only random crop + hflip) so the
encoder learns near-identity features and SupCon hurts downstream MW
accuracy.

## Hyperparameter rationale (linear scaling)

Two linear-scaling rules are stacked:

- **SupCon pretrain:** reference is `batch=256, lr=0.5` (Khosla et al. CIFAR
  recipe). With `train_examples=500` and `drop_last=True`, batch=256 gives
  only 1 SGD step/epoch, so we drop to **batch=64** (7 steps/epoch) and
  linear-scale lr to **0.5 × 64/256 = 0.125**.
- **Downstream MW:** SENN's reference yaml is `batch=128, lr=0.1`. At
  **batch=64** this linear-scales to **lr = 0.1 × 64/128 = 0.05**.

Strong (SimCLR-style) augs go on the SupCon pretrain stage; the downstream MW
stage uses the basic translation+hflip augs from `BasicAugmentation`.

## Commands

All three stages run from `/workspace/gem`. v3 is the configuration that
produced the conclusive numbers in the table at the bottom.

### Stage 1 — Scratch Memory Wrap (no pretraining)

```bash
python scripts/train.py cifair10 \
    --method memorywrap --architecture mobilenet \
    --rand-shift 4 \
    --train-split trainval0 --test-split test0 \
    --epochs 200 --batch-size 64 \
    --lr 0.05 --weight-decay 5e-4 \
    --param mem_batch_size 100 \
    --load-workers 4 --eval-interval 20 \
    --save runs/cifair_v3_mw_scratch.pth \
    --history runs/cifair_v3_mw_scratch.json
```

### Stage 2 — SupCon pretraining (strong augs)

`--target-size 32 --min-scale 0.2` switches the train-aug branch from
`RandomCrop(padding)` to `RandomResizedCrop(32, scale=(0.2, 1.0))`, matching
the SENN/SimCLR recipe. The ColorJitter / Grayscale / GaussianBlur are picked
up automatically from `ContrastiveAugmentation`'s defaults.

```bash
python scripts/train.py cifair10 \
    --method contrastive --architecture mobilenet \
    --target-size 32 --min-scale 0.2 --max-scale 1.0 \
    --train-split trainval0 --test-split test0 \
    --epochs 300 --batch-size 64 \
    --lr 0.125 --weight-decay 1e-4 \
    --param loss supcon --param temperature 0.07 \
    --param xent_weight 0 \
    --load-workers 4 \
    --save runs/cifair_v3_supcon_enc.pth \
    --history runs/cifair_v3_supcon.json
```

Notes:
- `--param xent_weight 0` makes this pure SupCon. The classifier head on
  `EncoderClassifierModel` is never trained, so the "val_accuracy" reported
  during this stage is random — ignore it. The encoder is what matters.
- `gem`'s `train.py` saves the full `state_dict` of the
  `EncoderClassifierModel`; the downstream loader in `MemoryWrapClassifier`
  picks up only the `encoder.*` keys via `strict=False`.

### Stage 3 — Frozen Memory Wrap on the SupCon encoder

```bash
python scripts/train.py cifair10 \
    --method memorywrap --architecture mobilenet \
    --rand-shift 4 \
    --train-split trainval0 --test-split test0 \
    --epochs 200 --batch-size 64 \
    --lr 0.05 --weight-decay 5e-4 \
    --param mem_batch_size 100 \
    --param pretrained_encoder runs/cifair_v3_supcon_enc.pth \
    --param freeze_encoder True \
    --load-workers 4 --eval-interval 20 \
    --save runs/cifair_v3_mw_supcon_frozen.pth \
    --history runs/cifair_v3_mw_supcon_frozen.json
```

`freeze_encoder=True` freezes every parameter whose name doesn't start with
`mw.`, so only the Memory Wrap MLP head trains. The encoder runs in
`model.train()` mode (BN running stats still update) which matches SENN's
behavior.

## Results

Single seed on `trainval0 / test0`. "Lift" is balanced-accuracy difference
between the frozen-SupCon-encoder MW and the scratch MW for the same budget.

| Run | epochs (MW / SupCon / MW) | Scratch MW | Frozen MW (SupCon) | Lift |
|---|---|---:|---:|---:|
| v1 (broken: weak augs, mis-scaled lr=0.1) | 50 / 30 / 50 | 41.31% | 31.94% | −9.37pp (HURTS) |
| v2 (fixed augs, mid-budget) | 50 / 100 / 50 | 43.55% | 44.58% | +1.02pp |
| **v3 (long budget)** | **200 / 300 / 200** | **47.61%** | **52.00%** | **+4.38pp** |

Reference: gem README reports the 500-epoch xent baseline on ciFAIR-10 as
55.18%. v3 frozen-MW at 200 epochs reaches 52.00% with a different head,
matching the qualitative ordering of SENN-CINIC (SupCon-pretrained encoder
beats training from scratch).

v3 SupCon loss curve (chance floor at batch=64 is `log(127) ≈ 4.84`):

```
ep   1: 4.91
ep  31: 4.64
ep  61: 4.57
ep  91: 4.43
ep 121: 4.17
ep 151: 3.99
ep 181: 3.68
ep 211: 3.49
ep 241: 3.29
ep 271: 3.12
ep 300: 3.11
```

A monotone descent ~1.7 nats below the chance floor — healthy contrastive
learning, no collapse.

## Artifacts

All produced under `runs/` from the repo root:

```
runs/cifair_v3_mw_scratch.pth          # scratch-MW weights
runs/cifair_v3_mw_scratch.json         # per-epoch history
runs/cifair_v3_supcon_enc.pth          # pretrained encoder + (unused) classifier
runs/cifair_v3_supcon.json             # SupCon loss history
runs/cifair_v3_mw_supcon_frozen.pth    # frozen-MW weights
runs/cifair_v3_mw_supcon_frozen.json   # per-epoch history
runs/bench_cifair_v3.log               # full stdout/stderr from the chained run
```

The earlier v1 and v2 artifacts are kept under the same naming pattern
(`cifair_mw_scratch.*`, `cifair_supcon.*`, `cifair_mw_supcon_frozen.*` for v1;
the `_v2_` prefix for v2) for comparison.
