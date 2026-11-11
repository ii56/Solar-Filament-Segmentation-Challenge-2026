# Solar Filament Segmentation Challenge 2026

My solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
on Kaggle: instance segmentation of solar filaments in GONG H-alpha full-disk images.

A U-Net with an ImageNet-pretrained ResNet-34 encoder is trained on CLAHE-equalised 512x512 tiles
(BCE + Tversky loss). At test time the tile predictions are averaged into a 2048x2048 probability
map, and filaments are extracted as connected regions with hysteresis thresholding: a region of
p > 0.4 is kept only if its peak probability reaches 0.995 and it covers at least 400 px.

| | held-out validation (half B) |
|---|---|
| Panoptic Quality | 0.361 ± 0.017 |
| Foreground Dice | 0.62 |
| Inference | 0.85 s / image on an RTX 2050 (4 GB), 1.2 GB peak VRAM |

The full write-up is in [report/](report/).

## Layout

```
src/filament.py        the whole pipeline: data, tiling, model, training, inference, PQ, CSV export
tests/                 tests for the metric, post-processing and submission writer
notebooks/
  01_eda_and_baseline  data exploration and the first baseline (kept as a record)
  02_experiments       evaluation, PQ analysis, post-processing tuning, training runs
  03_submission        test-set inference and the submission CSV
submissions/           the submitted CSV
report/                technical report (LaTeX), figures, and the scripts that produce its numbers
results/               logs and plots from the first training run
docs/competition/      material from the organisers: report template, self-evaluation notebook
```

Not in the repository (see `.gitignore`): `Data/` (the competition data), `checkpoints/` (model
weights) and `cache/` (cached tile indices and probability maps).

## Setup

Python 3.12 and an NVIDIA GPU with CUDA 12.6. Install PyTorch from the CUDA index, otherwise you
get the CPU-only wheel:

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

Put the competition data under `Data/` so the layout is:

```
Data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/
  train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  train/train_images/
  test/test_images/
```

## Reproducing the submission

1. Put the trained weights at `checkpoints/unet_resnet34_clahe_bce_tversky_cosine_best.pt`.
2. Run `notebooks/03_submission.ipynb`. It predicts the 180 test images (about 3 minutes),
   writes `submissions/submission_8_0.4508_high0.995_low0.4_min_area400.csv` and checks every
   row against the predicted masks.

## Training

Training ran on one RTX 2050 (4 GB) at about 27 minutes per epoch, roughly 9 GPU-hours in total:

1. Adam, learning rate 1e-4 (constant), batch size 4, mixed precision. Best checkpoint at epoch 12
   (selected on validation tile IoU).
2. From the epoch-12 weights, 10 more epochs with cosine decay from 1e-4 to 1e-6. The submitted
   model is epoch 8 of this stage.

Both stages use `F.train` in `src/filament.py`; the calls are in `notebooks/02_experiments.ipynb`.

## Tests

```
python tests/test_filament.py
```
