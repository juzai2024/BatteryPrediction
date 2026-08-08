# BatteryPrediction

This repository provides code for our paper, **"Computational Ultra-low-field MRI Enables Non-invasive Diagnostics of Commercial Lithium-ion Batteries."** It contains a minimal inference pipeline for predicting battery capacity directly from six-channel ultra-low-field MRI data.

Only the files required for inference are included: a shared ViT encoder, separate LCO and NMC capacity heads, and five toy-test samples for each battery type.

## Installation

Python 3.8 or newer and a CUDA-capable GPU are recommended. CPU inference is supported but considerably slower.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_weights.py
```

The shared encoder is approximately 608 MB and is downloaded from the [`v0.1.0` GitHub release](https://github.com/juzai2024/BatteryPrediction/releases/tag/v0.1.0). The two small capacity heads are included in `weights/`.

## Toy test

Run capacity prediction for the included LCO or NMC examples:

```bash
python predict.py --battery LCO
python predict.py --battery NMC
```

To predict one new sample or a directory of samples:

```bash
python predict.py --battery LCO --input path/to/sample.mat
python predict.py --battery NMC --input path/to/mat_directory
```

Each MAT file must contain a `data` array with shape `(128, 128, 6)`. An optional scalar `ah` field is treated as the reference capacity and used to display the absolute prediction error. Predicted capacity is reported in ampere-hours (Ah).

## Released model performance

Performance below is measured on the complete test set.

| Battery | R² | MAE (Ah) | RMSE (Ah) |
| --- | ---: | ---: | ---: |
| LCO | 0.8362 | 0.0968 | 0.1039 |
| NMC | 0.9763 | 0.0287 | 0.0395 |

Each `toy_data/` directory contains five example samples for quickly testing the inference pipeline.

## Model files

| File | Purpose |
| --- | --- |
| `battery_vit_encoder_fp16.pth` | Shared ViT-L/16 MRI encoder; downloaded separately |
| `lco_capacity_head.pth` | LCO capacity regression head |
| `nmc_capacity_head.pth` | NMC capacity regression head |

The encoder is stored in FP16 to reduce download size, then loaded for FP32 inference to preserve numerical stability. The architecture follows Vision Transformer and MAE conventions and uses [`timm`](https://github.com/huggingface/pytorch-image-models).
