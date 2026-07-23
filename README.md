# RSD-GLA

This repository contains the code for the RSD-GLA paper: Rank-Sparse Disentanglement with Global-Semantic and Local-Prototypical Alignment for Few-Shot Medical Anomaly Detection.

## Environment

Use Python 3.10+ and install from the repository root:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place datasets under `data/` using the following layout:

```text
data/BraTS_train_png/dataset_brats.csv
data/oct2017/dataset_oct2017.csv
data/liverct/dataset_livect.csv
data/resc/dataset_resc.csv
data/rsna-pneumonia-processed-dataset/dataset_rsna.csv
```

Each CSV should contain at least:

- `split`
- `image_path`
- `mask_path`
- `label`

Rules:

- `split` is used to select `train` and `test`.
- `label` should be `0` for normal and `1` for abnormal.
- Abnormal samples must have a valid `mask_path`.
- Images and masks should be stored relative to the corresponding `base_dir`.

## Training

Example:

```bash
python train_rsd_gla.py --dataset_name brats
```

Other supported dataset names:

- `brats`
- `oct`
- `liverct`
- `resc`
- `rsna`

The script writes checkpoints to `checkpoints/` and logs to `logs/`.

## Evaluation

Run evaluation only with a saved checkpoint:

```bash
python train_rsd_gla.py --dataset_name brats --eval_only --load_path checkpoints/best_rsd_gla_brats.pt
```

## Citation

If you use this repository, please cite the corresponding paper.
