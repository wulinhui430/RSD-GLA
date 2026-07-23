<div align="center">

# :rocket: RSD-GLA

### Rank-Sparse Disentanglement with Global-Semantic and Local-Prototypical Alignment for Few-Shot Medical Anomaly Detection

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-CLIP-FCC624?logo=huggingface&logoColor=black" alt="Transformers">
  <img src="https://img.shields.io/badge/Task-Few--Shot%20Medical%20Anomaly%20Detection-0A7B83" alt="Task">
</p>

<p>
  :medical_symbol: Medical Images ->
  :brain: Frozen CLIP ->
  :jigsaw: Rank-Sparse Adapters ->
  :compass: Sinkhorn LVPA + :globe_with_meridians: Cosine GTSA ->
  :fire: Anomaly Maps
</p>

</div>

RSD-GLA is a parameter-efficient few-shot medical anomaly detection framework. It keeps the CLIP visual backbone frozen and learns lightweight adapters for rank-sparse feature disentanglement, mask-guided local prototype alignment, and global text-semantic alignment.

This repository provides the minimal code required for training and evaluation. Raw datasets, model weights, logs, checkpoints, and visualization outputs are intentionally excluded.

## Highlights

- Parameter-efficient adaptation on top of frozen CLIP.
- Rank-sparse feature disentanglement with lightweight adapters.
- Mask-guided local visual prototype alignment via Sinkhorn optimal transport.
- Global text semantic alignment with cosine matching.
- Support for BraTS, OCT2017, LiverCT, RESC, and RSNA-style datasets.
- Generic dataset paths such as `data/brats/dataset.csv`, so the code is not tied to a private local directory structure.

## 1. Clone the Repository

Using SSH:

```bash
git clone git@github.com:wulinhui430/RSD-GLA.git
cd RSD-GLA
```

Using HTTPS:

```bash
git clone https://github.com/wulinhui430/RSD-GLA.git
cd RSD-GLA
```

## 2. Configure the Environment

Python 3.10 or newer is recommended. Create an isolated environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Check PyTorch and CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 3. Prepare CLIP Weights

The implementation uses a frozen CLIP ViT-L/14 visual encoder and a frozen OpenAI CLIP text encoder.

### 3.1 Vision encoder

The code loads the Hugging Face CLIP vision model locally by default. You can download it into `models/`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai/clip-vit-large-patch14', local_dir='models/clip-vit-large-patch14')"
```

Then provide the local path:

```bash
python train_rsd_gla.py --dataset_name brats --clip_model models/clip-vit-large-patch14
```

If the model already exists in the Hugging Face cache, the default `--clip_model openai/clip-vit-large-patch14` can also be used.

### 3.2 Text encoder

When text alignment is enabled, place the OpenAI CLIP checkpoint at:

```text
ViT-L-14-336px.pt
```

or pass a custom path:

```bash
python train_rsd_gla.py \
  --dataset_name brats \
  --openai_clip_ckpt models/ViT-L-14-336px.pt
```

The BPE vocabulary is included at `src/bpe_simple_vocab_16e6.txt.gz`.

## 4. Download and Organize Datasets

Download datasets from their official sources and preprocess them into image files, mask files, and CSV metadata. This repository does not redistribute medical data.

The default layout uses simple lowercase dataset names and a shared CSV filename:

```text
RSD-GLA/
|-- train_rsd_gla.py
|-- requirements.txt
|-- README.md
|-- src/
|-- data/
|   |-- brats/
|   |   |-- dataset.csv
|   |   |-- images/
|   |   |-- masks/
|   |-- oct/
|   |   |-- dataset.csv
|   |   |-- images/
|   |   |-- masks/
|   |-- liverct/
|   |   |-- dataset.csv
|   |   |-- images/
|   |   |-- masks/
|   |-- resc/
|   |   |-- dataset.csv
|   |   |-- images/
|   |   |-- masks/
|   |-- rsna/
|   |   |-- dataset.csv
|   |   |-- images/
|   |   |-- masks/
|-- models/
```

These names are only convenient defaults. Users may place data anywhere and use any dataset name by passing `--csv` and `--base_dir` explicitly.

### 4.1 CSV format

Each CSV must contain at least:

```text
split,image_path,mask_path,label
train,images/normal_0001.png,,0
train,images/abnormal_0001.png,masks/abnormal_0001.png,1
test,images/normal_0101.png,,0
test,images/abnormal_0101.png,masks/abnormal_0101.png,1
```

Required conventions:

- `split` must contain `train` and `test`.
- `label=0` denotes a normal image.
- `label=1` denotes an abnormal image.
- Abnormal samples must provide a valid `mask_path`, because lesion prototypes are mask-guided.
- `image_path` and `mask_path` are resolved relative to `base_dir`.
- For datasets with bounding boxes, convert boxes into rectangular masks before training.

### 4.2 Built-in aliases

The following aliases automatically resolve to generic default paths:

| Alias | CSV path | Base directory |
|---|---|---|
| `brats` | `data/brats/dataset.csv` | `data/brats` |
| `oct` | `data/oct/dataset.csv` | `data/oct` |
| `liverct` | `data/liverct/dataset.csv` | `data/liverct` |
| `resc` | `data/resc/dataset.csv` | `data/resc` |
| `rsna` | `data/rsna/dataset.csv` | `data/rsna` |

Custom dataset example:

```bash
python train_rsd_gla.py \
  --dataset_name my_dataset \
  --csv /path/to/my_dataset/metadata.csv \
  --base_dir /path/to/my_dataset
```

## 5. Train

Run commands from the repository root.

Default BraTS-style run:

```bash
python train_rsd_gla.py --dataset_name brats
```

Other built-in aliases:

```bash
python train_rsd_gla.py --dataset_name oct
python train_rsd_gla.py --dataset_name liverct
python train_rsd_gla.py --dataset_name resc
python train_rsd_gla.py --dataset_name rsna
```

Typical few-shot configuration:

```bash
python train_rsd_gla.py \
  --dataset_name brats \
  --k_shot 4 \
  --epochs 50 \
  --steps_per_epoch 10 \
  --device cuda \
  --seed 0
```

The implementation follows the paper protocol:

- Frozen CLIP backbone.
- Mask-guided abnormal prototypes.
- Sinkhorn-based local visual prototype alignment.
- Cosine-based global text semantic alignment.
- Direct support-set optimization, without a separate query-set branch.
- Top-k image-level anomaly scoring with default ratio `0.05`.

Outputs are written to:

```text
checkpoints/best_rsd_gla_<dataset>.pt
logs/rsd_gla_train_<dataset>_<timestamp>.log
```

## 6. Evaluate

Evaluate a checkpoint:

```bash
python train_rsd_gla.py \
  --dataset_name brats \
  --eval_only \
  --load_path checkpoints/best_rsd_gla_brats.pt
```

Save image-level anomaly scores:

```bash
python train_rsd_gla.py \
  --dataset_name brats \
  --eval_only \
  --load_path checkpoints/best_rsd_gla_brats.pt \
  --eval_save_scores_path scores/brats_scores.csv
```

The evaluator reports accuracy, precision, recall, F1-score, image-level AUROC, and, when enabled, pixel-level AUROC and Dice.

For datasets without pixel-level annotations, disable segmentation metrics:

```bash
python train_rsd_gla.py --dataset_name rsna --no-eval_seg_metrics
```

## 7. Useful Options

```bash
python train_rsd_gla.py --help
```

| Option | Purpose |
|---|---|
| `--dataset_name` | Built-in alias or custom dataset name |
| `--csv` | Custom metadata CSV path |
| `--base_dir` | Root directory for image and mask paths |
| `--k_shot` | Number of normal and abnormal support samples |
| `--epochs` | Number of training epochs |
| `--steps_per_epoch` | Optimization steps per epoch |
| `--device` | `cuda` or `cpu` |
| `--seed` | Random seed |
| `--eval_only` | Skip training and evaluate a checkpoint |
| `--load_path` | Checkpoint used for evaluation or initialization |
| `--eval_save_scores_path` | Save image-level scores to CSV |

## 8. Troubleshooting

### Dataset files cannot be found

Check that `image_path` and `mask_path` are relative to `--base_dir`, and that every abnormal sample has a valid mask.

### CLIP files cannot be loaded

Use local paths explicitly:

```bash
python train_rsd_gla.py \
  --clip_model models/clip-vit-large-patch14 \
  --cache_dir models \
  --openai_clip_ckpt models/ViT-L-14-336px.pt
```

### CUDA is unavailable

Run a CPU smoke test:

```bash
python train_rsd_gla.py --dataset_name brats --device cpu --epochs 1 --steps_per_epoch 1
```

## Citation

If you use this code, please cite the corresponding RSD-GLA paper.
