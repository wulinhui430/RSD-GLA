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

The released implementation supports:

- BraTS
- OCT2017
- LiverCT
- RESC
- RSNA

The repository contains the minimal training and evaluation code. Raw datasets, model weights, logs, checkpoints, and visualization outputs are intentionally excluded.

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

Install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

CUDA is recommended for training. To verify the PyTorch installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 3. Prepare the CLIP Weights

The implementation uses a frozen `CLIP ViT-L/14` visual encoder and a frozen OpenAI CLIP text encoder.

### 3.1 Vision encoder

Because the code uses local loading by default, make the Hugging Face CLIP files available locally. For example:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai/clip-vit-large-patch14', local_dir='models/clip-vit-large-patch14')"
```

Then train with:

```bash
python train_rsd_gla.py --dataset_name brats --clip_model models/clip-vit-large-patch14
```

Alternatively, place the model in your local Hugging Face cache and keep the default `--clip_model` value.

### 3.2 Text encoder

When text alignment is enabled, place the OpenAI CLIP checkpoint at:

```text
ViT-L-14-336px.pt
```

or provide another path explicitly:

```bash
python train_rsd_gla.py \
  --dataset_name brats \
  --openai_clip_ckpt models/ViT-L-14-336px.pt
```

The bundled BPE vocabulary is stored at `src/bpe_simple_vocab_16e6.txt.gz`.

## 4. Download and Organize the Datasets

Download the datasets from their official sources and preprocess them into PNG/JPEG images, masks, and CSV metadata. The repository does not redistribute raw medical data.

The default directory layout is:

```text
RSD-GLA/
├── train_rsd_gla.py
├── requirements.txt
├── README.md
├── src/
├── data/
│   ├── BraTS_train_png/
│   │   └── dataset_brats.csv
│   ├── oct2017/
│   │   └── dataset_oct2017.csv
│   ├── liverct/
│   │   └── dataset_livect.csv
│   ├── resc/
│   │   └── dataset_resc.csv
│   └── rsna-pneumonia-processed-dataset/
│       └── dataset_rsna.csv
└── models/
```

The folder and CSV names are conventions, not hard requirements. If you use different names, pass `--csv` and `--base_dir` explicitly.

### 4.1 CSV format

Each CSV must contain at least these columns:

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
- For datasets with bounding boxes, convert the boxes into rectangular masks before training.

### 4.2 Built-in dataset aliases

The following aliases automatically resolve to the default paths:

| Alias | CSV path | Base directory |
|---|---|---|
| `brats` | `data/BraTS_train_png/dataset_brats.csv` | `data/BraTS_train_png` |
| `oct` | `data/oct2017/dataset_oct2017.csv` | `data/oct2017` |
| `liverct` | `data/liverct/dataset_livect.csv` | `data/liverct` |
| `resc` | `data/resc/dataset_resc.csv` | `data/resc` |
| `rsna` | `data/rsna-pneumonia-processed-dataset/dataset_rsna.csv` | `data/rsna-pneumonia-processed-dataset` |

Example with a custom dataset location:

```bash
python train_rsd_gla.py \
  --dataset_name my_dataset \
  --csv /path/to/my_dataset.csv \
  --base_dir /path/to/my_dataset
```

## 5. Train the Model

Run commands from the repository root:

```bash
python train_rsd_gla.py --dataset_name brats
```

Other datasets:

```bash
python train_rsd_gla.py --dataset_name oct
python train_rsd_gla.py --dataset_name liverct
python train_rsd_gla.py --dataset_name resc
python train_rsd_gla.py --dataset_name rsna
```

A typical few-shot configuration is:

```bash
python train_rsd_gla.py \
  --dataset_name brats \
  --k_shot 4 \
  --epochs 50 \
  --steps_per_epoch 10 \
  --device cuda \
  --seed 0
```

The current implementation follows the paper protocol:

- The CLIP backbone is frozen.
- Prototypes are constructed from abnormal support samples using masks.
- Local prototype alignment uses Sinkhorn optimal transport only.
- Global text alignment uses cosine similarity only.
- Training directly uses the sampled support set; there is no separate query set.
- The image-level anomaly score uses top-k aggregation with a default ratio of `0.05`.

Checkpoints and logs are generated automatically:

```text
checkpoints/best_rsd_gla_<dataset>.pt
logs/rsd_gla_train_<dataset>_<timestamp>.log
```

## 6. Evaluate a Trained Checkpoint

Evaluate a saved checkpoint:

```bash
python train_rsd_gla.py \
  --dataset_name brats \
  --eval_only \
  --load_path checkpoints/best_rsd_gla_brats.pt
```

Save per-image anomaly scores:

```bash
python train_rsd_gla.py \
  --dataset_name brats \
  --eval_only \
  --load_path checkpoints/best_rsd_gla_brats.pt \
  --eval_save_scores_path scores/brats_scores.csv
```

The evaluator reports:

- Accuracy
- Precision
- Recall
- F1-score
- Image-level AUROC
- Pixel-level AUROC when segmentation metrics are enabled
- Dice score when segmentation metrics are enabled

For OCT2017, segmentation metrics are disabled by default when no explicit `--eval_seg_metrics` option is provided. Add `--eval_seg_metrics` when pixel-level evaluation is available.

## 7. Useful Options

```bash
python train_rsd_gla.py --help
```

Common options:

| Option | Purpose |
|---|---|
| `--dataset_name` | Dataset alias or custom dataset name |
| `--csv` | Custom metadata CSV path |
| `--base_dir` | Root directory for image and mask paths |
| `--k_shot` | Number of normal and abnormal support samples |
| `--epochs` | Number of training epochs |
| `--steps_per_epoch` | Support-set optimization steps per epoch |
| `--device` | `cuda` or `cpu` |
| `--seed` | Random seed |
| `--eval_only` | Skip training and evaluate a checkpoint |
| `--load_path` | Checkpoint used for evaluation or initialization |
| `--eval_save_scores_path` | CSV path for saving image-level scores |

## 8. Troubleshooting

### Dataset files cannot be found

Check that `image_path` and `mask_path` are relative to `--base_dir`, and that abnormal samples have masks.

### CLIP files cannot be loaded

Use local paths explicitly:

```bash
python train_rsd_gla.py \
  --clip_model models/clip-vit-large-patch14 \
  --cache_dir models \
  --openai_clip_ckpt models/ViT-L-14-336px.pt
```

### CUDA is unavailable

Run a small smoke test on CPU:

```bash
python train_rsd_gla.py --dataset_name brats --device cpu --epochs 1 --steps_per_epoch 1
```

## Citation

If you use this code, please cite the corresponding RSD-GLA paper.
