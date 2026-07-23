import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from src.datasets import ThymomaDataset
from src.models.losses import RSDGLALoss
from src.models.rsd_gla_visual import RSDGLAVisualEncoder


def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def _resolve_known_dataset(name: str) -> Optional[Tuple[str, str]]:
    root = _repo_root()
    mapping = {
        "brats": (os.path.join(root, "data", "brats", "dataset.csv"), os.path.join(root, "data", "brats")),
        "liverct": (os.path.join(root, "data", "liverct", "dataset.csv"), os.path.join(root, "data", "liverct")),
        "oct": (os.path.join(root, "data", "oct", "dataset.csv"), os.path.join(root, "data", "oct")),
        "resc": (os.path.join(root, "data", "resc", "dataset.csv"), os.path.join(root, "data", "resc")),
        "rsna": (os.path.join(root, "data", "rsna", "dataset.csv"), os.path.join(root, "data", "rsna")),
    }
    key = str(name).strip()
    return mapping.get(key)


@dataclass
class EvalMetrics:
    acc: float
    precision: float
    recall: float
    f1: float
    auc: float
    pixel_auroc: float
    dice: float
    tp: int
    fp: int
    tn: int
    fn: int
    cls_threshold_used: float


class _TimestampedTee:
    def __init__(self, original_stream, file_path: str, tz: ZoneInfo) -> None:
        self._orig = original_stream
        self._fh = open(file_path, "a", buffering=1, encoding="utf-8")
        self._tz = tz
        self._buf = ""

    def write(self, s: str) -> int:
        n = self._orig.write(s)
        self._orig.flush()

        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            ts = datetime.now(self._tz).strftime("%Y-%m-%d %H:%M:%S")
            self._fh.write(f"{ts} {line}\n")
            self._fh.flush()
        return n

    def flush(self) -> None:
        self._orig.flush()
        if self._buf:
            ts = datetime.now(self._tz).strftime("%Y-%m-%d %H:%M:%S")
            self._fh.write(f"{ts} {self._buf}")
            self._fh.flush()
            self._buf = ""

    def isatty(self) -> bool:
        return bool(getattr(self._orig, "isatty", lambda: False)())


def _setup_run_logging(log_dir: str, log_file: str) -> str:
    tz = ZoneInfo("Asia/Shanghai")
    os.makedirs(log_dir, exist_ok=True)
    abs_log_file = os.path.abspath(log_file)

    sys.stdout = _TimestampedTee(sys.stdout, abs_log_file, tz)  # type: ignore[assignment]
    sys.stderr = _TimestampedTee(sys.stderr, abs_log_file, tz)  # type: ignore[assignment]
    return abs_log_file


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_device(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but --device=cuda was requested")
    return dev


def sample_episode(
    rng: np.random.Generator,
    normal_indices: List[int],
    abnormal_indices_nonempty: List[int],
    k_shot: int,
) -> List[int]:
    if len(normal_indices) < k_shot:
        raise RuntimeError("Not enough normal samples for support")
    if len(abnormal_indices_nonempty) < k_shot:
        raise RuntimeError("Not enough abnormal non-empty-mask samples for support")

    support = rng.choice(normal_indices, size=k_shot, replace=False).tolist() + rng.choice(
        abnormal_indices_nonempty, size=k_shot, replace=False
    ).tolist()
    return support


def collate_by_indices(dataset: ThymomaDataset, indices: List[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs = []
    masks = []
    labels = []
    for i in indices:
        img, mask, y = dataset[int(i)]
        imgs.append(img)
        masks.append(mask)
        labels.append(y)
    return torch.stack(imgs).to(device), torch.stack(masks).to(device), torch.stack(labels).to(device)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    B = pred.size(0)
    pred_f = pred.view(B, -1)
    target_f = target.view(B, -1)
    inter = (pred_f * target_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + target_f.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.mean().detach().cpu())


def _best_threshold_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str,
    grid_size: int,
) -> float:
    if y_true.size == 0:
        return 0.5
    y_true = y_true.astype(np.int64)

    score_min = float(np.min(y_score))
    score_max = float(np.max(y_score))
    if not np.isfinite(score_min) or not np.isfinite(score_max) or score_min == score_max:
        return float(np.clip(score_min, 0.0, 1.0))

    grid_size = int(max(10, grid_size))
    thresholds = np.linspace(score_min, score_max, num=grid_size, dtype=np.float64)

    best_t = float(thresholds[0])
    best_v = -1.0

    for t in thresholds:
        y_pred = (y_score > float(t)).astype(np.int64)
        if metric == "acc":
            v = float(accuracy_score(y_true, y_pred))
        elif metric == "precision":
            v = float(precision_score(y_true, y_pred, zero_division=0))
        elif metric == "recall":
            v = float(recall_score(y_true, y_pred, zero_division=0))
        else:  # f1
            v = float(f1_score(y_true, y_pred, zero_division=0))
        if v > best_v:
            best_v = v
            best_t = float(t)

    return best_t


@torch.no_grad()
def evaluate(
    model: RSDGLAVisualEncoder,
    train_dataset: ThymomaDataset,
    test_loader: DataLoader,
    device: torch.device,
    cls_threshold: float,
    seg_threshold: float,
    cls_score: str,
    cls_topk_ratio: float,
    eval_seg_metrics: bool,
    k_shot: int,
    seed: int,
    use_proto_branch: bool = True,
    prototypes_override: Optional[torch.Tensor] = None,
    calib_loader: Optional[DataLoader] = None,
    cls_threshold_mode: str = "fixed",
    cls_threshold_metric: str = "f1",
    cls_threshold_grid: int = 200,
    save_scores_path: str = "",
) -> EvalMetrics:
    model.eval()

    rng = np.random.default_rng(seed)

    prototypes: Optional[torch.Tensor] = None
    if not bool(use_proto_branch):
        prototypes = None
    elif prototypes_override is not None:
        prototypes = prototypes_override
    else:
        normal_idx = [i for i in range(len(train_dataset)) if int(train_dataset.df.iloc[i]["label"]) == 0]
        abnormal_idx = [i for i in range(len(train_dataset)) if int(train_dataset.df.iloc[i]["label"]) == 1]

        support_idx = sample_episode(rng, normal_idx, abnormal_idx, k_shot=k_shot)
        support_imgs, support_masks, _ = collate_by_indices(train_dataset, support_idx, device=device)
        support_imgs_abn = support_imgs[k_shot:]
        support_masks_abn = support_masks[k_shot:]

        prototypes = model.extract_tumor_prototypes(
            support_imgs_abn,
            support_masks_abn,
            seed=seed,
        )
        if prototypes is None:
            raise RuntimeError("Prototype extraction returned None during evaluation")

    def _score_from_map(amap: torch.Tensor) -> torch.Tensor:
        if cls_score == "max":
            return amap.amax(dim=(2, 3)).squeeze(1)
        if cls_score == "mean":
            return amap.mean(dim=(2, 3)).squeeze(1)
        if cls_score == "topk_mean":
            flat = amap.flatten(2).squeeze(1)  # [B, H*W]
            ratio = float(cls_topk_ratio)
            if ratio <= 0.0 or ratio > 1.0:
                raise ValueError(f"cls_topk_ratio must be in (0,1] but got {ratio}")
            k = max(1, int(ratio * float(flat.size(1))))
            topk = flat.topk(k, dim=1).values
            return topk.mean(dim=1)
        raise ValueError(f"Unknown cls_score={cls_score!r}. Expected one of: max, mean, topk_mean")

    cls_threshold_used = float(cls_threshold)
    if cls_threshold_mode == "calibrate" and calib_loader is not None:
        calib_true: List[int] = []
        calib_score: List[float] = []
        for batch in calib_loader:
            imgs, masks, labels = batch
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(imgs, tumor_prototypes=prototypes)
            anomaly_map = outputs["anomaly_map"]
            score = _score_from_map(anomaly_map)
            calib_true.extend(labels.detach().cpu().numpy().astype(np.int64).tolist())
            calib_score.extend(score.detach().cpu().numpy().astype(np.float64).tolist())

        calib_true_arr = np.asarray(calib_true, dtype=np.int64)
        calib_score_arr = np.asarray(calib_score, dtype=np.float64)
        cls_threshold_used = _best_threshold_from_scores(
            calib_true_arr,
            calib_score_arr,
            metric=str(cls_threshold_metric),
            grid_size=int(cls_threshold_grid),
        )

    y_true = []
    y_score = []
    dice_scores = []

    pixel_true = []
    pixel_score = []

    for images, masks, labels in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        outputs = model(images, tumor_prototypes=prototypes)
        anomaly_map = outputs["anomaly_map"]
        score = _score_from_map(anomaly_map)

        if eval_seg_metrics:
            pixel_true.append(masks.detach().flatten().to(dtype=torch.float32).cpu().numpy())
            pixel_score.append(anomaly_map.detach().flatten().to(dtype=torch.float32).cpu().numpy())

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_score.extend(score.detach().cpu().numpy().tolist())

        if eval_seg_metrics:
            bin_map = (anomaly_map > seg_threshold).float()
            dice_scores.append(compute_dice(bin_map, masks))

    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    y_pred_arr = (y_score_arr > float(cls_threshold_used)).astype(np.int64)

    save_scores_path = str(save_scores_path).strip()
    if save_scores_path:
        out_df = pd.DataFrame({"label": y_true_arr.astype(np.int64), "score": y_score_arr.astype(np.float64)})
        os.makedirs(os.path.dirname(save_scores_path) or ".", exist_ok=True)
        out_df.to_csv(save_scores_path, index=False)
        print(f"[eval] saved_scores={save_scores_path} n={len(out_df)}", flush=True)

    tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    tn = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true_arr, y_score_arr))
    except Exception:
        auc = 0.0

    if eval_seg_metrics:
        pixel_true_arr = np.concatenate(pixel_true, axis=0) if pixel_true else np.asarray([], dtype=np.float32)
        pixel_score_arr = np.concatenate(pixel_score, axis=0) if pixel_score else np.asarray([], dtype=np.float32)

        try:
            pixel_auroc = float(roc_auc_score(pixel_true_arr, pixel_score_arr))
        except Exception:
            pixel_auroc = 0.0

        dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    else:
        pixel_auroc = 0.0
        dice = 0.0

    return EvalMetrics(
        acc=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        pixel_auroc=pixel_auroc,
        dice=dice,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        cls_threshold_used=float(cls_threshold_used),
    )


def save_checkpoint(path: str, model: RSDGLAVisualEncoder, optimizer: torch.optim.Optimizer, epoch: int, metrics: EvalMetrics) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics.__dict__,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    default_csv = "data/brats/dataset.csv"
    default_base_dir = "data/brats"
    parser.add_argument("--csv", default=default_csv)
    parser.add_argument("--base_dir", default=default_base_dir)
    parser.add_argument("--cache_dir", default="./models")
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--log_file", default="")

    parser.add_argument(
        "--eval_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, skip training and only run evaluation once on the specified dataset.",
    )

    parser.add_argument(
        "--eval_save_scores_path",
        default="",
        help="Optional. When set, save per-sample labels and fused anomaly scores (from --cls_score) on the test set to this CSV path during evaluation.",
    )
    parser.add_argument(
        "--load_path",
        default="",
        help="Optional. Path to a checkpoint created by this script to load weights from (model_state only).",
    )

    parser.add_argument(
        "--enable_text_align",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--openai_clip_ckpt", default="./ViT-L-14-336px.pt")
    parser.add_argument("--openai_bpe_path", default="")
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=8,
        help="Number of learnable context tokens (CoOp prompt length) used in GTSA when --enable_text_align is set.",
    )
    parser.add_argument("--fusion_weight", type=float, default=0.3)

    parser.add_argument("--proto_align_temperature", type=float, default=0.07)

    parser.add_argument(
        "--use_proto_branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use the prototype branch at all. Set --no-use_proto_branch to disable prototype extraction/alignment (e.g., text-only).",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=10)
    parser.add_argument("--k_shot", type=int, default=4)

    parser.add_argument("--proto_sampling", choices=["per_step", "per_epoch", "fixed"], default="per_epoch")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_rec", type=float, default=0.5)
    parser.add_argument("--lambda_rank", type=float, default=0.01)
    parser.add_argument("--lambda_sparse", type=float, default=0.001)
    parser.add_argument("--lambda_geo", type=float, default=0.1)
    parser.add_argument("--lambda_dice", type=float, default=1.0)
    parser.add_argument("--lambda_focal", type=float, default=1.0)

    parser.add_argument("--cls_threshold", type=float, default=0.4)
    parser.add_argument("--seg_threshold", type=float, default=0.5)
    parser.add_argument("--cls_score", choices=["max", "mean", "topk_mean"], default="topk_mean")
    parser.add_argument("--cls_topk_ratio", type=float, default=0.05)
    parser.add_argument(
        "--eval_seg_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compute segmentation metrics (pixel AUROC / Dice) on the test set.",
    )

    parser.add_argument(
        "--cls_threshold_mode",
        choices=["fixed", "calibrate"],
        default="calibrate",
        help="fixed: use --cls_threshold. calibrate: select threshold on a calibration subset of the training set each epoch.",
    )
    parser.add_argument(
        "--cls_threshold_metric",
        choices=["f1", "acc", "precision", "recall"],
        default="f1",
        help="Metric optimized on calibration set when --cls_threshold_mode=calibrate.",
    )
    parser.add_argument(
        "--cls_threshold_grid",
        type=int,
        default=200,
        help="Number of threshold candidates to scan between min/max score on calibration set.",
    )
    parser.add_argument(
        "--cls_threshold_calib_per_class",
        type=int,
        default=50,
        help="How many samples per class to use from train set for threshold calibration.",
    )
    parser.add_argument("--eval_proto_source", choices=["fixed_seed", "train_cached"], default="train_cached")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--resume_path",
        default="",
        help="Optional. Path to a checkpoint created by this script to resume training from (loads model_state; optionally optimizer_state).",
    )
    parser.add_argument(
        "--resume_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, whether to also load optimizer_state from the checkpoint.",
    )

    args = parser.parse_args()

    requested_dataset = str(args.dataset_name).strip()
    if requested_dataset and str(args.csv) == default_csv and str(args.base_dir) == default_base_dir:
        resolved = _resolve_known_dataset(requested_dataset)
        if resolved is not None:
            args.csv, args.base_dir = resolved

    tz = ZoneInfo("Asia/Shanghai")
    dataset_name = str(args.dataset_name).strip()
    if not dataset_name:
        dataset_name = os.path.splitext(os.path.basename(args.csv))[0].replace("dataset_", "")
    if dataset_name == "dataset":
        dataset_name = "brats"

    if dataset_name.lower() == "oct":
        explicit_eval_seg = any(a.startswith("--eval_seg_metrics") for a in sys.argv[1:])
        if not explicit_eval_seg:
            args.eval_seg_metrics = False

    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    log_file = str(args.log_file).strip()
    if not log_file:
        log_file = os.path.join(str(args.log_dir), f"rsd_gla_train_{dataset_name}_{ts}.log")

    abs_log_file = _setup_run_logging(str(args.log_dir), log_file)
    print(f"[log] file={abs_log_file}", flush=True)

    run_args = vars(args).copy()
    run_args["dataset_name"] = dataset_name
    run_args["log_file"] = abs_log_file
    print(f"[args] argv={' '.join(sys.argv)}", flush=True)
    print(f"[args] parsed={json.dumps(run_args, ensure_ascii=False, sort_keys=True)}", flush=True)

    set_seed(args.seed)
    device = ensure_device(args.device)

    import time

    t0 = time.time()
    print("[stage] read_csv...", flush=True)

    df = pd.read_csv(args.csv)

    print(f"[stage] read_csv done ({time.time() - t0:.1f}s)", flush=True)
    t1 = time.time()
    print("[stage] init datasets...", flush=True)

    train_dataset = ThymomaDataset(
        dataframe=df,
        split="train",
        base_dir=args.base_dir,
        use_clip=True,
        cache_dir=args.cache_dir,
        clip_model_name_or_path=args.clip_model,
        local_files_only=True,
        strict_paths=True,
    )
    test_dataset = ThymomaDataset(
        dataframe=df,
        split="test",
        base_dir=args.base_dir,
        use_clip=True,
        cache_dir=args.cache_dir,
        clip_model_name_or_path=args.clip_model,
        local_files_only=True,
        strict_paths=True,
    )

    print(f"[stage] init datasets done ({time.time() - t1:.1f}s)", flush=True)

    t2 = time.time()
    print("[stage] init model (CLIP load may take a while on first run)...", flush=True)

    dataset_to_obj = {
        "brats": "brain tumor",
        "liverct": "liver lesion",
        "oct": "retinal lesion",
        "resc": "retinal edema",
        "rsna": "chest pneumonia",
    }
    obj_name = dataset_to_obj.get(dataset_name, dataset_name)
    abnormal_prompt_texts = (
        f"damaged {obj_name}",
        f"broken {obj_name}",
        f"{obj_name} with flaw",
        f"{obj_name} with defect",
        f"{obj_name} with damage",
        f"disease {obj_name}",
        f"abnormal {obj_name}",
    )

    model = RSDGLAVisualEncoder(
        cache_dir=args.cache_dir,
        clip_model_name_or_path=args.clip_model,
        local_files_only=True,
        proto_align_temperature=float(args.proto_align_temperature),
        enable_text_align=bool(args.enable_text_align),
        openai_clip_ckpt_path=str(args.openai_clip_ckpt),
        openai_bpe_path=str(args.openai_bpe_path),
        abnormal_prompt_texts=abnormal_prompt_texts,
        prompt_length=int(args.prompt_length),
        fusion_weight=float(args.fusion_weight),
    ).to(device)

    print(f"[stage] init model done ({time.time() - t2:.1f}s)", flush=True)

    load_path = str(args.load_path).strip()
    if load_path:
        ckpt_obj = torch.load(load_path, map_location=device)
        if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
            missing, unexpected = model.load_state_dict(ckpt_obj["model_state"], strict=False)
            if missing or unexpected:
                print(f"[load] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            ckpt_epoch = ckpt_obj.get("epoch")
            try:
                ckpt_epoch_s = "?" if ckpt_epoch is None else str(int(ckpt_epoch))
            except Exception:
                ckpt_epoch_s = str(ckpt_epoch)
            print(f"[load] path={load_path} epoch={ckpt_epoch_s}", flush=True)
        else:
            raise RuntimeError(f"Unsupported load checkpoint format: {load_path}")

    if bool(args.eval_only):
        train_labels = train_dataset.df["label"].astype(int).tolist()
        rng = np.random.default_rng(args.seed)
        eval_batch_size = 4

        calib_loader = None
        if args.cls_threshold_mode == "calibrate":
            calib_indices = []
            for cls in [0, 1]:
                cls_indices = [i for i, y in enumerate(train_labels) if int(y) == cls]
                if not cls_indices:
                    continue
                calib_indices.extend(
                    rng.choice(
                        cls_indices,
                        size=min(args.cls_threshold_calib_per_class, len(cls_indices)),
                        replace=False,
                    )
                )
            if calib_indices:
                calib_dataset = Subset(train_dataset, calib_indices)
                calib_loader = DataLoader(
                    calib_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"),
                )

        metrics = evaluate(
            model=model,
            train_dataset=train_dataset,
            test_loader=DataLoader(
                test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            ),
            device=device,
            cls_threshold=args.cls_threshold,
            seg_threshold=args.seg_threshold,
            cls_score=args.cls_score,
            cls_topk_ratio=args.cls_topk_ratio,
            eval_seg_metrics=bool(args.eval_seg_metrics),
            k_shot=args.k_shot,
            seed=int(args.seed),
            use_proto_branch=bool(args.use_proto_branch),
            calib_loader=calib_loader,
            cls_threshold_mode=str(args.cls_threshold_mode),
            cls_threshold_metric=str(args.cls_threshold_metric),
            cls_threshold_grid=int(args.cls_threshold_grid),
            save_scores_path=str(args.eval_save_scores_path),
        )

        if bool(args.eval_seg_metrics):
            print(
                "[eval_only] "
                f"acc={metrics.acc:.4f} precision={metrics.precision:.4f} recall={metrics.recall:.4f} "
                f"f1={metrics.f1:.4f} auc={metrics.auc:.4f} pixel_auroc={metrics.pixel_auroc:.4f} "
                f"dice={metrics.dice:.4f} cls_threshold={metrics.cls_threshold_used:.4f}",
                flush=True,
            )
        else:
            print(
                "[eval_only] "
                f"acc={metrics.acc:.4f} precision={metrics.precision:.4f} recall={metrics.recall:.4f} "
                f"f1={metrics.f1:.4f} auc={metrics.auc:.4f} cls_threshold={metrics.cls_threshold_used:.4f}",
                flush=True,
            )
        return

    loss_fn = RSDGLALoss(
        lambda_rec=float(args.lambda_rec),
        lambda_rank=float(args.lambda_rank),
        lambda_sparse=float(args.lambda_sparse),
        lambda_geo=float(args.lambda_geo),
        lambda_dice=float(args.lambda_dice),
        lambda_focal=float(args.lambda_focal),
    ).to(device)

    opt_params = list(model.normal_adapter.parameters()) + list(model.abnormal_adapter.parameters())
    if bool(args.enable_text_align):
        if getattr(model, "abnormal_prompt", None) is not None:
            opt_params += list(model.abnormal_prompt.parameters())
        if getattr(model, "image_to_text_proj", None) is not None:
            opt_params += list(model.image_to_text_proj.parameters())

    opt = AdamW(
        opt_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_auc = -1.0
    resume_path = str(args.resume_path).strip()
    if resume_path:
        ckpt_obj = torch.load(resume_path, map_location=device)
        if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
            missing, unexpected = model.load_state_dict(ckpt_obj["model_state"], strict=False)
            if missing or unexpected:
                print(f"[resume] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            ckpt_epoch = ckpt_obj.get("epoch")
            if ckpt_epoch is not None:
                start_epoch = int(ckpt_epoch) + 1
            if bool(args.resume_optimizer) and "optimizer_state" in ckpt_obj:
                try:
                    opt.load_state_dict(ckpt_obj["optimizer_state"])
                except Exception as e:
                    print(f"[resume] WARNING: failed to load optimizer_state: {e}", flush=True)
            m = ckpt_obj.get("metrics")
            if isinstance(m, dict) and m.get("auc") is not None:
                try:
                    best_auc = float(m.get("auc"))
                except Exception:
                    best_auc = -1.0
            print(f"[resume] path={resume_path} start_epoch={start_epoch} best_auc_init={best_auc:.4f}", flush=True)
        else:
            raise RuntimeError(f"Unsupported resume checkpoint format: {resume_path}")

    train_labels = train_dataset.df["label"].astype(int).tolist()
    normal_indices = [i for i, y in enumerate(train_labels) if int(y) == 0]
    abnormal_indices_nonempty = [i for i, y in enumerate(train_labels) if int(y) == 1]

    print(
        f"[stage] indices ready: normal={len(normal_indices)} abnormal={len(abnormal_indices_nonempty)}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)

    cached_support_idx: Optional[List[int]] = None
    cached_support_imgs_abn: Optional[torch.Tensor] = None
    cached_support_masks_abn: Optional[torch.Tensor] = None
    eval_batch_size = 4

    best_epoch = -1
    best_cm: Tuple[int, int, int, int] = (0, 0, 0, 0)
    if start_epoch > int(args.epochs):
        raise ValueError(f"resume start_epoch={start_epoch} exceeds --epochs={int(args.epochs)}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        running_loss = 0.0
        if args.proto_sampling == "per_epoch":
            cached_support_idx = None
            cached_support_imgs_abn = None
            cached_support_masks_abn = None
        for step in range(args.steps_per_epoch):
            prototypes: Optional[torch.Tensor] = None
            if args.proto_sampling == "per_step":
                support_idx = sample_episode(
                    rng,
                    normal_indices=normal_indices,
                    abnormal_indices_nonempty=abnormal_indices_nonempty,
                    k_shot=args.k_shot,
                )
                support_imgs, support_masks, _ = collate_by_indices(train_dataset, support_idx, device=device)
                support_imgs_abn = support_imgs[args.k_shot:]
                support_masks_abn = support_masks[args.k_shot:]
            else:
                if cached_support_idx is None or cached_support_imgs_abn is None or cached_support_masks_abn is None:
                    cached_support_idx = sample_episode(
                        rng,
                        normal_indices=normal_indices,
                        abnormal_indices_nonempty=abnormal_indices_nonempty,
                        k_shot=args.k_shot,
                    )
                    support_imgs, support_masks, _ = collate_by_indices(train_dataset, cached_support_idx, device=device)
                    cached_support_imgs_abn = support_imgs[args.k_shot:]
                    cached_support_masks_abn = support_masks[args.k_shot:]
                else:
                    support_imgs, support_masks, _ = collate_by_indices(train_dataset, cached_support_idx, device=device)
                support_imgs_abn = cached_support_imgs_abn
                support_masks_abn = cached_support_masks_abn

            if bool(args.use_proto_branch):
                prototypes = model.extract_tumor_prototypes(
                    support_imgs_abn,
                    support_masks_abn,
                    seed=args.seed,
                )

            train_imgs = support_imgs
            train_masks = support_masks
            outputs = model(train_imgs, tumor_prototypes=prototypes)
            loss, _ = loss_fn(outputs, train_masks)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += float(loss.detach().cpu())

        avg_loss = running_loss / max(1, args.steps_per_epoch)

        calib_loader = None
        if args.cls_threshold_mode == "calibrate":
            calib_indices = []
            for cls in [0, 1]:
                cls_indices = [i for i, y in enumerate(train_labels) if int(y) == cls]
                calib_indices.extend(rng.choice(cls_indices, size=min(args.cls_threshold_calib_per_class, len(cls_indices)), replace=False))
            calib_dataset = Subset(train_dataset, calib_indices)
            calib_loader = DataLoader(
                calib_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )

        eval_prototypes = None
        if bool(args.use_proto_branch) and args.eval_proto_source == "train_cached":
            # Use the cached support to build eval prototypes (no_grad) so eval condition matches the current epoch.
            if cached_support_imgs_abn is not None and cached_support_masks_abn is not None:
                with torch.no_grad():
                    eval_prototypes = model.extract_tumor_prototypes(
                        cached_support_imgs_abn,
                        cached_support_masks_abn,
                        seed=args.seed,
                    )

        metrics = evaluate(
            model=model,
            train_dataset=train_dataset,
            test_loader=DataLoader(
                test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            ),
            device=device,
            cls_threshold=args.cls_threshold,
            seg_threshold=args.seg_threshold,
            cls_score=args.cls_score,
            cls_topk_ratio=args.cls_topk_ratio,
            eval_seg_metrics=bool(args.eval_seg_metrics),
            k_shot=args.k_shot,
            seed=int(args.seed),
            use_proto_branch=bool(args.use_proto_branch),
            calib_loader=calib_loader,
            cls_threshold_mode=str(args.cls_threshold_mode),
            cls_threshold_metric=str(args.cls_threshold_metric),
            cls_threshold_grid=int(args.cls_threshold_grid),
            prototypes_override=eval_prototypes,
        )

        if bool(args.eval_seg_metrics):
            print(
                f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
                f"acc={metrics.acc:.4f} precision={metrics.precision:.4f} recall={metrics.recall:.4f} "
                f"f1={metrics.f1:.4f} auc={metrics.auc:.4f} pixel_auroc={metrics.pixel_auroc:.4f} "
                f"dice={metrics.dice:.4f} cls_threshold={metrics.cls_threshold_used:.4f}",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
                f"acc={metrics.acc:.4f} precision={metrics.precision:.4f} recall={metrics.recall:.4f} "
                f"f1={metrics.f1:.4f} auc={metrics.auc:.4f} cls_threshold={metrics.cls_threshold_used:.4f}",
                flush=True,
            )

        if metrics.auc > best_auc:
            best_auc = metrics.auc
            best_epoch = epoch
            best_cm = (metrics.tp, metrics.fp, metrics.tn, metrics.fn)
            ckpt_path = os.path.join(args.save_dir, f"best_rsd_gla_{dataset_name}.pt")
            save_checkpoint(ckpt_path, model, opt, epoch, metrics)
            print(f"Saved best checkpoint to {ckpt_path}")

    if best_epoch > 0:
        tp, fp, tn, fn = best_cm
        print(f"[best] epoch={best_epoch} auc={best_auc:.4f}")
        print("[best] Confusion Matrix:")
        print("           Normal    Abnormal")
        print(f"Normal      {tn:3d}       {fp:3d}")
        print(f"Abnormal    {fn:3d}       {tp:3d}")
        print(f"[best] TP={tp}, FP={fp}, TN={tn}, FN={fn}", flush=True)


if __name__ == "__main__":
    main()
