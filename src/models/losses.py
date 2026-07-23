from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, 0.0, 1.0)
        target = (target > 0.5).float()

        B = pred.size(0)
        pred_f = pred.view(B, -1)
        target_f = target.view(B, -1)

        intersection = (pred_f * target_f).sum(dim=1)
        denom = pred_f.sum(dim=1) + target_f.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)
        target = (target > 0.5).float()

        pt = pred * target + (1.0 - pred) * (1.0 - target)
        w = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = -w * ((1.0 - pt) ** self.gamma) * torch.log(pt)
        return loss.mean()


class TVLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return dx.mean() + dy.mean()


def nuclear_norm_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError("Expected [B,N,D]")
    norms = []
    for b in range(x.size(0)):
        norms.append(torch.norm(x[b], p="nuc"))
    return torch.stack(norms).mean()


class RSDGLALoss(nn.Module):
    def __init__(
        self,
        lambda_rec: float = 0.5,
        lambda_rank: float = 0.01,
        lambda_sparse: float = 0.001,
        lambda_geo: float = 0.1,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ):
        super().__init__()
        self.lambda_rec = float(lambda_rec)
        self.lambda_rank = float(lambda_rank)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_geo = float(lambda_geo)
        self.lambda_dice = float(lambda_dice)
        self.lambda_focal = float(lambda_focal)

        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.tv = TVLoss()
        self.mse = nn.MSELoss()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        anomaly_map = outputs["anomaly_map"]
        feat_original = outputs["feat_original"]
        feat_normal = outputs["feat_normal"]
        feat_abnormal = outputs["feat_abnormal"]

        loss_dice = self.dice(anomaly_map, targets)
        loss_focal = self.focal(anomaly_map, targets)
        loss_task = loss_dice + loss_focal

        loss_rec = self.mse(feat_normal + feat_abnormal, feat_original)
        loss_rank = nuclear_norm_batch(feat_normal)
        loss_sparse = torch.mean(torch.abs(feat_abnormal))
        loss_geo = self.tv(anomaly_map)

        total = (
            self.lambda_dice * loss_dice
            + self.lambda_focal * loss_focal
            + self.lambda_rec * loss_rec
            + self.lambda_rank * loss_rank
            + self.lambda_sparse * loss_sparse
            + self.lambda_geo * loss_geo
        )

        loss_dict = {
            "loss_total": float(total.detach().cpu()),
            "loss_task": float(loss_task.detach().cpu()),
            "loss_dice": float(loss_dice.detach().cpu()),
            "loss_focal": float(loss_focal.detach().cpu()),
            "loss_rec": float(loss_rec.detach().cpu()),
            "loss_rank": float(loss_rank.detach().cpu()),
            "loss_sparse": float(loss_sparse.detach().cpu()),
            "loss_geo": float(loss_geo.detach().cpu()),
        }

        return total, loss_dict
