from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornDistance(nn.Module):
    def __init__(
        self,
        epsilon: float = 0.05,
        max_iter: int = 50,
        reduction: str = "none",
        verbose: bool = False,
    ):
        super().__init__()
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.reduction = reduction
        self.verbose = bool(verbose)

        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")

    @staticmethod
    def _compute_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or y.dim() != 3:
            raise ValueError("x and y must be 3D tensors: [B,n,d] and [B,m,d]")
        if x.size(0) != y.size(0) or x.size(-1) != y.size(-1):
            raise ValueError("x and y must share batch size and feature dim")

        x2 = (x ** 2).sum(dim=-1, keepdim=True)  # [B,n,1]
        y2 = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # [B,1,m]
        xy = x @ y.transpose(1, 2)  # [B,n,m]
        C = x2 + y2 - 2.0 * xy
        return torch.clamp(C, min=0.0)

    def _sinkhorn_algorithm(
        self,
        C: torch.Tensor,
        n: int,
        m: int,
        device: torch.device,
    ) -> torch.Tensor:
        if C.dim() != 3:
            raise ValueError("C must be [B,n,m]")

        B = C.size(0)
        mu = torch.full((B, n), 1.0 / float(n), device=device, dtype=C.dtype)
        nu = torch.full((B, m), 1.0 / float(m), device=device, dtype=C.dtype)

        log_u = torch.zeros_like(mu)
        log_v = torch.zeros_like(nu)

        K_log = -C / self.epsilon

        for _ in range(self.max_iter):
            log_u = torch.log(mu + 1e-8) - torch.logsumexp(K_log + log_v.unsqueeze(1), dim=2)
            log_v = torch.log(nu + 1e-8) - torch.logsumexp(K_log + log_u.unsqueeze(2), dim=1)

        P = torch.exp(K_log + log_u.unsqueeze(2) + log_v.unsqueeze(1))
        return P

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        C = self._compute_cost_matrix(x, y)
        P = self._sinkhorn_algorithm(C, n=x.size(1), m=y.size(1), device=x.device)
        distance = (P * C).sum(dim=(1, 2))

        if self.reduction == "mean":
            distance = distance.mean()
        elif self.reduction == "sum":
            distance = distance.sum()

        return distance, P

    def get_attention_map(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        reshape_size: Tuple[int, int] = (16, 16),
        temperature: float = 0.07,
    ) -> torch.Tensor:
        x_n = F.normalize(x, dim=-1)
        y_n = F.normalize(y, dim=-1)

        C = self._compute_cost_matrix(x_n, y_n)
        P = self._sinkhorn_algorithm(C, n=x.size(1), m=y.size(1), device=x.device)

        P_row = P / (P.sum(dim=2, keepdim=True) + 1e-8)

        sim = x_n @ y_n.transpose(1, 2)
        attention_logits = (P_row * sim).sum(dim=2)

        att = F.softmax(attention_logits / float(temperature), dim=1)

        h, w = reshape_size
        if att.size(1) != h * w:
            raise ValueError(f"Token count {att.size(1)} does not match reshape_size {reshape_size}")

        att = att.view(att.size(0), 1, h, w)

        att_min = att.amin(dim=(2, 3), keepdim=True)
        att_max = att.amax(dim=(2, 3), keepdim=True)
        att = (att - att_min) / (att_max - att_min + 1e-8)

        return att
