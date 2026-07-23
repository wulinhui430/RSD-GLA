from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.sinkhorn import SinkhornDistance

from src.openai_clip_text import CoOpAbnormalPrompt, CoOpAbnormalPromptConfig, OpenAIClipTextEncoder, load_openai_clip_state_dict


class TokenMLPAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(bottleneck, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RSDGLAVisualEncoder(nn.Module):
    def __init__(
        self,
        cache_dir: str = "./models",
        clip_model_name_or_path: str = "openai/clip-vit-large-patch14",
        local_files_only: bool = True,
        adapter_bottleneck: int = 256,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_max_iter: int = 50,
        proto_align_temperature: float = 0.07,
        enable_text_align: bool = False,
        openai_clip_ckpt_path: str = "./ViT-L-14-336px.pt",
        openai_bpe_path: str = "",
        abnormal_prompt_texts: Optional[Tuple[str, ...]] = None,
        prompt_length: int = 8,
        fusion_weight: float = 0.7,
    ):
        super().__init__()

        from transformers import CLIPVisionModel

        try:
            self.vision_model = CLIPVisionModel.from_pretrained(
                clip_model_name_or_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except Exception:
            self.vision_model = CLIPVisionModel.from_pretrained(
                clip_model_name_or_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                use_safetensors=False,
            )

        for p in self.vision_model.parameters():
            p.requires_grad = False

        self.feature_dim = int(self.vision_model.config.hidden_size)
        self.n_tokens = int((self.vision_model.config.image_size // self.vision_model.config.patch_size) ** 2)
        self.grid_size = int(self.vision_model.config.image_size // self.vision_model.config.patch_size)

        self.normal_adapter = TokenMLPAdapter(self.feature_dim, adapter_bottleneck)
        self.abnormal_adapter = TokenMLPAdapter(self.feature_dim, adapter_bottleneck)

        self.sinkhorn = SinkhornDistance(epsilon=sinkhorn_epsilon, max_iter=sinkhorn_max_iter, reduction="none")

        self.proto_align_temperature = float(proto_align_temperature)

        self.enable_text_align = bool(enable_text_align)
        self.fusion_weight = float(fusion_weight)

        self.text_encoder = None
        self.abnormal_prompt = None
        self.image_to_text_proj = None

        if self.enable_text_align:
            if abnormal_prompt_texts is None:
                abnormal_prompt_texts = ("abnormal mass",)

            if not openai_bpe_path:
                openai_bpe_path = None
            else:
                openai_bpe_path = str(openai_bpe_path)

            sd = load_openai_clip_state_dict(str(openai_clip_ckpt_path), map_location="cpu")
            self.text_encoder = OpenAIClipTextEncoder(sd)
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            cfg = CoOpAbnormalPromptConfig(n_ctx=int(prompt_length), class_token_position="end")
            self.abnormal_prompt = CoOpAbnormalPrompt(
                text_encoder=self.text_encoder,
                prompt_texts=list(abnormal_prompt_texts),
                cfg=cfg,
                bpe_path=openai_bpe_path,
            )

            text_dim = int(self.text_encoder.text_projection.shape[1])
            self.image_to_text_proj = nn.Linear(self.feature_dim, text_dim)

    def _extract_tokens(self, images: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values=images)
        tokens = out.last_hidden_state  # [B, 1+N, D]
        feat = tokens[:, 1:, :]  # [B, N, D]
        return feat

    def extract_tumor_prototypes(
        self,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        seed: int = 0,
    ) -> Optional[torch.Tensor]:
        feat_original = self._extract_tokens(support_images)
        feat_abnormal = self.abnormal_adapter(feat_original)

        protos = []
        mask_tokens = F.interpolate(support_masks, size=(self.grid_size, self.grid_size), mode="nearest")
        mask_tokens = mask_tokens.flatten(2).squeeze(1)  # [B, N]

        for b in range(feat_abnormal.size(0)):
            idx = mask_tokens[b] > 0.5
            if not idx.any():
                continue
            protos.append(feat_abnormal[b, idx, :].mean(dim=0))

        if not protos:
            return None

        proto = torch.stack(protos, dim=0)  # [K, D]
        return proto

    def forward(
        self,
        images: torch.Tensor,
        tumor_prototypes: Optional[torch.Tensor] = None,
    ) -> dict:
        feat_original = self._extract_tokens(images)  # [B,N,D]
        feat_normal = self.normal_adapter(feat_original)
        feat_abnormal = self.abnormal_adapter(feat_original)

        B = images.size(0)
        anomaly_map_proto = torch.zeros((B, 1, 224, 224), device=images.device, dtype=images.dtype)
        anomaly_map_text = torch.zeros((B, 1, 224, 224), device=images.device, dtype=images.dtype)
        anomaly_map = torch.zeros((B, 1, 224, 224), device=images.device, dtype=images.dtype)

        if tumor_prototypes is not None:
            if tumor_prototypes.dim() != 2 or tumor_prototypes.size(1) != self.feature_dim:
                raise ValueError(
                    f"tumor_prototypes must be [K,{self.feature_dim}] but got {tuple(tumor_prototypes.shape)}"
                )

            y = tumor_prototypes.unsqueeze(0).expand(B, -1, -1)  # [B,K,D]

            att_tokens = self.sinkhorn.get_attention_map(
                feat_abnormal,
                y,
                reshape_size=(self.grid_size, self.grid_size),
                temperature=self.proto_align_temperature,
            )  # [B,1,gs,gs]

            anomaly_map_proto = F.interpolate(att_tokens, size=(224, 224), mode="bilinear", align_corners=False)
            anomaly_map_proto = torch.clamp(anomaly_map_proto, 0.0, 1.0)

        if self.enable_text_align and self.text_encoder is not None and self.abnormal_prompt is not None:
            text_feat = self.abnormal_prompt().to(device=images.device, dtype=feat_abnormal.dtype)  # [Dt]
            if self.image_to_text_proj is None:
                raise RuntimeError("Text alignment is enabled but projection modules are missing")

            feat_text = self.image_to_text_proj(feat_abnormal)
            feat_text = F.normalize(feat_text, dim=-1)

            t = F.normalize(text_feat, dim=0)
            sim = feat_text @ t.view(-1, 1)
            sim = sim.squeeze(-1)
            cos_map = (sim + 1.0) * 0.5
            cos_map = cos_map.view(B, 1, self.grid_size, self.grid_size)
            anomaly_map_text = F.interpolate(cos_map, size=(224, 224), mode="bilinear", align_corners=False)
            anomaly_map_text = torch.clamp(anomaly_map_text, 0.0, 1.0)

        if self.enable_text_align:
            anomaly_map = self.fusion_weight * anomaly_map_proto + (1.0 - self.fusion_weight) * anomaly_map_text
        else:
            anomaly_map = anomaly_map_proto

        anomaly_map = torch.clamp(anomaly_map, 0.0, 1.0)

        return {
            "anomaly_map": anomaly_map,
            "anomaly_map_proto": anomaly_map_proto,
            "anomaly_map_text": anomaly_map_text,
            "feat_original": feat_original,
            "feat_normal": feat_normal,
            "feat_abnormal": feat_abnormal,
        }
