import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSamplePaths:
    image_path: str
    mask_path: str


class ThymomaDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str,
        base_dir: str,
        use_clip: bool = True,
        transform=None,
        cache_dir: str = "./models",
        clip_model_name_or_path: str = "openai/clip-vit-large-patch14",
        local_files_only: bool = True,
        strict_paths: bool = True,
    ):
        self.df = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.split = split
        self.base_dir = base_dir
        self.use_clip = use_clip
        self.transform = transform
        self.cache_dir = cache_dir
        self.clip_model_name_or_path = clip_model_name_or_path
        self.local_files_only = local_files_only
        self.strict_paths = strict_paths

        self._clip_processor = None

        if not self.use_clip and self.transform is None:
            from torchvision import transforms

            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        if self.transform is None and not self.use_clip:
            raise ValueError("transform must be provided when use_clip=False")

        from torchvision import transforms

        self._mask_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def _get_clip_processor(self):
        if self._clip_processor is not None:
            return self._clip_processor
        from transformers import CLIPImageProcessor

        try:
            self._clip_processor = CLIPImageProcessor.from_pretrained(
                self.clip_model_name_or_path,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            raise OSError(
                "Failed to load CLIPImageProcessor locally. "
                "If you're offline, make sure the model files exist under cache_dir, "
                "or pass clip_model_name_or_path to a local directory. "
                f"clip_model_name_or_path={self.clip_model_name_or_path!r} cache_dir={self.cache_dir!r} "
                f"local_files_only={self.local_files_only}. Original error: {e}"
            ) from e
        return self._clip_processor

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_paths(self, idx: int) -> DatasetSamplePaths:
        row = self.df.iloc[idx]
        image_rel = str(row["image_path"])
        mask_rel_raw = row.get("mask_path", "")
        mask_rel = "" if pd.isna(mask_rel_raw) else str(mask_rel_raw)
        if mask_rel.lower() in ("", "nan", "none"):
            mask_rel = ""
        image_path = os.path.join(self.base_dir, image_rel)
        mask_path = os.path.join(self.base_dir, mask_rel) if mask_rel else ""
        return DatasetSamplePaths(image_path=image_path, mask_path=mask_path)

    def _check_paths(self, paths: DatasetSamplePaths, label: int) -> None:
        missing = []
        if not os.path.exists(paths.image_path):
            missing.append(paths.image_path)
        if int(label) == 1:
            if not paths.mask_path or not os.path.exists(paths.mask_path):
                missing.append(paths.mask_path or "<empty mask_path>")
        if missing and self.strict_paths:
            raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        label = int(row["label"])
        paths = self._resolve_paths(idx)
        self._check_paths(paths, label=label)

        image = Image.open(paths.image_path).convert("RGB")
        if int(label) == 1 and paths.mask_path:
            mask = Image.open(paths.mask_path).convert("L")
            mask_tensor = self._mask_transform(mask)
        else:
            mask_tensor = torch.zeros((1, 224, 224), dtype=torch.float32)

        if self.use_clip:
            processed = self._get_clip_processor()(images=image, return_tensors="pt")
            image_tensor = processed["pixel_values"].squeeze(0)
        else:
            image_tensor = self.transform(image)

        mask_tensor = (mask_tensor > 0.5).float()

        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, mask_tensor, label_tensor
