from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch


MODALITY_ORDER = ["imu", "gesture", "audio", "text", "scene"]

_SAMPLE_KEY_COLUMNS = (
    "sample_id",
    "cache_sample_id",
    "sample",
    "sample_dir",
    "cache_dir",
    "cache_path",
    "sample_path",
    "relative_path",
    "directory",
    "name",
    "path",
)
_MISSING_COLUMNS = (
    "missing_modalities",
    "missing_modality",
    "dropped_modalities",
    "dropped_modality",
    "modalities",
    "modality",
)
_TRUE_VALUES = {"1", "true", "yes", "y", "missing", "dropped", "drop"}


def infer_modality_mask(features: Mapping[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
    mask = []
    for modality in MODALITY_ORDER:
        value = features[modality]
        if isinstance(value, torch.Tensor):
            available = bool(torch.count_nonzero(value).item())
        else:
            available = bool(np.count_nonzero(np.asarray(value)))
        mask.append(float(available))
    return torch.tensor(mask, dtype=torch.float32)


def _split_modalities(value: object) -> set[str]:
    tokens = re.split(r"[\s,;|+/\[\]'\"()]+", str(value).strip().lower())
    return {token for token in tokens if token in MODALITY_ORDER}


def _row_sample_keys(row: Mapping[str, str]) -> set[str]:
    keys: set[str] = set()
    for column in _SAMPLE_KEY_COLUMNS:
        value = str(row.get(column, "")).strip()
        if value:
            path = Path(value)
            keys.update({value, path.name, path.stem})
    video_name = str(row.get("video_name", row.get("video", ""))).strip()
    segment = str(
        row.get("segment_index", row.get("segment_id", row.get("index", "")))
    ).strip()
    if video_name and segment:
        keys.add(f"{video_name}|{segment}")
        keys.add(f"{Path(video_name).stem}|{segment}")
    return keys


def load_missing_manifest(path: str | Path | None) -> Dict[str, torch.Tensor]:
    if path is None:
        return {}
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing-modality manifest does not exist: {manifest_path}")

    lookup: Dict[str, torch.Tensor] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        normalized_columns = [str(column).strip().lower() for column in columns]
        recognized_identity = any(column in normalized_columns for column in _SAMPLE_KEY_COLUMNS) or (
            any(column in normalized_columns for column in ("video_name", "video"))
            and any(column in normalized_columns for column in ("segment_index", "segment_id", "index"))
        )
        recognized_missing = any(column in normalized_columns for column in _MISSING_COLUMNS) or any(
            "missing" in column or "dropped" in column for column in normalized_columns
        ) or any(
            modality in normalized_columns
            or f"{modality}_missing" in normalized_columns
            or f"missing_{modality}" in normalized_columns
            for modality in MODALITY_ORDER
        )
        if not recognized_identity or not recognized_missing:
            warnings.warn(
                "Missing-modality manifest columns are not fully recognized. "
                f"Available columns: {normalized_columns}. "
                f"Expected an identity column from {_SAMPLE_KEY_COLUMNS} (or video + segment), "
                f"and missing modalities from {_MISSING_COLUMNS} or per-modality columns.",
                RuntimeWarning,
            )

        for row_number, raw_row in enumerate(reader, start=2):
            row = {
                str(key).strip().lower(): str(value).strip()
                for key, value in raw_row.items()
                if key is not None
            }
            sample_keys = _row_sample_keys(row)
            if not sample_keys:
                warnings.warn(
                    f"Skipping manifest row {row_number}: no recognizable sample identity; "
                    f"columns={normalized_columns}",
                    RuntimeWarning,
                )
                continue

            missing: set[str] = set()
            for column in _MISSING_COLUMNS:
                if row.get(column):
                    missing.update(_split_modalities(row[column]))
            for column, value in row.items():
                if "missing" in column or "dropped" in column:
                    missing.update(_split_modalities(value))
            for modality in MODALITY_ORDER:
                for column in (
                    modality,
                    f"{modality}_missing",
                    f"missing_{modality}",
                    f"is_{modality}_missing",
                    f"is_missing_{modality}",
                ):
                    if str(row.get(column, "")).strip().lower() in _TRUE_VALUES:
                        missing.add(modality)

            mask = torch.tensor(
                [0.0 if modality in missing else 1.0 for modality in MODALITY_ORDER],
                dtype=torch.float32,
            )
            for key in sample_keys:
                lookup[key] = mask
    return lookup


def manifest_mask_for_sample(
    manifest: Mapping[str, torch.Tensor],
    *,
    sample_id: str,
    video_name: str,
    segment_index: int,
) -> torch.Tensor | None:
    candidates = (
        sample_id,
        Path(sample_id).name,
        f"{video_name}|{segment_index}",
        f"{Path(video_name).stem}|{segment_index}",
    )
    for key in candidates:
        if key in manifest:
            return manifest[key].clone()
    return None


def build_sample_mask(
    features: Mapping[str, np.ndarray | torch.Tensor],
    manifest: Mapping[str, torch.Tensor],
    *,
    sample_id: str,
    video_name: str,
    segment_index: int,
) -> torch.Tensor:
    manifest_mask = manifest_mask_for_sample(
        manifest,
        sample_id=sample_id,
        video_name=video_name,
        segment_index=segment_index,
    )
    return manifest_mask if manifest_mask is not None else infer_modality_mask(features)


def apply_random_modality_dropout(
    batch: Dict[str, torch.Tensor],
    drop_one_prob: float,
    drop_two_prob: float,
) -> Dict[str, torch.Tensor]:
    if drop_one_prob < 0.0 or drop_two_prob < 0.0 or drop_one_prob + drop_two_prob > 1.0:
        raise ValueError("drop_one_prob and drop_two_prob must be non-negative and sum to at most 1")
    if "modality_mask" not in batch:
        raise KeyError("Random modality dropout requires batch['modality_mask']")

    result = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    mask = result["modality_mask"]
    for sample_index in range(mask.shape[0]):
        draw = float(torch.rand((), device=mask.device).item())
        drop_count = 1 if draw < drop_one_prob else 2 if draw < drop_one_prob + drop_two_prob else 0
        available = torch.nonzero(mask[sample_index] > 0, as_tuple=False).flatten()
        drop_count = min(drop_count, max(int(available.numel()) - 1, 0))
        if drop_count == 0:
            continue
        selected = available[torch.randperm(available.numel(), device=mask.device)[:drop_count]]
        for modality_index in selected.tolist():
            modality = MODALITY_ORDER[modality_index]
            result[modality][sample_index].zero_()
            mask[sample_index, modality_index] = 0.0
    return result


def masks_to_numpy(masks: Iterable[torch.Tensor]) -> np.ndarray:
    rows = [mask.detach().cpu().numpy().astype(np.float32) for mask in masks]
    return np.stack(rows, axis=0) if rows else np.empty((0, len(MODALITY_ORDER)), dtype=np.float32)
