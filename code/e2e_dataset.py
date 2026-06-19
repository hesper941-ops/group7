from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from torch.utils.data import Dataset

from e2e_config import (
    INTENT_NAMES,
    SCENE_BY_VIDEO,
    SCENE_NAME_TO_ID,
    TEST_VIDEO_NAMES,
    TRAIN_VIDEO_NAMES,
    VIDEO_LABELS,
    E2EConfig,
)
from e2e_feature_pipeline import E2EFeaturePipeline, sample_cache_key, stack_feature_dicts


@dataclass(frozen=True)
class E2ESample:
    sample_id: str
    video_name: str
    scene: str
    segment_index: int
    raw_paths: Dict[str, str]
    intent_label: int
    scene_label: int
    joint_label: str


def make_joint_label(scene: str, intent_label: int) -> str:
    return f"{scene}_{INTENT_NAMES[int(intent_label)]}"


def get_split_video_names(split: str) -> List[str]:
    if split in {"train", "val"}:
        return list(TRAIN_VIDEO_NAMES)
    if split == "test":
        return list(TEST_VIDEO_NAMES)
    raise ValueError(f"Unknown split: {split}")


def build_video_samples(config: E2EConfig, split: str, pipeline: E2EFeaturePipeline) -> List[E2ESample]:
    samples: List[E2ESample] = []
    for video_name in get_split_video_names(split):
        pipeline.extract_video_features(video_name)
        meta = pipeline.get_video_metadata(video_name)
        timestamps = meta["approx_timestamps"]
        labels = meta["labels"]
        raw_paths = {
            key: str(value)
            for key, value in pipeline.resolve_raw_paths(video_name).items()
            if value is not None
        }
        scene = SCENE_BY_VIDEO[video_name]
        for index, timestamp in enumerate(timestamps.tolist()):
            intent_label = int(labels[index]) if index < len(labels) else int(VIDEO_LABELS[video_name])
            samples.append(
                E2ESample(
                    sample_id=sample_cache_key(video_name, index, timestamp),
                    video_name=video_name,
                    scene=scene,
                    segment_index=index,
                    raw_paths=raw_paths,
                    intent_label=intent_label,
                    scene_label=SCENE_NAME_TO_ID[scene],
                    joint_label=make_joint_label(scene, intent_label),
                )
            )
    return samples


class E2EMultimodalDataset(Dataset):
    def __init__(self, config: E2EConfig, split: str, pipeline: E2EFeaturePipeline | None = None):
        self.config = config
        self.split = split
        self.pipeline = pipeline or E2EFeaturePipeline(config)
        self.samples = build_video_samples(config, split, self.pipeline)
        if not self.samples:
            raise RuntimeError(f"No samples were built for split={split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        payload = {
            "sample_id": sample.sample_id,
            "video_name": sample.video_name,
            "scene": sample.scene,
            "segment_index": sample.segment_index,
            "raw_paths": sample.raw_paths,
            "joint_label": sample.joint_label,
            "intent_label": sample.intent_label,
            "scene_label": sample.scene_label,
        }
        payload["features"] = self.pipeline.extract_sample_features(payload)
        return payload


def dataset_to_arrays(dataset: E2EMultimodalDataset) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    feature_rows = []
    intent_labels = []
    scene_labels = []
    joint_labels = []
    for index in range(len(dataset)):
        item = dataset[index]
        feature_rows.append(item["features"])
        intent_labels.append(item["intent_label"])
        scene_labels.append(item["scene_label"])
        joint_labels.append(item["joint_label"])
    return (
        stack_feature_dicts(feature_rows),
        np.asarray(intent_labels, dtype=np.int64),
        np.asarray(scene_labels, dtype=np.int64),
        np.asarray(joint_labels, dtype=object),
    )


def describe_samples(samples: Iterable[E2ESample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.joint_label] = counts.get(sample.joint_label, 0) + 1
    return dict(sorted(counts.items()))
