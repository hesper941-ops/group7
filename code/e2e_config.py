from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


MODALITY_KEYS: Tuple[str, ...] = ("imu", "gesture", "audio", "text", "scene")
FEATURE_DIMS: Dict[str, int] = {
    "imu": 12,
    "gesture": 768,
    "audio": 39,
    "text": 384,
    "scene": 768,
}
TARGET_TIMESTEPS = 10

INTENT_NAMES = {
    0: "menu",
    1: "select",
    2: "magnify",
    3: "narrow",
    4: "brush",
    5: "cancel",
}
SCENE_NAME_TO_ID = {"office": 0, "museum": 1}
SCENE_ID_TO_NAME = {value: key for key, value in SCENE_NAME_TO_ID.items()}

VIDEO_LABELS = {
    "interaction_20260306_072344.mp4": 0,
    "interaction_20260227_122606.mp4": 1,
    "interaction_20260227_122952.mp4": 2,
    "interaction_20260227_123354.mp4": 3,
    "interaction_20260227_124559.mp4": 4,
    "interaction_20260227_123745.mp4": 5,
    "interaction_20260131_120024.mp4": 0,
    "interaction_20260227_132951.mp4": 1,
    "interaction_20260227_133408.mp4": 2,
    "interaction_20260131_114156.mp4": 3,
    "interaction_20260131_115150.mp4": 4,
    "interaction_20260131_114852.mp4": 5,
    "interaction_20260301_073041.mp4": 0,
    "interaction_20260301_064753.mp4": 1,
    "interaction_20260306_072721.mp4": 2,
    "interaction_20260301_071948.mp4": 3,
    "interaction_20260131_121548.mp4": 3,
    "interaction_20260301_073435.mp4": 4,
    "interaction_20260301_072503.mp4": 5,
    "interaction_20260131_071552.mp4": 0,
    "interaction_20260131_072412.mp4": 1,
    "interaction_20260131_084300.mp4": 1,
    "interaction_20260131_085611.mp4": 2,
    "interaction_20260131_090139.mp4": 3,
    "interaction_20260131_085207.mp4": 4,
    "interaction_20260131_084732.mp4": 5,
    "interaction_20260131_090917.mp4": 0,
    "interaction_20260131_090541.mp4": 1,
    "interaction_20260131_065459.mp4": 2,
    "interaction_20260131_070722.mp4": 3,
    "interaction_20260131_091657.mp4": 4,
    "interaction_20260131_091249.mp4": 5,
    "interaction_20260306_082346.mp4": 2,
    "interaction_20260306_083107.mp4": 3,
    "interaction_20260306_083434.mp4": 1,
    "interaction_20260306_084406.mp4": 0,
    "interaction_20260306_084853.mp4": 5,
    "interaction_20260306_085830.mp4": 4,
    "interaction_20260306_090441.mp4": 1,
}

TEST_VIDEO_NAMES = [
    "interaction_20260306_072344.mp4",
    "interaction_20260227_122606.mp4",
    "interaction_20260227_122952.mp4",
    "interaction_20260227_123354.mp4",
    "interaction_20260227_124559.mp4",
    "interaction_20260227_123745.mp4",
    "interaction_20260306_082346.mp4",
    "interaction_20260306_083107.mp4",
    "interaction_20260306_083434.mp4",
    "interaction_20260306_084406.mp4",
    "interaction_20260306_084853.mp4",
    "interaction_20260306_085830.mp4",
    "interaction_20260306_090441.mp4",
]
TRAIN_VIDEO_NAMES = [name for name in VIDEO_LABELS if name not in TEST_VIDEO_NAMES]

OFFICE_VIDEO_NAMES = {
    "interaction_20260306_072344.mp4",
    "interaction_20260227_122606.mp4",
    "interaction_20260227_122952.mp4",
    "interaction_20260227_123354.mp4",
    "interaction_20260227_124559.mp4",
    "interaction_20260227_123745.mp4",
    "interaction_20260131_120024.mp4",
    "interaction_20260227_132951.mp4",
    "interaction_20260227_133408.mp4",
    "interaction_20260131_114156.mp4",
    "interaction_20260131_115150.mp4",
    "interaction_20260131_114852.mp4",
    "interaction_20260301_073041.mp4",
    "interaction_20260301_064753.mp4",
    "interaction_20260306_072721.mp4",
    "interaction_20260301_071948.mp4",
    "interaction_20260131_121548.mp4",
    "interaction_20260301_073435.mp4",
    "interaction_20260301_072503.mp4",
}
SCENE_BY_VIDEO = {name: "office" for name in OFFICE_VIDEO_NAMES}
SCENE_BY_VIDEO.update({name: "museum" for name in set(VIDEO_LABELS) - OFFICE_VIDEO_NAMES})
ALL_JOINT_CLASS_NAMES = [
    f"{scene}_{INTENT_NAMES[intent]}"
    for scene in ("office", "museum")
    for intent in sorted(INTENT_NAMES)
]


@dataclass(frozen=True)
class E2EConfig:
    project_root: Path
    data_root: Path
    output_dir: Path
    cache_dir: Path
    model: str = "baseline"
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    val_split: float = 0.2
    patience: int = 10
    model_dim: int = 128
    num_latents: int = 32
    depth: int = 2
    num_heads: int = 4
    dropout: float = 0.1

    @property
    def processed_data_dir(self) -> Path:
        direct = self.data_root / "data"
        return direct if direct.exists() else self.data_root

    @property
    def hololens_dir(self) -> Path:
        return self.data_root.parent / "HoloLens"

    @property
    def fisheye_dir(self) -> Path:
        return self.data_root.parent / "fisheye"

    @property
    def imu_csv_path(self) -> Path:
        return self.data_root.parent / "imu.csv"

    @property
    def clip_model_path(self) -> Path:
        return self.data_root / "models" / "clip_teacher_model"

    @property
    def sentence_model_path(self) -> Path:
        return self.data_root / "models" / "all-MiniLM-L6-v2"


def default_project_root() -> Path:
    return Path(os.getenv("SMART_AR_ROOT", Path(__file__).resolve().parents[1]))


def resolve_path(value: str | None, env_name: str, default: Path) -> Path:
    raw_value = value or os.getenv(env_name)
    return Path(raw_value).expanduser().resolve() if raw_value else default.resolve()


def build_config(
    *,
    data_root: str | None = None,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    model: str = "baseline",
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> E2EConfig:
    project_root = resolve_path(None, "SMART_AR_ROOT", default_project_root())
    default_data = project_root / "dataset" / "AR_Data_process3.0"
    default_output = project_root / "outputs" / "e2e" / model
    default_cache = project_root / "outputs" / "e2e" / "cache"
    return E2EConfig(
        project_root=project_root,
        data_root=resolve_path(data_root, "SMART_AR_DATA_ROOT", default_data),
        output_dir=resolve_path(output_dir, "SMART_AR_OUTPUT_DIR", default_output),
        cache_dir=resolve_path(cache_dir, "SMART_AR_CACHE_DIR", default_cache),
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
