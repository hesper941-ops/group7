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
UNKNOWN_LABELS = {"未知", "unknown", "Unknown"}

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

# User mapping from README:
#   A = Luo, B = Gu, C = Bian
# The required split is users A+B for training and user C for testing.
LUO_VIDEO_NAMES = {
    "interaction_20260131_120024.mp4",
    "interaction_20260227_132951.mp4",
    "interaction_20260227_133408.mp4",
    "interaction_20260131_114156.mp4",
    "interaction_20260131_115150.mp4",
    "interaction_20260131_114852.mp4",
    "interaction_20260131_071552.mp4",
    "interaction_20260131_072412.mp4",
    "interaction_20260131_084300.mp4",
    "interaction_20260131_084732.mp4",
    "interaction_20260131_085207.mp4",
    "interaction_20260131_085611.mp4",
    "interaction_20260131_090139.mp4",
}
GU_VIDEO_NAMES = {
    "interaction_20260301_073041.mp4",
    "interaction_20260301_064753.mp4",
    "interaction_20260306_072721.mp4",
    "interaction_20260301_071948.mp4",
    "interaction_20260131_121548.mp4",
    "interaction_20260301_073435.mp4",
    "interaction_20260301_072503.mp4",
    "interaction_20260131_065459.mp4",
    "interaction_20260131_070722.mp4",
    "interaction_20260131_090541.mp4",
    "interaction_20260131_090917.mp4",
    "interaction_20260131_091249.mp4",
    "interaction_20260131_091657.mp4",
}
BIAN_VIDEO_NAMES = set(TEST_VIDEO_NAMES)
USER_BY_VIDEO = {name: "A" for name in LUO_VIDEO_NAMES}
USER_BY_VIDEO.update({name: "B" for name in GU_VIDEO_NAMES})
USER_BY_VIDEO.update({name: "C" for name in BIAN_VIDEO_NAMES})
TRAIN_USERS = ("A", "B")
TEST_USERS = ("C",)

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
FISHEYE_AVI_BY_VIDEO = {
    "interaction_20260306_072344.mp4": "Video_20260306_152340690.avi",
    "interaction_20260227_122606.mp4": "Video_20260227_202553335.avi",
    "interaction_20260227_122952.mp4": "Video_20260227_202953348.avi",
    "interaction_20260227_123354.mp4": "Video_20260227_203348219.avi",
    "interaction_20260227_124559.mp4": "Video_20260227_204553897.avi",
    "interaction_20260227_123745.mp4": "Video_20260227_203753817.avi",
    "interaction_20260131_120024.mp4": "Video_20260131_200029359.avi",
    "interaction_20260227_132951.mp4": "Video_20260227_213001434.avi",
    "interaction_20260227_133408.mp4": "Video_20260227_213404452.avi",
    "interaction_20260131_114156.mp4": "Video_20260131_194205407.avi",
    "interaction_20260131_115150.mp4": "Video_20260131_195202906.avi",
    "interaction_20260131_114852.mp4": "Video_20260131_194854095.avi",
    "interaction_20260301_073041.mp4": "Video_20260301_153037623.avi",
    "interaction_20260301_064753.mp4": "Video_20260301_144803454.avi",
    "interaction_20260306_072721.mp4": "Video_20260306_152721366.avi",
    "interaction_20260301_071948.mp4": "Video_20260301_151942635.avi",
    "interaction_20260131_121548.mp4": "Video_20260131_201556629.avi",
    "interaction_20260301_073435.mp4": "Video_20260301_153434856.avi",
    "interaction_20260301_072503.mp4": "Video_20260301_152459131.avi",
    "interaction_20260131_071552.mp4": "Video_20260131_151559270.avi",
    "interaction_20260131_072412.mp4": "Video_20260131_152410916.avi",
    "interaction_20260131_084300.mp4": "Video_20260131_164304016.avi",
    "interaction_20260131_084732.mp4": "Video_20260131_164745532.avi",
    "interaction_20260131_085207.mp4": "Video_20260131_165208524.avi",
    "interaction_20260131_085611.mp4": "Video_20260131_165614756.avi",
    "interaction_20260131_090139.mp4": "Video_20260131_170142792.avi",
    "interaction_20260131_065459.mp4": "Video_20260131_145524524.avi",
    "interaction_20260131_070722.mp4": "Video_20260131_150734369.avi",
    "interaction_20260131_090541.mp4": "Video_20260131_170539636.avi",
    "interaction_20260131_090917.mp4": "Video_20260131_170919896.avi",
    "interaction_20260131_091249.mp4": "Video_20260131_171253889.avi",
    "interaction_20260131_091657.mp4": "Video_20260131_171648040.avi",
    "interaction_20260306_082346.mp4": "Video_20260306_162401599.avi",
    "interaction_20260306_083107.mp4": "Video_20260306_163105571.avi",
    "interaction_20260306_083434.mp4": "Video_20260306_163434878.avi",
    "interaction_20260306_084406.mp4": "Video_20260306_164407883.avi",
    "interaction_20260306_084853.mp4": "Video_20260306_164902044.avi",
    "interaction_20260306_085830.mp4": "Video_20260306_165839689.avi",
    "interaction_20260306_090441.mp4": "Video_20260306_170449073.avi",
}
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
    learning_rate: float = 5e-4
    weight_decay: float = 3e-4
    seed: int = 42
    val_split: float = 0.2
    patience: int = 4
    model_dim: int = 128
    num_latents: int = 32
    depth: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    intent_aux_weight: float = 0.35
    scene_aux_weight: float = 0.15
    gesture_intent_aux_weight: float = 0.25
    base_intent_aux_weight: float = 0.10
    selection_intent_weight: float = 0.35
    selection_scene_weight: float = 0.05
    label_smoothing: float = 0.03
    grad_clip_norm: float = 1.0
    no_early_stop: bool = False
    min_gate: float = 0.02
    imu_drop_prob: float = 0.35
    audio_drop_prob: float = 0.20
    imu_max_scale: float = 0.15
    audio_max_scale: float = 0.10
    intent_refine_scale: float = 0.35
    gesture_logit_blend: float = 0.30

    @property
    def ar_data_process_dir(self) -> Path:
        if self.data_root.name.lower() == "data" and self.data_root.parent.name.lower() in {
            "ar_data_process3.0",
            "ar_data_process3.0",
        }:
            return self.data_root.parent
        if self.data_root.name.lower() in {"ar_data_process3.0", "ar_data_process3.0"}:
            return self.data_root
        candidate = self.data_root / "AR_Data_process3.0"
        if candidate.exists():
            return candidate
        candidate = self.data_root / "AR_Data_Process3.0"
        if candidate.exists():
            return candidate
        return self.data_root / "AR_Data_process3.0"

    @property
    def processed_data_dir(self) -> Path:
        if self.data_root.name.lower() == "data":
            return self.data_root
        return self.ar_data_process_dir / "data"

    @property
    def dataset_root(self) -> Path:
        if self.data_root.name.lower() == "data":
            return self.data_root.parent.parent
        if self.data_root.name.lower() in {"ar_data_process3.0", "ar_data_process3.0"}:
            return self.data_root.parent
        if (self.data_root / "HoloLens").exists() or (self.data_root / "fisheye").exists():
            return self.data_root
        return self.ar_data_process_dir.parent

    @property
    def hololens_dir(self) -> Path:
        return self.dataset_root / "HoloLens"

    @property
    def fisheye_dir(self) -> Path:
        override = os.getenv("SMART_AR_FISHEYE_DIR") or os.getenv("REAL_SCENE_VIDEO_DIR")
        return Path(override).expanduser().resolve() if override else self.dataset_root / "fisheye"

    @property
    def legacy_scene_cache_dir(self) -> Path:
        override = os.getenv("SMART_AR_LEGACY_SCENE_CACHE_DIR")
        if override:
            return Path(override).expanduser().resolve()
        return Path("/share/home/tm1078571822880000/a904903640/group7/dataset/scene_cache_real_vit")

    @property
    def imu_csv_path(self) -> Path:
        return self.dataset_root / "imu.csv"

    @property
    def clip_model_path(self) -> Path:
        return self.ar_data_process_dir / "models" / "clip_teacher_model"

    @property
    def sentence_model_path(self) -> Path:
        return self.ar_data_process_dir / "models" / "all-MiniLM-L6-v2"

    @property
    def whisper_cache_dir(self) -> Path:
        return self.cache_dir / "whisper"

    def describe_data_layout(self) -> Dict[str, str]:
        return {
            "data_root": str(self.data_root),
            "dataset_root": str(self.dataset_root),
            "ar_data_process_dir": str(self.ar_data_process_dir),
            "processed_data_dir": str(self.processed_data_dir),
            "hololens_dir": str(self.hololens_dir),
            "fisheye_dir": str(self.fisheye_dir),
            "legacy_scene_cache_dir": str(self.legacy_scene_cache_dir),
            "imu_csv_path": str(self.imu_csv_path),
            "clip_model_path": str(self.clip_model_path),
            "sentence_model_path": str(self.sentence_model_path),
        }


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
    learning_rate: float = 5e-4,
    weight_decay: float = 3e-4,
    seed: int = 42,
    patience: int = 4,
    no_early_stop: bool = False,
    min_gate: float = 0.02,
    imu_drop_prob: float = 0.35,
    audio_drop_prob: float = 0.20,
    imu_max_scale: float = 0.15,
    audio_max_scale: float = 0.10,
    intent_refine_scale: float = 0.35,
    gesture_logit_blend: float = 0.30,
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
        patience=patience,
        no_early_stop=no_early_stop,
        min_gate=min_gate,
        imu_drop_prob=imu_drop_prob,
        audio_drop_prob=audio_drop_prob,
        imu_max_scale=imu_max_scale,
        audio_max_scale=audio_max_scale,
        intent_refine_scale=intent_refine_scale,
        gesture_logit_blend=gesture_logit_blend,
    )
