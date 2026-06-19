from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from e2e_config import E2EConfig, FEATURE_DIMS, MODALITY_KEYS, TARGET_TIMESTEPS
from e2e_utils import ensure_dir


class FeatureExtractionError(RuntimeError):
    pass


def normalize_sequence_length(
    sequence: np.ndarray,
    target_steps: int,
    feat_dim: int,
    long_mode: str = "truncate",
) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D sequence, got shape {array.shape}")
    if array.shape[1] != feat_dim:
        raise ValueError(f"Expected feature dim {feat_dim}, got shape {array.shape}")
    if array.shape[0] == target_steps:
        return array
    if array.shape[0] > target_steps:
        if long_mode == "even":
            indices = np.linspace(0, array.shape[0] - 1, target_steps, dtype=int)
            return array[indices]
        return array[:target_steps]
    pad = np.zeros((target_steps - array.shape[0], feat_dim), dtype=np.float32)
    return np.vstack((array, pad))


def normalize_audio_payload(audio_payload: np.ndarray) -> np.ndarray:
    rows = []
    for item in audio_payload:
        feature = item["feature"] if isinstance(item, dict) else item
        rows.append(normalize_sequence_length(feature, TARGET_TIMESTEPS, FEATURE_DIMS["audio"], "even"))
    return np.stack(rows).astype(np.float32)


def normalize_dense_payload(payload: np.ndarray, modality: str) -> np.ndarray:
    return np.stack(
        [
            normalize_sequence_length(row, TARGET_TIMESTEPS, FEATURE_DIMS[modality], "truncate")
            for row in np.asarray(payload)
        ]
    ).astype(np.float32)


class E2EFeaturePipeline:
    def __init__(self, config: E2EConfig):
        self.config = config
        ensure_dir(config.cache_dir)
        self._video_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._meta_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._scene_cache = None

    def extract_sample_features(self, sample: dict) -> dict:
        sample_id = sample["sample_id"]
        cache_dir = ensure_dir(self.config.cache_dir / sample_id)
        cached = self._load_sample_cache(cache_dir)
        if cached is not None:
            return cached

        video_features = self.extract_video_features(sample["video_name"])
        index = int(sample["segment_index"])
        features = {key: np.asarray(video_features[key][index], dtype=np.float32) for key in MODALITY_KEYS}
        for key, value in features.items():
            path = cache_dir / f"{key}.npy"
            print(f"[cache] writing {key} feature for {sample_id}: {path}")
            np.save(path, value)
        return features

    def extract_video_features(self, video_name: str) -> Dict[str, np.ndarray]:
        if video_name not in self._video_cache:
            self._video_cache[video_name] = self._build_video_features(video_name)
        return self._video_cache[video_name]

    def get_video_metadata(self, video_name: str) -> Dict[str, np.ndarray]:
        if video_name not in self._meta_cache:
            self.extract_video_features(video_name)
        return self._meta_cache[video_name]

    def _load_sample_cache(self, cache_dir: Path) -> Dict[str, np.ndarray] | None:
        paths = {key: cache_dir / f"{key}.npy" for key in MODALITY_KEYS}
        if not all(path.exists() for path in paths.values()):
            return None
        print(f"[cache] using existing features for {cache_dir.name}")
        return {key: np.load(path).astype(np.float32) for key, path in paths.items()}

    def _build_video_features(self, video_name: str) -> Dict[str, np.ndarray]:
        print(f"[extract] building video-level features for {video_name}")
        self._ensure_video_level_feature_files(video_name)
        paths = self.get_feature_paths(video_name)
        missing = [f"{key}: {path}" for key, path in paths.items() if key != "scene" and not path.exists()]
        if missing:
            raise FeatureExtractionError(
                "Missing extracted modality files after automatic preparation:\n"
                + "\n".join(missing)
                + "\nRun this command on the server so raw extraction can access videos, models, and dependencies:\n"
                + "python code/train.py --model baseline --data-root <server dataset>/AR_Data_Process3.0"
            )

        gesture_data = np.load(paths["gesture"], allow_pickle=True).item()
        imu_data = np.load(paths["imu"], allow_pickle=True).item()
        audio_data = np.load(paths["audio"], allow_pickle=True)
        text_data = np.load(paths["text"], allow_pickle=True).item()

        lengths = [
            len(gesture_data["labels"]),
            len(gesture_data["features"]),
            len(gesture_data["approx_timestamps"]),
            len(imu_data["features"]),
            len(audio_data),
            len(text_data["features"]),
        ]
        count = min(lengths)
        if count <= 0:
            raise FeatureExtractionError(f"No aligned samples are available for {video_name}")

        labels = np.asarray(gesture_data["labels"][:count], dtype=np.int64)
        timestamps = np.asarray(gesture_data["approx_timestamps"][:count], dtype=object)
        features = {
            "imu": normalize_dense_payload(np.asarray(imu_data["features"][:count]), "imu"),
            "gesture": normalize_dense_payload(np.asarray(gesture_data["features"][:count]), "gesture"),
            "audio": normalize_audio_payload(audio_data[:count]),
            "text": normalize_dense_payload(np.asarray(text_data["features"][:count]), "text"),
            "scene": self._load_scene_features(video_name, timestamps),
        }
        self._meta_cache[video_name] = {
            "labels": labels,
            "approx_timestamps": timestamps,
            "count": np.asarray(count, dtype=np.int64),
        }
        return features

    def _ensure_video_level_feature_files(self, video_name: str) -> None:
        paths = self.get_feature_paths(video_name)
        if all(path.exists() for key, path in paths.items() if key != "scene"):
            return

        missing_raw = self.missing_raw_paths(video_name)
        if missing_raw:
            details = "\n".join(f"{key}: {path}" for key, path in missing_raw.items())
            raise FeatureExtractionError(f"Cannot extract {video_name}; missing raw files:\n{details}")

        missing_features = "\n".join(
            f"{key}: {path}" for key, path in paths.items() if key != "scene" and not path.exists()
        )
        raise FeatureExtractionError(
            f"Automatic raw extraction is required for {video_name}, but the server-only "
            "CLIP/Whisper/MoviePy/IMU pipeline is not available in this local environment.\n"
            f"Missing modality feature files:\n{missing_features}"
        )

    def missing_raw_paths(self, video_name: str) -> Dict[str, Path]:
        raw_paths = self.resolve_raw_paths(video_name)
        return {key: path for key, path in raw_paths.items() if path is not None and not path.exists()}

    def resolve_raw_paths(self, video_name: str) -> Dict[str, Path | None]:
        return {
            "hololens_video": self.config.hololens_dir / video_name,
            "fisheye_video": self._resolve_fisheye_path(video_name),
            "imu_csv": self.config.imu_csv_path,
        }

    def get_feature_paths(self, video_name: str) -> Dict[str, Path]:
        stem = Path(video_name).stem
        data_dir = self.config.processed_data_dir
        return {
            "gesture": data_dir / "strong_gesture_features" / f"strong_gesture_features_{stem}.npy",
            "imu": data_dir / "imu_features" / f"imu_features_{stem}.npy",
            "audio": data_dir / "audio_features" / f"audio_features_{stem}.npy",
            "text": data_dir / "text_features" / f"text_features_{stem}.npy",
            "scene": self.config.cache_dir / stem / "scene.npy",
        }

    def _load_scene_features(self, video_name: str, timestamps: np.ndarray) -> np.ndarray:
        try:
            import real_scene_utils as scene_utils
        except Exception as exc:
            raise FeatureExtractionError(f"Cannot import real_scene_utils for scene extraction: {exc}") from exc

        scene_utils.DATASET_VIDEO_DIR = self.config.fisheye_dir
        scene_utils.REAL_SCENE_CACHE_DIR = self.config.cache_dir / "real_scene_vit"
        if (self.config.project_root / "ViTModel").exists():
            scene_utils.LOCAL_VIT_PATH = self.config.project_root / "ViTModel"
        if self._scene_cache is None:
            self._scene_cache = scene_utils.RealSceneFeatureCache(scene_utils.REAL_SCENE_CACHE_DIR)
        print(f"[extract] building scene feature for {video_name}")
        scene_features, _ = scene_utils.load_real_scene_features(video_name, timestamps, self._scene_cache)
        return scene_features.astype(np.float32)

    def _resolve_fisheye_path(self, video_name: str) -> Path | None:
        try:
            import real_scene_utils as scene_utils
        except Exception:
            return None
        avi_name = scene_utils.MP4_TO_AVI_MAP.get(video_name)
        return self.config.fisheye_dir / avi_name if avi_name else None


def sample_cache_key(video_name: str, segment_index: int, timestamp_value: object) -> str:
    raw = f"{video_name}|{segment_index}|{timestamp_value}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"{Path(video_name).stem}_{segment_index:04d}_{digest}"


def stack_feature_dicts(items: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    rows = list(items)
    if not rows:
        raise ValueError("No feature rows to stack")
    return {key: np.stack([row[key] for row in rows]).astype(np.float32) for key in MODALITY_KEYS}
