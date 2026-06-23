# 截取视频中的真实场景，为训练模型提供场景模态，并缓存场景编码后的特征

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

os.environ.setdefault("HF_HOME", r"/share/home/tm1078571822880000/a904903640/group7/.hf_cache")
os.environ.setdefault("HF_HUB_CACHE", r"/share/home/tm1078571822880000/a904903640/group7/.hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", r"/share/home/tm1078571822880000/a904903640/group7/.hf_cache/transformers")
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

ROOT_DIR = Path(r"/share/home/tm1078571822880000/a904903640/group7/")
DATASET_VIDEO_DIR = Path(r"/share/home/tm1078571822880000/a944494510/课程项目/dataset/fisheye")
LOCAL_VIT_PATH = ROOT_DIR / "ViTModel"
REAL_SCENE_CACHE_DIR = ROOT_DIR / "dataset" / "scene_cache_real_vit"
SCENE_FEAT_DIM = 768


def _env_path(*names: str, default: Path) -> Path:
    for name in names:
        value = os.getenv(name)
        if value:
            return Path(value).expanduser().resolve()
    return default


DATASET_VIDEO_DIR = _env_path(
    "SMART_AR_FISHEYE_DIR",
    "REAL_SCENE_VIDEO_DIR",
    default=Path("/share/home/tm1078571822880000/a944494510/课程项目/dataset/fisheye"),
)
REAL_SCENE_CACHE_DIR = _env_path(
    "SMART_AR_REAL_SCENE_CACHE_DIR",
    "REAL_SCENE_CACHE_DIR",
    default=ROOT_DIR / "dataset" / "scene_cache_real_vit",
)
LEGACY_REAL_SCENE_CACHE_DIR = _env_path(
    "SMART_AR_LEGACY_SCENE_CACHE_DIR",
    default=ROOT_DIR / "dataset" / "scene_cache_real_vit",
)

AVI_TO_MP4_MAP = {
    "Video_20260306_152340690.avi": "interaction_20260306_072344.mp4",
    "Video_20260227_202553335.avi": "interaction_20260227_122606.mp4",
    "Video_20260227_202953348.avi": "interaction_20260227_122952.mp4",
    "Video_20260227_203348219.avi": "interaction_20260227_123354.mp4",
    "Video_20260227_204553897.avi": "interaction_20260227_124559.mp4",
    "Video_20260227_203753817.avi": "interaction_20260227_123745.mp4",
    "Video_20260131_200029359.avi": "interaction_20260131_120024.mp4",
    "Video_20260227_213001434.avi": "interaction_20260227_132951.mp4",
    "Video_20260227_213404452.avi": "interaction_20260227_133408.mp4",
    "Video_20260131_194205407.avi": "interaction_20260131_114156.mp4",
    "Video_20260131_195202906.avi": "interaction_20260131_115150.mp4",
    "Video_20260131_194854095.avi": "interaction_20260131_114852.mp4",
    "Video_20260301_153037623.avi": "interaction_20260301_073041.mp4",
    "Video_20260301_144803454.avi": "interaction_20260301_064753.mp4",
    "Video_20260306_152721366.avi": "interaction_20260306_072721.mp4",
    "Video_20260301_151942635.avi": "interaction_20260301_071948.mp4",
    "Video_20260131_201556629.avi": "interaction_20260131_121548.mp4",
    "Video_20260301_153434856.avi": "interaction_20260301_073435.mp4",
    "Video_20260301_152459131.avi": "interaction_20260301_072503.mp4",
    "Video_20260131_151559270.avi": "interaction_20260131_071552.mp4",
    "Video_20260131_152410916.avi": "interaction_20260131_072412.mp4",
    "Video_20260131_164304016.avi": "interaction_20260131_084300.mp4",
    "Video_20260131_164745532.avi": "interaction_20260131_084732.mp4",
    "Video_20260131_165208524.avi": "interaction_20260131_085207.mp4",
    "Video_20260131_165614756.avi": "interaction_20260131_085611.mp4",
    "Video_20260131_170142792.avi": "interaction_20260131_090139.mp4",
    "Video_20260131_145524524.avi": "interaction_20260131_065459.mp4",
    "Video_20260131_150734369.avi": "interaction_20260131_070722.mp4",
    "Video_20260131_170539636.avi": "interaction_20260131_090541.mp4",
    "Video_20260131_170919896.avi": "interaction_20260131_090917.mp4",
    "Video_20260131_171253889.avi": "interaction_20260131_091249.mp4",
    "Video_20260131_171648040.avi": "interaction_20260131_091657.mp4",
    "Video_20260306_162401599.avi": "interaction_20260306_082346.mp4",
    "Video_20260306_163105571.avi": "interaction_20260306_083107.mp4",
    "Video_20260306_163434878.avi": "interaction_20260306_083434.mp4",
    "Video_20260306_164407883.avi": "interaction_20260306_084406.mp4",
    "Video_20260306_164902044.avi": "interaction_20260306_084853.mp4",
    "Video_20260306_165839689.avi": "interaction_20260306_085830.mp4",
    "Video_20260306_170449073.avi": "interaction_20260306_090441.mp4",
}
MP4_TO_AVI_MAP = {mp4_name: avi_name for avi_name, mp4_name in AVI_TO_MP4_MAP.items()}

_PROCESSOR: Optional[ViTImageProcessor] = None
_MODEL: Optional[ViTModel] = None


def get_scene_backbone() -> Tuple[ViTImageProcessor, ViTModel]:
    global _PROCESSOR, _MODEL
    if _PROCESSOR is not None and _MODEL is not None:
        return _PROCESSOR, _MODEL
    _PROCESSOR = ViTImageProcessor.from_pretrained(str(LOCAL_VIT_PATH), local_files_only=True)
    _MODEL = ViTModel.from_pretrained(str(LOCAL_VIT_PATH), local_files_only=True, add_pooling_layer=False)
    _MODEL.eval()
    _MODEL.to("cpu")
    return _PROCESSOR, _MODEL


@torch.no_grad()
def encode_scene_pil_image(image: Image.Image) -> np.ndarray:
    processor, model = get_scene_backbone()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


def parse_utc_timestamp(timestamp_value: str) -> datetime:
    return datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00")).replace(tzinfo=None)


def avi_start_utc_from_name(avi_name: str) -> datetime:
    parts = avi_name.split("_")
    time_part = f"{parts[1]}_{parts[2].split('.')[0]}"
    return datetime.strptime(time_part, "%Y%m%d_%H%M%S%f") - timedelta(hours=8)


def resolve_avi_path(video_name: str) -> Path:
    avi_name = MP4_TO_AVI_MAP[video_name]
    return DATASET_VIDEO_DIR / avi_name


def read_real_scene_frame(video_name: str, timestamp_value: str) -> Optional[Image.Image]:
    try:
        avi_path = resolve_avi_path(video_name)
        utc_target = parse_utc_timestamp(timestamp_value)
        avi_utc_start = avi_start_utc_from_name(avi_path.name)
        offset_ms = (utc_target - avi_utc_start).total_seconds() * 1000.0
    except Exception as exc:
        print(f"[scene-miss] cannot resolve scene timestamp video_name={video_name}, timestamp={timestamp_value}, error={exc}")
        return None

    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        print(f"[scene-miss] cannot open video_name={video_name}, timestamp={timestamp_value}, avi_path={avi_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        print(f"[scene-miss] invalid video metadata video_name={video_name}, timestamp={timestamp_value}, avi_path={avi_path}, fps={fps}, frames={frame_count}")
        return None

    duration_ms = frame_count / fps * 1000.0
    if offset_ms < 0 or offset_ms > duration_ms:
        cap.release()
        print(f"[scene-miss] timestamp out of range video_name={video_name}, timestamp={timestamp_value}, avi_path={avi_path}, offset_ms={offset_ms:.2f}, duration_ms={duration_ms:.2f}")
        return None

    frame_index = int(round(offset_ms * fps / 1000.0))
    frame_index = min(max(frame_index, 0), frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"[scene-miss] cannot read frame video_name={video_name}, timestamp={timestamp_value}, avi_path={avi_path}, frame_index={frame_index}")
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


class RealSceneFeatureCache:
    def __init__(self, cache_dir: Path = REAL_SCENE_CACHE_DIR, legacy_cache_dir: Optional[Path] = LEGACY_REAL_SCENE_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.legacy_cache_dir = Path(legacy_cache_dir) if legacy_cache_dir else None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"[scene-cache-error] cannot create cache dir: {self.cache_dir}: {exc}")
        self.memory_cache: Dict[str, np.ndarray] = {}
        self.last_records: Dict[str, Dict[str, object]] = {}

    def _cache_key(self, video_name: str, timestamp_value: str) -> str:
        return f"{video_name}|{timestamp_value}"

    def _cache_path(self, video_name: str, timestamp_value: str, cache_dir: Optional[Path] = None) -> Path:
        key = hashlib.md5(self._cache_key(video_name, timestamp_value).encode("utf-8")).hexdigest()
        return (cache_dir or self.cache_dir) / f"{key}.npy"

    def _avi_path_text(self, video_name: str) -> str:
        try:
            return str(resolve_avi_path(video_name))
        except Exception:
            return "<unresolved>"

    def _load_cache_file(self, path: Path, cache_key: str, source: str) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        try:
            feature = np.load(path).astype(np.float32)
        except Exception as exc:
            print(f"[scene-cache-error] cannot read cache: {path}: {exc}")
            return None
        self.memory_cache[cache_key] = feature
        self.last_records[cache_key] = {
            "scene_source": source,
            "scene_cache_path": str(path),
            "scene_nonzero": int(np.count_nonzero(feature)),
        }
        if source == "legacy":
            print(f"[scene-cache-hit] source=legacy path={path}")
        return feature

    def get(self, video_name: str, timestamp_value: str) -> np.ndarray:
        cache_key = self._cache_key(video_name, timestamp_value)
        if cache_key in self.memory_cache:
            feature = self.memory_cache[cache_key]
            record = dict(self.last_records.get(cache_key, {}))
            record.update({"scene_source": "memory", "scene_nonzero": int(np.count_nonzero(feature))})
            self.last_records[cache_key] = record
            return feature

        writable_cache_path = self._cache_path(video_name, timestamp_value)
        feature = self._load_cache_file(writable_cache_path, cache_key, "writable")
        if feature is not None and np.any(feature):
            return feature
        if feature is not None:
            print(
                f"[scene-zero] writable cache is zero; trying legacy video_name={video_name}, "
                f"timestamp={timestamp_value}, cache_path={writable_cache_path}"
            )

        legacy_cache_path = None
        if self.legacy_cache_dir is not None:
            legacy_cache_path = self._cache_path(video_name, timestamp_value, self.legacy_cache_dir)
            if legacy_cache_path != writable_cache_path:
                feature = self._load_cache_file(legacy_cache_path, cache_key, "legacy")
                if feature is not None and np.any(feature):
                    return feature
                if feature is not None:
                    print(
                        f"[scene-zero] legacy cache is zero; falling back to video extract video_name={video_name}, "
                        f"timestamp={timestamp_value}, cache_path={legacy_cache_path}"
                    )

        image = read_real_scene_frame(video_name, timestamp_value)
        feature = encode_scene_pil_image(image) if image is not None else np.zeros(SCENE_FEAT_DIM, dtype=np.float32)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(writable_cache_path, feature)
            scene_source = "generated"
        except Exception as exc:
            print(f"[scene-cache-error] cannot write cache: {writable_cache_path}: {exc}")
            scene_source = "generated_unwritten"
        self.memory_cache[cache_key] = feature
        self.last_records[cache_key] = {
            "scene_source": scene_source,
            "scene_cache_path": str(writable_cache_path),
            "legacy_scene_cache_path": str(legacy_cache_path) if legacy_cache_path else None,
            "scene_nonzero": int(np.count_nonzero(feature)),
        }
        if not np.any(feature):
            print(
                f"[scene-zero] video_name={video_name}, timestamp={timestamp_value}, "
                f"avi_path={self._avi_path_text(video_name)}"
            )
        return feature

    def get_record(self, video_name: str, timestamp_value: str) -> Dict[str, object]:
        return dict(self.last_records.get(self._cache_key(video_name, timestamp_value), {}))


def load_real_scene_features(
    video_name: str,
    approx_timestamps: np.ndarray,
    cache: RealSceneFeatureCache,
) -> Tuple[np.ndarray, Dict[str, object]]:
    features = []
    failed_count = 0
    sample_records = []
    for timestamp_value in approx_timestamps.tolist():
        feature = cache.get(video_name, str(timestamp_value))
        if not np.any(feature):
            failed_count += 1
        sample_records.append(cache.get_record(video_name, str(timestamp_value)))
        features.append(feature)

    stacked = np.stack(features).astype(np.float32) if features else np.zeros((0, SCENE_FEAT_DIM), dtype=np.float32)
    source_counts: Dict[str, int] = {}
    for sample_record in sample_records:
        source = str(sample_record.get("scene_source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    record = {
        "scene_source": "real",
        "avi_path": str(resolve_avi_path(video_name)),
        "sample_count": int(len(approx_timestamps)),
        "failed_scene_frames": int(failed_count),
        "scene_source_counts": source_counts,
        "sample_records": sample_records,
        "scene_cache_dir": str(cache.cache_dir),
        "legacy_scene_cache_dir": str(cache.legacy_cache_dir) if cache.legacy_cache_dir else None,
        "example_timestamps": [str(value) for value in approx_timestamps[:5].tolist()],
    }
    return stacked, record
