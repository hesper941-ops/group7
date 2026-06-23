from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable

import numpy as np

from e2e_config import (
    E2EConfig,
    FEATURE_DIMS,
    FISHEYE_AVI_BY_VIDEO,
    MODALITY_KEYS,
    TARGET_TIMESTEPS,
    UNKNOWN_LABELS,
    VIDEO_LABELS,
)
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
        self._scene_records: Dict[str, Dict[str, object]] = {}
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
        self._write_sample_source(cache_dir, sample, features)
        return features

    def extract_video_features(self, video_name: str) -> Dict[str, np.ndarray]:
        if video_name not in self._video_cache:
            self._video_cache[video_name] = self._build_video_features(video_name)
        return self._video_cache[video_name]

    def ensure_modality_feature(self, video_name: str, modality: str) -> Path:
        if modality not in {"imu", "gesture", "audio", "text"}:
            raise ValueError(f"Unsupported raw extraction modality: {modality}")
        path = self.get_feature_paths(video_name)[modality]
        if path.exists():
            print(f"[cache] using existing legacy {modality} feature for {video_name}: {path}")
            return path
        print(f"[extract] building {modality} feature for {video_name}")
        if modality == "gesture":
            return self.extract_gesture_from_raw(video_name)
        if modality == "imu":
            return self.extract_imu_from_raw(video_name)
        if modality == "audio":
            return self.extract_audio_from_raw(video_name)
        if modality == "text":
            return self.extract_text_from_raw(video_name)
        raise AssertionError("unreachable")

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

        labels_raw = np.asarray(gesture_data["labels"][:count], dtype=object)
        timestamps_raw = np.asarray(gesture_data["approx_timestamps"][:count], dtype=object)
        valid_mask = np.array([str(label) not in UNKNOWN_LABELS for label in labels_raw], dtype=bool)
        if not valid_mask.any():
            raise FeatureExtractionError(f"All labels are filtered as unknown for {video_name}")

        labels = labels_raw[valid_mask].astype(np.int64)
        timestamps = timestamps_raw[valid_mask]
        if len(labels) != count:
            print(f"[filter] {video_name}: kept {len(labels)}/{count} samples after legacy unknown-label filter")
        features = {
            "imu": normalize_dense_payload(np.asarray(imu_data["features"][:count]), "imu")[valid_mask],
            "gesture": normalize_dense_payload(np.asarray(gesture_data["features"][:count]), "gesture")[valid_mask],
            "audio": normalize_audio_payload(audio_data[:count])[valid_mask],
            "text": normalize_dense_payload(np.asarray(text_data["features"][:count]), "text")[valid_mask],
            "scene": self._load_scene_features(video_name, timestamps),
        }
        self._meta_cache[video_name] = {
            "labels": labels,
            "approx_timestamps": timestamps,
            "count": np.asarray(len(labels), dtype=np.int64),
            "raw_count": np.asarray(count, dtype=np.int64),
        }
        return features

    def _ensure_video_level_feature_files(self, video_name: str) -> None:
        paths = self.get_feature_paths(video_name)
        if all(path.exists() for key, path in paths.items() if key != "scene"):
            return

        for modality in ("gesture", "imu", "audio", "text"):
            self.ensure_modality_feature(video_name, modality)

        missing_features = [f"{key}: {path}" for key, path in paths.items() if key != "scene" and not path.exists()]
        if missing_features:
            raise FeatureExtractionError(
                "Raw extraction finished but these modality files are still missing:\n"
                + "\n".join(missing_features)
            )

    def extract_imu_from_raw(self, video_name: str) -> Path:
        self._ensure_gesture_metadata(video_name)
        self._require_paths(video_name, {"imu_csv": self.config.imu_csv_path})
        try:
            import pandas as pd
            from scipy.spatial.transform import Rotation as R
        except Exception as exc:
            raise FeatureExtractionError(f"IMU extraction requires pandas and scipy on the server: {exc}") from exc

        paths = self.get_feature_paths(video_name)
        metadata = np.load(self._gesture_metadata_path(video_name), allow_pickle=True).item()
        approx_timestamps = metadata["approx_timestamps"]
        labels = metadata["labels"]
        df = pd.read_csv(self.config.imu_csv_path).dropna().sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp_sec"] = df["timestamp"].astype(np.int64) / 1e9
        rotation = R.from_quat(df[["rot_x", "rot_y", "rot_z", "rot_w"]].values)
        df[["roll", "pitch", "yaw"]] = rotation.as_euler("xyz", degrees=True)
        dt = df["timestamp_sec"].diff().fillna(0.01).clip(lower=0.001)
        for axis in ["x", "y", "z"]:
            df[f"vel_{axis}"] = df[f"pos_{axis}"].diff() / dt
            df[f"acc_{axis}"] = df[f"vel_{axis}"].diff() / dt
        df = df.fillna(0)

        features = []
        for timestamp_value in approx_timestamps:
            mid_dt = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))
            mid_abs = mid_dt.timestamp()
            segment = df[(df["timestamp_sec"] >= mid_abs - 0.75) & (df["timestamp_sec"] <= mid_abs + 0.75)]
            features.append(self._sample_imu_segment(segment))
        features_array = np.asarray(features, dtype=np.float32)
        if len(features_array) > 0:
            mean = np.mean(features_array, axis=(0, 1), keepdims=True)
            std = np.std(features_array, axis=(0, 1), keepdims=True) + 1e-8
            features_array = (features_array - mean) / std

        payload = {
            "imu_id": np.arange(len(features)),
            "approx_timestamps": approx_timestamps,
            "features": features_array.astype(np.float32),
            "labels": np.asarray(labels),
        }
        ensure_dir(paths["imu"].parent)
        np.save(paths["imu"], payload)
        return paths["imu"]

    def extract_audio_from_raw(self, video_name: str) -> Path:
        self._ensure_gesture_metadata(video_name)
        self._require_paths(video_name, {"hololens_video": self.config.hololens_dir / video_name})
        try:
            import librosa
            from moviepy.editor import VideoFileClip
        except Exception as exc:
            raise FeatureExtractionError(f"Audio extraction requires librosa and moviepy on the server: {exc}") from exc

        paths = self.get_feature_paths(video_name)
        metadata = np.load(self._gesture_metadata_path(video_name), allow_pickle=True).item()
        wav_path = self.config.cache_dir / "temp_audio" / f"{Path(video_name).stem}.wav"
        ensure_dir(wav_path.parent)
        clip = VideoFileClip(str(self.config.hololens_dir / video_name))
        clip.audio.write_audiofile(str(wav_path), fps=16000, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)
        clip.close()
        audio, sample_rate = librosa.load(wav_path, sr=16000, mono=True)
        video_start = self._video_start_datetime(video_name)
        rows = []
        for index, timestamp_value in enumerate(metadata["approx_timestamps"]):
            mid_dt = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00")).replace(tzinfo=None)
            mid_sec = (mid_dt - video_start).total_seconds()
            start_sample = int(max(0.0, mid_sec - 0.75) * sample_rate)
            end_sample = int((mid_sec + 0.75) * sample_rate)
            segment = audio[start_sample:end_sample]
            if len(segment) < sample_rate * 0.1:
                feature = np.zeros((TARGET_TIMESTEPS, FEATURE_DIMS["audio"]), dtype=np.float32)
            else:
                mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
                width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
                if width < 3:
                    combined = np.vstack([mfcc, np.zeros_like(mfcc), np.zeros_like(mfcc)])
                else:
                    combined = np.vstack([
                        mfcc,
                        librosa.feature.delta(mfcc, width=width),
                        librosa.feature.delta(mfcc, order=2, width=width),
                    ])
                feature = combined.T.astype(np.float32)
            rows.append({"id": index, "timestamp": [float(max(0.0, mid_sec - 0.75)), float(mid_sec + 0.75)], "feature": feature})
        ensure_dir(paths["audio"].parent)
        np.save(paths["audio"], rows)
        self._remove_temp_file(wav_path)
        return paths["audio"]

    def extract_text_from_raw(self, video_name: str) -> Path:
        self._ensure_gesture_metadata(video_name)
        self._require_paths(
            video_name,
            {
                "hololens_video": self.config.hololens_dir / video_name,
                "sentence_model_path": self.config.sentence_model_path,
            },
        )
        try:
            import librosa
            import whisper
            from moviepy.editor import VideoFileClip
            from pypinyin import Style, pinyin
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise FeatureExtractionError(
                f"Text extraction requires whisper, sentence_transformers, pypinyin, librosa, and moviepy on the server: {exc}"
            ) from exc

        paths = self.get_feature_paths(video_name)
        metadata = np.load(self._gesture_metadata_path(video_name), allow_pickle=True).item()
        wav_path = self.config.cache_dir / "temp_text" / f"{Path(video_name).stem}.wav"
        ensure_dir(wav_path.parent)
        clip = VideoFileClip(str(self.config.hololens_dir / video_name))
        clip.audio.write_audiofile(str(wav_path), fps=16000, nbytes=2, codec="pcm_s16le", ffmpeg_params=["-ac", "1"], verbose=False, logger=None)
        clip.close()
        audio, sample_rate = librosa.load(wav_path, sr=16000, mono=True)
        whisper_model = whisper.load_model("small", download_root=str(self.config.whisper_cache_dir))
        st_model = SentenceTransformer(str(self.config.sentence_model_path))
        video_start = self._video_start_datetime(video_name)
        embeddings = []
        meta_rows = []
        for index, timestamp_value in enumerate(metadata["approx_timestamps"]):
            mid_dt = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00")).replace(tzinfo=None)
            mid_sec = (mid_dt - video_start).total_seconds()
            start = max(0.0, mid_sec - 0.75)
            end = mid_sec + 0.75
            segment = audio[int(start * sample_rate): int(end * sample_rate)]
            text = ""
            if len(segment) >= sample_rate * 0.1:
                try:
                    text = whisper_model.transcribe(segment, language="zh", fp16=False)["text"].strip()
                except Exception:
                    text = ""
            pinyin_text = " ".join(item[0] for item in pinyin(text, style=Style.TONE)) if text else ""
            embeddings.append(st_model.encode(pinyin_text, normalize_embeddings=True))
            meta_rows.append({"id": index, "abs_timestamp": [float(start), float(end)], "text": text, "pinyin": pinyin_text})
        embedding_array = np.asarray(embeddings, dtype=np.float32)
        features = np.tile(embedding_array[:, np.newaxis, :], (1, TARGET_TIMESTEPS, 1))
        ensure_dir(paths["text"].parent)
        np.save(paths["text"], {"features": features.astype(np.float32), "metadata": meta_rows})
        self._remove_temp_file(wav_path)
        return paths["text"]

    def extract_gesture_from_raw(self, video_name: str) -> Path:
        timestamp_path = self._ensure_timestamp_file(video_name)
        fisheye_path = self._resolve_fisheye_path(video_name)
        self._require_paths(
            video_name,
            {
                "fisheye_video": fisheye_path,
                "clip_model_path": self.config.clip_model_path,
                "timestamp_file": timestamp_path,
            },
        )
        try:
            import cv2
            import mediapipe as mp
            import torch
            from PIL import Image
            from transformers import CLIPImageProcessor, CLIPVisionModel
        except Exception as exc:
            raise FeatureExtractionError(
                f"Gesture extraction requires cv2, mediapipe, torch, PIL, and transformers on the server: {exc}"
            ) from exc

        paths = self.get_feature_paths(video_name)
        payload = np.load(timestamp_path, allow_pickle=True).item()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = CLIPImageProcessor.from_pretrained(str(self.config.clip_model_path), local_files_only=True)
        vision = CLIPVisionModel.from_pretrained(str(self.config.clip_model_path), local_files_only=True).to(device).eval()
        hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

        valid_features = []
        valid_labels = []
        valid_timestamps = []
        debug_log = {}
        try:
            for index, timestamp_value in enumerate(payload["approx_timestamps"]):
                utc_dt = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00")).replace(tzinfo=None)
                center_ms = self._fisheye_offset_ms(Path(fisheye_path), utc_dt)
                if center_ms is None:
                    continue
                sequence = self._extract_clip_sequence(Path(fisheye_path), center_ms, processor, vision, device, hands)
                if sequence is None:
                    continue
                valid_features.append(sequence)
                valid_labels.append(payload["labels"][index])
                valid_timestamps.append(timestamp_value)
                debug_log[str(len(valid_features) - 1)] = {
                    "original_idx": index,
                    "utc_time": str(timestamp_value),
                    "msec_center": round(center_ms, 2),
                }
        finally:
            hands.close()
        if not valid_features:
            raise FeatureExtractionError(f"Gesture extraction produced no valid samples for {video_name}")
        final_payload = {
            "features": np.asarray(valid_features, dtype=np.float32),
            "labels": np.asarray(valid_labels),
            "video_names": np.asarray([video_name] * len(valid_labels)),
            "approx_timestamps": valid_timestamps,
        }
        ensure_dir(paths["gesture"].parent)
        np.save(paths["gesture"], final_payload)
        np.save(self._gesture_metadata_path(video_name), {key: value for key, value in final_payload.items() if key != "features"})
        with (self.config.processed_data_dir / f"debug_strong_gesture_{Path(video_name).stem}.json").open("w", encoding="utf-8") as file:
            json.dump(debug_log, file, indent=2)
        return paths["gesture"]

    def _write_sample_source(self, cache_dir: Path, sample: dict, features: Dict[str, np.ndarray]) -> None:
        video_name = str(sample["video_name"])
        segment_index = int(sample["segment_index"])
        scene_record = self._scene_records.get(video_name, {})
        sample_records = scene_record.get("sample_records", [])
        sample_scene_record = sample_records[segment_index] if segment_index < len(sample_records) else {}
        timestamps = self._meta_cache.get(video_name, {}).get("approx_timestamps", [])
        timestamp_value = timestamps[segment_index] if segment_index < len(timestamps) else sample.get("timestamp", "")
        scene_feature = np.asarray(features["scene"])
        source_payload = {
            "scene_source": sample_scene_record.get("scene_source", scene_record.get("scene_source", "unknown")),
            "scene_nonzero": int(np.count_nonzero(scene_feature)),
            "scene_failed_count": int(scene_record.get("failed_scene_frames", 0)),
            "scene_cache_dir": str(self.config.cache_dir / "real_scene_vit"),
            "legacy_scene_cache_dir": str(self.config.legacy_scene_cache_dir),
            "fisheye_dir": str(self.config.fisheye_dir),
            "video_name": video_name,
            "timestamp": str(timestamp_value),
            "scene_cache_path": sample_scene_record.get("scene_cache_path"),
            "legacy_scene_cache_path": sample_scene_record.get("legacy_scene_cache_path"),
            "avi_path": scene_record.get("avi_path"),
        }
        with (cache_dir / "source.json").open("w", encoding="utf-8") as file:
            json.dump(source_payload, file, indent=2, ensure_ascii=False)

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

    def _gesture_metadata_path(self, video_name: str) -> Path:
        return self.config.processed_data_dir / f"metadata_strong_gesture_{Path(video_name).stem}.npy"

    def _timestamp_path(self, video_name: str) -> Path:
        return self.config.processed_data_dir / f"features_timestamp_{Path(video_name).stem}.npy"

    def _ensure_gesture_metadata(self, video_name: str) -> Path:
        metadata_path = self._gesture_metadata_path(video_name)
        if metadata_path.exists():
            return metadata_path
        self.ensure_modality_feature(video_name, "gesture")
        if not metadata_path.exists():
            raise FeatureExtractionError(f"Gesture metadata was not produced: {metadata_path}")
        return metadata_path

    def _ensure_timestamp_file(self, video_name: str) -> Path:
        timestamp_path = self._timestamp_path(video_name)
        if timestamp_path.exists():
            print(f"[cache] using existing timestamp file for {video_name}: {timestamp_path}")
            return timestamp_path
        print(f"[extract] building timestamp file for {video_name}")
        self._extract_timestamps_from_raw(video_name, timestamp_path)
        return timestamp_path

    def _extract_timestamps_from_raw(self, video_name: str, output_path: Path) -> None:
        self._require_paths(video_name, {"hololens_video": self.config.hololens_dir / video_name})
        try:
            import contextlib
            import wave
            import webrtcvad
            from moviepy import editor as mp_edit
        except Exception as exc:
            raise FeatureExtractionError(f"Timestamp extraction requires moviepy and webrtcvad on the server: {exc}") from exc

        ensure_dir(output_path.parent)
        temp_wav = self.config.cache_dir / "temp_timestamp" / f"{Path(video_name).stem}.wav"
        ensure_dir(temp_wav.parent)
        clip = mp_edit.VideoFileClip(str(self.config.hololens_dir / video_name))
        clip.audio.write_audiofile(str(temp_wav), fps=16000, codec="pcm_s16le", verbose=False, logger=None)
        clip.close()
        with contextlib.closing(wave.open(str(temp_wav), "rb")) as wav_file:
            sample_rate = wav_file.getframerate()
            audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
            if wav_file.getnchannels() == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

        vad = webrtcvad.Vad(2)
        frame_ms = 30
        frame_len = int(sample_rate * frame_ms / 1000.0)
        frames = [audio[i:i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
        q70_rms = np.percentile(np.abs(audio.astype(np.float32)), 70) if len(audio) else 0.0
        segments = []
        start = None
        for index, frame in enumerate(frames):
            is_speech = vad.is_speech(frame.tobytes(), sample_rate)
            frame_rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
            valid_speech = is_speech and frame_rms > q70_rms * 0.6
            t = index * frame_ms / 1000.0
            if valid_speech and start is None:
                start = t
            elif not valid_speech and start is not None:
                duration = t - start
                if 0.3 <= duration <= 5.0:
                    segments.append((round(max(0.0, start - 0.4), 3), round(t + 0.4, 3)))
                start = None
        timestamps = self._segments_to_iso_timestamps(video_name, segments)
        if not timestamps:
            raise FeatureExtractionError(f"Timestamp extraction produced no speech/action segments for {video_name}")
        np.save(
            output_path,
            {
                "features": np.array([]),
                "labels": np.full(len(timestamps), VIDEO_LABELS[video_name]),
                "video_names": np.asarray([video_name] * len(timestamps)),
                "approx_timestamps": timestamps,
            },
        )
        with (self.config.processed_data_dir / f"segments_info_{Path(video_name).stem}.json").open("w", encoding="utf-8") as file:
            json.dump({str(i): {"range": seg, "mid_utc": timestamps[i]} for i, seg in enumerate(segments)}, file, indent=2)
        self._remove_temp_file(temp_wav)

    def _segments_to_iso_timestamps(self, video_name: str, segments: list[tuple[float, float]]) -> list[str]:
        start_dt = self._video_start_datetime(video_name).replace(tzinfo=timezone.utc)
        timestamps = []
        for start, end in segments:
            mid_dt = start_dt + timedelta(seconds=(start + end) / 2.0)
            timestamps.append(mid_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
        return timestamps

    def _require_paths(self, video_name: str, paths: Dict[str, Path | None]) -> None:
        missing = {key: path for key, path in paths.items() if path is None or not Path(path).exists()}
        if missing:
            layout = "\n".join(f"  {key}: {value}" for key, value in self.config.describe_data_layout().items())
            details = "\n".join(f"  {key}: {path}" for key, path in missing.items())
            raise FeatureExtractionError(
                f"Cannot extract raw features for {video_name}; missing required paths:\n"
                f"{details}\nSearched data layout:\n{layout}"
            )

    def _sample_imu_segment(self, segment) -> np.ndarray:
        columns = ["pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw", "vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"]
        if segment.empty:
            return np.zeros((TARGET_TIMESTEPS, FEATURE_DIMS["imu"]), dtype=np.float32)
        values = segment[columns].values.astype(np.float32)
        if len(values) < 2:
            base = values[0] if len(values) == 1 else np.zeros(FEATURE_DIMS["imu"], dtype=np.float32)
            return np.tile(base, (TARGET_TIMESTEPS, 1)).astype(np.float32)
        new_idx = np.linspace(0, len(values) - 1, TARGET_TIMESTEPS)
        sampled = np.zeros((TARGET_TIMESTEPS, FEATURE_DIMS["imu"]), dtype=np.float32)
        for column_index in range(FEATURE_DIMS["imu"]):
            sampled[:, column_index] = np.interp(new_idx, np.arange(len(values)), values[:, column_index])
        return sampled.astype(np.float32)

    def _video_start_datetime(self, video_name: str) -> datetime:
        match_str = video_name.replace("interaction_", "").split(".")[0]
        return datetime.strptime(match_str, "%Y%m%d_%H%M%S")

    def _fisheye_offset_ms(self, avi_path: Path, utc_target: datetime) -> float | None:
        try:
            import cv2
        except Exception as exc:
            raise FeatureExtractionError(f"Fisheye offset calculation requires cv2 on the server: {exc}") from exc
        time_part = avi_path.name.split("_")[1] + "_" + avi_path.name.split("_")[2].split(".")[0]
        avi_utc_start = datetime.strptime(time_part, "%Y%m%d_%H%M%S%f") - timedelta(hours=8)
        offset_ms = (utc_target - avi_utc_start).total_seconds() * 1000.0
        cap = cv2.VideoCapture(str(avi_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps <= 0 or frame_count <= 0:
            return None
        duration_ms = frame_count / fps * 1000.0
        return offset_ms if 0 <= offset_ms <= duration_ms else None

    def _crop_hand_like_legacy(self, image, hands):
        import cv2
        from PIL import Image

        image_np = np.asarray(image)
        height, width = image_np.shape[:2]
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            xs = [int(landmark.x * width) for hand in results.multi_hand_landmarks for landmark in hand.landmark]
            ys = [int(landmark.y * height) for hand in results.multi_hand_landmarks for landmark in hand.landmark]
            x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(width, max(xs)), min(height, max(ys))
            crop_width, crop_height = x2 - x1, y2 - y1
            pad = 0.4
            x1 = max(0, int(x1 - crop_width * pad))
            y1 = max(0, int(y1 - crop_height * pad))
            x2 = min(width, int(x2 + crop_width * pad))
            y2 = min(height, int(y2 + crop_height * pad))
            return image.crop((x1, y1, x2, y2)).resize((224, 224), Image.LANCZOS)
        return image.rotate(0).resize((224, 224), Image.LANCZOS)

    def _extract_clip_sequence(self, video_path: Path, center_ms: float, processor, vision, device, hands) -> np.ndarray | None:
        import cv2
        import torch
        from PIL import Image

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            cap.release()
            return None
        duration_ms = frame_count / fps * 1000.0
        start_ms = center_ms - 750.0
        end_ms = center_ms + 750.0
        if start_ms < 0 or end_ms > duration_ms:
            cap.release()
            return None
        seq_features = []
        with torch.no_grad():
            for msec in np.linspace(start_ms, end_ms, TARGET_TIMESTEPS):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(msec))
                ok, frame = cap.read()
                if not ok:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand_box = self._crop_hand_like_legacy(image, hands)
                inputs = processor(images=hand_box.convert("RGB"), return_tensors="pt").to(device)
                outputs = vision(**inputs)
                seq_features.append(outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy())
        cap.release()
        return np.asarray(seq_features, dtype=np.float32) if len(seq_features) == TARGET_TIMESTEPS else None

    def _remove_temp_file(self, path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            print(f"[warn] could not remove temporary file: {path}")

    def _load_scene_features(self, video_name: str, timestamps: np.ndarray) -> np.ndarray:
        try:
            import real_scene_utils as scene_utils
        except Exception as exc:
            raise FeatureExtractionError(f"Cannot import real_scene_utils for scene extraction: {exc}") from exc

        writable_scene_cache_dir = self.config.cache_dir / "real_scene_vit"
        legacy_scene_cache_dir = self.config.legacy_scene_cache_dir
        scene_utils.DATASET_VIDEO_DIR = self.config.fisheye_dir
        scene_utils.REAL_SCENE_CACHE_DIR = writable_scene_cache_dir
        if hasattr(scene_utils, "LEGACY_REAL_SCENE_CACHE_DIR"):
            scene_utils.LEGACY_REAL_SCENE_CACHE_DIR = legacy_scene_cache_dir
        if (self.config.project_root / "ViTModel").exists():
            scene_utils.LOCAL_VIT_PATH = self.config.project_root / "ViTModel"
        if self._scene_cache is None:
            self._scene_cache = scene_utils.RealSceneFeatureCache(writable_scene_cache_dir, legacy_scene_cache_dir)
        print(
            f"[extract] building scene feature for {video_name} "
            f"writable_cache={writable_scene_cache_dir} legacy_cache={legacy_scene_cache_dir}"
        )
        scene_features, record = scene_utils.load_real_scene_features(video_name, timestamps, self._scene_cache)
        record["fisheye_dir"] = str(self.config.fisheye_dir)
        record["scene_cache_dir"] = str(writable_scene_cache_dir)
        record["legacy_scene_cache_dir"] = str(legacy_scene_cache_dir)
        self._scene_records[video_name] = record
        source_counts = record.get("scene_source_counts", {})
        failed_count = int(record.get("failed_scene_frames", 0))
        print(f"[scene-source] video_name={video_name} sources={source_counts} failed_scene_frames={failed_count}")
        if scene_features.size and not np.any(scene_features):
            print(
                f"[scene-zero] all scene features are zero video_name={video_name}, "
                f"fisheye_dir={self.config.fisheye_dir}, legacy_scene_cache_dir={legacy_scene_cache_dir}"
            )
        return scene_features.astype(np.float32)

    def _resolve_fisheye_path(self, video_name: str) -> Path | None:
        avi_name = FISHEYE_AVI_BY_VIDEO.get(video_name)
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
