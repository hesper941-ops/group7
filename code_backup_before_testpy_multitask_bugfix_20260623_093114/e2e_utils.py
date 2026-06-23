from __future__ import annotations

import csv
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_pickle(payload: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as file:
        pickle.dump(payload, file)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def compute_accuracy(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    true_arr = np.asarray(list(y_true))
    pred_arr = np.asarray(list(y_pred))
    if true_arr.size == 0:
        return 0.0
    return float(np.mean(true_arr == pred_arr))


def split_joint_label(label_name: str) -> tuple[str, str]:
    scene_name, intent_name = str(label_name).split("_", 1)
    return scene_name, intent_name


def compute_selection_score(
    joint_acc: float,
    intent_acc: float,
    scene_acc: float,
    intent_weight: float = 0.35,
    scene_weight: float = 0.05,
) -> float:
    return float(joint_acc + intent_weight * intent_acc + scene_weight * scene_acc)


def write_csv_row(path: Path, row: Dict[str, Any], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def get_run_id() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def format_metrics(metrics: Dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    return value
