from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from e2e_config import (
    ALL_JOINT_CLASS_NAMES,
    E2EConfig,
    INTENT_NAMES,
    SCENE_ID_TO_NAME,
    TEST_USERS,
    TRAIN_USERS,
    build_config,
    MODALITY_KEYS,
    FEATURE_DIMS,
)
from e2e_dataset import E2EMultimodalDataset, dataset_to_arrays, describe_samples
from e2e_models import build_model, model_config_dict
from e2e_utils import ensure_dir, get_run_id, save_json, save_pickle, set_seed, write_csv_row, make_jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end multimodal AR intent training")
    parser.add_argument("--model", default="baseline", choices=["baseline", "improved"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--no-early-stop", action="store_true")
    return parser.parse_args()


def fit_scalers(features: dict) -> dict:
    scalers = {}
    for key in MODALITY_KEYS:
        scaler = StandardScaler()
        scaler.fit(features[key].reshape(-1, FEATURE_DIMS[key]))
        scalers[key] = scaler
    return scalers


def apply_scalers(features: dict, scalers: dict) -> dict:
    transformed = {}
    for key in MODALITY_KEYS:
        flat = features[key].reshape(-1, FEATURE_DIMS[key])
        transformed[key] = scalers[key].transform(flat).reshape(features[key].shape).astype(np.float32)
    return transformed


def make_loader(
    features: dict,
    joint_labels: np.ndarray,
    intent_labels: np.ndarray,
    scene_labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(
        MultimodalSceneDataset(features, joint_labels, intent_labels, scene_labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


class MultimodalSceneDataset(Dataset):
    def __init__(self, features: dict, joint_labels: np.ndarray, intent_labels: np.ndarray, scene_labels: np.ndarray):
        self.imu = torch.from_numpy(features["imu"].astype(np.float32))
        self.gesture = torch.from_numpy(features["gesture"].astype(np.float32))
        self.audio = torch.from_numpy(features["audio"].astype(np.float32))
        self.text = torch.from_numpy(features["text"].astype(np.float32))
        self.scene = torch.from_numpy(features["scene"].astype(np.float32))
        self.joint_labels = torch.from_numpy(joint_labels.astype(np.int64))
        self.intent_labels = torch.from_numpy(intent_labels.astype(np.int64))
        self.scene_labels = torch.from_numpy(scene_labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.joint_labels)

    def __getitem__(self, index: int):
        return (
            self.imu[index],
            self.gesture[index],
            self.audio[index],
            self.text[index],
            self.scene[index],
            self.joint_labels[index],
            self.intent_labels[index],
            self.scene_labels[index],
        )


def build_class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    weights = np.ones(num_classes, dtype=np.float32)
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(class_weight="balanced", classes=unique_labels, y=labels)
    for label_value, weight in zip(unique_labels, class_weights):
        weights[int(label_value)] = float(weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_loss(outputs: dict, joint_y, intent_y, scene_y, criteria: dict, config: E2EConfig) -> tuple[torch.Tensor, dict]:
    joint_loss = criteria["joint"](outputs["joint_logits"], joint_y)
    intent_loss = criteria["intent"](outputs["intent_logits"], intent_y)
    base_intent_loss = criteria["intent"](outputs["base_intent_logits"], intent_y)
    scene_loss = criteria["scene"](outputs["scene_logits"], scene_y)
    gesture_intent_loss = criteria["intent"](outputs["gesture_intent_logits"], intent_y)
    total_loss = (
        joint_loss
        + config.intent_aux_weight * intent_loss
        + config.base_intent_aux_weight * base_intent_loss
        + config.scene_aux_weight * scene_loss
        + config.gesture_intent_aux_weight * gesture_intent_loss
    )
    return total_loss, {
        "joint_loss": float(joint_loss.item()),
        "intent_loss": float(intent_loss.item()),
        "base_intent_loss": float(base_intent_loss.item()),
        "scene_loss": float(scene_loss.item()),
        "gesture_intent_loss": float(gesture_intent_loss.item()),
    }


def compute_selection_score(metrics: dict, config: E2EConfig) -> float:
    return (
        float(metrics["joint_acc"])
        + config.selection_intent_weight * float(metrics["intent_acc"])
        + config.selection_scene_weight * float(metrics["scene_acc"])
    )


def train_one_epoch(model, loader, criteria: dict, optimizer, device: torch.device, config: E2EConfig) -> dict:
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = {"joint": 0, "intent": 0, "scene": 0}
    loss_meter = {"joint_loss": 0.0, "intent_loss": 0.0, "base_intent_loss": 0.0, "scene_loss": 0.0, "gesture_intent_loss": 0.0}
    for batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, batch_joint_y, batch_intent_y, batch_scene_y in loader:
        batch_imu = batch_imu.to(device)
        batch_gesture = batch_gesture.to(device)
        batch_audio = batch_audio.to(device)
        batch_text = batch_text.to(device)
        batch_scene = batch_scene.to(device)
        batch_joint_y = batch_joint_y.to(device)
        batch_intent_y = batch_intent_y.to(device)
        batch_scene_y = batch_scene_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_imu, batch_gesture, batch_audio, batch_text, batch_scene)
        loss, loss_parts = compute_loss(outputs, batch_joint_y, batch_intent_y, batch_scene_y, criteria, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        batch_size = int(batch_joint_y.size(0))
        total_loss += float(loss.item()) * batch_size
        correct["joint"] += int((outputs["joint_logits"].argmax(dim=1) == batch_joint_y).sum().item())
        correct["intent"] += int((outputs["intent_logits"].argmax(dim=1) == batch_intent_y).sum().item())
        correct["scene"] += int((outputs["scene_logits"].argmax(dim=1) == batch_scene_y).sum().item())
        total_samples += batch_size
        for key in loss_meter:
            loss_meter[key] += loss_parts[key] * batch_size
    denom = max(total_samples, 1)
    return {
        "loss": total_loss / denom,
        "joint_acc": correct["joint"] / denom,
        "intent_acc": correct["intent"] / denom,
        "scene_acc": correct["scene"] / denom,
        **{key: value / denom for key, value in loss_meter.items()},
    }


@torch.no_grad()
def evaluate(model, loader, criteria: dict, device: torch.device, config: E2EConfig) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    loss_meter = {"joint_loss": 0.0, "intent_loss": 0.0, "base_intent_loss": 0.0, "scene_loss": 0.0, "gesture_intent_loss": 0.0}
    y_true = {"joint": [], "intent": [], "scene": []}
    y_pred = {"joint": [], "intent": [], "scene": []}
    all_gates = []
    for batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, batch_joint_y, batch_intent_y, batch_scene_y in loader:
        batch_imu = batch_imu.to(device)
        batch_gesture = batch_gesture.to(device)
        batch_audio = batch_audio.to(device)
        batch_text = batch_text.to(device)
        batch_scene = batch_scene.to(device)
        batch_joint_y = batch_joint_y.to(device)
        batch_intent_y = batch_intent_y.to(device)
        batch_scene_y = batch_scene_y.to(device)
        outputs = model(batch_imu, batch_gesture, batch_audio, batch_text, batch_scene)
        loss, loss_parts = compute_loss(outputs, batch_joint_y, batch_intent_y, batch_scene_y, criteria, config)
        batch_size = int(batch_joint_y.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        for key in loss_meter:
            loss_meter[key] += loss_parts[key] * batch_size
        pred_joint = outputs["joint_logits"].argmax(dim=1)
        pred_intent = outputs["intent_logits"].argmax(dim=1)
        pred_scene = outputs["scene_logits"].argmax(dim=1)
        y_true["joint"].append(batch_joint_y.cpu().numpy())
        y_true["intent"].append(batch_intent_y.cpu().numpy())
        y_true["scene"].append(batch_scene_y.cpu().numpy())
        y_pred["joint"].append(pred_joint.cpu().numpy())
        y_pred["intent"].append(pred_intent.cpu().numpy())
        y_pred["scene"].append(pred_scene.cpu().numpy())
        if "modality_gates" in outputs:
            all_gates.append(outputs["modality_gates"].cpu().numpy())
    denom = max(total_samples, 1)
    arrays_true = {key: np.concatenate(value) if value else np.array([], dtype=np.int64) for key, value in y_true.items()}
    arrays_pred = {key: np.concatenate(value) if value else np.array([], dtype=np.int64) for key, value in y_pred.items()}
    return {
        "loss": total_loss / denom,
        "joint_acc": float(np.mean(arrays_true["joint"] == arrays_pred["joint"])) if len(arrays_true["joint"]) else 0.0,
        "intent_acc": float(np.mean(arrays_true["intent"] == arrays_pred["intent"])) if len(arrays_true["intent"]) else 0.0,
        "scene_acc": float(np.mean(arrays_true["scene"] == arrays_pred["scene"])) if len(arrays_true["scene"]) else 0.0,
        "joint_true": arrays_true["joint"],
        "joint_pred": arrays_pred["joint"],
        "intent_true": arrays_true["intent"],
        "intent_pred": arrays_pred["intent"],
        "scene_true": arrays_true["scene"],
        "scene_pred": arrays_pred["scene"],
        "avg_modality_gates": np.concatenate(all_gates, axis=0).mean(axis=0).tolist() if all_gates else [0.0] * len(MODALITY_KEYS),
        **{key: value / denom for key, value in loss_meter.items()},
    }


def encode_joint_labels(labels: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoder.fit(ALL_JOINT_CLASS_NAMES)
    return encoder.transform(labels), encoder


def split_train_val(config: E2EConfig, features: dict, joint_labels: np.ndarray, intent_labels: np.ndarray, scene_labels: np.ndarray):
    indices = np.arange(len(joint_labels))
    unique, counts = np.unique(joint_labels, return_counts=True)
    stratify = joint_labels if len(unique) > 1 and np.all(counts >= 2) else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.val_split,
        random_state=config.seed,
        stratify=stratify,
    )
    return (
        {key: value[train_idx] for key, value in features.items()},
        {key: value[val_idx] for key, value in features.items()},
        joint_labels[train_idx],
        joint_labels[val_idx],
        intent_labels[train_idx],
        intent_labels[val_idx],
        scene_labels[train_idx],
        scene_labels[val_idx],
    )


def train(config: E2EConfig) -> Path:
    set_seed(config.seed)
    run_dir = ensure_dir(config.output_dir / get_run_id())
    save_json(make_jsonable(config.__dict__), run_dir / "run_config.json")

    print("[step] build train dataset from raw data paths")
    full_dataset = E2EMultimodalDataset(config, "train")
    features, intent_labels, scene_labels, joint_labels = dataset_to_arrays(full_dataset)
    print("train users: A, B")
    print(f"[split] train videos -> {describe_samples(full_dataset.samples)}")

    (
        train_features_raw,
        val_features_raw,
        y_train_joint_raw,
        y_val_joint_raw,
        y_train_intent,
        y_val_intent,
        y_train_scene,
        y_val_scene,
    ) = split_train_val(config, features, joint_labels, intent_labels, scene_labels)

    scalers = fit_scalers(train_features_raw)
    train_features = apply_scalers(train_features_raw, scalers)
    val_features = apply_scalers(val_features_raw, scalers)
    y_train, label_encoder = encode_joint_labels(y_train_joint_raw)
    y_val = label_encoder.transform(y_val_joint_raw)
    intent_names = [INTENT_NAMES[index] for index in sorted(INTENT_NAMES)]
    scene_names = [SCENE_ID_TO_NAME[index] for index in sorted(SCENE_ID_TO_NAME)]
    joint_class_names = label_encoder.classes_.tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(train_features, y_train, y_train_intent, y_train_scene, config.batch_size, True)
    val_loader = make_loader(val_features, y_val, y_val_intent, y_val_scene, config.batch_size, False)
    model = build_model(
        config,
        len(label_encoder.classes_),
        num_intent_classes=len(intent_names),
        num_scene_classes=len(scene_names),
        joint_class_names=joint_class_names,
    ).to(device)
    criteria = {
        "joint": nn.CrossEntropyLoss(
            weight=build_class_weights(y_train, len(label_encoder.classes_), device),
            label_smoothing=config.label_smoothing,
        ),
        "intent": nn.CrossEntropyLoss(
            weight=build_class_weights(y_train_intent, len(intent_names), device),
            label_smoothing=config.label_smoothing,
        ),
        "scene": nn.CrossEntropyLoss(
            weight=build_class_weights(y_train_scene, len(scene_names), device),
            label_smoothing=config.label_smoothing,
        ),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_joint_acc = -1.0
    best_val_intent_acc = -1.0
    best_val_scene_acc = -1.0
    best_val_loss = float("inf")
    best_selection_score = -1.0
    best_epoch = 0
    patience_counter = 0
    train_log_path = run_dir / "train_log.csv"
    train_log_fields = [
        "epoch",
        "train_loss",
        "train_joint_acc",
        "train_intent_acc",
        "train_scene_acc",
        "val_loss",
        "val_joint_acc",
        "val_intent_acc",
        "val_scene_acc",
        "val_selection_score",
        "lr",
        "train_joint_loss",
        "train_intent_loss",
        "train_scene_loss",
        "train_base_intent_loss",
        "train_gesture_intent_loss",
        "val_joint_loss",
        "val_intent_loss",
        "val_scene_loss",
        "val_base_intent_loss",
        "val_gesture_intent_loss",
    ]
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criteria, optimizer, device, config)
        val_metrics = evaluate(model, val_loader, criteria, device, config)
        selection_score = compute_selection_score(val_metrics, config)
        lr = optimizer.param_groups[0]["lr"]
        write_csv_row(
            train_log_path,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_joint_acc": train_metrics["joint_acc"],
                "train_intent_acc": train_metrics["intent_acc"],
                "train_scene_acc": train_metrics["scene_acc"],
                "val_loss": val_metrics["loss"],
                "val_joint_acc": val_metrics["joint_acc"],
                "val_intent_acc": val_metrics["intent_acc"],
                "val_scene_acc": val_metrics["scene_acc"],
                "val_selection_score": selection_score,
                "lr": lr,
                "train_joint_loss": train_metrics["joint_loss"],
                "train_intent_loss": train_metrics["intent_loss"],
                "train_scene_loss": train_metrics["scene_loss"],
                "train_base_intent_loss": train_metrics["base_intent_loss"],
                "train_gesture_intent_loss": train_metrics["gesture_intent_loss"],
                "val_joint_loss": val_metrics["joint_loss"],
                "val_intent_loss": val_metrics["intent_loss"],
                "val_scene_loss": val_metrics["scene_loss"],
                "val_base_intent_loss": val_metrics["base_intent_loss"],
                "val_gesture_intent_loss": val_metrics["gesture_intent_loss"],
            },
            train_log_fields,
        )
        print(
            f"epoch {epoch:03d}/{config.epochs:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_joint={train_metrics['joint_acc']:.4f} "
            f"train_intent={train_metrics['intent_acc']:.4f} train_scene={train_metrics['scene_acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_joint={val_metrics['joint_acc']:.4f} "
            f"val_intent={val_metrics['intent_acc']:.4f} val_scene={val_metrics['scene_acc']:.4f} "
            f"select={selection_score:.4f}"
        )
        improved = (selection_score > best_selection_score) or (
            np.isclose(selection_score, best_selection_score) and float(val_metrics["loss"]) < best_val_loss
        )
        if improved:
            best_selection_score = float(selection_score)
            best_val_joint_acc = float(val_metrics["joint_acc"])
            best_val_intent_acc = float(val_metrics["intent_acc"])
            best_val_scene_acc = float(val_metrics["scene_acc"])
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config_dict(config, len(label_encoder.classes_)),
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_joint_acc,
                    "best_val_joint_acc": best_val_joint_acc,
                    "best_val_intent_acc": best_val_intent_acc,
                    "best_val_scene_acc": best_val_scene_acc,
                    "best_val_loss": best_val_loss,
                    "best_selection_score": best_selection_score,
                    "best_val_selection_score": best_selection_score,
                },
                run_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            if not config.no_early_stop and patience_counter >= config.patience:
                print(f"[early_stop] no validation selection_score improvement for {config.patience} epochs")
                break

    save_pickle(scalers, run_dir / "scalers.pkl")
    save_pickle(label_encoder, run_dir / "label_encoder.pkl")
    save_pickle(
        {
            "joint_class_names": joint_class_names,
            "intent_names": intent_names,
            "scene_names": scene_names,
        },
        run_dir / "label_mappings.pkl",
    )
    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_joint_acc,
        "best_val_joint_acc": best_val_joint_acc,
        "best_val_intent_acc": best_val_intent_acc,
        "best_val_scene_acc": best_val_scene_acc,
        "best_val_loss": best_val_loss,
        "best_selection_score": best_selection_score,
        "best_val_selection_score": best_selection_score,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "class_names": joint_class_names,
        "intent_names": intent_names,
        "scene_names": scene_names,
        "model_config": model_config_dict(config, len(label_encoder.classes_)),
        "selection_rule": (
            "val_joint_acc "
            f"+ {config.selection_intent_weight} * val_intent_acc "
            f"+ {config.selection_scene_weight} * val_scene_acc"
        ),
        "loss_weights": {
            "intent_aux_weight": config.intent_aux_weight,
            "scene_aux_weight": config.scene_aux_weight,
            "gesture_intent_aux_weight": config.gesture_intent_aux_weight,
            "base_intent_aux_weight": config.base_intent_aux_weight,
            "label_smoothing": config.label_smoothing,
        },
        "train_users": list(TRAIN_USERS),
        "test_users": list(TEST_USERS),
    }
    save_json(metrics, run_dir / "metrics.json")
    print(f"[saved] {run_dir}")
    return run_dir


def main() -> None:
    args = parse_args()
    config = build_config(
        model=args.model,
        data_root=args.data_root,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        patience=args.patience,
        no_early_stop=args.no_early_stop,
    )
    train(config)


if __name__ == "__main__":
    main()
