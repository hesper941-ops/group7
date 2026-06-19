from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from e2e_config import ALL_JOINT_CLASS_NAMES, E2EConfig, build_config, MODALITY_KEYS, FEATURE_DIMS
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
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


def make_loader(features: dict, labels: np.ndarray, scene_labels: np.ndarray, batch_size: int, shuffle: bool):
    import baseline_real_scene as base

    return base.make_loader(features, labels, scene_labels, batch_size, shuffle)


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, batch_y, _ in loader:
        batch_imu = batch_imu.to(device)
        batch_gesture = batch_gesture.to(device)
        batch_audio = batch_audio.to(device)
        batch_text = batch_text.to(device)
        batch_scene = batch_scene.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_imu, batch_gesture, batch_audio, batch_text, batch_scene)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch_y.size(0)
        total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        total_samples += int(batch_y.size(0))
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []
    for batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, batch_y, _ in loader:
        batch_imu = batch_imu.to(device)
        batch_gesture = batch_gesture.to(device)
        batch_audio = batch_audio.to(device)
        batch_text = batch_text.to(device)
        batch_scene = batch_scene.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_imu, batch_gesture, batch_audio, batch_text, batch_scene)
        loss = criterion(logits, batch_y)
        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * batch_y.size(0)
        total_correct += int((preds == batch_y).sum().item())
        total_samples += int(batch_y.size(0))
        y_true.append(batch_y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
    return (
        total_loss / max(total_samples, 1),
        total_correct / max(total_samples, 1),
        np.concatenate(y_true) if y_true else np.array([], dtype=np.int64),
        np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64),
    )


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
    print(f"[split] train videos -> {describe_samples(full_dataset.samples)}")

    (
        train_features_raw,
        val_features_raw,
        y_train_joint_raw,
        y_val_joint_raw,
        _y_train_intent,
        _y_val_intent,
        y_train_scene,
        y_val_scene,
    ) = split_train_val(config, features, joint_labels, intent_labels, scene_labels)

    scalers = fit_scalers(train_features_raw)
    train_features = apply_scalers(train_features_raw, scalers)
    val_features = apply_scalers(val_features_raw, scalers)
    y_train, label_encoder = encode_joint_labels(y_train_joint_raw)
    y_val = label_encoder.transform(y_val_joint_raw)

    import baseline_real_scene as base

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(train_features, y_train, y_train_scene, config.batch_size, True)
    val_loader = make_loader(val_features, y_val, y_val_scene, config.batch_size, False)
    model = build_model(config, len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss(weight=base.build_class_weights(y_train, len(label_encoder.classes_)).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    train_log_path = run_dir / "train_log.csv"
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        write_csv_row(
            train_log_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        print(f"epoch {epoch:03d}/{config.epochs:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        improved = val_acc > best_val_acc or (np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss)
        if improved:
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config_dict(config, len(label_encoder.classes_)),
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                },
                run_dir / "best_model.pt",
            )

    save_pickle(scalers, run_dir / "scalers.pkl")
    save_pickle(label_encoder, run_dir / "label_encoder.pkl")
    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "class_names": label_encoder.classes_.tolist(),
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
    )
    train(config)


if __name__ == "__main__":
    main()
