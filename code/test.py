from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from e2e_config import INTENT_NAMES, SCENE_ID_TO_NAME, build_config
from e2e_dataset import E2EMultimodalDataset, dataset_to_arrays, describe_samples
from e2e_models import build_model
from e2e_utils import compute_accuracy, ensure_dir, load_pickle, save_json, make_jsonable
from train import apply_scalers, evaluate, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end multimodal AR intent testing")
    parser.add_argument("--model", default="baseline", choices=["baseline", "improved"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_joint_names(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scenes = []
    intents = []
    for label in labels.tolist():
        scene, intent = str(label).split("_", 1)
        scenes.append(scene)
        intents.append(intent)
    return np.asarray(scenes, dtype=object), np.asarray(intents, dtype=object)


def save_predictions(
    path: Path,
    dataset: E2EMultimodalDataset,
    true_joint: np.ndarray,
    pred_joint: np.ndarray,
    pred_intent_head: np.ndarray | None = None,
    pred_scene_head: np.ndarray | None = None,
) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "sample_id",
            "video_name",
            "scene",
            "true_joint",
            "pred_joint",
            "true_intent",
            "pred_intent_from_joint",
            "pred_intent_from_head",
            "true_scene",
            "pred_scene_from_joint",
            "pred_scene_from_head",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (sample, true_label, pred_label) in enumerate(zip(dataset.samples, true_joint.tolist(), pred_joint.tolist())):
            true_scene, true_intent = str(true_label).split("_", 1)
            pred_scene, pred_intent = str(pred_label).split("_", 1)
            head_intent = pred_intent_head[index] if pred_intent_head is not None else ""
            head_scene = pred_scene_head[index] if pred_scene_head is not None else ""
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "video_name": sample.video_name,
                    "scene": sample.scene,
                    "true_joint": true_label,
                    "pred_joint": pred_label,
                    "true_intent": true_intent,
                    "pred_intent_from_joint": pred_intent,
                    "pred_intent_from_head": head_intent,
                    "true_scene": true_scene,
                    "pred_scene_from_joint": pred_scene,
                    "pred_scene_from_head": head_scene,
                }
            )


def apply_checkpoint_model_config(config, checkpoint: dict):
    model_config = checkpoint.get("model_config")
    if not model_config:
        print("[warn] checkpoint has no model_config; falling back to command-line/default model parameters")
        return config
    return replace(
        config,
        model=model_config.get("model", config.model),
        model_dim=int(model_config.get("model_dim", config.model_dim)),
        num_latents=int(model_config.get("num_latents", config.num_latents)),
        depth=int(model_config.get("depth", config.depth)),
        num_heads=int(model_config.get("num_heads", config.num_heads)),
        dropout=float(model_config.get("dropout", config.dropout)),
        learning_rate=float(model_config.get("learning_rate", config.learning_rate)),
        weight_decay=float(model_config.get("weight_decay", config.weight_decay)),
        label_smoothing=float(model_config.get("label_smoothing", config.label_smoothing)),
        grad_clip_norm=float(model_config.get("grad_clip_norm", config.grad_clip_norm)),
        intent_aux_weight=float(model_config.get("intent_aux_weight", config.intent_aux_weight)),
        scene_aux_weight=float(model_config.get("scene_aux_weight", config.scene_aux_weight)),
        gesture_intent_aux_weight=float(model_config.get("gesture_intent_aux_weight", config.gesture_intent_aux_weight)),
        base_intent_aux_weight=float(model_config.get("base_intent_aux_weight", config.base_intent_aux_weight)),
        selection_intent_weight=float(model_config.get("selection_intent_weight", config.selection_intent_weight)),
        selection_scene_weight=float(model_config.get("selection_scene_weight", config.selection_scene_weight)),
        min_gate=float(model_config.get("min_gate", config.min_gate)),
        imu_drop_prob=float(model_config.get("imu_drop_prob", config.imu_drop_prob)),
        audio_drop_prob=float(model_config.get("audio_drop_prob", config.audio_drop_prob)),
        imu_max_scale=float(model_config.get("imu_max_scale", config.imu_max_scale)),
        audio_max_scale=float(model_config.get("audio_max_scale", config.audio_max_scale)),
        intent_refine_scale=float(model_config.get("intent_refine_scale", config.intent_refine_scale)),
        gesture_logit_blend=float(model_config.get("gesture_logit_blend", config.gesture_logit_blend)),
    )


def test(config, checkpoint_path: Path) -> Path:
    run_dir = checkpoint_path.parent
    output_dir = ensure_dir(run_dir / "test")
    scalers = load_pickle(run_dir / "scalers.pkl")
    label_encoder = load_pickle(run_dir / "label_encoder.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = apply_checkpoint_model_config(config, checkpoint)

    print("[step] build test dataset from raw data paths")
    test_dataset = E2EMultimodalDataset(config, "test")
    features_raw, intent_labels, scene_labels, joint_labels_raw = dataset_to_arrays(test_dataset)
    print("test users: C")
    print(f"[split] test videos -> {describe_samples(test_dataset.samples)}")
    features = apply_scalers(features_raw, scalers)
    y_test = label_encoder.transform(joint_labels_raw)

    intent_names = [INTENT_NAMES[index] for index in sorted(INTENT_NAMES)]
    scene_names = [SCENE_ID_TO_NAME[index] for index in sorted(SCENE_ID_TO_NAME)]
    joint_names = label_encoder.classes_.tolist()
    model = build_model(
        config,
        len(label_encoder.classes_),
        num_intent_classes=len(intent_names),
        num_scene_classes=len(scene_names),
        joint_class_names=joint_names,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = make_loader(features, y_test, intent_labels, scene_labels, config.batch_size, False)
    criteria = {
        "joint": nn.CrossEntropyLoss(),
        "intent": nn.CrossEntropyLoss(),
        "scene": nn.CrossEntropyLoss(),
    }
    test_metrics = evaluate(model, loader, criteria, device, config)
    test_loss = test_metrics["loss"]
    joint_acc = test_metrics["joint_acc"]
    y_true = test_metrics["joint_true"]
    y_pred = test_metrics["joint_pred"]

    true_joint = label_encoder.inverse_transform(y_true)
    pred_joint = label_encoder.inverse_transform(y_pred)
    true_scene, true_intent = split_joint_names(true_joint)
    pred_scene, pred_intent = split_joint_names(pred_joint)
    pred_intent_head = np.array([intent_names[index] for index in test_metrics["intent_pred"]], dtype=object)
    pred_scene_head = np.array([scene_names[index] for index in test_metrics["scene_pred"]], dtype=object)

    joint_report = classification_report(
        true_joint,
        pred_joint,
        labels=joint_names,
        target_names=joint_names,
        zero_division=0,
        digits=4,
    )
    intent_report = classification_report(
        true_intent,
        pred_intent,
        labels=intent_names,
        target_names=intent_names,
        zero_division=0,
        digits=4,
    )
    scene_report = classification_report(
        true_scene,
        pred_scene,
        labels=scene_names,
        target_names=scene_names,
        zero_division=0,
        digits=4,
    )
    intent_head_report = classification_report(
        true_intent,
        pred_intent_head,
        labels=intent_names,
        target_names=intent_names,
        zero_division=0,
        digits=4,
    )
    scene_head_report = classification_report(
        true_scene,
        pred_scene_head,
        labels=scene_names,
        target_names=scene_names,
        zero_division=0,
        digits=4,
    )

    print("[joint]")
    print(joint_report)
    print("[intent]")
    print(intent_report)
    print("[scene]")
    print(scene_report)
    print("[intent_head]")
    print(intent_head_report)
    print("[scene_head]")
    print(scene_head_report)

    joint_cm = confusion_matrix(true_joint, pred_joint, labels=joint_names)
    intent_cm = confusion_matrix(true_intent, pred_intent, labels=intent_names)
    scene_cm = confusion_matrix(true_scene, pred_scene, labels=scene_names)
    intent_head_cm = confusion_matrix(true_intent, pred_intent_head, labels=intent_names)
    scene_head_cm = confusion_matrix(true_scene, pred_scene_head, labels=scene_names)
    metrics = {
        "test_loss": float(test_loss),
        "joint_accuracy": float(joint_acc),
        "intent_accuracy": compute_accuracy(true_intent, pred_intent),
        "scene_accuracy": compute_accuracy(true_scene, pred_scene),
        "intent_accuracy_from_head": compute_accuracy(true_intent, pred_intent_head),
        "scene_accuracy_from_head": compute_accuracy(true_scene, pred_scene_head),
        "classification_report": joint_report,
        "intent_classification_report": intent_report,
        "scene_classification_report": scene_report,
        "intent_head_classification_report": intent_head_report,
        "scene_head_classification_report": scene_head_report,
        "joint_confusion_matrix": joint_cm.tolist(),
        "intent_confusion_matrix": intent_cm.tolist(),
        "scene_confusion_matrix": scene_cm.tolist(),
        "intent_head_confusion_matrix": intent_head_cm.tolist(),
        "scene_head_confusion_matrix": scene_head_cm.tolist(),
        "joint_class_names": joint_names,
        "intent_class_names": intent_names,
        "scene_class_names": scene_names,
    }
    save_json(make_jsonable(metrics), output_dir / "test_metrics.json")
    save_predictions(output_dir / "test_predictions.csv", test_dataset, true_joint, pred_joint, pred_intent_head, pred_scene_head)
    print(f"[saved] {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    config = build_config(
        model=args.model,
        data_root=args.data_root,
        output_dir=str(checkpoint_path.parent),
        cache_dir=args.cache_dir,
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    test(config, checkpoint_path)


if __name__ == "__main__":
    main()
