from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from e2e_config import E2EConfig
from e2e_models import E2EMultitaskPerceiver
from e2e_modality_mask import MODALITY_ORDER


class RobustMaskMultimodalModel(E2EMultitaskPerceiver):
    """Baseline-compatible task heads with mask-aware gated modality fusion."""

    def __init__(
        self,
        num_joint_classes: int,
        num_intent_classes: int,
        num_scene_classes: int,
        joint_class_names: List[str],
        model_dim: int,
        num_latents: int,
        depth: int,
        num_heads: int,
        dropout: float,
        **baseline_kwargs,
    ):
        super().__init__(
            num_joint_classes=num_joint_classes,
            num_intent_classes=num_intent_classes,
            num_scene_classes=num_scene_classes,
            joint_class_names=joint_class_names,
            model_dim=model_dim,
            num_latents=num_latents,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            **baseline_kwargs,
        )
        gate_hidden_dim = max(model_dim // 2, 32)
        self.mask_aware_gates = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.LayerNorm(model_dim),
                    nn.Linear(model_dim, gate_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gate_hidden_dim, 1),
                )
                for modality in MODALITY_ORDER
            }
        )

    def forward(
        self,
        imu: torch.Tensor,
        gesture: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        scene: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        modality_tokens = {
            "imu": self._add_embeddings(self.imu_proj(imu), 0),
            "gesture": self._add_embeddings(self.gesture_proj(gesture), 1),
            "audio": self._add_embeddings(self.audio_proj(audio), 2),
            "text": self._add_embeddings(self.text_proj(text), 3),
            "scene": self._add_single_token_embedding(self.scene_proj(scene).unsqueeze(1), 4),
        }
        representations = {
            key: self.anchor_encoder(self.input_dropout(tokens)).mean(dim=1)
            for key, tokens in modality_tokens.items()
        }
        stacked = torch.stack([representations[key] for key in MODALITY_ORDER], dim=1)

        mask = modality_mask.to(device=stacked.device, dtype=stacked.dtype)
        if mask.ndim != 2 or mask.shape != stacked.shape[:2]:
            raise ValueError(
                f"modality_mask must have shape {tuple(stacked.shape[:2])}, got {tuple(mask.shape)}"
            )
        all_missing = mask.sum(dim=1, keepdim=True) <= 0
        safe_mask = torch.where(all_missing, torch.ones_like(mask), mask)

        gate_logits = torch.cat(
            [self.mask_aware_gates[key](representations[key]) for key in MODALITY_ORDER],
            dim=1,
        )
        gate_logits = gate_logits.masked_fill(safe_mask <= 0, -1e9)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        fused = self.fusion_norm((gate_weights.unsqueeze(-1) * stacked).sum(dim=1))

        base_intent_logits = self.classifier(fused)
        scene_logits = self.scene_classifier(fused)
        gesture_feature = representations["gesture"]
        gesture_intent_logits = self.gesture_intent_head(gesture_feature)
        intent_refine_input = torch.cat(
            [fused, fused, gesture_feature, fused - gesture_feature],
            dim=-1,
        )
        intent_refine_logits = self.intent_refine_head(intent_refine_input)
        intent_refine_gate = self.intent_refine_gate(gesture_feature, fused)
        intent_logits = base_intent_logits + intent_refine_gate * (
            self.intent_refine_scale * intent_refine_logits
            + self.gesture_logit_blend * gesture_intent_logits
        )
        joint_logits = scene_logits.index_select(1, self.joint_scene_index) + intent_logits.index_select(
            1, self.joint_intent_index
        )
        return {
            "joint_logits": joint_logits,
            "intent_logits": intent_logits,
            "base_intent_logits": base_intent_logits,
            "scene_logits": scene_logits,
            "gesture_intent_logits": gesture_intent_logits,
            "intent_refine_logits": intent_refine_logits,
            "intent_refine_gate": intent_refine_gate,
            "modality_gates": gate_weights,
        }


def build_robust_mask_model(
    config: E2EConfig,
    num_classes: int,
    num_intent_classes: int,
    num_scene_classes: int,
    joint_class_names: List[str],
) -> RobustMaskMultimodalModel:
    return RobustMaskMultimodalModel(
        num_joint_classes=num_classes,
        num_intent_classes=num_intent_classes,
        num_scene_classes=num_scene_classes,
        joint_class_names=joint_class_names,
        model_dim=config.model_dim,
        num_latents=config.num_latents,
        depth=config.depth,
        num_heads=config.num_heads,
        dropout=config.dropout,
        min_gate=config.min_gate,
        imu_drop_prob=config.imu_drop_prob,
        audio_drop_prob=config.audio_drop_prob,
        imu_max_scale=config.imu_max_scale,
        audio_max_scale=config.audio_max_scale,
        intent_refine_scale=config.intent_refine_scale,
        gesture_logit_blend=config.gesture_logit_blend,
    )
