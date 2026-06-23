from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from e2e_config import E2EConfig, FEATURE_DIMS, MODALITY_KEYS, SCENE_ID_TO_NAME, INTENT_NAMES


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim * multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForward(query_dim, dropout=dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        x = query + attn_out
        return x + self.ff(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        return x + self.ff(x)


class PerceiverEncoder(nn.Module):
    def __init__(self, latent_dim: int, num_latents: int, depth: int, num_heads: int, dropout: float):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.cross_attn = CrossAttentionBlock(latent_dim, latent_dim, num_heads, dropout)
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(latent_dim, num_heads, dropout) for _ in range(depth)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        latents = self.cross_attn(latents, tokens)
        for block in self.blocks:
            latents = block(latents)
        return latents


class ScalarGate(nn.Module):
    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        hidden_dim = max(model_dim // 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, modality_summary: torch.Tensor, anchor_summary: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([modality_summary, anchor_summary], dim=-1)))


class CrossReadoutBlock(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, query_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 2, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        x = attn_out
        return x + self.ff(x)


class E2EMultitaskPerceiver(nn.Module):
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
        min_gate: float = 0.02,
    ):
        super().__init__()
        self.min_gate = min_gate
        self.modality_drop_probs = {"imu": 0.35, "audio": 0.20}
        self.support_scales = {"imu": 0.15, "audio": 0.10}

        self.imu_proj = nn.Sequential(nn.Linear(FEATURE_DIMS["imu"], model_dim), nn.LayerNorm(model_dim))
        self.gesture_proj = nn.Sequential(nn.Linear(FEATURE_DIMS["gesture"], model_dim), nn.LayerNorm(model_dim))
        self.audio_proj = nn.Sequential(nn.Linear(FEATURE_DIMS["audio"], model_dim), nn.LayerNorm(model_dim))
        self.text_proj = nn.Sequential(nn.Linear(FEATURE_DIMS["text"], model_dim), nn.LayerNorm(model_dim))
        self.scene_proj = nn.Sequential(nn.Linear(FEATURE_DIMS["scene"], model_dim), nn.LayerNorm(model_dim))

        self.time_embedding = nn.Parameter(torch.randn(1, 10, model_dim) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(len(MODALITY_KEYS), 1, model_dim) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        self.anchor_encoder = PerceiverEncoder(model_dim, num_latents, depth, num_heads, dropout)
        self.output_query = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.anchor_decoder = CrossAttentionBlock(model_dim, model_dim, num_heads, dropout)
        self.support_readers = nn.ModuleDict(
            {key: CrossReadoutBlock(model_dim, model_dim, num_heads, dropout) for key in ("imu", "audio")}
        )
        self.gates = nn.ModuleDict({key: ScalarGate(model_dim, dropout) for key in ("imu", "audio")})
        self.gesture_readout = CrossReadoutBlock(model_dim, model_dim, num_heads, dropout)

        self.fusion_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_intent_classes),
        )
        self.intent_refine_gate = ScalarGate(model_dim, dropout)
        self.intent_refine_head = nn.Sequential(
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, num_intent_classes),
        )
        self.scene_classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_scene_classes),
        )
        self.gesture_intent_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_intent_classes),
        )

        scene_name_to_idx = {SCENE_ID_TO_NAME[i]: i for i in range(num_scene_classes)}
        intent_name_to_idx = {INTENT_NAMES[i]: i for i in range(num_intent_classes)}
        joint_scene_index = []
        joint_intent_index = []
        for joint_name in joint_class_names:
            scene_name, intent_name = joint_name.split("_", 1)
            joint_scene_index.append(scene_name_to_idx[scene_name])
            joint_intent_index.append(intent_name_to_idx[intent_name])
        self.register_buffer("joint_scene_index", torch.tensor(joint_scene_index, dtype=torch.long))
        self.register_buffer("joint_intent_index", torch.tensor(joint_intent_index, dtype=torch.long))

    def _add_embeddings(self, tokens: torch.Tensor, modality_index: int) -> torch.Tensor:
        return tokens + self.time_embedding + self.modality_embedding[modality_index]

    def _add_single_token_embedding(self, token: torch.Tensor, modality_index: int) -> torch.Tensor:
        return token + self.modality_embedding[modality_index]

    def _sample_keep_masks(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        masks: Dict[str, torch.Tensor] = {}
        for key in ("imu", "audio"):
            drop_prob = self.modality_drop_probs[key] if self.training else 0.0
            if drop_prob <= 0.0:
                keep = torch.ones(batch_size, 1, 1, device=device)
            else:
                keep = (torch.rand(batch_size, 1, 1, device=device) > drop_prob).float()
            masks[key] = keep
        return masks

    def _gate_value(self, key: str, modality_summary: torch.Tensor, anchor_summary: torch.Tensor) -> torch.Tensor:
        gate = self.gates[key](modality_summary, anchor_summary)
        return self.min_gate + (1.0 - self.min_gate) * gate

    def forward(
        self,
        imu: torch.Tensor,
        gesture: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        scene: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        imu_tokens = self._add_embeddings(self.imu_proj(imu), 0)
        gesture_tokens = self._add_embeddings(self.gesture_proj(gesture), 1)
        audio_tokens = self._add_embeddings(self.audio_proj(audio), 2)
        text_tokens = self._add_embeddings(self.text_proj(text), 3)
        scene_token = self._add_single_token_embedding(self.scene_proj(scene).unsqueeze(1), 4)

        keep_masks = self._sample_keep_masks(imu_tokens.shape[0], imu_tokens.device)
        imu_tokens = imu_tokens * keep_masks["imu"]
        audio_tokens = audio_tokens * keep_masks["audio"]

        anchor_tokens = torch.cat([gesture_tokens, text_tokens, scene_token], dim=1)
        anchor_tokens = self.input_dropout(anchor_tokens)
        latents = self.anchor_encoder(anchor_tokens)
        query = self.output_query.expand(anchor_tokens.shape[0], -1, -1)
        anchor_repr = self.anchor_decoder(query, latents).squeeze(1)
        gesture_support_feature = self.gesture_readout(anchor_repr.unsqueeze(1), gesture_tokens).squeeze(1)

        residual = torch.zeros_like(anchor_repr)
        gate_values: Dict[str, torch.Tensor] = {}
        for key, tokens in {"imu": imu_tokens, "audio": audio_tokens}.items():
            summary = tokens.mean(dim=1)
            support_feature = self.support_readers[key](anchor_repr.unsqueeze(1), tokens).squeeze(1)
            gate = self._gate_value(key, summary, anchor_repr)
            residual = residual + self.support_scales[key] * gate * support_feature
            gate_values[key] = gate

        fused = self.fusion_norm(anchor_repr + residual)
        base_intent_logits = self.classifier(fused)
        scene_logits = self.scene_classifier(anchor_repr)
        gesture_intent_logits = self.gesture_intent_head(gesture_support_feature)
        intent_refine_input = torch.cat(
            [anchor_repr, fused, gesture_support_feature, fused - gesture_support_feature],
            dim=-1,
        )
        intent_refine_logits = self.intent_refine_head(intent_refine_input)
        intent_refine_gate = self.intent_refine_gate(gesture_support_feature, fused)
        intent_logits = base_intent_logits + intent_refine_gate * (
            0.35 * intent_refine_logits + 0.30 * gesture_intent_logits
        )
        joint_logits = scene_logits.index_select(1, self.joint_scene_index) + intent_logits.index_select(1, self.joint_intent_index)
        modality_gates = torch.cat(
            [
                gate_values["imu"],
                torch.ones_like(gate_values["imu"]),
                gate_values["audio"],
                torch.ones_like(gate_values["imu"]),
                torch.ones_like(gate_values["imu"]),
            ],
            dim=1,
        )
        return {
            "joint_logits": joint_logits,
            "intent_logits": intent_logits,
            "base_intent_logits": base_intent_logits,
            "scene_logits": scene_logits,
            "gesture_intent_logits": gesture_intent_logits,
            "intent_refine_logits": intent_refine_logits,
            "intent_refine_gate": intent_refine_gate,
            "modality_gates": modality_gates,
        }


def build_model(
    config: E2EConfig,
    num_classes: int,
    num_intent_classes: int = 6,
    num_scene_classes: int = 2,
    joint_class_names: List[str] | None = None,
) -> nn.Module:
    if config.model != "baseline":
        raise NotImplementedError("--model improved is reserved; this refactor currently trains --model baseline only")
    if joint_class_names is None:
        raise ValueError("joint_class_names is required for the multitask joint head mapping")
    return E2EMultitaskPerceiver(
        num_joint_classes=num_classes,
        num_intent_classes=num_intent_classes,
        num_scene_classes=num_scene_classes,
        joint_class_names=joint_class_names,
        model_dim=config.model_dim,
        num_latents=config.num_latents,
        depth=config.depth,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )


def model_config_dict(config: E2EConfig, num_classes: int) -> Dict[str, Any]:
    return {
        "model": config.model,
        "num_classes": int(num_classes),
        "num_intent_classes": 6,
        "num_scene_classes": 2,
        "model_dim": int(config.model_dim),
        "num_latents": int(config.num_latents),
        "depth": int(config.depth),
        "num_heads": int(config.num_heads),
        "dropout": float(config.dropout),
    }
