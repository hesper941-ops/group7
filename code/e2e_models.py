from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from e2e_config import E2EConfig


def build_model(config: E2EConfig, num_classes: int) -> nn.Module:
    if config.model != "baseline":
        raise NotImplementedError("--model improved is reserved; this refactor currently trains --model baseline only")

    import baseline_real_scene as base

    return base.PerceiverIOSceneBaseline(
        num_classes=num_classes,
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
        "model_dim": int(config.model_dim),
        "num_latents": int(config.num_latents),
        "depth": int(config.depth),
        "num_heads": int(config.num_heads),
        "dropout": float(config.dropout),
    }
