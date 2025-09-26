import torch
from dion import Muon

from zeroband.config import OptimizerConfig
from zeroband.model import Transformer


def setup_muon_optimizer(model: Transformer, config: OptimizerConfig):
    def muon_enabled(n, p):
        if p.ndim < 2:
            return False
        if "lm_head" in n:
            return False
        if "embed_tokens" in n:
            return False
        return True

    muon_params = [p for n, p in model.named_parameters() if muon_enabled(n, p)]
    adamw_params = [p for n, p in model.named_parameters() if not muon_enabled(n, p)]

    return Muon(
        params=[
            dict(
                params=muon_params,
                algorithm="muon",
                lr=config.lr,
                weight_decay=config.wd,
                adjust_lr="rms_norm",
            ),
            dict(params=adamw_params, algorithm="adamw", lr=config.lr, weight_decay=config.wd),
        ]
    )


def setup_optimizer(model: Transformer, config: OptimizerConfig):
    match config.type:
        case "adamw":
            return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
        case "muon":
            return setup_muon_optimizer(model, config)
        case _:
            raise ValueError(f"Invalid optimizer type: {config.type}")
