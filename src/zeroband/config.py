from typing import Literal

from pydantic import model_validator
from pydantic_config import BaseConfig


class DataConfig(BaseConfig):
    name: str = "allenai/c4"
    fake: bool = False

    micro_batch_size: int
    batch_size: int
    seq_len: int

    name: str

    num_workers: int = 4


class OptimizerConfig(BaseConfig):
    type: Literal["adamw", "muon"] = "muon"

    lr: float = 1e-3
    wd: float = 0.01


class CheckpointConfig(BaseConfig):
    enable: bool = False
    interval: int = 100
    keep: int = 5
    load_step: int | None = None


class LRSChedulerConfig(BaseConfig):
    warmup_steps: int = 100
    decay_steps: int = 100


class WandbConfig(BaseConfig):
    project: str = "zeroband"
    name: str | None = None
    group: str | None = None


class ModelConfig(BaseConfig):
    name: str
    compile: bool = True


class LRSchedulerConfig(BaseConfig):
    """Configuration for linear learning rate scheduler."""

    type: Literal["linear", "cosine"] = "linear"
    warmup_steps: int = 10
    decay_steps: int = 0

    @model_validator(mode="after")
    def no_decay_with_cosine(self):
        if self.type == "cosine" and not self.decay_steps == 0:
            raise ValueError("Cosine scheduler should not have decay steps")
        return self


class Config(BaseConfig):
    data: DataConfig
    model: ModelConfig
    total_steps: int
    optim: OptimizerConfig = OptimizerConfig()
    scheduler: LRSchedulerConfig = LRSchedulerConfig()
    wandb: WandbConfig | None = None
