from typing import Literal

from pydantic_config import BaseConfig


class DataConfig(BaseConfig):
    name: str = "allenai/c4"
    fake: bool = True

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


class Config(BaseConfig):
    data: DataConfig
    model: ModelConfig
    total_steps: int
    optim: OptimizerConfig = OptimizerConfig()
    wandb: WandbConfig | None = None
