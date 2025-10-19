from typing import Literal, Union, TypeAlias

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
    

class LocalSGDConfig(BaseConfig):
    type: Literal["local_sgd"] = "local_sgd"
    inner_step: int = 10

class DilocoConfig(BaseConfig):
    type: Literal["diloco"] = "diloco"
    inner_step: int = 10
    outer_lr: float = 0.7
    nesterov: bool = True  

SemiSyncType: TypeAlias = Union[LocalSGDConfig, DilocoConfig]     


class Config(BaseConfig):
    data: DataConfig
    model: ModelConfig
    total_steps: int
    optim: OptimizerConfig = OptimizerConfig()
    scheduler: LRSchedulerConfig = LRSchedulerConfig()
    wandb: WandbConfig | None = None
    cpu: bool = False # use for dev in plane 
    
    semi_sync: LocalSGDConfig | None = None
    
    
    def wandb_name_and_group(self) -> str:
        name = f"lr-{self.optim.lr}"
        group = f"{self.model.name}-{self.optim.type}-lr-{self.optim.lr}-bs-{self.data.batch_size}-total_steps-{self.total_steps}"
        return name, group
