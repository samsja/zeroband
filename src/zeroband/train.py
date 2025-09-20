import os
import sys

import torch
import torch.distributed as dist
from loguru import logger
from pydantic_config import BaseConfig, parse_argv
from rich import print as rprint

from zeroband.model import Transformer, llama_configs

# Remove default handler
logger.remove()

# Add with simpler format
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
)


class DataConfig(BaseConfig):
    fake: bool = True

    micro_batch_size: int
    batch_size: int
    seq_len: int


class OptimizerConfig(BaseConfig):
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


class Config(BaseConfig):
    data: DataConfig
    model: str
    total_steps: int = 100


class World:
    def __init__(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))


def train(config: Config):
    world = World()

    logger.info(f"Starting training on device: {world.local_rank} with world size: {world.world_size}")

    torch.cuda.set_device(world.local_rank)
    dist.init_process_group(backend="cuda:nccl", device_id=torch.device("cuda", world.local_rank))

    ##################
    ### model init ###
    ##################

    model_config = llama_configs[config.model]
    model = Transformer(model_config).cuda()
    logger.info("Model initialized")

    model = torch.compile(model, fullgraph=True)
    logger.info("Model compiled")

    ######################
    ### optimizer init ###
    #####################

    for step in range(config.total_steps):
        ...

    logger.success("Training finished")

    dist.destroy_process_group()


if __name__ == "__main__":
    config = Config(**parse_argv())
    rprint(config)
    train(config)
