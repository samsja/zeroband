import torch
import torch.distributed as dist
from dion import Muon
from pydantic_config import parse_argv
from rich import print as rprint
from torch.distributed._composable.replicate import replicate
from torch.nn import functional as F
from transformers import AutoTokenizer

import wandb
from zeroband.config import Config
from zeroband.data import setup_dataloader
from zeroband.logger import logger
from zeroband.model import Transformer, llama_configs
from zeroband.utils import FakeTokenizer, World


def train(config: Config):
    #########################
    ### pytorch init ###
    #########################

    world = World()

    logger.info(f"Starting training on device: {world.local_rank} with world size: {world.world_size}")

    torch.cuda.set_device(world.local_rank)
    dist.init_process_group(backend="cuda:nccl", device_id=torch.device("cuda", world.local_rank))

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"

    # gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(world.local_rank))

    ##################
    ### model init ###
    ##################

    model_config = llama_configs[config.model]
    model = Transformer(model_config).cuda()
    logger.info("Model initialized")

    model = torch.compile(model, fullgraph=True)
    logger.info("Model compiled")

    model = replicate(model, bucket_cap_mb=100)
    logger.info("Applied DDP to model")

    #########################
    ### other init ###
    #########################

    if world.rank == 0 and config.wandb:
        wandb.init(
            project=config.wandb.project, name=config.wandb.name, group=config.wandb.group, config=config.model_dump()
        )

    max_memory = torch.cuda.mem_get_info()[1] / 1024**3  # GiB

    ######################
    ### optimizer init ###
    #####################

    # optimizer = AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.wd)
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

    optimizer = Muon(
        params=[
            dict(
                params=muon_params,
                algorithm="muon",
                lr=config.optim.lr,
                weight_decay=config.optim.wd,
                adjust_lr="rms_norm",
            ),
            dict(params=adamw_params, algorithm="adamw", lr=config.optim.lr, weight_decay=config.optim.wd),
        ]
    )

    ##################
    ### data init ###
    ##################
    assert config.data.batch_size % (world.world_size * config.data.micro_batch_size) == 0, (
        f"batch_size must be divisible by world_size * micro_batch_size, but got {config.data.batch_size} and {world.world_size} and {config.data.micro_batch_size}"
    )
    num_grad_acc = config.data.batch_size // (world.world_size * config.data.micro_batch_size)

    tokenizer = (
        AutoTokenizer.from_pretrained(config.model) if not config.data.fake else FakeTokenizer(model_config.vocab_size)
    )
    assert len(tokenizer) == model_config.vocab_size, (
        f"tokenizer vocab size {len(tokenizer)} does not match model vocab size {model_config.vocab_size}"
    )

    dataloader = setup_dataloader(config.data, tokenizer)

    data_iter = iter(dataloader)

    #####################
    ### training loop ###
    #####################

    for step in range(config.total_steps):
        torch.cuda.reset_peak_memory_stats()

        batch_loss = torch.tensor(0.0).cuda()
        max_loss = torch.tensor(0.0).cuda()
        ###################
        ### gradd accum ##
        ###################
        for _ in range(num_grad_acc):
            batch = next(data_iter)
            inputs_ids = batch["input_ids"].cuda()
            targets = batch["labels"].cuda()

            logger.info(f"inputs_ids shape: {inputs_ids.shape}")
            logger.info(f"targets shape: {targets.shape}")

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs_ids)
                loss = F.cross_entropy(outputs.view(-1, model_config.vocab_size), targets.view(-1)) / num_grad_acc

            del outputs
            loss.backward()
            batch_loss += loss
            max_loss = torch.max(max_loss, loss)

        ######################
        ### optimizer step ###
        ######################
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        ####################
        ### log metrics ###
        ####################

        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(max_loss, op=dist.ReduceOp.MAX)

        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        peak_memory_pct = peak_memory / max_memory * 100

        pplx = torch.exp(batch_loss).item()

        logger.success(
            f"step {step} | loss {batch_loss.item():.4f} | max_loss {max_loss.item():.4f} | grad_norm {grad_norm.item():.4f} | peak_memory {peak_memory:.4f} GiB  {peak_memory_pct:.1f}% | pplx {pplx:.4f}"
        )

        if world.rank == 0 and config.wandb:
            wandb.log(
                {
                    "train/loss": batch_loss.item(),
                    "train/max_loss": max_loss.item(),
                    "train/perplexity": pplx,
                    "optim/grad_norm": grad_norm.item(),
                    "optim/lr": optimizer.param_groups[0]["lr"],
                    "perf/peak_memory": peak_memory,
                    "step": step,
                }
            )

    logger.success("Training finished")

    dist.destroy_process_group()


if __name__ == "__main__":
    config = Config(**parse_argv())
    rprint(config)
    train(config)
