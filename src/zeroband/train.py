import torch
import torch.distributed as dist
from pydantic_config import parse_argv
from rich import print as rprint
from torch.distributed._composable.replicate import replicate
from torch.nn import functional as F
from transformers import AutoTokenizer

import wandb
from zeroband.config import Config
from zeroband.data import setup_dataloader
from zeroband.logger import logger
from zeroband.lr_scheduler import setup_scheduler
from zeroband.model import Transformer, llama_configs
from zeroband.optim import setup_optimizer
from zeroband.utils import FakeTokenizer, PerfCounter, World


def train(config: Config):
    #########################
    ### pytorch init ###
    #########################

    world = World()

    logger.info(f"Starting training on device: {world.local_rank} with world size: {world.world_size}")

    if not config.cpu:
        torch.cuda.set_device(world.local_rank)
        device = torch.device("cuda", world.local_rank)
        dist.init_process_group(backend="cuda:nccl", device_id=device)

    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="cpu:gloo")

        

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"
    torch._dynamo.config.compiled_autograd = True

    ##################
    ### model init ###
    ##################

    model_config = llama_configs[config.model.name]
    model = Transformer(model_config).to(device)
    logger.info("Model initialized")

    if config.model.compile:
        model = torch.compile(model, fullgraph=True)
        logger.info("Model compiled")

    model = replicate(model, bucket_cap_mb=100)  # TODO make sure we don't all reduce at each grad acc step
    logger.info("Applied DDP to model")

    #########################
    ### other init ###
    #########################

    if world.rank == 0 and config.wandb:
        name, group = config.wandb_name_and_group()
        wandb.init(project=config.wandb.project, name=name, group=group, config=config.model_dump())


    if not config.cpu:
        max_memory = torch.cuda.mem_get_info()[1] / 1024**3  # GiB 
    else:
        max_memory = 1

    ######################
    ### optimizer init ###
    #####################

    optimizer = setup_optimizer(model, config.optim)
    scheduler = setup_scheduler(optimizer, config.scheduler, config.total_steps, config.optim.lr)

    ##################
    ### data init ###
    ##################
    assert config.data.batch_size % (world.world_size * config.data.micro_batch_size) == 0, (
        f"batch_size must be divisible by world_size * micro_batch_size, but got {config.data.batch_size} and {world.world_size} and {config.data.micro_batch_size}"
    )
    num_grad_acc = config.data.batch_size // (world.world_size * config.data.micro_batch_size)

    tokenizer = (
        AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b", legacy=False)
        if not config.data.fake
        else FakeTokenizer(model_config.vocab_size)
    )
    assert len(tokenizer) == model_config.vocab_size, (
        f"tokenizer vocab size {len(tokenizer)} does not match model vocab size {model_config.vocab_size}"
    )

    dataloader = setup_dataloader(config.data, tokenizer)

    data_iter = iter(dataloader)

    #####################
    ### training loop ###
    #####################

    perf_counter = PerfCounter(model, config.data.seq_len)

    total_tokens = 0

    for step in range(config.total_steps):
        
        if not config.cpu:
            torch.cuda.reset_peak_memory_stats()

        batch_loss = torch.tensor(0.0).to(device)
        max_loss = torch.tensor(0.0).to(device)
        perf_counter.start()
        ###################
        ### gradd accum ##
        ###################
        for grad_acc_step in range(num_grad_acc):
            model.set_requires_gradient_sync(grad_acc_step == num_grad_acc - 1)
            batch = next(data_iter)
            inputs_ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

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
        scheduler.step()
        optimizer.zero_grad()

        ####################
        ### log metrics ###
        ####################

        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(max_loss, op=dist.ReduceOp.MAX)

        if not config.cpu:
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        else:
            peak_memory = 1
            
        peak_memory_pct = peak_memory / max_memory * 100

        pplx = torch.exp(batch_loss).item()

        total_tokens += config.data.seq_len * config.data.batch_size
        perf_data = perf_counter.get_perf(
            num_tokens_per_device=config.data.seq_len * config.data.batch_size // world.world_size
        )

        logger.info(
            f"[green]step {step}[/green] | [yellow]loss {batch_loss.item():.4f}[/yellow] | [red]grad_norm {grad_norm.item():.4f}[/red] | [blue]peak_memory {peak_memory:.4f} GiB {peak_memory_pct:.1f}%[/blue] | [cyan]tps {perf_data['tps']:.4f}[/cyan] | [yellow]mfu {perf_data['mfu']:.4f}%[/yellow], [purple]lr {optimizer.param_groups[0]['lr']:.4f}[/purple]"
        )

        if world.rank == 0 and config.wandb:
            wandb.log(
                {
                    "train/loss": batch_loss.item(),
                    "train/max_loss": max_loss.item(),
                    "train/perplexity": pplx,
                    "train/total_tokens": total_tokens,
                    "optim/grad_norm": grad_norm.item(),
                    "optim/lr": optimizer.param_groups[0]["lr"],
                    "perf/peak_memory": peak_memory,
                    "perf/tps": perf_data["tps"],
                    "perf/tps_global": perf_data["tps_global"],
                    "perf/mfu": perf_data["mfu"],
                    "perf/tflops": perf_data["tflops"],
                    "perf/step_time": perf_data["step_time"],
                    "step": step,
                }
            )

    logger.success("Training finished")

    dist.destroy_process_group()


if __name__ == "__main__":
    config = Config(**parse_argv())
    rprint(config)
    train(config)
