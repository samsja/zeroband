from zeroband.config import SemiSyncType, LocalSGDConfig, DilocoConfig
from torch.optim import Optimizer
import torch.distributed as dist
import torch
from zeroband.logger import logger


class LocalSGDHook:
    def __init__(self, config: LocalSGDConfig):
        self.config = config
        self.step = 0  # todo check how this value is saved in ckpt

    def _sync(self, optimizer: Optimizer):
        handles = []
        for group in optimizer.param_groups:
            # this code is not optimized for actualy low bandwidth and is doing naive communicaiton
            # optimized way would bucket all this param together
            # this is mainly for ablation
            for param in group["params"]:
                if param.grad is not None:
                    handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
                    handles.append(handle)

        for handle in handles:
            handle.wait()

    def __call__(self, optimizer: Optimizer, *args, **kwargs):
        self.step += 1

        if self.step % self.config.inner_steps:
            logger.info(f"all reduce sync {self.step=} {self.config.inner_steps=}")
            self._sync(optimizer)

        else:
            logger.info(f"skip sync {self.step=} {self.config.inner_steps=}")

    def state_dict(self):
        return {"step": step}

    def load_state_dict(self, states: dict):
        self.step = states["step"]


class DilocoHook:
    def __init__(self, config: DilocoConfig, optimizer: Optimizer):
        self.config = config
        # todo(sami): check hyper param
        self.inner_optimizer = torch.optim.SGD(
            self._get_param_from_opt(optimizer), lr=config.outer_lr, nesterov=True, momentum=0.9
        )
        self._save_params()
        self.step = 0

    def _get_param_from_opt(self, optimizer):
        for group in optimizer.param_groups:
            for param in group["params"]:
                yield param

    def _save_params(self):
        self.params = [param.detach().clone() for param in self._get_param_from_opt(self.inner_optimizer)]

    def _sync(self):
        handles = []
        params = self._get_param_from_opt(self.inner_optimizer)
        for param, old_param in zip(params, self.params):
            param.grad.copy_(old_param.data - param.data)  # todo: check if old_param - param or the other way around
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            param.data = old_param
            handles.append(handle)

        for handle in handles:
            handle.wait()

        self.inner_optimizer.step()

        self._save_params()

    def __call__(self, optimizer: Optimizer, *args, **kwargs):
        self.step += 1

        if self.step % self.config.inner_steps:
            logger.info(f"all reduce sync {self.step=} {self.config.inner_steps=}")
            self._sync()

        else:
            logger.info(f"skip sync {self.step=} {self.config.inner_steps=}")

    def state_dict(self):
        return {
            "step": step,
            "inner_optimizer": self.inner_optimizer.state_dict(),
            "params": [param.data for param in self.params],
        }

    def load_state_dict(self, states: dict):
        self.step = states["step"]
        self.inner_optimizer.load_state_dict(states["inner_optimizer"])

        for param, data in zip(self.params, states["params"]):
            param.data.copy_(data)


def apply_semi_sync_opt(optimizer: Optimizer, config: SemiSyncType):
    match config.type:
        case "local_sgd":
            semi_sync_hook = LocalSGDHook(config)
        case "diloco":
            semi_sync_hook = DilocoHook(config, optimizer)
        case _:
            raise ValueError(f"{config.type=} is not supported for semi sync")

    optimizer.register_step_post_hook(semi_sync_hook)
