from zeroband.config import SemiSyncType, LocalSGDConfig, DilocoConfig
from torch.optim import Optimizer
import torch.distributed as dist
from zeroband.logger import logger



class LocalSGDHook:

    def __init__(self, config: LocalSGDConfig):
        self.config = config
        self.step = 0 # todo check how this value is saved in ckpt


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
        
        


class DilocoHook:
    
    def __init__(self, config: DilocoConfig):
        self.config = config
        
    def __call__(self, optimizer: Optimizer, *args, **kwargs):
        raise NotImplementedError("DilocoHook not yet implemented")


def apply_semi_sync_opt(optimizer: Optimizer, config: SemiSyncType):
    
    match config.type:
        case "local_sgd":
            semi_sync_hook = LocalSGDHook(config)
        case "diloco":
            semi_sync_hook = DilocoHook(config)
        case _:
            raise ValueError(f"{config.type=} is not supported for semi sync")

    
    optimizer.register_step_post_hook(semi_sync_hook)
