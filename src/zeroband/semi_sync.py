from zeroband.config import SemiSyncType, LocalSGDConfig, DilocoConfig
from torch.optim import Optimizer



class LocalSGDHook:

    def __init__(self, config: LocalSGDConfig):
        self.config = config
        
    def __call__(self, optimizer: Optimizer, *args, **kwargs):
        print(f"LocalSGDHook: semi sync applied, not yet implemented")


class DilocoHook:
    
    def __init__(self, config: DilocoConfig):
        self.config = config
        
    def __call__(self, optimizer: Optimizer, *args, **kwargs):
        print(f"DilocoHook: semi sync applied, not yet implemented")
        

def apply_semi_sync_opt(optimizer: Optimizer, config: SemiSyncType):
    
    match config.type:
        case "local_sgd":
            semi_sync_hook = LocalSGDHook(config)
        case "diloco":
            semi_sync_hook = DilocoHook(config)
        case _:
            raise ValueError(f"{config.type=} is not supported for semi sync")

    
    optimizer.register_step_post_hook(semi_sync_hook)
