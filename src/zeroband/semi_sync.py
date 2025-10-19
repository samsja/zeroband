from zeroband.config import LocalSGDConfig
from torch.optim import Optimizer


def apply_semi_sync_opt(optimizer: Optimizer):
    def dummy_hook(optimizer, *args, **kwargs):
        print("semi sync applied, not yet implemented")
    optimizer.register_step_post_hook(dummy_hook)

# class LocalSGD

#     def __init__(self, config: LocalSGDConfig)
#         self.config = config
    

