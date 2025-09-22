import os
import time

import torch

from zeroband.logger import logger
from zeroband.model import Transformer


class World:
    def __init__(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))


# hardcoded BF16 type peak flops for NVIDIA A100, H100, H200, B200 GPU and AMD MI250, MI300X, AMD MI325X and Intel PVC
def get_peak_flops(device_name: str) -> int:
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "B200" in device_name:
        # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
        return 2.25e15
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1300e12
    elif "MI250X" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html (per GCD)
        return 191.5e12
    elif "Data Center GPU Max 1550" in device_name:
        # Also known as Ponte Vecchio (PVC).
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Dot Product Accumulate Systolic (DPAS):
        # - Freq: 1300MHz
        # - #ops: 512
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name:
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return 362e12

    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


class PerfCounter:
    def __init__(self, model: Transformer, seq_len: int):
        self.model_config = model.model_args
        self.world = World()
        self.peak_flops = get_peak_flops(torch.cuda.get_device_name(self.world.local_rank))
        self.nparams, self.num_flops_per_token = self.model_config.get_nparams_and_flops(model, seq_len=seq_len)

        self._start_time = None

    def start(self):
        self._start_time = time.time()

    def get_perf(self, num_tokens_per_device: int):
        # tokens per second per device, abbreviated as tps

        step_time = time.time() - self._start_time
        tps = num_tokens_per_device / step_time

        # model FLOPS utilization
        # For its definition and calculation, please refer to the PaLM paper:
        # https://arxiv.org/abs/2204.02311
        mfu = 100 * self.num_flops_per_token * tps / self.peak_flops
        tflops = self.num_flops_per_token * tps / 1e12

        tps_global = tps * self.world.world_size

        return {
            "tps": tps,
            "mfu": mfu,
            "tflops": tflops,
            "tps_global": tps_global,
            "step_time": step_time,
        }


class FakeTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size
