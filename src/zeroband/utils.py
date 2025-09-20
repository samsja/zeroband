from zeroband.logger import logger


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
