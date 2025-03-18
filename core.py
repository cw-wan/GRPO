import gc
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Optional, Union
import numpy as np
import torch
from transformers import is_torch_npu_available, is_torch_xpu_available

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """根据给定遮罩计算张量的均值。"""
    if axis is None:
        total_sum = (values * mask).sum()
        mask_sum = mask.sum()
        return total_sum / mask_sum
    else:
        total_sum = (values * mask).sum(dim=axis)
        mask_sum = mask.sum(dim=axis)
        return total_sum / mask_sum

def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """根据给定遮罩计算张量的方差。"""
    avg_val = masked_mean(values, mask)
    shifted = values - avg_val
    var_result = masked_mean(shifted ** 2, mask)
    if unbiased:
        total_mask = mask.sum()
        if total_mask == 0:
            raise ValueError("遮罩的总和为零，可能在 `mini_batch_size=1` 时出现；尝试增大 `mini_batch_size` 或 `gradient_accumulation_steps`")
        correction = total_mask / (total_mask - 1)
        var_result *= correction
    return var_result

def randn_tensor(
    shape: Union[tuple, list],
    generator: Optional[Union[list[torch.Generator], torch.Generator]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None
) -> torch.Tensor:
    chosen_dev = device or torch.device("cpu")
    chosen_layout = layout if layout is not None else torch.strided
    rand_dev = chosen_dev
    b_size = shape[0]
    if generator is not None:
        g_dev_type = generator[0].device.type if isinstance(generator, list) else generator.device.type
        if g_dev_type != chosen_dev.type and g_dev_type == "cpu":
            rand_dev = "cpu"
            if chosen_dev != torch.device("mps"):
                warnings.warn(
                    f"传入的 generator 在 'cpu' 上创建，但期望在 {chosen_dev} 上创建张量。"
                    f"会先在 'cpu' 上创建后再移动到 {chosen_dev}。"
                    f"如果在 {chosen_dev} 上创建 generator，可能可以略微加速该函数。",
                    UserWarning
                )
        elif g_dev_type != chosen_dev.type and g_dev_type == "cuda":
            raise ValueError(f"无法从类型为 {g_dev_type} 的 generator 生成 {chosen_dev} 张量。")
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]
    if isinstance(generator, list):
        alt_shape = (1,) + shape[1:]
        tmp_list = []
        for i in range(b_size):
            tmp_list.append(
                torch.randn(alt_shape, generator=generator[i], device=rand_dev, dtype=dtype, layout=chosen_layout)
            )
        return torch.cat(tmp_list, dim=0).to(chosen_dev)
    else:
        rand_out = torch.randn(shape, generator=generator, device=rand_dev, dtype=dtype, layout=chosen_layout)
        return rand_out.to(chosen_dev)
