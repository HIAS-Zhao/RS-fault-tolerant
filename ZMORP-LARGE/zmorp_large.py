"""ZMORP-LARGE 对外接口：只输入模型即可完成保护、注错与恢复。

用法::

    import sys, torch
    sys.path.insert(0, "ZMORP-LARGE")
    from zmorp_large import ZMORPLarge

    model = ...                                # 任意 float32 torch 模型
    zmorp = ZMORPLarge()                       # 可选 excluded_layer_prefixes
    zmorp.protect(model)                       # 保护：改写参数位布局
    zmorp.inject(model, ber=1e-4, seed=42)     # 注错：稠密 Bernoulli 位翻转
    zmorp.recover(model)                       # 解码恢复：校验 + 重建 FP32
    # 之后可直接用 model 推理

本文件只是薄封装：全部位级逻辑来自同目录 ``zmorp_large_codec.py``
（即原桌面 ``ZMORP_LARGE.py`` 的原样更名副本——面向大模型的打包版
``OptimizedZMORP``，protect/recover 把全部受保护参数拼成单个平面张量
一次处理），注错器 ``inject_error_to_tensor`` 与
``fault_tolerance_tools/injectors.py`` 语义一致。
"""

from __future__ import annotations

from typing import Sequence

import torch

from zmorp_large_codec import OptimizedZMORP

# 与 fault_tolerance_tools/bitops.integer_dtype_and_width 一致的位视图映射。
_INTEGER_VIEW = {
    torch.float16: (torch.int16, 16),
    torch.bfloat16: (torch.int16, 16),
    torch.float32: (torch.int32, 32),
    torch.float64: (torch.int64, 64),
    torch.int16: (torch.int16, 16),
    torch.int32: (torch.int32, 32),
    torch.int64: (torch.int64, 64),
}


def inject_error_to_tensor(
    tensor: torch.Tensor,
    error_rate: float = 1e-6,
    seed: int | None = None,
    chunk_size: int = 2048 * 2048,
) -> torch.Tensor:
    """稠密 Bernoulli 位翻转，返回新张量（不修改输入）。

    与 ``fault_tolerance_tools/injectors.py`` 的同名函数语义一致：
    float64 随机数按 chunk 采样后折叠成整数 XOR 掩码；传入 seed 会重置
    全局 torch RNG（CUDA 张量同时重置 CUDA RNG）。
    """
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError(f"error_rate must be in [0,1], got {error_rate}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if seed is not None:
        torch.manual_seed(seed)
        if tensor.is_cuda:
            torch.cuda.manual_seed(seed)
    if tensor.dtype not in _INTEGER_VIEW:
        raise TypeError(f"unsupported dtype: {tensor.dtype}")
    integer_dtype, bit_width = _INTEGER_VIEW[tensor.dtype]

    original_shape = tensor.shape
    flat = tensor.view(-1)
    corrupted = flat.clone()
    positions = torch.arange(bit_width, dtype=torch.int64, device=tensor.device)
    for start in range(0, flat.numel(), chunk_size):
        end = min(start + chunk_size, flat.numel())
        chunk = flat[start:end]
        random_bits = torch.rand(
            chunk.numel() * bit_width, device=tensor.device, dtype=torch.float64
        )
        flips = random_bits < error_rate
        if flips.any():
            masks = (
                (flips.view(chunk.numel(), bit_width).to(torch.int64) << positions)
                .sum(dim=1)
                .to(integer_dtype)
            )
            corrupted[start:end] = (chunk.view(integer_dtype) ^ masks).view(
                tensor.dtype
            )
    return corrupted.view(original_shape).to(tensor.dtype)


class ZMORPLarge:
    """ZMORP 零冗余指数保护（大模型打包版）的对外接口。

    Parameters
    ----------
    excluded_layer_prefixes:
        参数名以任一前缀开头时跳过保护与注错。

    三个方法均直接接收 ``torch.nn.Module``：``protect``/``recover`` 返回
    ``ZMORPStats``，``inject`` 返回注错统计字典。位布局与恢复规则和
    ZMORP-SMALL 完全一致，差别只在实现：参数被拼接为单个平面张量统一
    处理，避免大模型上数百次小 CUDA 启动与主机同步。
    """

    def __init__(
        self, excluded_layer_prefixes: Sequence[str] = ()
    ) -> None:
        self._impl = OptimizedZMORP(excluded_layer_prefixes)

    def protect(self, model: torch.nn.Module):
        """保护模型：写入冗余指数表示（无存储开销）。"""
        return self._impl.protect_model(model)

    def inject(self, model: torch.nn.Module, ber: float = 1e-4, seed: int = 42):
        """对 ``module.weight`` 型参数注入稠密 Bernoulli 位翻转。"""
        injected = self._impl.inject_weight_errors(
            model, ber=ber, seed=seed, inject_error_fn=inject_error_to_tensor
        )
        return {"ber": ber, "seed": seed, "injected_tensors": injected}

    def recover(self, model: torch.nn.Module):
        """解码恢复模型：校验两份指数副本并重建 FP32。"""
        return self._impl.recover_model(model)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.Conv2d(4, 2, 1)
    )
    clean = {k: v.clone() for k, v in model.state_dict().items()}

    zmorp = ZMORPLarge()
    print("protect:", zmorp.protect(model))
    print("inject :", zmorp.inject(model, ber=1e-4, seed=42))
    print("recover:", zmorp.recover(model))

    # 恢复后每个参数等于量化期望：尾数低 8 位清零、指数 bit 30 清零。
    for name, value in model.state_dict().items():
        if value.dtype != torch.float32:
            continue
        bits = clean[name].contiguous().view(torch.int32)
        expected = (
            (bits & -2147483648)
            | (bits & 0x007FFF00)
            | (((bits >> 23) & 0x7F) << 23)
        )
        assert torch.equal(value.view(torch.int32), expected), name
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 16, 16))
    assert torch.isfinite(output).all()
    print("ZMORPLarge interface self-test passed.")
