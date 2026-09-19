"""VHPS-LARGE 对外接口：敏感层 RS(24,9)、其余参数 ZMORP 的融合保护。

用法::

    import sys, torch
    sys.path.insert(0, "VHPS-LARGE")
    from vhps_large import VHPSLarge

    model = ...                              # 混合精度模型：敏感层 float16，其余 float32
    vhps = VHPSLarge(layer_prefixes=["0."])  # 敏感层前缀（RS 组）
    vhps.protect(model)                      # 其余参数先 ZMORP，敏感层再 RS(24,9)
    vhps.inject(model, ber=1e-5, seed=100)   # RS 码字 + ZMORP 权重分别注错
    stats = vhps.decode(model)               # RS 译码敏感层，再 ZMORP 恢复
    # decode 之后直接用 model 推理

与 VHPS-SMALL 的差异：

1. RS 组换成 RS(24,9) high9（float16，来自 ASRP-LARGE）。9-bit 符号负载
   覆盖 float16 的符号 + 全部指数位（含最高位 bit 10）+ 尾数高 3 位，
   随码字精确恢复，**指数最高位不强制恢复为零**（SMALL 的 RS(48,23)
   译码会把 fp32 bit 30 清零，LARGE 不做这一步）；
2. ZMORP 组换成打包版 ``OptimizedZMORP``（float32，来自 ZMORP-LARGE），
   全部受保护参数拼成单个平面张量一次处理。

三个方法副本即 ASRP-LARGE / ZMORP-LARGE 文件夹内的 ``rs249_codec.py``、
``rs249_high9_codec.py``、``zmorp_large_codec.py``。ZMORP 注错器
``inject_error_to_tensor`` 与 ``fault_tolerance_tools/injectors.py``
语义一致；RS 组注错器内置于 ``RS249High9ModelProtector``（低 BER 几何
稀疏、高 BER 分块稠密 Bernoulli）。VHPS-SMALL 依赖的 ``vhps_layers``
敏感层前缀预设（60/75/80）未随该文件夹提供，此处 ``layer_prefixes``
必须显式传入。
"""

from __future__ import annotations

from typing import Sequence

import torch

from rs249_high9_codec import RS249High9ModelProtector
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


class VHPSLarge:

    def __init__(
        self,
        layer_prefixes: Sequence[str],
        device: torch.device | str | None = None,
        weights_only: bool = True,
        exclude_name_fragments: Sequence[str] = ("mask", "classwise"),
        **protector_kwargs,
    ) -> None:
        self.layer_prefixes = tuple(layer_prefixes)
        self._device = device
        self._weights_only = bool(weights_only)
        self._exclude_name_fragments = tuple(exclude_name_fragments)
        self._protector_kwargs = dict(protector_kwargs)
        self._zmorp = OptimizedZMORP(excluded_layer_prefixes=self.layer_prefixes)
        self._rs: RS249High9ModelProtector | None = None
        self._protected = False

    def uses_rs(self, parameter_name: str) -> bool:
        """名称匹配敏感层前缀的参数走 RS(24,9)。"""
        return parameter_name.startswith(self.layer_prefixes)

    def _rs_parameter_filter(self):
        fragments = self._exclude_name_fragments

        def parameter_filter(name, parameter):
            if not self.uses_rs(name):
                return False
            if any(fragment in name for fragment in fragments):
                return False
            if self._weights_only and not (
                name == "weight" or name.endswith(".weight")
            ):
                return False
            return True

        return parameter_filter

    def protect(self, model: torch.nn.Module):
        """保护模型：其余参数先 ZMORP，敏感层再 RS(24,9) 编码。

        参数按名称前缀与 dtype 自动路由：敏感层前缀下的 float16 权重走
        RS（默认 ``weights_only=True``）；敏感层前缀以外的 float32 参数走
        ZMORP；两组都不覆盖的参数（如敏感层的 float32 参数、非敏感层的
        float16 参数、敏感层 bias）保持原样。
        """
        if self._protected:
            raise RuntimeError("protect() called twice without decode()")
        device = self._device
        if device is None:
            device = next(model.parameters()).device
        self._rs = RS249High9ModelProtector(device, **self._protector_kwargs)
        zmorp_stats = self._zmorp.protect_model(model)
        rs_stats = self._rs.encode(
            model, parameter_filter=self._rs_parameter_filter()
        )
        self._protected = True
        return {"zmorp": zmorp_stats, "rs": rs_stats}

    def inject(self, model: torch.nn.Module, ber: float = 1e-5, seed: int = 100):
        """对 RS 码字容器与 ZMORP 权重分别注入 Bernoulli 位翻转。"""
        if not self._protected:
            raise RuntimeError("protect(model) must run before inject")
        rs_stats = self._rs.inject_errors(ber=ber, seed=seed)
        zmorp_tensors = self._zmorp.inject_weight_errors(
            model, ber=ber, seed=seed, inject_error_fn=inject_error_to_tensor
        )
        return {
            "ber": ber,
            "seed": seed,
            "rs": rs_stats,
            "zmorp_weight_tensors": zmorp_tensors,
        }

    def decode(self, model: torch.nn.Module):
        """解码模型：先 RS 译码敏感层，再 ZMORP 恢复其余参数。

        与 VHPS-SMALL 不同，**指数最高位不强制恢复为零**：RS(24,9) 的
        9-bit 负载包含 float16 的全部指数位（含最高位 bit 10），译码后
        按 ``high9 << 7`` 精确重建，不做任何指数位置零。
        """
        if not self._protected:
            raise RuntimeError("protect(model) must run before decode")
        rs_stats = self._rs.decode(model)
        zmorp_stats = self._zmorp.recover_model(model)
        self._protected = False
        return {"rs": rs_stats, "zmorp": zmorp_stats}


if __name__ == "__main__":
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Linear(128, 32))
    model[0].half()  # 敏感层 float16 → RS 组；其余保持 float32 → ZMORP 组
    with torch.no_grad():
        model[0].weight.mul_(8.0)  # 拉大取值范围，确保有权重的指数最高位为 1
    clean = {k: v.clone() for k, v in model.state_dict().items()}

    vhps = VHPSLarge(layer_prefixes=["0."])
    stats = vhps.protect(model)
    print("protect:", {"zmorp_tensors": stats["zmorp"].parameter_tensors,
                       "rs_tensors": stats["rs"]["parameter_tensors"],
                       "rs_codewords": stats["rs"]["codewords"]})
    inject_stats = vhps.inject(model, ber=1e-5, seed=7)
    print("inject: dirty_codewords =", inject_stats["rs"]["dirty_codewords"])
    decode_stats = vhps.decode(model)
    rs = decode_stats["rs"]
    print("decode:", {k: rs[k] for k in ("corrected", "uncorrectable")})
    assert rs["uncorrectable"] == 0

    # 敏感层（RS 组）== high9 截断期望：低 7 位为 0，高 9 位原样保留。
    # 指数最高位（bit 10）在 9-bit 负载内随码字精确恢复，不强制为零。
    reference = clean["0.weight"]
    raw16 = reference.contiguous().view(torch.int16)
    assert ((raw16 >> 10) & 1).any(), "self-test needs weights with exponent MSB set"
    expected_rs = (raw16 >> 7) << 7
    got_rs = model.state_dict()["0.weight"].contiguous().view(torch.int16)
    assert torch.equal(got_rs, expected_rs), "RS group"
    assert torch.equal(got_rs >> 7, raw16 >> 7), "exponent MSB must survive"

    # 其余参数（ZMORP 组）：尾数低 8 位与 bit 30 清零的量化期望。
    # 1.bias 被保护但未注错（注错只针对 module.weight），应精确等于期望；
    # 1.weight 被注错，ZMORP 只保护指数域，落在尾数高 15 位的翻转按设计
    # 保留，因此只检查指数域完全恢复 + 失配比例极小。
    bits = clean["1.bias"].contiguous().view(torch.int32)
    expected_bias = (
        (bits & -2147483648)
        | (bits & 0x007FFF00)
        | (((bits >> 23) & 0x7F) << 23)
    )
    assert torch.equal(
        model.state_dict()["1.bias"].view(torch.int32), expected_bias
    ), "1.bias"

    bits = clean["1.weight"].contiguous().view(torch.int32)
    expected_weight = (
        (bits & -2147483648)
        | (bits & 0x007FFF00)
        | (((bits >> 23) & 0x7F) << 23)
    )
    got_weight = model.state_dict()["1.weight"].contiguous().view(torch.int32)
    assert torch.equal(
        (got_weight >> 23) & 0xFF, (expected_weight >> 23) & 0x7F
    ), "ZMORP exponent"
    mismatch = int((got_weight != expected_weight).sum())
    print(f"ZMORP weight residual mismatches (mantissa-only by design): {mismatch}/{got_weight.numel()}")
    assert mismatch <= max(4, got_weight.numel() // 200)

    # 敏感层 bias：被 ZMORP 前缀排除、又不是 module.weight，与 VHPS-SMALL
    # 语义一致——既不保护也不注错，保持原值。
    assert torch.equal(model.state_dict()["0.bias"], clean["0.bias"])

    model.eval()
    with torch.no_grad():
        features = model[0](torch.randn(4, 64, dtype=torch.float16))
        output = model[1](features.float())
    assert torch.isfinite(output).all()
    print("VHPSLarge interface self-test passed.")
