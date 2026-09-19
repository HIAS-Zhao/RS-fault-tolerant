from __future__ import annotations

from typing import Sequence

import torch

from rs4823_high23_codec import RS4823High23ModelProtector
from vhps_layers import layer60, layer75, layer80
from zmorp_codec import OptimizedZMORP

RATIO_LAYERS = {
    "60": tuple(layer60),
    "75": tuple(layer75),
    "80": tuple(layer80),
}

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


class VHPSSmall:

    def __init__(
        self,
        layer_prefixes: Sequence[str] | None = None,
        ratio: str = "75",
        device: torch.device | str | None = None,
        weights_only: bool = True,
        exclude_name_fragments: Sequence[str] = ("mask", "classwise"),
        **protector_kwargs,
    ) -> None:
        if layer_prefixes is not None:
            self.layer_prefixes = tuple(layer_prefixes)
        else:
            if str(ratio) not in RATIO_LAYERS:
                raise ValueError(
                    f"ratio must be one of {sorted(RATIO_LAYERS)} or pass "
                    "layer_prefixes explicitly"
                )
            self.layer_prefixes = RATIO_LAYERS[str(ratio)]
        self._device = device
        self._weights_only = bool(weights_only)
        self._exclude_name_fragments = tuple(exclude_name_fragments)
        self._protector_kwargs = dict(protector_kwargs)
        self._zmorp = OptimizedZMORP(excluded_layer_prefixes=self.layer_prefixes)
        self._rs: RS4823High23ModelProtector | None = None
        self._protected = False

    def uses_rs(self, parameter_name: str) -> bool:
        """名称匹配敏感层前缀的参数走 RS(48,23)。"""
        return parameter_name.startswith(self.layer_prefixes)

    def protect(self, model: torch.nn.Module):
        """保护模型：其余参数先 ZMORP，敏感层再 RS 编码。"""
        if self._protected:
            raise RuntimeError("protect() called twice without decode()")
        device = self._device
        if device is None:
            device = next(model.parameters()).device
        self._rs = RS4823High23ModelProtector(
            device,
            weights_only=self._weights_only,
            exclude_name_fragments=self._exclude_name_fragments,
            show_progress=False,
            **self._protector_kwargs,
        )
        zmorp_stats = self._zmorp.protect_model(model)
        rs_stats = self._rs.encode(
            model, parameter_filter=lambda name, _: self.uses_rs(name)
        )
        self._protected = True
        return {"zmorp": zmorp_stats, "rs": rs_stats}

    def inject(self, model: torch.nn.Module, ber: float = 1e-5, seed: int = 100):
        """对 RS 码字容器与 ZMORP 权重分别注入稠密 Bernoulli 位翻转。"""
        if not self._protected:
            raise RuntimeError("protect(model) must run before inject")
        rs_stats = self._rs.inject_errors(
            ber=ber,
            seed=seed,
            inject_error_fn=inject_error_to_tensor,
        )
        zmorp_tensors = self._zmorp.inject_weight_errors(
            model, ber=ber, seed=seed, inject_error_fn=inject_error_to_tensor
        )
        return {"ber": ber, "seed": seed, "rs": rs_stats, "zmorp_weight_tensors": zmorp_tensors}

    def decode(self, model: torch.nn.Module, decode_mode: str = "auto"):
        """解码模型：RS 译码敏感层（含指数最高位置零），再 ZMORP 恢复。

        ``decode_mode``: ``"auto"``（按脏码字比例自动 tracked/full）、
        ``"tracked"``（只译注错记录的脏码字）、``"full"``（全量扫描）。
        """
        if not self._protected:
            raise RuntimeError("protect(model) must run before decode")
        if decode_mode not in ("auto", "tracked", "full"):
            raise ValueError("decode_mode must be auto, tracked, or full")
        rs_stats = self._rs.decode(
            model,
            decode_all={"auto": None, "tracked": False, "full": True}[decode_mode],
            failure_policy="systematic",
        )
        zmorp_stats = self._zmorp.recover_model(model)
        self._protected = False
        return {"rs": rs_stats, "zmorp": zmorp_stats}


if __name__ == "__main__":
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.Linear(128, 32)
    )
    clean = {k: v.clone() for k, v in model.state_dict().items()}

    # 敏感层 = 第一个 Linear 的 weight（前缀 "0."），其余参数走 ZMORP。
    vhps = VHPSSmall(layer_prefixes=["0."])
    stats = vhps.protect(model)
    print("protect:", {"zmorp_tensors": stats["zmorp"].parameter_tensors,
                       "rs_tensors": stats["rs"]["parameter_tensors"],
                       "rs_codewords": stats["rs"]["codewords"]})
    inject_stats = vhps.inject(model, ber=1e-4, seed=7)
    print("inject: dirty_codewords =", inject_stats["rs"]["dirty_codewords"])
    decode_stats = vhps.decode(model)
    rs = decode_stats["rs"]
    print("decode:", {k: rs[k] for k in ("corrected_codewords", "uncorrectable_codewords")})

    # 敏感层（RS 组）== high23 截断期望：低 9 位与 bit 30 为 0。
    reference = clean["0.weight"]
    raw = reference.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    expected_rs = ((raw >> 9) << 9) & 0xBFFFFFFF
    expected_rs = (
        expected_rs - ((expected_rs >> 31) << 32)
    ).to(torch.int32).view(torch.float32)
    assert torch.equal(model.state_dict()["0.weight"], expected_rs), "RS group"

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

    # 敏感层 bias：被 ZMORP 前缀排除、又不是 module.weight，与 SHPS 语义
    # 一致——既不保护也不注错，保持原值。
    assert torch.equal(model.state_dict()["0.bias"], clean["0.bias"])

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(4, 64))
    assert torch.isfinite(output).all()
    print("VHPSSmall interface self-test passed.")
