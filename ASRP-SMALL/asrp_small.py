"""ASRP-SMALL 对外接口：只输入模型即可完成 RS(48,23) 保护、注错与解码。

用法::

    import sys, torch
    sys.path.insert(0, "exp/eval_scripts/ASRP-SMALL")
    from asrp_small import ASRPSmall

    model = ...                          # 任意 float32 torch 模型（已在目标设备上）
    asrp = ASRPSmall()                   # 设备默认取模型参数所在设备
    asrp.protect(model)                  # 编码：参数替换为 int32 码字容器
    asrp.inject(model, ber=1e-5, seed=100)  # 注错：稠密 Bernoulli 位翻转
    stats = asrp.decode(model)           # 译码：恢复 FP32 参数（含统计字典）
    # 之后可直接用 model 推理

本文件只是薄封装：全部编译码逻辑来自同目录 ``rs4823_codec.py`` 与
``rs4823_high23_codec.py``（即 ``fault_tolerance_tools/codecs/`` 下两个
文件的原样副本），注错器 ``inject_error_to_tensor`` 与
``fault_tolerance_tools/injectors.py`` 语义一致。
"""

from __future__ import annotations

from typing import Sequence

import torch

from rs4823_high23_codec import RS4823High23ModelProtector

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
    全局 torch RNG（CUDA 张量同时重置 CUDA RNG）。int32 码字容器按
    整数位翻转处理，不做 dtype 转换。
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


class ASRPSmall:
    """RS(48,23) high23 保护的对外接口。

    Parameters
    ----------
    device:
        编解码设备；``None`` 时在 ``protect`` 阶段取模型首个参数的设备。
    weights_only:
        ``True`` 只保护 ``module.weight``；``False`` 保护所有非空 FP32
        参数（FPS-U2Net/Point2RBox 全模型实验使用后者）。
    exclude_name_fragments:
        参数名包含任一片段时跳过。
    protector_kwargs:
        透传 ``RS4823High23ModelProtector`` 的其余参数（如
        ``encode_chunk_codewords``、``decode_chunk_codewords``、
        ``injection_chunk_words``）。
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        weights_only: bool = False,
        exclude_name_fragments: Sequence[str] = ("mask", "classwise"),
        **protector_kwargs,
    ) -> None:
        self.device = device
        self.weights_only = bool(weights_only)
        self.exclude_name_fragments = tuple(exclude_name_fragments)
        self._protector_kwargs = dict(protector_kwargs)
        self._impl: RS4823High23ModelProtector | None = None

    def protect(self, model: torch.nn.Module):
        """编码模型：受保护参数替换为 int32 RS 码字容器。"""
        device = self.device
        if device is None:
            device = next(model.parameters()).device
        self._impl = RS4823High23ModelProtector(
            device,
            weights_only=self.weights_only,
            exclude_name_fragments=self.exclude_name_fragments,
            show_progress=False,
            **self._protector_kwargs,
        )
        return self._impl.encode(model)

    def inject(self, model: torch.nn.Module, ber: float = 1e-5, seed: int = 100):
        """对全部码字容器注入稠密 Bernoulli 位翻转。"""
        if self._impl is None:
            raise RuntimeError("protect(model) must run before inject")
        del model
        return self._impl.inject_errors(
            ber=ber,
            seed=seed,
            inject_error_fn=inject_error_to_tensor,
        )

    def decode(self, model: torch.nn.Module, clear_exponent_msb: bool = True):
        """译码模型：纠正字节错并恢复 FP32 参数，返回统计字典。

        纠不动的码字按 systematic 策略保留收到的负载字节；恢复值恒有
        低 9 位为 0，``clear_exponent_msb=True`` 时指数最高位 bit 30
        也置 0。
        """
        if self._impl is None:
            raise RuntimeError("protect(model) must run before decode")
        return self._impl.decode(model, clear_exponent_msb=clear_exponent_msb)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.Conv2d(4, 2, 1)
    )
    clean = {k: v.clone() for k, v in model.state_dict().items()}

    asrp = ASRPSmall()
    encode_stats = asrp.protect(model)
    inject_stats = asrp.inject(model, ber=1e-5, seed=100)
    decode_stats = asrp.decode(model)
    print("protect:", {k: encode_stats[k] for k in ("parameter_tensors", "codewords")})
    print("inject :", inject_stats["ber"], inject_stats["seed"])
    print("decode :", decode_stats)

    # 恢复后每个受保护参数等于 high23 截断期望：低 9 位与 bit 30 为 0。
    checked = 0
    for name, value in model.state_dict().items():
        reference = clean[name]
        if reference.dtype != torch.float32 or value.dtype != torch.float32:
            continue
        if not torch.equal(value.view(torch.int32), reference.view(torch.int32)):
            continue  # 未受保护或未注错的参数原样保留即可
        raw = reference.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        expected = ((raw >> 9) << 9) & 0xBFFFFFFF
        expected = (
            expected - ((expected >> 31) << 32)
        ).to(torch.int32).view(torch.float32)
        assert torch.equal(value, expected), name
        checked += 1
    print(f"recovered parameters matching high23 expectation: {checked}")
    assert decode_stats["uncorrectable_codewords"] == 0
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 16, 16))
    assert torch.isfinite(output).all()
    print("ASRPSmall interface self-test passed.")
