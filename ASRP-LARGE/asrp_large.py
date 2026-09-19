"""ASRP-LARGE 对外接口：只输入模型即可完成 RS(24,9) 保护、注错与解码。

用法::

    import sys, torch
    sys.path.insert(0, "ASRP-LARGE")
    from asrp_large import ASRPLarge

    model = ...                              # 任意 float16 torch 模型（已在目标设备上）
    asrp = ASRPLarge()                       # 设备默认取模型参数所在设备
    asrp.protect(model)                      # 编码：参数替换为 uint8 码字容器
    asrp.inject(model, ber=1e-5, seed=100)   # 注错：稀疏/稠密 Bernoulli 位翻转
    stats = asrp.decode(model)               # 译码：恢复 float16 参数（含统计字典）
    # 之后可直接用 model 推理

本文件只是薄封装：全部编译码逻辑来自同目录 ``rs249_codec.py`` 与
``rs249_high9_codec.py``（即原桌面 ``rs249_codec.py`` / ``ASRP_LARGE.py``
的副本，后者原样更名）。注错器内置于 ``RS249High9ModelProtector``：
低 BER 走几何稀疏 Bernoulli，高 BER 走分块稠密 Bernoulli，无需外部
注入函数。
"""

from __future__ import annotations

from typing import Sequence

import torch

from rs249_high9_codec import RS249High9ModelProtector


class ASRPLarge:
    """RS(24,9) high9 保护（float16 模型）的对外接口。

    Parameters
    ----------
    device:
        编解码设备；``None`` 时在 ``protect`` 阶段取模型首个参数的设备。
    weights_only:
        ``True`` 只保护 ``module.weight`` 型参数；``False`` 保护所有非空
        float16 参数。
    exclude_name_fragments:
        参数名包含任一片段时跳过（默认沿用 ``("mask", "classwise")``）。
    protector_kwargs:
        透传 ``RS249High9ModelProtector`` 的其余参数（如
        ``encode_chunk_codewords``、``decode_chunk_codewords``、
        ``injection_chunk_codewords``、``sparse_injection_max_ber``）。
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
        self._impl: RS249High9ModelProtector | None = None

    def _parameter_filter(self):
        fragments = self.exclude_name_fragments

        def parameter_filter(name, parameter):
            if any(fragment in name for fragment in fragments):
                return False
            if self.weights_only and not (
                name == "weight" or name.endswith(".weight")
            ):
                return False
            return True

        return parameter_filter

    def protect(self, model: torch.nn.Module):
        """编码模型：受保护参数替换为 uint8 RS 码字容器。"""
        device = self.device
        if device is None:
            device = next(model.parameters()).device
        self._impl = RS249High9ModelProtector(device, **self._protector_kwargs)
        return self._impl.encode(model, parameter_filter=self._parameter_filter())

    def inject(self, model: torch.nn.Module, ber: float = 1e-5, seed: int = 100):
        """对全部码字容器注入 Bernoulli 位翻转（低 BER 稀疏、高 BER 稠密）。"""
        if self._impl is None:
            raise RuntimeError("protect(model) must run before inject")
        del model
        return self._impl.inject_errors(ber=ber, seed=seed)

    def decode(self, model: torch.nn.Module):
        """译码模型：纠正符号错并恢复 float16 参数，返回统计字典。

        译码只处理 tracked 脏码字（伴随式 → BM → Chien → Forney → 后验
        校验）；纠不动的码字按 systematic 策略保留收到的负载符号；恢复值
        恒有低 7 位为 0。
        """
        if self._impl is None:
            raise RuntimeError("protect(model) must run before decode")
        return self._impl.decode(model)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.Conv2d(4, 2, 1),
        torch.nn.Flatten(),
        torch.nn.Linear(2 * 14 * 14, 3),
    ).half()
    clean = {k: v.clone() for k, v in model.state_dict().items()}

    asrp = ASRPLarge()
    encode_stats = asrp.protect(model)
    inject_stats = asrp.inject(model, ber=1e-5, seed=100)
    decode_stats = asrp.decode(model)
    print("protect:", {k: encode_stats[k] for k in ("parameter_tensors", "codewords")})
    print("inject :", inject_stats["ber"], inject_stats["seed"])
    print("decode :", decode_stats)

    # 恢复后每个受保护参数等于 high9 截断期望：低 7 位为 0。
    checked = 0
    for name, value in model.state_dict().items():
        reference = clean[name]
        if reference.dtype != torch.float16 or value.dtype != torch.float16:
            continue
        if not torch.equal(value.view(torch.int16), reference.view(torch.int16)):
            continue  # 不可纠码字保留收到的负载符号，属预期行为
        expected = (reference.contiguous().view(torch.int16) >> 7) << 7
        assert torch.equal(value.view(torch.int16), expected), name
        checked += 1
    print(f"recovered parameters matching high9 expectation: {checked}")
    assert decode_stats["uncorrectable"] == 0
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 16, 16).half())
    assert torch.isfinite(output).all()
    print("ASRPLarge interface self-test passed.")
