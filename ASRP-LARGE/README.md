# ASRP-LARGE（RS(24,9) high9，float16 模型）

RS(24,9) 符号保护（float16 模型）的方法代码 + 对外接口。目录内只有
方法本身，不依赖 `fault_tolerance_tools`、数据集或任何评测脚本；核心
实现是原桌面 `rs249_codec.py` 与 `ASRP_LARGE.py` 的副本（后者原样更名
为 `rs249_high9_codec.py`；前者仅给 pack/unpack/restore 增加了无 CUDA
环境下的纯 torch 回退，见下文「运行环境」）。

## 对外接口：输入模型即可

```python
import sys, torch
sys.path.insert(0, "ASRP-LARGE")
from asrp_large import ASRPLarge

model = ...                                   # 任意 float16 模型（在目标设备上）
asrp = ASRPLarge()                            # 设备默认取模型参数所在设备
asrp.protect(model)                           # 编码：参数替换为 uint8 码字容器
asrp.inject(model, ber=1e-5, seed=100)        # 注错：低 BER 稀疏 / 高 BER 稠密 Bernoulli
stats = asrp.decode(model)                    # 译码：恢复 float16 参数
# decode 之后直接用 model 推理
```

- `protect(model)` → 编码统计（保护参数数、码字数、冗余度等）；默认
  `weights_only=False` 保护所有非空 float16 参数（BN 的
  `num_batches_tracked` 等非 float16 参数自动跳过），可通过
  `exclude_name_fragments` 排除（默认 `("mask", "classwise")`）；
- `inject(model, ber, seed)` → 注错统计（脏码字数、翻转位数、超纠错
  能力码字数等）；注错器内置于方法本体：`ber <=
  sparse_injection_max_ber`（默认 1e-4）走几何稀疏 Bernoulli，否则走
  分块稠密 Bernoulli，单个连续生成器跨参数采样；
- `decode(model)` → 译码统计字典：`total_codewords` / `dirty_codewords`
  / `corrected` / `uncorrectable` 等，只译 tracked 脏码字。

直接运行自检：`python ASRP-LARGE/asrp_large.py`。

## 方法原理

```text
9 个 float16 的 raw bit 15..7（符号+指数+尾数高位，各 9 bit）
  -> 每个 9 bit 直接作为一个 GF(2^9) 符号，9 个符号构成系统式负载
  -> GF(2^9)：本原多项式 0x211（x^9 + x^4 + 1）、连续根 alpha^1..alpha^15、
     15 个校验符号，共 24 符号码字
  -> 24 个 9-bit 符号紧打包成 27 字节，交给稀疏/稠密 Bernoulli 注错器
```

- 每个码字最多纠正 **7 个错误符号**（GF(2^9) 符号，不是字节）；
- 译码严格执行：伴随式 → Berlekamp–Massey → Chien 搜索 → Forney →
  错误修正，并做全 15 项 syndrome 后验校验；纠不动的码字按 systematic
  策略保留收到的负载符号；
- 恢复的 float16 恒满足 raw bit 6..0 = 0（低 7 位编码时未存储）；
- 每 9 个 float16（18 字节）编码后占 27 字节，**编码容器/原始 float16 =
  1.5 倍**。

## 运行环境

`rs249_codec.py` 的 pack/unpack/restore 在 **CUDA 设备且同目录存在
`rs249_cuda.cpp` / `rs249_cuda_kernel.cu`** 时走编译扩展（这两个源
文件未随目录提供）；否则自动回退纯 torch 路径（复用文件内置的
`pack_9bit_symbols` / `unpack_9bit_symbols` 与等价的 restore 实现），
结果逐位一致，可在 CPU 上完整自检。依赖仅 torch / tqdm（triton 为
可选导入，缺失时不影响）。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `asrp_large.py` | 对外接口 `ASRPLarge`（protect / inject / decode），注错器内置于方法本体 |
| `rs249_codec.py` | 核心 codec（原桌面文件副本 + 纯 torch 回退）：GF(2^9) 编码、伴随式/BM/Chien/Forney 译码、9-bit 符号打包解包，只依赖 torch（triton 可选） |
| `rs249_high9_codec.py` | 模型级 protector（原 `ASRP_LARGE.py` 原样更名）：`encode(model) / inject_errors / decode(model)` 全生命周期与 tracked 脏码字译码 |

## 验证记录

2026-09-19 整理完成；按要求未执行。可在任意装有 torch/tqdm 的机器
运行 `python ASRP-LARGE/asrp_large.py`（float16 小模型闭环：
protect → inject(BER=1e-5) → decode，断言 0 个不可纠码字、恢复值满足
high9 截断期望）后在此补充记录。
