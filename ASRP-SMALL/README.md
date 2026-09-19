# ASRP-SMALL（RS(48,23) high23）

RS(48,23) 字节符号保护的方法代码 + 对外接口。目录内只有方法本身，
不依赖 `fault_tolerance_tools`、数据集或任何评测脚本；核心实现是
`fault_tolerance_tools/codecs/rs4823_codec.py` 与
`rs4823_high23_codec.py` 的原样副本（仅把后者的包内相对导入改为同目录
导入）。

## 对外接口：输入模型即可

```python
import sys, torch
sys.path.insert(0, "exp/eval_scripts/ASRP-SMALL")
from asrp_small import ASRPSmall

model = ...                                   # 任意 float32 模型（在目标设备上）
asrp = ASRPSmall()                            # 设备默认取模型参数所在设备
asrp.protect(model)                           # 编码：参数替换为 int32 码字容器
asrp.inject(model, ber=1e-5, seed=100)        # 注错：稠密 Bernoulli 位翻转
stats = asrp.decode(model)                    # 译码：恢复 FP32 参数
# decode 之后直接用 model 推理
```

- `protect(model)` → 编码统计（保护参数数、码字数、冗余度等）；
  默认 `weights_only=False` 保护所有非空 FP32 参数（BN 的
  `num_batches_tracked` 等非 FP32 参数自动跳过），可通过
  `exclude_name_fragments` 排除（默认 `("mask", "classwise")`）；
- `inject(model, ber, seed)` → 注错统计（脏码字数、翻转位数等）；
- `decode(model)` → 译码统计字典：`corrected_codewords` /
  `uncorrectable_codewords` / `corrected_symbols` 等。

直接运行自检：`python exp/eval_scripts/ASRP-SMALL/asrp_small.py`。

## 方法原理

```text
8 个 FP32 的 raw bit 31..9（符号+指数+尾数高 14 位，各 23 bit）
  -> MSB-first 拼成 184 bit = 23 个负载字节
  -> GF(256) 系统式 RS(48,23)：本原多项式 0x11D、生成元 2、
     连续根 alpha^0..alpha^24、25 个校验字节
  -> 48 字节码字按小端装入 12 个 int32，交给稠密 Bernoulli 注错器
```

- 每个码字最多纠正 **12 个错误字节符号**；
- 译码严格执行：伴随式 → Berlekamp–Massey → Chien 搜索 → Forney →
  错误修正，并做全 25 项 syndrome 后验校验；纠不动的码字按
  `failure_policy="systematic"` 保留收到的负载字节；
- 恢复的 FP32 恒满足 raw bit 8..0 = 0（低 9 位编码时未存储）且指数
  最高位 bit 30 = 0；
- 每 8 个 FP32（32 字节）编码后占 48 字节，**编码容器/原始 FP32 =
  1.5 倍**。

当前译码器标识 `gf256-lut-forney-v6-adaptive-bm`（LUT 化 GF 乘法与
syndrome、FCR=0 Forney、分层 BM、tracked 脏码字译码），编码布局与
纠错语义对所有加速版本不变。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `asrp_small.py` | 对外接口 `ASRPSmall`（protect / inject / decode），内置与 `fault_tolerance_tools/injectors.py` 同语义的稠密 Bernoulli 注错器 |
| `rs4823_codec.py` | 核心 codec（原样副本）：GF(256) 编码、伴随式/BM/Chien/Forney 译码、FP32 high23 打包解包，只依赖 torch/tqdm |
| `rs4823_high23_codec.py` | 模型级 protector（原样副本）：`encode(model) / inject_errors / decode(model)` 全生命周期与 tracked 译码 |

需要完整 FPS-U2Net 评测入口（BER × seed 扫描、指标、断点续跑）时，
使用原始脚本 `exp/eval_scripts/FPSU2Net_rs4823_high23.py` 与
`FPSU2Net_rs4823_high23_latency.py`。

## 验证记录（2026-09-19）

- `rs4823_codec.py`：4096 个随机 FP32 编码后注入 200 个随机位翻转再
  译码：166 个脏码字全部纠正、0 个不可纠、恢复值与 high23 截断期望
  （low9=0、bit30=0）逐元素一致。
- `asrp_small.py` 接口：8192 权重的 Linear 模型 protect →
  inject(BER=1e-3) → decode 闭环：1556 个码字中 533 个脏码字全部
  纠正、0 个不可纠、权重与期望逐位相等。
