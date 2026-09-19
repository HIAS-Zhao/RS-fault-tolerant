# ZMORP-SMALL

ZMORP（零冗余指数复用保护）方法代码 + 对外接口。目录内只有方法本身，
不依赖 `fault_tolerance_tools`、数据集或任何评测脚本；核心实现是
`fault_tolerance_tools/codecs/zmorp.py` 的原样副本。

## 对外接口：输入模型即可

```python
import sys, torch
sys.path.insert(0, "exp/eval_scripts/ZMORP-SMALL")
from zmorp_small import ZMORPSmall

model = ...                                # 任意 float32 torch 模型
zmorp = ZMORPSmall()                       # 可选 excluded_layer_prefixes
zmorp.protect(model)                       # 保护：改写参数位布局
zmorp.inject(model, ber=1e-4, seed=42)     # 注错：稠密 Bernoulli 位翻转
zmorp.recover(model)                       # 解码：校验 + 重建 FP32
# recover 之后直接用 model 推理
```

- `protect(model)` → `ZMORPStats`：保护所有 float32 参数（`excluded_layer_prefixes` 除外）；
- `inject(model, ber, seed)` → dict：只对 `module.weight` 型参数注错；
- `recover(model)` → `ZMORPStats`：恢复所有受保护参数。

直接运行自检：`python exp/eval_scripts/ZMORP-SMALL/zmorp_small.py`。

## 方法原理

ZMORP 不增加任何存储字节（冗余度 1.0）。它把 FP32 指数低 7 位
（raw bit 29..23，记 e6..e0）连同其偶校验位复制进尾数低 8 位（raw
bit 7..0），指数域最高位 raw bit 30（e7）也改存同一个校验位。代价是
权重被轻度量化（丢失尾数低 8 位），换来指数域单比特错的检错/纠错能力。

protect 之后的单个 FP32 位布局：

```text
bit 31 | bit 30         | bit 29..23 | bit 22..8 | bit 7         | bit 6..0
符号   | parity(e6..e0) | e6..e0     | 尾数高15位 | parity(e6..e0) | e6..e0 副本
       \____________________ 指数域 _______________________________/
```

recover 规则（两份 (e6..e0, parity) 各自做偶校验）：

- 原份通过校验 → 使用原份；
- 原份坏、副本好 → 使用副本（纠正 1 bit 指数错）；
- 两份都坏 → 逐位 AND：相同位保留、不同位置 0；
- 重建时丢弃尾数低 8 位，并强制 e7（bit 30）为 0，杜绝 NaN/Inf 级别的
  指数灾难值。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `zmorp_small.py` | 对外接口 `ZMORPSmall`（protect / inject / recover），内置与 `fault_tolerance_tools/injectors.py` 同语义的稠密 Bernoulli 注错器 |
| `zmorp_codec.py` | 核心 codec（原样副本）：`OptimizedZMORP`（FP32）与 `OptimizedFloat16ZMORP`（LHRS 布局），只依赖 torch |

需要完整 FPS-U2Net 评测入口（BER × seed 扫描、指标、断点续跑）时，
使用原始脚本 `exp/eval_scripts/zmorp.py` 与 `zmorp_latency.py`。

## 验证记录（2026-09-19）

- `zmorp_codec.py`：FP32 无错往返 == 量化期望值（尾数低 8 位清零、
  bit 30 清零）；对 bit 0..7 与 23..31 的全部单比特翻转，指数域
  （含 bit 30）全部正确恢复。
- `zmorp_small.py` 接口：Linear 模型 protect → inject(BER=1e-3) →
  recover 闭环正常，恢复值等于量化期望（多比特错误导致的少量偏离为
  方法固有行为）。
