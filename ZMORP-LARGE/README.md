# ZMORP-LARGE

ZMORP（零冗余指数复用保护）大模型打包版的方法代码 + 对外接口。目录内
只有方法本身，不依赖 `fault_tolerance_tools`、数据集或任何评测脚本；
核心实现是原桌面 `ZMORP_LARGE.py` 的原样更名副本
`zmorp_large_codec.py`。

## 对外接口：输入模型即可

```python
import sys, torch
sys.path.insert(0, "ZMORP-LARGE")
from zmorp_large import ZMORPLarge

model = ...                                # 任意 float32 torch 模型
zmorp = ZMORPLarge()                       # 可选 excluded_layer_prefixes
zmorp.protect(model)                       # 保护：改写参数位布局
zmorp.inject(model, ber=1e-4, seed=42)     # 注错：稠密 Bernoulli 位翻转
zmorp.recover(model)                       # 解码：校验 + 重建 FP32
# recover 之后直接用 model 推理
```

- `protect(model)` → `ZMORPStats`：保护所有 float32 参数
  （`excluded_layer_prefixes` 除外）；
- `inject(model, ber, seed)` → dict：只对 `module.weight` 型参数注错；
- `recover(model)` → `ZMORPStats`：恢复所有受保护参数。

直接运行自检：`python ZMORP-LARGE/zmorp_large.py`。

## 方法原理

位布局与恢复规则和 ZMORP-SMALL 完全一致：FP32 指数低 7 位（raw bit
29..23，记 e6..e0）连同其偶校验位复制进尾数低 8 位（bit 7 存校验、
bit 6..0 存副本），指数域最高位 raw bit 30（e7）也改存同一个校验位。
代价是权重被轻度量化（丢失尾数低 8 位），换来指数域单比特错的检错/
纠错能力；冗余度为 1.0（不增加任何存储字节）。

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

LARGE 版与 SMALL 版的差别只在实现：protect/recover 先把全部受保护参数
拼接为单个平面张量一次性做位级变换，再按视图重新绑定回各参数，避免
大模型上数百次小 CUDA 启动与主机同步；两份都坏分支用 `torch.where` +
逐位 AND 表达，不引入 `mask.any()` 或布尔索引同步点。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `zmorp_large.py` | 对外接口 `ZMORPLarge`（protect / inject / recover），内置与 `fault_tolerance_tools/injectors.py` 同语义的稠密 Bernoulli 注错器 |
| `zmorp_large_codec.py` | 核心 codec（原 `ZMORP_LARGE.py` 原样更名）：打包版 `OptimizedZMORP`，只依赖 torch |

## 验证记录

2026-09-19 整理完成；按要求未执行。可在任意装有 torch 的机器运行
`python ZMORP-LARGE/zmorp_large.py`（float32 小模型闭环：protect →
inject(BER=1e-4) → recover，断言恢复值等于量化期望）后在此补充记录。
