# RS-fault-tolerant

This repository contains the PyTorch implementations accompanying **VHPS**, a hardware fault-tolerance framework for protecting neural network model weights against bit errors.

## Overview

**VHPS** combines two complementary protection components:

- **ZMORP** — Zero-Memory-Overhead Redundancy Protection. It encodes a redundant copy of the exponent's low bits along with parity checks into unused mantissa bits, enabling bit exponent error detection and correction without expanding the parameter footprint.
- **FRP** — Full Redundancy Protection. A coding-theoretic method that encodes each weight value into a longer codeword capable of correcting up to **3 bit errors** per codeword.

The two schemes are applied to different layers of the same model, forming the complete VHPS protection pipeline.

## File Structure

| File | Description |
|------|-------------|
| `eject_error.py` | Bit error injection utility for evaluation. Simulates hardware BER (Bit Error Rate) by randomly flipping bits in model weights. |
| `zmorp_little_model.py` | ZMORP standalone implementation for **small** models. |
| `zmorp_large_model.py` | ZMORP standalone implementation for **large** models. |
| `frp_little_model.py` | FRP standalone implementation for **small** models.. |
| `frp_large_model.py` | FRP standalone implementation for **large** models.. |
| `vhps_little_model.py` | Combined VHPS for **small** models (ZMORP + FRP applied to different layers). |
| `vhps_large_model.py` | Combined VHPS for **large** models (ZMORP + FRP applied to different layers). |





## Usage


### Error Injection (`eject_error.py`)

Used to benchmark protection schemes under simulated hardware faults:

```python
from eject_error import inject_error_to_model

inject_error_to_model(model, ber=BER, seed=SEEDS)
```


### VHPS

```python
from vhps_little_model import protect, recover  


protect(model, layer=["YOUR_VULNERABILITY_LAYERS"], device="cuda")


recover(model, layer=["YOUR_VULNERABILITY_LAYERS"], device="cuda")
```

### ZMORP

```python
from zmorp_little_model import ZMORP

ZMORP.protect_model(model)

ZMORP.recover_model(model)
```

### Standalone FRP

```python
from frp_little_model import FRP 

frp = FRP(device="cuda")
frp.encode(model)
frp.decode(model)
```

## Requirements

- Python 3.10+
- PyTorch (with CUDA support recommended)
- tqdm
