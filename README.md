# VHPS: A Vulnerability-Aware Hybrid Protection Framework for Space-Borne Remote Sensing Models
Official PyTorch implementation of **VHPS**, an algorithm-level fault-tolerance framework that protects the weights of space-borne remote sensing models against radiation-induced bit flips.

> **Paper status:** *VHPS: A Vulnerability-Aware Hybrid Protection Framework for Space-Borne Remote Sensing Models* has been submitted to **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**.  
> **Manuscript:** [Download the submitted PDF](./TGRS_VHPS.pdf)

## Why VHPS?

Neural networks deployed on satellites operate in radiation-prone environments. Even a small number of bit flips in model weights can propagate through the network and cause severe performance degradation across segmentation, super-resolution, object detection, and multimodal understanding tasks.

<p align="center">
  <img src="assets/radiation-impact.png" width="82%" alt="Impact of radiation-induced bit flips without protection">
</p>

VHPS protects a model according to the vulnerability of its modules instead of applying the same level of redundancy everywhere. It combines two complementary mechanisms:

- **ZMORP — Zero-Memory-Overhead Redundancy Protection.** Stores parity and redundant exponent information inside available mantissa bits, enabling exponent-error detection and recovery without increasing the parameter footprint.
- **ASRP — Adaptive Semi-Redundant Protection.** a coding-theoretic approach delivering provable multi-bit error correction with bounded overhead.
- **Vulnerability-aware hybrid protection strategy.** Assigns ASRP to the most vulnerable modules and ZMORP to moderately vulnerable modules, balancing robustness and storage cost.

## Method at a Glance

### ZMORP

ZMORP reuses selected mantissa bits to hold lightweight error-correction information. At recovery time, parity checks detect corruption and the redundant exponent bits restore the protected value.

<p align="center">
  <img src="assets/zmorp-framework.png" width="100%" alt="Zero-Memory-Overhead Redundancy Protection framework">
</p>

### ASRP

ASRP appends parity codes to the original weight representation. When radiation-induced bit flips corrupt the stored codeword, the decoder uses this redundancy to correct the errors and recover the protected parameter.

<p align="center">
  <img src="assets/asrp-frame.png" width="78%" alt="Full Redundancy Protection framework">
</p>

Together with module-level vulnerability analysis, ZMORP and ASRP form the complete VHPS pipeline. Protection and recovery are performed entirely at the algorithm level and do not require hardware self-checking support.

## Qualitative Results

The following examples cover four representative remote sensing tasks under a high bit-error rate. Without protection, corrupted weights lead to missing segmentation regions, degraded reconstruction, incorrect detections, and broken multimodal responses. VHPS substantially restores the original outputs.

<p align="center">
  <img src="assets/qualitative-results.png" width="58%" alt="Qualitative comparison of remote sensing models with and without VHPS">
</p>

## Repository Structure

| File | Description |
| --- | --- |
| `eject_error.py` | Injects random bit errors into model weights for evaluation at a specified bit-error rate (BER). |
| `ZMORP—SMALL/` | Standalone ZMORP implementation for small models. |
| `ZMORP—LARGE/` | Standalone ZMORP implementation for large models. |
| `ASRP-SMALL/` | Standalone ASRP implementation for small models. |
| `ASRP-LARGE/` | Standalone ASRP implementation for large models. |
| `VHPS-SMALL/` | Combined VHPS pipeline for small models. |
| `VHPS-LARGE` | Combined VHPS pipeline for large models. |

Use the `SMALL` implementation for smaller networks and the `LARGE` implementation for models that require the corresponding large-model protection path.

## Requirements

- Python 3.10+
- PyTorch (CUDA is recommended)
- tqdm

Install the runtime dependencies with:

```bash
pip install torch tqdm
```

## Quick Start

### 1. Inject bit errors

Use the fault injector to evaluate an unprotected or protected model under a controlled BER:

```python
from eject_error import inject_error_to_model

inject_error_to_model(model, ber=BER)
```

### 2. Protect and recover with VHPS

Provide the model modules identified as vulnerable. VHPS applies its hybrid protection strategy to those modules:
### 

```python

from vhps_large import VHPSLarge

vhps = VHPSLarge(layer_prefixes=["layers.0."])  
vhps.protect(model)                           
vhps.inject(model, ber=1e-5, seed=100)        
stats = vhps.decode(model)                    

```


### 3. Use ZMORP independently

```python
   
from zmorp_large import ZMORPLarge           

zmorp = ZMORPLarge()                          
zmorp.protect(model)                          
zmorp.inject(model, ber=1e-4, seed=42)
zmorp.recover(model)
```



### 4. Use ASRP independently

### 

```python
            
from asrp_large import ASRPLarge              

asrp = ASRPLarge()                            
asrp.protect(model)                           
asrp.inject(model, ber=1e-5, seed=100)
stats = asrp.decode(model)
```

### 

## Citation

The paper is currently under submission. Citation information will be added after publication. In the meantime, please refer to the [submitted manuscript](./VHPS_TGRS_manuscript.pdf).

## Contact

For questions about the code or paper, please open an issue in this repository.
