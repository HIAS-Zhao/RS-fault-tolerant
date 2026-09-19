"""Optimized ZMORP protection for float32 model parameters.

The exponent's low seven bits are copied into mantissa bits 0..6.  The two
copies each have an even-parity bit.  Recovery uses the measured rule for the
case where both copies fail parity:

    same bit -> keep it; different bit -> set it to zero

For a bit pair this truth table is exactly bitwise AND.  Parameters are packed
and processed as one tensor during protection/recovery to avoid hundreds of
small CUDA launches and host synchronizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class ZMORPStats:
    parameter_tensors: int
    elements: int


class OptimizedZMORP:
    """Protect all float32 parameters outside ``excluded_layer_prefixes``."""

    def __init__(self, excluded_layer_prefixes: Sequence[str] = ()) -> None:
        self.excluded_layer_prefixes = tuple(excluded_layer_prefixes)

    def uses_zmorp(self, parameter_name: str) -> bool:
        return not any(
            parameter_name.startswith(prefix)
            for prefix in self.excluded_layer_prefixes
        )

    @staticmethod
    def _parity7(values: torch.Tensor) -> torch.Tensor:
        """Return even-parity check values for the low seven bits."""
        folded = values ^ (values >> 4)
        folded ^= folded >> 2
        folded ^= folded >> 1
        return folded & 1

    @classmethod
    def protect_tensor(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Add the seven-bit redundant exponent representation."""
        if tensor.dtype != torch.float32:
            return tensor

        original_shape = tensor.shape
        bits = tensor.contiguous().view(torch.int32)
        sign_bits = bits & -2147483648
        exponent_low7 = (bits >> 23) & 0x7F
        parity = cls._parity7(exponent_low7)
        mantissa = bits & 0x7FFFFF

        # Mantissa bits 0..6 hold the redundant exponent, bit 7 its parity.
        protected_mantissa = (
            (mantissa & -256) | exponent_low7 | (parity << 7)
        )
        protected_exponent = (parity << 7) | exponent_low7
        protected_bits = (
            sign_bits | (protected_exponent << 23) | protected_mantissa
        )
        return protected_bits.view(torch.float32).view(original_shape)

    @classmethod
    def recover_tensor(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Recover ZMORP data and force the exponent MSB to zero."""
        if tensor.dtype != torch.float32:
            return tensor

        original_shape = tensor.shape
        bits = tensor.contiguous().view(torch.int32)
        sign_bits = bits & -2147483648
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF

        exponent_low7 = exponent & 0x7F
        exponent_parity = (exponent >> 7) & 1
        redundant_low7 = mantissa & 0x7F
        redundant_parity = (mantissa >> 7) & 1

        exponent_ok = exponent_parity == cls._parity7(exponent_low7)
        redundant_ok = redundant_parity == cls._parity7(redundant_low7)
        original_bad_redundant_good = (~exponent_ok) & redundant_ok
        both_bad = (~exponent_ok) & (~redundant_ok)

        # Both bad: equal bits survive and unequal bits become zero.  This is
        # exactly exponent_low7 & redundant_low7, computed without mask.any()
        # or boolean indexing, so no CPU/GPU synchronization is introduced.
        corrected_low7 = torch.where(
            both_bad,
            exponent_low7 & redundant_low7,
            torch.where(
                original_bad_redundant_good,
                redundant_low7,
                exponent_low7,
            ),
        )

        # Bit 30 (exponent e7) is intentionally absent from the rebuilt word.
        clean_mantissa = mantissa & -256
        recovered_bits = (
            sign_bits | ((corrected_low7 & 0x7F) << 23) | clean_mantissa
        )
        return recovered_bits.view(torch.float32).view(original_shape)

    def _parameters(self, model: torch.nn.Module):
        return [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.dtype == torch.float32 and self.uses_zmorp(name)
        ]

    @staticmethod
    def _bind_packed_views(parameters, packed: torch.Tensor) -> ZMORPStats:
        offset = 0
        for _, parameter in parameters:
            end = offset + parameter.numel()
            parameter.data = packed[offset:end].view(parameter.shape)
            offset = end
        return ZMORPStats(len(parameters), offset)

    @torch.no_grad()
    def protect_model(self, model: torch.nn.Module) -> ZMORPStats:
        """Pack and protect the configured model parameter group."""
        parameters = self._parameters(model)
        if not parameters:
            return ZMORPStats(0, 0)
        packed = torch.cat(
            [parameter.data.reshape(-1) for _, parameter in parameters]
        )
        protected = self.protect_tensor(packed)
        return self._bind_packed_views(parameters, protected)

    @torch.no_grad()
    def recover_model(self, model: torch.nn.Module) -> ZMORPStats:
        """Pack and recover the configured group with no per-tensor sync."""
        parameters = self._parameters(model)
        if not parameters:
            return ZMORPStats(0, 0)
        packed = torch.cat(
            [parameter.data.reshape(-1) for _, parameter in parameters]
        )
        recovered = self.recover_tensor(packed)
        return self._bind_packed_views(parameters, recovered)

    @torch.no_grad()
    def inject_weight_errors(
        self,
        model: torch.nn.Module,
        ber: float,
        seed: int,
        inject_error_fn: Callable,
    ) -> int:
        """Inject only non-RS module weights with the original injector."""
        injected = 0
        for module_name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.nn.Parameter):
                continue
            parameter_name = (
                f"{module_name}.weight" if module_name else "weight"
            )
            if not self.uses_zmorp(parameter_name):
                continue
            corrupted = inject_error_fn(
                weight.data,
                error_rate=ber,
                seed=seed,
            )
            weight.data.copy_(corrupted)
            injected += 1
        return injected
