"""Optimized ZMORP protection for float32 and float16 parameters.

The exponent's low seven bits are copied into mantissa bits 0..6.  The two
copies each have an even-parity bit.  Recovery uses the measured rule for the
case where both copies fail parity:

    same bit -> keep it; different bit -> set it to zero

For a bit pair this truth table is exactly bitwise AND.  Parameters are packed
and processed as one tensor during protection/recovery to avoid hundreds of
small CUDA launches and host synchronizations.

The float16 variant follows the LHRS implementation: all five exponent bits
are copied into mantissa bits 1..5, while mantissa bits 0 and 6 store the
parities of the original and redundant exponent copies respectively.  Mantissa
bits 7..9 are retained.  Unlike the float32 layout, no exponent bit is
repurposed as parity, so the float16 exponent MSB is not forcibly cleared.
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


class OptimizedFloat16ZMORP:
    """Protect non-excluded float16 parameters using the LHRS ZMORP layout.

    Protection and recovery aggregate all selected parameters into one packed
    tensor, matching :class:`OptimizedZMORP`'s low-launch model path.  Fault
    injection remains weight-only and resets the caller-provided injector with
    the same seed for every unique module weight, as in the LHRS runners.
    """

    def __init__(self, excluded_layer_prefixes: Sequence[str] = ()) -> None:
        self.excluded_layer_prefixes = tuple(excluded_layer_prefixes)

    def uses_zmorp(self, parameter_name: str) -> bool:
        return not any(
            parameter_name.startswith(prefix)
            for prefix in self.excluded_layer_prefixes
        )

    @staticmethod
    def _parity5(values: torch.Tensor) -> torch.Tensor:
        """Return the even-parity check value for each five-bit exponent."""
        folded = values ^ (values >> 4)
        folded ^= folded >> 2
        folded ^= folded >> 1
        return folded & 1

    @staticmethod
    def _as_words(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF

    @staticmethod
    def _as_float16(words: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return (words & 0xFFFF).to(torch.int16).view(torch.float16).view(shape)

    @classmethod
    def protect_tensor(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Embed a five-bit exponent copy and the two parity bits."""
        if tensor.dtype != torch.float16:
            return tensor

        original_shape = tensor.shape
        words = cls._as_words(tensor)
        sign_bits = words & 0x8000
        exponent = (words >> 10) & 0x1F
        mantissa = words & 0x03FF
        parity = cls._parity5(exponent)

        # LHRS layout, from low to high: original parity at mantissa bit 0,
        # redundant exponent at bits 1..5, redundant parity at bit 6.  The
        # original mantissa's high three bits (7..9) remain available.
        protection_bits = parity | (exponent << 1) | (parity << 6)
        protected_mantissa = (mantissa & 0x0380) | protection_bits
        protected_words = sign_bits | (exponent << 10) | protected_mantissa
        return cls._as_float16(protected_words, original_shape)

    @classmethod
    def recover_tensor(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Recover the exponent with the original LHRS decision truth table."""
        if tensor.dtype != torch.float16:
            return tensor

        original_shape = tensor.shape
        words = cls._as_words(tensor)
        sign_bits = words & 0x8000
        exponent = (words >> 10) & 0x1F
        mantissa = words & 0x03FF
        redundant_exponent = (mantissa >> 1) & 0x1F
        exponent_parity = mantissa & 1
        redundant_parity = (mantissa >> 6) & 1

        exponent_ok = exponent_parity == cls._parity5(exponent)
        redundant_ok = redundant_parity == cls._parity5(redundant_exponent)
        original_bad_redundant_good = (~exponent_ok) & redundant_ok
        both_bad = (~exponent_ok) & (~redundant_ok)

        # Original truth table:
        #   original bad, redundant good -> redundant copy
        #   both bad -> equal bits retained, differing bits cleared
        # The latter is exactly a bitwise AND over the two exponent copies.
        corrected_exponent = torch.where(
            both_bad,
            exponent & redundant_exponent,
            torch.where(
                original_bad_redundant_good,
                redundant_exponent,
                exponent,
            ),
        )

        clean_mantissa = mantissa & 0x0380
        recovered_words = (
            sign_bits | ((corrected_exponent & 0x1F) << 10) | clean_mantissa
        )
        return cls._as_float16(recovered_words, original_shape)

    def _parameters(self, model: torch.nn.Module):
        return [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.dtype == torch.float16 and self.uses_zmorp(name)
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
        """Pack and protect every selected float16 model parameter."""
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
        """Pack and recover every selected float16 parameter."""
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
        """Inject each unique, selected float16 ``module.weight`` once."""
        injected = 0
        seen: set[int] = set()
        for module_name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.nn.Parameter) or id(weight) in seen:
                continue
            seen.add(id(weight))
            parameter_name = (
                f"{module_name}.weight" if module_name else "weight"
            )
            if weight.dtype != torch.float16 or not self.uses_zmorp(parameter_name):
                continue
            corrupted = inject_error_fn(
                weight.data.clone(),
                error_rate=ber,
                seed=seed,
            )
            weight.data.copy_(corrupted)
            injected += 1
        return injected

    @classmethod
    @torch.no_grad()
    def self_test(cls, device: torch.device | str = "cpu") -> dict[str, int | str]:
        """Check lossless canonical round trips and all protected single faults."""
        device = torch.device(device)
        signs = torch.arange(2, device=device, dtype=torch.int32)
        exponents = torch.arange(32, device=device, dtype=torch.int32)
        mantissa_high = torch.arange(8, device=device, dtype=torch.int32)
        sign_grid, exponent_grid, mantissa_grid = torch.meshgrid(
            signs, exponents, mantissa_high, indexing="ij"
        )
        words = (
            (sign_grid.reshape(-1) << 15)
            | (exponent_grid.reshape(-1) << 10)
            | (mantissa_grid.reshape(-1) << 7)
        )
        canonical = cls._as_float16(words, words.shape)
        protected = cls.protect_tensor(canonical)
        recovered = cls.recover_tensor(protected)
        if not torch.equal(cls._as_words(recovered), words):
            raise AssertionError("float16 ZMORP canonical round trip failed")

        protected_words = cls._as_words(protected)
        protected_positions = tuple(range(7)) + tuple(range(10, 15))
        faulty = protected_words.repeat(len(protected_positions), 1)
        for row, position in enumerate(protected_positions):
            faulty[row] ^= 1 << position
        faulty_values = cls._as_float16(faulty, faulty.shape)
        single_recovered = cls._as_words(cls.recover_tensor(faulty_values))
        expected = words.expand_as(single_recovered)
        if not torch.equal(single_recovered, expected):
            raise AssertionError("float16 ZMORP protected-field single-error test failed")

        # Explicitly exercise the double-parity-failure rule for all pairs of
        # five-bit values.  Invalid parity bits force the both-bad branch.
        original = torch.arange(32, device=device, dtype=torch.int32).repeat_interleave(32)
        redundant = torch.arange(32, device=device, dtype=torch.int32).repeat(32)
        invalid_original_parity = cls._parity5(original) ^ 1
        invalid_redundant_parity = cls._parity5(redundant) ^ 1
        both_bad_words = (
            (original << 10)
            | invalid_original_parity
            | (redundant << 1)
            | (invalid_redundant_parity << 6)
        )
        both_bad_values = cls._as_float16(both_bad_words, both_bad_words.shape)
        both_bad_recovered = (cls._as_words(cls.recover_tensor(both_bad_values)) >> 10) & 0x1F
        if not torch.equal(both_bad_recovered, original & redundant):
            raise AssertionError("float16 ZMORP double-failure truth table failed")

        return {
            "dtype": str(torch.float16),
            "canonical_round_trip_words": int(words.numel()),
            "protected_single_error_cases": int(faulty.numel()),
            "double_failure_truth_table_cases": int(original.numel()),
        }


__all__ = [
    "OptimizedFloat16ZMORP",
    "OptimizedZMORP",
    "ZMORPStats",
]
