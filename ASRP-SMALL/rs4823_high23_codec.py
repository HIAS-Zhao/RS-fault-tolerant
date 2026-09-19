"""Model lifecycle for byte-symbol RS(48,23) FP32 high-bit protection."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from tqdm import tqdm

from rs4823_codec import (
    DECODER_IMPLEMENTATION,
    FLOATS_PER_CODEWORD,
    PACKED_BYTES,
    RS4823Codec,
    RSDecodeStats,
)


@dataclass
class EncodedHigh23Parameter:
    name: str
    parameter: torch.nn.Parameter
    original_shape: tuple[int, ...]
    original_elements: int
    original_requires_grad: bool
    codewords: int
    storage_bytes: int
    is_module_weight: bool
    error_symbol_counts: torch.Tensor


class RS4823High23ModelProtector:
    """Encode, inject, decode, and restore selected FP32 parameters.

    The implementation uses byte symbols over GF(256): every eight FP32
    high-23 fields form one 23-byte information payload and one 48-byte
    codeword. ``decoder_backend`` remains accepted so older entry points do
    not break, but both legacy values select the new PyTorch GF(256) decoder.
    """

    def __init__(
        self,
        device: torch.device | str,
        encode_chunk_codewords: int = 16_384,
        decode_chunk_codewords: int = 16_384,
        injection_chunk_words: int = 262_144,
        high_density_decode_chunk_codewords: int = 65_536,
        decoder_backend: str | None = None,
        weights_only: bool = True,
        exclude_name_fragments: Sequence[str] = (),
        block_chunk_size: int | None = None,
        tracked_decode_max_dirty_fraction: float = 0.5,
        show_progress: bool = True,
        **_ignored,
    ):
        self.device = torch.device(device)
        if block_chunk_size is not None:
            encode_chunk_codewords = block_chunk_size
            decode_chunk_codewords = block_chunk_size
        self.encode_chunk_codewords = self._positive_int(
            encode_chunk_codewords, "encode_chunk_codewords"
        )
        self.decode_chunk_codewords = self._positive_int(
            decode_chunk_codewords, "decode_chunk_codewords"
        )
        self.high_density_decode_chunk_codewords = self._positive_int(
            high_density_decode_chunk_codewords,
            "high_density_decode_chunk_codewords",
        )
        self.injection_chunk_words = self._positive_int(
            injection_chunk_words, "injection_chunk_words"
        )
        if not 0.0 <= tracked_decode_max_dirty_fraction <= 1.0:
            raise ValueError(
                "tracked_decode_max_dirty_fraction must be in [0,1]"
            )
        self.tracked_decode_max_dirty_fraction = float(
            tracked_decode_max_dirty_fraction
        )
        self.show_progress = bool(show_progress)
        requested_backend = (decoder_backend or "gf256-batched").lower()
        if requested_backend not in {
            "batched",
            "fused",
            "byte",
            "gf256",
            "gf256-batched",
            "fast",
            "gf256-fast",
            "gf256-lut-forney",
        }:
            raise ValueError(
                "decoder_backend must be batched, fused, byte, gf256, "
                "gf256-batched, fast, gf256-fast, or gf256-lut-forney"
            )
        self.requested_decoder_backend = requested_backend
        self.decoder_backend = "gf256-batched"
        if requested_backend == "fused":
            warnings.warn(
                "The new GF(256) RS(48,23) decoder has no fused/Triton "
                "backend; 'fused' is a compatibility alias for gf256-batched.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.weights_only = bool(weights_only)
        self.exclude_name_fragments = tuple(exclude_name_fragments)
        self.codec = RS4823Codec(
            self.device, block_chunk_size=self.decode_chunk_codewords
        )
        self.device = self.codec.device
        self.parameters: list[EncodedHigh23Parameter] = []
        self._encoded = False
        self._tracked_dirty_codewords = 0

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _weight_parameter_ids(model: torch.nn.Module) -> set[int]:
        return {
            id(weight)
            for module in model.modules()
            for weight in (getattr(module, "weight", None),)
            if isinstance(weight, torch.nn.Parameter)
        }

    def selected_parameters(
        self, model: torch.nn.Module
    ) -> Iterable[tuple[str, torch.nn.Parameter, bool]]:
        weight_ids = self._weight_parameter_ids(model)
        for name, parameter in model.named_parameters():
            if parameter.dtype != torch.float32 or parameter.numel() == 0:
                continue
            if any(fragment in name for fragment in self.exclude_name_fragments):
                continue
            is_weight = id(parameter) in weight_ids
            if self.weights_only and not is_weight:
                continue
            yield name, parameter, is_weight

    @torch.no_grad()
    def encode(self, model: torch.nn.Module, parameter_filter=None) -> dict:
        if self._encoded:
            raise RuntimeError("RS(48,23) parameters are already encoded")
        selected = [
            item
            for item in self.selected_parameters(model)
            if parameter_filter is None or parameter_filter(item[0], item[1])
        ]
        totals = {
            "parameter_tensors": 0,
            "float32_elements": 0,
            "codewords": 0,
            "information_bits": 0,
            "encoded_bytes": 0,
            "storage_bytes": 0,
            "symbol_field": "GF(256)",
            "symbol_bits": 8,
            "floats_per_codeword": FLOATS_PER_CODEWORD,
        }
        for name, parameter, is_weight in tqdm(
            selected,
            desc="RS(48,23) GF(256) high23 encoding",
            disable=not self.show_progress,
        ):
            if parameter.device != self.device:
                raise ValueError(
                    f"parameter {name!r} is on {parameter.device}, but protector "
                    f"is on {self.device}"
                )
            elements = parameter.numel()
            encoded = self.codec.encode_float32(
                parameter, block_chunk_size=self.encode_chunk_codewords
            )
            codewords = (elements + FLOATS_PER_CODEWORD - 1) // FLOATS_PER_CODEWORD
            storage_bytes = encoded.numel() * encoded.element_size()
            metadata = EncodedHigh23Parameter(
                name=name,
                parameter=parameter,
                original_shape=tuple(parameter.shape),
                original_elements=elements,
                original_requires_grad=parameter.requires_grad,
                codewords=codewords,
                storage_bytes=storage_bytes,
                is_module_weight=is_weight,
                error_symbol_counts=torch.zeros(
                    codewords, dtype=torch.uint8, device=self.device
                ),
            )
            parameter.requires_grad_(False)
            parameter.data = encoded
            parameter._rs4823_high23_encoded = True
            self.parameters.append(metadata)
            totals["parameter_tensors"] += 1
            totals["float32_elements"] += elements
            totals["codewords"] += codewords
            totals["information_bits"] += elements * 23
            totals["encoded_bytes"] += storage_bytes
            totals["storage_bytes"] += storage_bytes
        self._encoded = True
        self._tracked_dirty_codewords = 0
        original_bytes = totals["float32_elements"] * 4
        totals["encoded_to_original_float32_ratio"] = (
            totals["storage_bytes"] / original_bytes if original_bytes else None
        )
        totals["encoded_to_information_bit_ratio"] = (
            totals["encoded_bytes"] * 8 / totals["information_bits"]
            if totals["information_bits"]
            else None
        )
        return totals

    @staticmethod
    def _popcount_lut(device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [value.bit_count() for value in range(256)],
            dtype=torch.uint8,
            device=device,
        )

    @torch.no_grad()
    def inject_errors(
        self,
        ber: float,
        seed: int,
        inject_error_fn: Callable[..., torch.Tensor],
        weights_only: bool | None = None,
        reset_seed_per_parameter: bool = True,
    ) -> dict:
        """Inject dense Bernoulli flips into the packed int32 codewords."""

        if not self._encoded:
            raise RuntimeError("RS(48,23) injection called before encode")
        if not 0.0 <= ber <= 1.0:
            raise ValueError(f"BER must be in [0,1], got {ber}")
        inject_weights_only = self.weights_only if weights_only is None else bool(
            weights_only
        )
        popcount = self._popcount_lut(self.device)
        totals = {
            "ber": float(ber),
            "seed": int(seed),
            "injection_function": getattr(
                inject_error_fn, "__name__", type(inject_error_fn).__name__
            ),
            "reset_seed_per_parameter": bool(reset_seed_per_parameter),
            "parameter_tensors": 0,
            "candidate_bits": 0,
            "flipped_bits": 0,
            "dirty_codewords": 0,
            "codewords_over_t": 0,
            "max_error_symbols_per_codeword": 0,
            "error_symbol_unit": "byte",
        }
        first_parameter = True
        for metadata in tqdm(
            self.parameters,
            desc=f"Injecting RS(48,23) GF(256) BER={ber:g}, seed={seed}",
            disable=not self.show_progress,
        ):
            metadata.error_symbol_counts.zero_()
            if inject_weights_only and not metadata.is_module_weight:
                continue
            clean_storage = metadata.parameter.data
            corrupted = inject_error_fn(
                clean_storage,
                error_rate=ber,
                seed=(
                    seed
                    if reset_seed_per_parameter or first_parameter
                    else None
                ),
                chunk_size=self.injection_chunk_words,
            )
            first_parameter = False
            if (
                corrupted.dtype != torch.int32
                or corrupted.device != clean_storage.device
                or corrupted.shape != clean_storage.shape
            ):
                raise RuntimeError(
                    "inject_error_fn must preserve packed int32 dtype, device, and shape"
                )
            if (
                corrupted.untyped_storage().data_ptr()
                == clean_storage.untyped_storage().data_ptr()
            ):
                raise RuntimeError(
                    "inject_error_fn must return out-of-place storage; an in-place "
                    "injector destroys the clean bytes required for the exact dirty mask"
                )
            clean_bytes = clean_storage.view(torch.uint8).reshape(
                metadata.codewords, PACKED_BYTES
            )
            corrupted_bytes = corrupted.view(torch.uint8).reshape(
                metadata.codewords, PACKED_BYTES
            )
            xor_bytes = clean_bytes ^ corrupted_bytes
            counts = (xor_bytes != 0).sum(dim=1).to(torch.uint8)
            metadata.error_symbol_counts.copy_(counts)
            totals["candidate_bits"] += clean_storage.numel() * 32
            totals["flipped_bits"] += int(
                popcount[xor_bytes.long()].sum(dtype=torch.int64).item()
            )
            totals["dirty_codewords"] += int((counts > 0).sum().item())
            totals["codewords_over_t"] += int(
                (counts > self.codec.t).sum().item()
            )
            if counts.numel():
                totals["max_error_symbols_per_codeword"] = max(
                    totals["max_error_symbols_per_codeword"],
                    int(counts.max().item()),
                )
            clean_storage.copy_(corrupted)
            totals["parameter_tensors"] += 1
        self._tracked_dirty_codewords = totals["dirty_codewords"]
        return totals

    @torch.no_grad()
    def decode(
        self,
        model: torch.nn.Module | None = None,
        decode_all: bool | None = None,
        failure_policy: str = "systematic",
        clear_exponent_msb: bool = True,
    ) -> dict:
        """Decode and restore all encoded parameters.

        The default automatic mode uses the exact dirty-codeword mask recorded
        by :meth:`inject_errors` at low dirty fractions and switches to a full,
        contiguous scan when gathering nearly every codeword would cost more.
        Pass ``False`` to force tracked mode or ``True`` to scan all codewords
        when storage may have been mutated outside the protector's injector.
        """

        del model
        if not self._encoded:
            raise RuntimeError("RS(48,23) decode called before encode")
        aggregate = RSDecodeStats.empty()
        total_codewords = sum(item.codewords for item in self.parameters)
        requested_decode_all = decode_all
        dirty_fraction = (
            self._tracked_dirty_codewords / total_codewords
            if total_codewords
            else 0.0
        )
        if decode_all is None:
            decode_all = dirty_fraction > self.tracked_decode_max_dirty_fraction
        active_decode_chunk_codewords = (
            self.high_density_decode_chunk_codewords
            if decode_all
            else self.decode_chunk_codewords
        )
        expected_selected_codewords = (
            total_codewords if decode_all else self._tracked_dirty_codewords
        )
        decode_batch_buffer_codewords = max(
            1,
            min(active_decode_chunk_codewords, expected_selected_codewords),
        )
        selected_codewords = 0
        single_symbol_fast_path_codewords = 0
        staged_bm_codewords = 0
        full_bm_input_codewords = 0
        pending_codewords = torch.empty(
            (decode_batch_buffer_codewords, self.codec.N),
            dtype=torch.uint8,
            device=self.device,
        )
        pending_targets: list[
            tuple[torch.Tensor, slice | torch.Tensor, int]
        ] = []
        pending_size = 0

        def flush() -> None:
            nonlocal aggregate, pending_targets, pending_size
            nonlocal single_symbol_fast_path_codewords
            nonlocal staged_bm_codewords, full_bm_input_codewords
            if pending_size == 0:
                return
            payload, stats = self.codec.decode_bytes(
                pending_codewords[:pending_size], failure_policy=failure_policy
            )
            single_symbol_fast_path_codewords += (
                self.codec.last_single_symbol_fast_path_codewords
            )
            staged_bm_codewords += self.codec.last_staged_bm_codewords
            full_bm_input_codewords += self.codec.last_full_bm_input_codewords
            cursor = 0
            for target, selector, length in pending_targets:
                target[selector, : self.codec.K] = payload[cursor : cursor + length]
                cursor += length
            aggregate += stats
            pending_targets = []
            pending_size = 0

        def queue(
            storage: torch.Tensor,
            selector: slice | torch.Tensor,
            count: int,
        ) -> None:
            nonlocal pending_size
            offset = 0
            while offset < count:
                capacity = decode_batch_buffer_codewords - pending_size
                take = min(capacity, count - offset)
                if isinstance(selector, slice):
                    base = 0 if selector.start is None else selector.start
                    local_selector: slice | torch.Tensor = slice(
                        base + offset, base + offset + take
                    )
                else:
                    local_selector = selector[offset : offset + take]
                destination = pending_codewords[
                    pending_size : pending_size + take
                ]
                if isinstance(local_selector, slice):
                    destination.copy_(storage[local_selector])
                else:
                    torch.index_select(
                        storage, 0, local_selector, out=destination
                    )
                pending_targets.append((storage, local_selector, take))
                pending_size += take
                offset += take
                if pending_size == decode_batch_buffer_codewords:
                    flush()

        for metadata in tqdm(
            self.parameters,
            desc="Collecting RS(48,23) GF(256) codewords",
            disable=not self.show_progress,
        ):
            storage = self.codec._unpack_int32_to_bytes(
                metadata.parameter.data.reshape(
                    metadata.codewords, self.codec.PACKED_INT32_PER_CODEWORD
                )
            )
            if decode_all:
                selected_codewords += metadata.codewords
                queue(storage, slice(0, metadata.codewords), metadata.codewords)
                continue
            dirty_indices = torch.nonzero(
                metadata.error_symbol_counts != 0, as_tuple=False
            ).flatten()
            dirty_count = dirty_indices.numel()
            selected_codewords += dirty_count
            if dirty_count:
                queue(storage, dirty_indices, dirty_count)
        flush()

        if not decode_all:
            skipped = total_codewords - selected_codewords
            aggregate += RSDecodeStats(
                codewords=skipped,
                clean_codewords=skipped,
                corrected_codewords=0,
                uncorrectable_codewords=0,
                corrected_symbols=0,
            )

        for metadata in tqdm(
            self.parameters,
            desc="Restoring RS(48,23) FP32 parameters",
            disable=not self.show_progress,
        ):
            recovered = self.codec.restore_float32_systematic(
                metadata.parameter.data,
                numel=metadata.original_elements,
                shape=metadata.original_shape,
                block_chunk_size=active_decode_chunk_codewords,
                clear_exponent_msb=clear_exponent_msb,
            )
            metadata.parameter.data = recovered
            metadata.parameter.requires_grad_(metadata.original_requires_grad)
            if hasattr(metadata.parameter, "_rs4823_high23_encoded"):
                delattr(metadata.parameter, "_rs4823_high23_encoded")
        self.parameters.clear()
        self._encoded = False
        self._tracked_dirty_codewords = 0
        totals = aggregate.as_dict()
        totals.update(
            {
                "decoder_backend": self.decoder_backend,
                "decoder_implementation": DECODER_IMPLEMENTATION,
                "requested_decoder_backend": self.requested_decoder_backend,
                "failure_policy": failure_policy,
                "decode_mode": (
                    ("auto-full-syndrome" if decode_all else "auto-tracked-dirty")
                    if requested_decode_all is None
                    else ("full-syndrome" if decode_all else "tracked-dirty")
                ),
                "tracked_dirty_fraction": dirty_fraction,
                "tracked_decode_max_dirty_fraction": (
                    self.tracked_decode_max_dirty_fraction
                ),
                "tracked_decode_chunk_codewords": self.decode_chunk_codewords,
                "high_density_decode_chunk_codewords": (
                    self.high_density_decode_chunk_codewords
                ),
                "active_decode_chunk_codewords": active_decode_chunk_codewords,
                "decode_batch_buffer_codewords": (
                    decode_batch_buffer_codewords
                ),
                "vectorized_decode_min_batch": self.codec.vectorized_min_batch,
                "single_symbol_fast_path": (
                    self.codec.enable_single_symbol_fast_path
                ),
                "single_symbol_fast_path_codewords": (
                    single_symbol_fast_path_codewords
                ),
                "adaptive_bm_fast_path": (
                    self.codec.enable_adaptive_bm_fast_path
                ),
                "compact_bm_state": self.codec.enable_compact_bm_state,
                "staged_bm_single_fraction_min": (
                    self.codec.staged_bm_single_fraction_min
                ),
                "staged_bm_codewords": staged_bm_codewords,
                "full_bm_input_codewords": full_bm_input_codewords,
                "vectorized_dense_cuda_path": (
                    self.device.type == "cuda"
                    and active_decode_chunk_codewords
                    >= self.codec.vectorized_min_batch
                ),
                "rs_decoded_codewords": selected_codewords,
                "rs_skipped_clean_codewords": total_codewords - selected_codewords,
                "cross_parameter_batching": True,
                "corrected_symbol_unit": "byte",
                "low_bits_cleared": 9,
                "exponent_msb_cleared": bool(clear_exponent_msb),
            }
        )
        return totals


def storage_statistics(
    model: torch.nn.Module,
    weights_only: bool = True,
    *,
    exclude_name_fragments: Sequence[str] = (),
) -> dict:
    """Return exact storage overhead for the byte-packed RS representation."""

    weight_ids = RS4823High23ModelProtector._weight_parameter_ids(model)
    excluded = tuple(exclude_name_fragments)
    original_bytes = 0
    protected_data_bytes = 0
    encoded_weight_bytes = 0
    protected_parameters = 0
    protected_elements = 0
    codewords = 0
    for name, parameter in model.named_parameters():
        data_bytes = parameter.numel() * parameter.element_size()
        original_bytes += data_bytes
        if parameter.dtype != torch.float32 or parameter.numel() == 0:
            continue
        if any(fragment in name for fragment in excluded):
            continue
        if weights_only and id(parameter) not in weight_ids:
            continue
        parameter_codewords = (
            parameter.numel() + FLOATS_PER_CODEWORD - 1
        ) // FLOATS_PER_CODEWORD
        encoded_bytes = parameter_codewords * PACKED_BYTES
        protected_data_bytes += data_bytes
        encoded_weight_bytes += encoded_bytes
        protected_parameters += 1
        protected_elements += parameter.numel()
        codewords += parameter_codewords
    unprotected_bytes = original_bytes - protected_data_bytes
    protected_model_bytes = unprotected_bytes + encoded_weight_bytes
    information_bits = protected_elements * 23
    return {
        "original_bytes": original_bytes,
        "protected_data_bytes": protected_data_bytes,
        "encoded_weight_bytes": encoded_weight_bytes,
        "encoded_bytes": encoded_weight_bytes,
        "unprotected_bytes": unprotected_bytes,
        "protected_model_bytes": protected_model_bytes,
        "protected_parameters": protected_parameters,
        "protected_elements": protected_elements,
        "codewords": codewords,
        "information_bits": information_bits,
        "symbol_field": "GF(256)",
        "symbol_bits": 8,
        "floats_per_codeword": FLOATS_PER_CODEWORD,
        "bytes_per_codeword": PACKED_BYTES,
        "redundancy_ratio": (
            protected_model_bytes / original_bytes if original_bytes else 1.0
        ),
        "encoded_to_information_bit_ratio": (
            encoded_weight_bytes * 8 / information_bits
            if information_bits
            else None
        ),
    }


__all__ = [
    "EncodedHigh23Parameter",
    "RS4823High23ModelProtector",
    "storage_statistics",
]
