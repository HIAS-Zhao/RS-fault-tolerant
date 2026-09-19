"""RS(24,9) protection for the nine most-significant float16 bits.

Each float16 contributes raw bits 15..7 as one GF(2^9) symbol.  Nine symbols
form the systematic payload of a 24-symbol RS word.  Decoding reconstructs
each float16 as ``recovered_high9 << 7``, so raw bits 6..0 are exactly zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tqdm import tqdm

from rs249_codec import RS249Codec


@dataclass
class EncodedHigh9Parameter:
    name: str
    parameter: torch.nn.Parameter
    original_shape: tuple[int, ...]
    original_elements: int
    original_requires_grad: bool
    codewords: int
    error_symbol_counts: torch.Tensor


class RS249High9ModelProtector:
    def __init__(
        self,
        device: torch.device | str,
        encode_chunk_codewords: int = 1_048_576,
        decode_chunk_codewords: int = 1_048_576,
        injection_chunk_codewords: int = 131_072,
        sparse_injection_max_ber: float = 1e-4,
    ):
        self.device = torch.device(device)
        self.codec = RS249Codec(self.device)
        self.encode_chunk_codewords = int(encode_chunk_codewords)
        self.decode_chunk_codewords = int(decode_chunk_codewords)
        self.injection_chunk_codewords = int(injection_chunk_codewords)
        self.sparse_injection_max_ber = float(sparse_injection_max_ber)
        self.parameters: list[EncodedHigh9Parameter] = []

    @staticmethod
    def selected_parameters(model: torch.nn.Module):
        for name, parameter in model.named_parameters():
            if "mask" in name or "classwise" in name:
                continue
            if parameter.dtype == torch.float16:
                yield name, parameter

    @torch.no_grad()
    def encode(self, model: torch.nn.Module, parameter_filter=None):
        """Encode selected float16 parameters.

        ``parameter_filter`` receives ``(name, parameter)``.  Keeping it as
        ``None`` preserves the original full-model RS(24,9) experiment.
        """
        if self.parameters:
            raise RuntimeError("RS(24,9) parameters are already encoded")
        total_elements = 0
        total_codewords = 0
        selected = [
            (name, parameter)
            for name, parameter in self.selected_parameters(model)
            if parameter_filter is None or parameter_filter(name, parameter)
        ]
        for name, parameter in tqdm(selected, desc="RS(24,9) high9 encoding"):
            elements = parameter.numel()
            codewords = (elements + 8) // 9
            raw = parameter.detach().contiguous().view(torch.int16).reshape(-1)
            packed = torch.empty((codewords, 27), dtype=torch.uint8, device=self.device)
            for start in range(0, codewords, self.encode_chunk_codewords):
                end = min(start + self.encode_chunk_codewords, codewords)
                element_start = start * 9
                element_end = min(end * 9, elements)
                data = torch.zeros((end - start, 9), dtype=torch.int32, device=self.device)
                high9 = ((raw[element_start:element_end].to(torch.int32) & 0xffff) >> 7)
                data.reshape(-1)[: high9.numel()].copy_(high9)
                packed[start:end].copy_(self.codec.pack_blocks(self.codec.encode_blocks(data)))
            metadata = EncodedHigh9Parameter(
                name=name,
                parameter=parameter,
                original_shape=tuple(parameter.shape),
                original_elements=elements,
                original_requires_grad=parameter.requires_grad,
                codewords=codewords,
                error_symbol_counts=torch.zeros(codewords, dtype=torch.uint8, device=self.device),
            )
            parameter.requires_grad_(False)
            parameter.data = packed.reshape(-1)
            parameter._rs249_high9_encoded = True
            self.parameters.append(metadata)
            total_elements += elements
            total_codewords += codewords
        return {
            "parameter_tensors": len(self.parameters),
            "float16_elements": total_elements,
            "codewords": total_codewords,
            "information_bits": total_elements * 9,
            "encoded_bytes": total_codewords * 27,
            "encoded_to_original_float16_ratio": (
                total_codewords * 27 / (total_elements * 2) if total_elements else None
            ),
        }

    @torch.no_grad()
    def _inject_sparse(self, packed, symbol_counts, ber, generator):
        total_bits = packed.numel() * 8
        if total_bits == 0 or ber == 0:
            return 0
        log_survival = math.log1p(-ber)
        last = -1
        flipped = 0
        flat = packed.reshape(-1)
        while True:
            remaining = total_bits - last - 1
            expected = remaining * ber
            sample_count = min(1_048_576, max(1024, int(expected * 1.1) + 128))
            uniform = torch.rand(
                sample_count, dtype=torch.float64, device=self.device, generator=generator
            ).clamp_min_(torch.finfo(torch.float64).tiny)
            gaps = torch.floor(torch.log(uniform) / log_survival).to(torch.int64)
            positions = last + torch.cumsum(gaps + 1, dim=0)
            valid = positions < total_bits
            positions = positions[valid]
            if positions.numel():
                byte_indices = torch.div(positions, 8, rounding_mode="floor")
                bit_indices = positions.remainder(8)
                unique_bytes, inverse = torch.unique_consecutive(
                    byte_indices, return_inverse=True
                )
                byte_masks = torch.zeros(
                    unique_bytes.numel(), dtype=torch.int64, device=self.device
                )
                byte_masks.scatter_add_(
                    0, inverse, torch.ones_like(bit_indices) << bit_indices
                )
                flat[unique_bytes] ^= byte_masks.to(torch.uint8)

                unique_symbols = torch.unique_consecutive(
                    torch.div(positions, 9, rounding_mode="floor")
                )
                word_indices = torch.div(unique_symbols, 24, rounding_mode="floor")
                unique_words, counts = torch.unique_consecutive(
                    word_indices, return_counts=True
                )
                symbol_counts[unique_words] += counts.to(torch.uint8)
            flipped += int(positions.numel())
            if not bool(valid.all()):
                break
            last = int(positions[-1].item())
        return flipped

    @torch.no_grad()
    def _inject_dense(self, packed, symbol_counts, ber, generator):
        flipped = 0
        bit_values = torch.arange(9, dtype=torch.int64, device=self.device)
        for start in range(0, packed.shape[0], self.injection_chunk_codewords):
            end = min(start + self.injection_chunk_codewords, packed.shape[0])
            flips = torch.rand(
                (end - start, 24, 9),
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            ) < ber
            masks = (
                flips.to(torch.int64) << bit_values.view(1, 1, 9)
            ).sum(dim=2).to(torch.int32)
            symbols = self.codec.unpack_blocks(packed[start:end])
            symbols ^= masks
            packed[start:end].copy_(self.codec.pack_blocks(symbols))
            symbol_counts[start:end].copy_((masks != 0).sum(dim=1).to(torch.uint8))
            flipped += int(flips.sum(dtype=torch.int64).item())
        return flipped

    @torch.no_grad()
    def inject_errors(self, ber: float, seed: int):
        if not 0 <= ber <= 1:
            raise ValueError(f"BER must be in [0,1], got {ber}")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        sparse = ber <= self.sparse_injection_max_ber
        flipped_bits = 0
        dirty_codewords = 0
        codewords_over_t = 0
        max_error_symbols = 0
        for metadata in tqdm(
            self.parameters, desc=f"Injecting RS(24,9) BER={ber:g}, seed={seed}"
        ):
            packed = metadata.parameter.data.view(torch.uint8).view(-1, 27)
            metadata.error_symbol_counts.zero_()
            if sparse:
                flipped_bits += self._inject_sparse(
                    packed, metadata.error_symbol_counts, ber, generator
                )
            else:
                flipped_bits += self._inject_dense(
                    packed, metadata.error_symbol_counts, ber, generator
                )
            counts = metadata.error_symbol_counts
            dirty_codewords += int((counts > 0).sum().item())
            codewords_over_t += int((counts > self.codec.t).sum().item())
            if counts.numel():
                max_error_symbols = max(max_error_symbols, int(counts.max().item()))
        return {
            "ber": float(ber),
            "seed": int(seed),
            "method": "geometric sparse Bernoulli" if sparse else "chunked dense Bernoulli",
            "rng_stream": "one continuous CUDA generator across parameters",
            "flipped_bits": flipped_bits,
            "dirty_codewords": dirty_codewords,
            "codewords_over_t": codewords_over_t,
            "max_error_symbols_per_codeword": max_error_symbols,
        }

    @torch.no_grad()
    def decode(self, model=None):
        del model
        if not self.parameters:
            raise RuntimeError("RS(24,9) decode called before encode")
        total_codewords = sum(item.codewords for item in self.parameters)
        dirty_codewords = sum(
            int((item.error_symbol_counts > 0).sum().item()) for item in self.parameters
        )
        totals = {
            "total_codewords": total_codewords,
            "clean_codewords": total_codewords - dirty_codewords,
            "dirty_codewords": dirty_codewords,
            "corrected": 0,
            "uncorrectable": 0,
        }
        pending_packed = []
        pending_targets = []
        pending_size = 0

        def flush():
            nonlocal pending_packed, pending_targets, pending_size
            if pending_size == 0:
                return
            received_packed = (
                pending_packed[0] if len(pending_packed) == 1 else torch.cat(pending_packed)
            )
            received = self.codec.unpack_blocks(received_packed)
            corrected, success = self.codec.decode_blocks_batched(received)
            corrected_packed = self.codec.pack_blocks(corrected)
            successes = int(success.sum().item())
            totals["corrected"] += successes
            totals["uncorrectable"] += pending_size - successes
            cursor = 0
            for target, indices in pending_targets:
                length = indices.numel()
                target[indices] = corrected_packed[cursor : cursor + length]
                cursor += length
            pending_packed = []
            pending_targets = []
            pending_size = 0

        for metadata in tqdm(self.parameters, desc="Collecting/decoding RS(24,9)"):
            words = metadata.parameter.data.view(torch.uint8).view(-1, 27)
            for start in range(0, metadata.codewords, self.decode_chunk_codewords):
                end = min(start + self.decode_chunk_codewords, metadata.codewords)
                local = (metadata.error_symbol_counts[start:end] > 0).nonzero().flatten()
                if local.numel() == 0:
                    continue
                local += start
                offset = 0
                while offset < local.numel():
                    capacity = self.decode_chunk_codewords - pending_size
                    take = min(capacity, local.numel() - offset)
                    indices = local[offset : offset + take]
                    pending_packed.append(words[indices])
                    pending_targets.append((words, indices))
                    pending_size += take
                    offset += take
                    if pending_size == self.decode_chunk_codewords:
                        flush()
        flush()

        for metadata in tqdm(self.parameters, desc="Restoring float16 (low7=0)"):
            words = metadata.parameter.data.view(torch.uint8).view(-1, 27)
            recovered = self.codec.restore_high9(words, metadata.original_elements)
            metadata.parameter.data = recovered.view(metadata.original_shape)
            metadata.parameter.requires_grad_(metadata.original_requires_grad)
            if hasattr(metadata.parameter, "_rs249_high9_encoded"):
                delattr(metadata.parameter, "_rs249_high9_encoded")
        self.parameters.clear()
        return totals


@torch.no_grad()
def zero_float16_low7(model: torch.nn.Module):
    for _, parameter in RS249High9ModelProtector.selected_parameters(model):
        raw = parameter.data.contiguous().view(torch.int16).to(torch.int32) & 0xffff
        parameter.data.copy_((raw & 0xff80).to(torch.int16).view(torch.float16))
