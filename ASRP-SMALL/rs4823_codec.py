"""Byte-symbol RS(48,23) codec for packed FP32 high-23-bit payloads.

Eight float32 values contribute their raw bits 31..9, giving exactly 184
payload bits (23 bytes). A shortened systematic Reed-Solomon code over
GF(2**8) appends 25 parity bytes. The resulting 48-byte codeword is stored as
12 int32 words so it can be passed through the dense Bernoulli bit injector.

Decoded values have raw bits 8..0 and exponent bit 30 cleared before they are
reinterpreted as float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


N = 48
K = 23
NSYM = N - K
T = NSYM // 2
FLOATS_PER_CODEWORD = 8
PACKED_INT32_PER_CODEWORD = 12
PACKED_BYTES = N
DECODER_IMPLEMENTATION = "gf256-lut-forney-v6-adaptive-bm"


class _GF256:
    """Tensor-oriented GF(256) using x^8+x^4+x^3+x^2+1."""

    PRIMITIVE_POLYNOMIAL = 0x11D
    GENERATOR = 2
    ORDER = 255

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        exp_values = [0] * (2 * self.ORDER)
        log_values = [0] * 256
        value = 1
        for exponent in range(self.ORDER):
            exp_values[exponent] = value
            log_values[value] = exponent
            value <<= 1
            if value & 0x100:
                value ^= self.PRIMITIVE_POLYNOMIAL
        exp_values[self.ORDER :] = exp_values[: self.ORDER]
        self.exp = torch.tensor(exp_values, dtype=torch.int64, device=self.device)
        self.log = torch.tensor(log_values, dtype=torch.int64, device=self.device)
        values = torch.arange(256, dtype=torch.int64, device=self.device)
        left, right = torch.meshgrid(values, values, indexing="ij")
        multiplication = self.exp[self.log[left] + self.log[right]]
        self.multiplication = torch.where(
            (left == 0) | (right == 0),
            torch.zeros((), dtype=torch.int64, device=self.device),
            multiplication,
        )
        self.inverse = torch.zeros(256, dtype=torch.int64, device=self.device)
        self.inverse[1:] = self.exp[self.ORDER - self.log[1:]]

    def mul(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left, right = torch.broadcast_tensors(
            left.to(torch.int64), right.to(torch.int64)
        )
        return self.multiplication[left, right]

    def div(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        numerator, denominator = torch.broadcast_tensors(
            numerator.to(torch.int64), denominator.to(torch.int64)
        )
        if bool((denominator == 0).any()):
            raise ZeroDivisionError("division by zero in GF(256)")
        return self.div_unchecked(numerator, denominator)

    def div_unchecked(
        self, numerator: torch.Tensor, denominator: torch.Tensor
    ) -> torch.Tensor:
        """GF division when the caller already guarantees nonzero divisors."""

        numerator, denominator = torch.broadcast_tensors(
            numerator.to(torch.int64), denominator.to(torch.int64)
        )
        return self.multiplication[numerator, self.inverse[denominator]]

    def inv(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(torch.int64)
        if bool((value == 0).any()):
            raise ZeroDivisionError("inverse of zero in GF(256)")
        return self.inverse[value]

    def inv_unchecked(self, value: torch.Tensor) -> torch.Tensor:
        """GF inverse when the caller already guarantees nonzero values."""

        return self.inverse[value.to(torch.int64)]


def _gf_mul_scalar(left: int, right: int) -> int:
    result = 0
    for _ in range(8):
        if right & 1:
            result ^= left
        right >>= 1
        left <<= 1
        if left & 0x100:
            left ^= _GF256.PRIMITIVE_POLYNOMIAL
    return result & 0xFF


def _gf_pow_scalar(value: int, exponent: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = _gf_mul_scalar(result, value)
        value = _gf_mul_scalar(value, value)
        exponent >>= 1
    return result


def _poly_mul_scalar(left: Sequence[int], right: Sequence[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] ^= _gf_mul_scalar(
                left_value, right_value
            )
    return result


def _xor_reduce_last(values: torch.Tensor) -> torch.Tensor:
    """XOR-reduce the final dimension with a balanced reduction tree."""

    if values.shape[-1] == 0:
        return torch.zeros(
            values.shape[:-1], dtype=values.dtype, device=values.device
        )
    reduced = values
    while reduced.shape[-1] > 1:
        pair_count = reduced.shape[-1] // 2
        paired = reduced[..., : 2 * pair_count].reshape(
            *reduced.shape[:-1], pair_count, 2
        )
        next_values = paired[..., 0] ^ paired[..., 1]
        if reduced.shape[-1] % 2:
            next_values = torch.cat((next_values, reduced[..., -1:]), dim=-1)
        reduced = next_values
    return reduced[..., 0]


@dataclass(frozen=True)
class RSDecodeStats:
    codewords: int
    clean_codewords: int
    corrected_codewords: int
    uncorrectable_codewords: int
    corrected_symbols: int

    def __add__(self, other: "RSDecodeStats") -> "RSDecodeStats":
        if not isinstance(other, RSDecodeStats):
            return NotImplemented
        return RSDecodeStats(
            self.codewords + other.codewords,
            self.clean_codewords + other.clean_codewords,
            self.corrected_codewords + other.corrected_codewords,
            self.uncorrectable_codewords + other.uncorrectable_codewords,
            self.corrected_symbols + other.corrected_symbols,
        )

    @classmethod
    def empty(cls) -> "RSDecodeStats":
        return cls(0, 0, 0, 0, 0)

    def as_dict(self) -> dict[str, int]:
        return {
            "codewords": self.codewords,
            "total_codewords": self.codewords,
            "clean_codewords": self.clean_codewords,
            "corrected_codewords": self.corrected_codewords,
            "uncorrectable_codewords": self.uncorrectable_codewords,
            "corrected_symbols": self.corrected_symbols,
            # Compatibility keys used by the older model entry points.
            "corrected": self.corrected_codewords,
            "uncorrectable": self.uncorrectable_codewords,
        }


class RS4823Codec:
    """Shortened systematic RS(48,23) over GF(256).

    The consecutive roots are alpha**0 through alpha**24, using primitive
    polynomial 0x11d and generator element 2. At most 12 erroneous byte
    symbols can be corrected in each 48-byte codeword.
    """

    N = N
    K = K
    NSYM = NSYM
    T = T
    FLOATS_PER_CODEWORD = FLOATS_PER_CODEWORD
    PACKED_INT32_PER_CODEWORD = PACKED_INT32_PER_CODEWORD
    PACKED_BYTES = PACKED_BYTES
    FCR = 0

    # Lower-case aliases keep existing experiment entry points compatible.
    n = N
    k = K
    nsym = NSYM
    t = T
    symbol_bits = 8
    packed_bytes = PACKED_BYTES
    primitive_polynomial = _GF256.PRIMITIVE_POLYNOMIAL
    floats_per_codeword = FLOATS_PER_CODEWORD

    def __init__(
        self,
        device: torch.device | str,
        block_chunk_size: int = 16_384,
        vectorized_min_batch: int = 1_024,
        enable_single_symbol_fast_path: bool = True,
        enable_adaptive_bm_fast_path: bool = True,
        enable_compact_bm_state: bool = True,
        staged_bm_single_fraction_min: float = 0.02,
    ):
        if block_chunk_size <= 0:
            raise ValueError("block_chunk_size must be positive")
        if vectorized_min_batch <= 0:
            raise ValueError("vectorized_min_batch must be positive")
        if not 0.0 <= staged_bm_single_fraction_min <= 1.0:
            raise ValueError("staged_bm_single_fraction_min must be in [0,1]")
        self.block_chunk_size = int(block_chunk_size)
        self.vectorized_min_batch = int(vectorized_min_batch)
        self.enable_single_symbol_fast_path = bool(
            enable_single_symbol_fast_path
        )
        self.enable_adaptive_bm_fast_path = bool(
            enable_adaptive_bm_fast_path
        )
        self.enable_compact_bm_state = bool(enable_compact_bm_state)
        self.staged_bm_single_fraction_min = float(
            staged_bm_single_fraction_min
        )
        self.last_single_symbol_fast_path_codewords = 0
        self.last_staged_bm_codewords = 0
        self.last_full_bm_input_codewords = 0
        self.device = torch.device(device)
        self.gf = _GF256(self.device)
        # Resolve an index-less ``cuda`` request to the concrete device used
        # by the lookup tables, so later device validation accepts cuda:0.
        self.device = self.gf.exp.device
        self.gf.device = self.device
        self.generator_polynomial = self._make_generator_polynomial()
        self.parity_matrix = self._make_parity_matrix().to(self.device)
        self.check_matrix = self._make_check_matrix().to(self.device)
        symbol_values = torch.arange(256, dtype=torch.int64, device=self.device)
        self.syndrome_lut = torch.empty(
            (self.N, 256, self.NSYM), dtype=torch.uint8, device=self.device
        )
        for position in range(self.N):
            self.syndrome_lut[position] = self.gf.mul(
                symbol_values.view(-1, 1),
                self.check_matrix[:, position].view(1, -1),
            ).to(torch.uint8)

        # BER=1e-2 dirties roughly 98% of 48-byte codewords.  For those large
        # batches, one broadcasted LUT gather followed by a balanced XOR tree
        # launches far fewer CUDA kernels than scanning all 48 positions in
        # Python.  The compact Chien LUT applies the same idea to all 48 roots.
        self.syndrome_vector_lut = self.syndrome_lut.permute(2, 0, 1).contiguous()
        self._syndrome_checks = torch.arange(
            self.NSYM, dtype=torch.int64, device=self.device
        ).view(1, self.NSYM, 1)
        self._syndrome_positions = torch.arange(
            self.N, dtype=torch.int64, device=self.device
        ).view(1, 1, self.N)
        positions = torch.arange(self.N, dtype=torch.int64, device=self.device)
        inverse_locations = self.gf.exp[
            torch.remainder(-(self.N - 1 - positions), self.gf.ORDER)
        ]
        chien_powers = torch.ones(
            (self.T + 1, self.N), dtype=torch.int64, device=self.device
        )
        for power in range(1, self.T + 1):
            chien_powers[power] = self.gf.mul(
                chien_powers[power - 1], inverse_locations
            )
        self.chien_term_lut = torch.empty(
            (self.N, self.T + 1, 256), dtype=torch.uint8, device=self.device
        )
        for power in range(self.T + 1):
            self.chien_term_lut[:, power] = self.gf.mul(
                chien_powers[power].view(-1, 1), symbol_values.view(1, -1)
            ).to(torch.uint8)
        self._chien_degrees = torch.arange(
            self.T + 1, dtype=torch.int64, device=self.device
        ).view(1, 1, self.T + 1)
        self._chien_positions = torch.arange(
            self.N, dtype=torch.int64, device=self.device
        ).view(1, self.N, 1)
        self.single_error_position_lut = torch.full(
            (256,), -1, dtype=torch.int64, device=self.device
        )
        self.single_error_position_lut[
            self.check_matrix[1].to(torch.int64)
        ] = positions

        payload_bit_offsets = torch.arange(
            self.K, dtype=torch.int64, device=self.device
        ) * 8
        self._pack_source_indices = torch.div(
            payload_bit_offsets, 23, rounding_mode="floor"
        )
        source_bit_offsets = torch.remainder(payload_bit_offsets, 23)
        available_source_bits = 23 - source_bit_offsets
        self._pack_second_bits = (8 - available_source_bits).clamp_min(0)
        self._pack_first_right_shifts = (available_source_bits - 8).clamp_min(0)
        self._pack_first_left_shifts = self._pack_second_bits
        self._pack_second_right_shifts = 23 - self._pack_second_bits

        high23_bit_offsets = torch.arange(
            self.FLOATS_PER_CODEWORD, dtype=torch.int64, device=self.device
        ) * 23
        high23_byte_starts = torch.div(
            high23_bit_offsets, 8, rounding_mode="floor"
        )
        self._unpack_byte_indices = high23_byte_starts.view(-1, 1) + torch.arange(
            4, dtype=torch.int64, device=self.device
        ).view(1, -1)
        self._unpack_right_shifts = 9 - torch.remainder(high23_bit_offsets, 8)
        self._unpack_byte_shifts = torch.tensor(
            (24, 16, 8, 0), dtype=torch.int64, device=self.device
        )

    def _make_generator_polynomial(self) -> list[int]:
        generator = [1]
        for root_index in range(self.FCR, self.FCR + self.NSYM):
            root = _gf_pow_scalar(_GF256.GENERATOR, root_index)
            generator = _poly_mul_scalar(generator, [1, root])
        return generator

    def _encode_scalar(self, message: Sequence[int]) -> list[int]:
        if len(message) != self.K:
            raise ValueError(f"expected {self.K} message bytes, got {len(message)}")
        work = list(message) + [0] * self.NSYM
        for index in range(self.K):
            coefficient = work[index]
            if coefficient == 0:
                continue
            for offset in range(1, len(self.generator_polynomial)):
                work[index + offset] ^= _gf_mul_scalar(
                    self.generator_polynomial[offset], coefficient
                )
        return list(message) + work[self.K :]

    def _make_parity_matrix(self) -> torch.Tensor:
        rows: list[list[int]] = []
        for message_index in range(self.K):
            basis = [0] * self.K
            basis[message_index] = 1
            rows.append(self._encode_scalar(basis)[self.K :])
        return torch.tensor(rows, dtype=torch.int64)

    def _make_check_matrix(self) -> torch.Tensor:
        rows: list[list[int]] = []
        for syndrome_index in range(self.NSYM):
            root_power = self.FCR + syndrome_index
            rows.append(
                [
                    _gf_pow_scalar(
                        _GF256.GENERATOR,
                        (root_power * (self.N - 1 - position)) % _GF256.ORDER,
                    )
                    for position in range(self.N)
                ]
            )
        return torch.tensor(rows, dtype=torch.int64)

    def _require_codec_device(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.device != self.device:
            raise ValueError(
                f"{name} is on {tensor.device}, but codec is on {self.device}"
            )

    def encode_bytes(self, messages: torch.Tensor) -> torch.Tensor:
        """Encode ``[batch, 23]`` byte messages to ``[batch, 48]``."""

        if messages.ndim != 2 or messages.shape[1] != self.K:
            raise ValueError(f"messages must have shape [batch, {self.K}]")
        self._require_codec_device(messages, "messages")
        messages_i64 = messages.to(dtype=torch.int64)
        if bool(((messages_i64 < 0) | (messages_i64 > 255)).any()):
            raise ValueError("message symbols must be bytes")
        parity = torch.zeros(
            (messages_i64.shape[0], self.NSYM),
            dtype=torch.int64,
            device=self.device,
        )
        for index in range(self.K):
            parity ^= self.gf.mul(
                messages_i64[:, index : index + 1], self.parity_matrix[index]
            )
        return torch.cat((messages_i64, parity), dim=1).to(torch.uint8)

    def _use_vectorized_decode(self, batch: int) -> bool:
        return self.device.type == "cuda" and batch >= self.vectorized_min_batch

    def _syndromes_loop(self, codewords_u8: torch.Tensor) -> torch.Tensor:
        """Low-memory syndrome path for CPU and small CUDA batches."""

        syndromes = torch.zeros(
            (codewords_u8.shape[0], self.NSYM),
            dtype=torch.uint8,
            device=self.device,
        )
        for position in range(self.N):
            syndromes ^= self.syndrome_lut[
                position, codewords_u8[:, position].to(torch.int64)
            ]
        return syndromes.to(torch.int64)

    def _syndromes_vectorized(self, codewords_u8: torch.Tensor) -> torch.Tensor:
        """Large-batch syndrome path with one LUT gather and six XOR stages."""

        contributions = self.syndrome_vector_lut[
            self._syndrome_checks,
            self._syndrome_positions,
            codewords_u8.to(torch.int64).unsqueeze(1),
        ]
        return _xor_reduce_last(contributions).to(torch.int64)

    def _syndromes(self, codewords: torch.Tensor) -> torch.Tensor:
        codewords_u8 = codewords.to(device=self.device, dtype=torch.uint8)
        if self._use_vectorized_decode(codewords_u8.shape[0]):
            return self._syndromes_vectorized(codewords_u8)
        return self._syndromes_loop(codewords_u8)

    def _berlekamp_massey(
        self,
        syndromes: torch.Tensor,
        syndrome_count: int | None = None,
        locator_capacity: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        iterations = self.NSYM if syndrome_count is None else int(syndrome_count)
        if not 1 <= iterations <= self.NSYM:
            raise ValueError(f"syndrome_count must be in [1,{self.NSYM}]")
        capacity = self.T if locator_capacity is None else int(locator_capacity)
        if not 1 <= capacity <= self.T:
            raise ValueError(f"locator_capacity must be in [1,{self.T}]")
        batch = syndromes.shape[0]
        # Only coefficients through the largest correctable degree are useful.
        # A degree overflow is monotonic in BM, so the overflowing coefficient
        # never needs to be materialized: that row will be rejected before
        # Chien search.  This cuts the hot BM state from 26 coefficients to
        # 13 for the full decoder, and to 5/7 in the staged tiers.
        width = capacity + 1 if self.enable_compact_bm_state else self.NSYM + 1
        locator = torch.zeros((batch, width), dtype=torch.int64, device=self.device)
        previous = torch.zeros_like(locator)
        locator[:, 0] = 1
        previous[:, 0] = 1
        degree = torch.zeros(batch, dtype=torch.int64, device=self.device)
        shift = torch.ones(batch, dtype=torch.int64, device=self.device)
        previous_discrepancy = torch.ones(
            batch, dtype=torch.int64, device=self.device
        )
        polynomial_indices = torch.arange(width, device=self.device).view(1, -1)

        for syndrome_index in range(iterations):
            discrepancy = syndromes[:, syndrome_index].clone()
            if syndrome_index:
                coefficient_count = min(syndrome_index, width - 1)
                products = self.gf.mul(
                    locator[:, 1 : coefficient_count + 1],
                    torch.flip(
                        syndromes[
                            :,
                            syndrome_index
                            - coefficient_count : syndrome_index,
                        ],
                        dims=(1,),
                    ),
                )
                discrepancy ^= _xor_reduce_last(products)
            nonzero = discrepancy != 0
            # ``previous_discrepancy`` starts at one and is only replaced by
            # a nonzero discrepancy.  The unchecked operation therefore
            # avoids two CUDA host synchronizations per BM iteration.
            scale = self.gf.div_unchecked(discrepancy, previous_discrepancy)

            old_locator = locator.clone()
            source_indices = polynomial_indices - shift.view(-1, 1)
            valid_source = source_indices >= 0
            shifted_previous = previous.gather(
                1, source_indices.clamp(min=0, max=width - 1)
            )
            shifted_previous = torch.where(
                valid_source,
                shifted_previous,
                torch.zeros_like(shifted_previous),
            )
            locator ^= self.gf.mul(scale.view(-1, 1), shifted_previous)

            increases_degree = nonzero & (2 * degree <= syndrome_index)
            previous = torch.where(
                increases_degree.view(-1, 1), old_locator, previous
            )
            previous_discrepancy = torch.where(
                increases_degree, discrepancy, previous_discrepancy
            )
            degree = torch.where(
                increases_degree, syndrome_index + 1 - degree, degree
            )
            shift = torch.where(
                increases_degree, torch.ones_like(shift), shift + 1
            )

        locator = torch.where(
            polynomial_indices <= degree.view(-1, 1),
            locator,
            torch.zeros_like(locator),
        )
        return locator, degree

    def _find_error_positions_loop(self, locator: torch.Tensor) -> torch.Tensor:
        """Low-memory Horner Chien search for small batches."""

        positions = torch.arange(self.N, dtype=torch.int64, device=self.device)
        location_exponents = torch.remainder(
            -(self.N - 1 - positions), _GF256.ORDER
        )
        inverse_locations = self.gf.exp[location_exponents]
        # Staged BM intentionally returns narrower locator tensors (degree 4
        # or 6), while the final tier returns the full correctable degree 12.
        locator_capacity = locator.shape[1] - 1
        values = locator[
            :, locator_capacity : locator_capacity + 1
        ].expand(-1, self.N).clone()
        for coefficient_index in range(locator_capacity - 1, -1, -1):
            values = self.gf.mul(values, inverse_locations) ^ locator[
                :, coefficient_index : coefficient_index + 1
            ]
        return values == 0

    def _find_error_positions_vectorized(
        self, locator: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate all correctable locator coefficients and roots at once."""

        width = locator.shape[1]
        coefficients = locator.to(torch.int64).unsqueeze(1)
        terms = self.chien_term_lut[
            self._chien_positions,
            self._chien_degrees[:, :, :width],
            coefficients,
        ]
        values = _xor_reduce_last(terms)
        return values == 0

    def _find_error_positions(self, locator: torch.Tensor) -> torch.Tensor:
        if self._use_vectorized_decode(locator.shape[0]):
            return self._find_error_positions_vectorized(locator)
        return self._find_error_positions_loop(locator)

    def _solve_error_magnitudes(
        self, positions: torch.Tensor, syndromes: torch.Tensor
    ) -> torch.Tensor:
        error_count = positions.shape[1]
        if error_count == 0:
            return torch.empty_like(positions)
        matrix = self.check_matrix[:error_count, :].transpose(0, 1)[positions]
        matrix = matrix.transpose(1, 2).contiguous()
        right = syndromes[:, :error_count].clone()
        for column in range(error_count):
            pivot = matrix[:, column, column]
            # Distinct Chien roots form a nonsingular Vandermonde system.
            # Avoid synchronizing the CUDA stream merely to recheck that
            # invariant for every elimination column.
            inverse = self.gf.inv_unchecked(pivot)
            matrix[:, column, column:] = self.gf.mul(
                matrix[:, column, column:], inverse.view(-1, 1)
            )
            right[:, column] = self.gf.mul(right[:, column], inverse)
            for row in range(error_count):
                if row == column:
                    continue
                factor = matrix[:, row, column].clone()
                matrix[:, row, column:] ^= self.gf.mul(
                    factor.view(-1, 1), matrix[:, column, column:]
                )
                right[:, row] ^= self.gf.mul(factor, right[:, column])
        return right

    def _forney_error_magnitudes(
        self,
        positions: torch.Tensor,
        syndromes: torch.Tensor,
        locator: torch.Tensor,
    ) -> torch.Tensor:
        """Return error magnitudes using the FCR=0 Forney equation."""

        batch, error_count = positions.shape
        if error_count == 0:
            return torch.empty_like(positions)

        location_exponents = torch.remainder(
            self.N - 1 - positions, self.gf.ORDER
        )
        locations = self.gf.exp[location_exponents]
        inverse_locations = self.gf.exp[
            torch.remainder(-location_exponents, self.gf.ORDER)
        ]

        omega = torch.empty(
            (batch, error_count), dtype=torch.int64, device=self.device
        )
        for power in range(error_count):
            products = self.gf.mul(
                locator[:, : power + 1],
                torch.flip(syndromes[:, : power + 1], dims=(1,)),
            )
            omega[:, power] = _xor_reduce_last(products)

        omega_at_roots = omega[:, -1:].expand(-1, error_count).clone()
        for power in range(error_count - 2, -1, -1):
            omega_at_roots = (
                self.gf.mul(omega_at_roots, inverse_locations)
                ^ omega[:, power : power + 1]
            )

        derivative = torch.zeros(
            (batch, error_count), dtype=torch.int64, device=self.device
        )
        derivative[:, 0::2] = locator[:, 1 : error_count + 1 : 2]
        derivative_at_roots = derivative[:, -1:].expand(
            -1, error_count
        ).clone()
        for power in range(error_count - 2, -1, -1):
            derivative_at_roots = (
                self.gf.mul(derivative_at_roots, inverse_locations)
                ^ derivative[:, power : power + 1]
            )

        # The general factor is X_i**(1-FCR).  FCR is zero here, so X_i is
        # mandatory; omitting it would silently implement an FCR=1 decoder.
        ratios = self.gf.div_unchecked(omega_at_roots, derivative_at_roots)
        return self.gf.mul(locations, ratios)

    def _verify_sparse_corrections(
        self,
        original_syndromes: torch.Tensor,
        positions: torch.Tensor,
        magnitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Check all 25 post-correction syndromes from sparse deltas."""

        contributions = self.syndrome_lut[
            positions.to(torch.int64), magnitudes.to(torch.int64)
        ]
        delta = _xor_reduce_last(contributions.transpose(1, 2)).to(torch.int64)
        return ((original_syndromes ^ delta) == 0).all(dim=1)

    def _single_symbol_errors(
        self, syndromes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recognize and solve exact one-symbol syndromes without BM/Chien.

        With FCR=0, S0 is the error magnitude and S1/S0 is its field
        location.  All 25 syndrome components are compared against the LUT
        before accepting the correction, so this is a genuine syndrome decode
        rather than information from the fault injector.
        """

        magnitudes = syndromes[:, 0].to(torch.int64)
        locations = self.gf.div_unchecked(syndromes[:, 1], magnitudes)
        positions = self.single_error_position_lut[locations]
        safe_positions = positions.clamp_min(0)
        expected = self.syndrome_lut[safe_positions, magnitudes].to(torch.int64)
        valid = (
            (magnitudes != 0)
            & (positions >= 0)
            & (expected == syndromes).all(dim=1)
        )
        return valid, positions, magnitudes

    def _apply_bm_stage(
        self,
        syndromes: torch.Tensor,
        global_rows: torch.Tensor,
        corrected: torch.Tensor,
        corrected_symbols: torch.Tensor,
        *,
        syndrome_count: int,
        max_degree: int,
    ) -> torch.Tensor:
        """Attempt one bounded BM tier and apply only verified corrections."""

        stage_success = torch.zeros(
            syndromes.shape[0], dtype=torch.bool, device=self.device
        )
        if syndromes.shape[0] == 0:
            return stage_success
        locator, degree = self._berlekamp_massey(
            syndromes,
            syndrome_count=syndrome_count,
            locator_capacity=max_degree,
        )

        # Filtering before Chien is especially important at BER=1e-1, where
        # the 25th BM update normally exposes degree 13 (> t=12).
        degree_candidate = (degree > 0) & (degree <= max_degree)
        candidate_rows = torch.nonzero(
            degree_candidate, as_tuple=False
        ).flatten()
        if candidate_rows.numel() == 0:
            return stage_success

        candidate_locator = locator[candidate_rows, : max_degree + 1]
        candidate_roots = self._find_error_positions(candidate_locator)
        root_count = candidate_roots.sum(dim=1).to(torch.int64)
        has_expected_roots = root_count == degree[candidate_rows]
        locatable_candidates = torch.nonzero(
            has_expected_roots, as_tuple=False
        ).flatten()
        if locatable_candidates.numel() == 0:
            return stage_success

        locatable_rows = candidate_rows[locatable_candidates]
        locatable_roots = candidate_roots[locatable_candidates]
        ordering = torch.argsort(degree[locatable_rows])
        locatable_rows = locatable_rows[ordering]
        locatable_roots = locatable_roots[ordering]
        degree_counts = torch.bincount(
            degree[locatable_rows], minlength=max_degree + 1
        ).cpu().tolist()

        group_start = 0
        for error_count in range(1, max_degree + 1):
            group_size = degree_counts[error_count]
            if group_size == 0:
                continue
            local_rows = locatable_rows[
                group_start : group_start + group_size
            ]
            local_roots = locatable_roots[
                group_start : group_start + group_size
            ]
            group_start += group_size
            positions = torch.nonzero(local_roots, as_tuple=False)[
                :, 1
            ].reshape(-1, error_count)
            magnitudes = self._forney_error_magnitudes(
                positions,
                syndromes[local_rows],
                locator[local_rows],
            )
            verified = self._verify_sparse_corrections(
                syndromes[local_rows], positions, magnitudes
            )
            verified_group_rows = torch.nonzero(
                verified, as_tuple=False
            ).flatten()
            if verified_group_rows.numel() == 0:
                continue
            successful_rows = local_rows[verified_group_rows]
            successful_positions = positions[verified_group_rows]
            successful_magnitudes = magnitudes[verified_group_rows]
            successful_global_rows = global_rows[successful_rows]
            row_grid = successful_global_rows.view(-1, 1).expand_as(
                successful_positions
            )
            corrected[row_grid, successful_positions] ^= (
                successful_magnitudes.to(torch.uint8)
            )
            corrected_symbols[successful_global_rows] = error_count
            stage_success[successful_rows] = True
        return stage_success

    def decode_bytes(
        self,
        codewords: torch.Tensor,
        *,
        failure_policy: str = "systematic",
    ) -> tuple[torch.Tensor, RSDecodeStats]:
        """Decode ``[batch,48]`` codewords and return 23 payload bytes."""

        if failure_policy not in {"systematic", "zero", "raise"}:
            raise ValueError("failure_policy must be systematic, zero, or raise")
        self.last_single_symbol_fast_path_codewords = 0
        self.last_staged_bm_codewords = 0
        self.last_full_bm_input_codewords = 0
        if codewords.ndim != 2 or codewords.shape[1] != self.N:
            raise ValueError(f"codewords must have shape [batch, {self.N}]")
        self._require_codec_device(codewords, "codewords")
        if codewords.dtype == torch.uint8:
            received = codewords
        else:
            codewords_i64 = codewords.to(dtype=torch.int64)
            if bool(((codewords_i64 < 0) | (codewords_i64 > 255)).any()):
                raise ValueError("codeword symbols must be bytes")
            received = codewords_i64.to(torch.uint8)

        syndromes = self._syndromes(received)
        has_errors = (syndromes != 0).any(dim=1)
        error_rows = torch.nonzero(has_errors, as_tuple=False).flatten()
        if error_rows.numel() == 0:
            stats = RSDecodeStats(
                codewords=received.shape[0],
                clean_codewords=received.shape[0],
                corrected_codewords=0,
                uncorrectable_codewords=0,
                corrected_symbols=0,
            )
            return received[:, : self.K].contiguous(), stats

        corrected = received.clone()
        corrected_symbols = torch.zeros(
            received.shape[0], dtype=torch.int64, device=self.device
        )
        error_syndromes = syndromes[error_rows]
        error_success = torch.zeros(
            error_rows.numel(), dtype=torch.bool, device=self.device
        )
        if self.enable_single_symbol_fast_path:
            single_mask, single_positions, single_magnitudes = (
                self._single_symbol_errors(error_syndromes)
            )
        else:
            single_mask = torch.zeros(
                error_rows.numel(), dtype=torch.bool, device=self.device
            )
            single_positions = torch.zeros(
                error_rows.numel(), dtype=torch.int64, device=self.device
            )
            single_magnitudes = torch.zeros_like(single_positions)
        single_rows = torch.nonzero(single_mask, as_tuple=False).flatten()
        self.last_single_symbol_fast_path_codewords = single_rows.numel()
        if single_rows.numel():
            single_global_rows = error_rows[single_rows]
            corrected[
                single_global_rows, single_positions[single_rows]
            ] ^= single_magnitudes[single_rows].to(torch.uint8)
            corrected_symbols[single_global_rows] = 1
            error_success[single_rows] = True

        remaining_rows = torch.nonzero(~single_mask, as_tuple=False).flatten()
        single_fraction = single_rows.numel() / error_rows.numel()
        use_staged_bm = (
            self.enable_adaptive_bm_fast_path
            and remaining_rows.numel() > 0
            and single_fraction >= self.staged_bm_single_fraction_min
        )
        if use_staged_bm:
            # Two tiers minimize expected BM iterations for the BER=1e-2
            # binomial symbol-error distribution: <=4 errors are solved from
            # 8 syndromes, then <=6 from 12; every proposal is checked against
            # all 25 syndromes before it is accepted.
            for max_degree, syndrome_count in ((4, 8), (6, 12)):
                if remaining_rows.numel() == 0:
                    break
                stage_rows = remaining_rows
                stage_success = self._apply_bm_stage(
                    error_syndromes[stage_rows],
                    error_rows[stage_rows],
                    corrected,
                    corrected_symbols,
                    syndrome_count=syndrome_count,
                    max_degree=max_degree,
                )
                successful_error_rows = stage_rows[stage_success]
                error_success[successful_error_rows] = True
                self.last_staged_bm_codewords += (
                    successful_error_rows.numel()
                )
                remaining_rows = stage_rows[~stage_success]

        self.last_full_bm_input_codewords = remaining_rows.numel()
        if remaining_rows.numel():
            final_success = self._apply_bm_stage(
                error_syndromes[remaining_rows],
                error_rows[remaining_rows],
                corrected,
                corrected_symbols,
                syndrome_count=self.NSYM,
                max_degree=self.T,
            )
            error_success[remaining_rows[final_success]] = True

        # Syndrome linearity lets us verify all 25 checks using only the
        # sparse corrections above.  This is equivalent to rescanning the
        # complete 48-byte candidate, without a second full syndrome pass.
        failed_rows = error_rows[~error_success]
        if failed_rows.numel():
            if failure_policy == "raise":
                raise RuntimeError(
                    f"RS(48,23) could not decode {failed_rows.numel()} "
                    f"of {received.shape[0]} codewords"
                )
            if failure_policy == "zero":
                corrected[failed_rows, : self.K] = 0
            else:
                corrected[failed_rows, : self.K] = received[failed_rows, : self.K]

        clean_count, corrected_count, corrected_symbol_count = torch.stack(
            (
                (~has_errors).sum(dtype=torch.int64),
                error_success.sum(dtype=torch.int64),
                corrected_symbols[error_rows][error_success].sum(dtype=torch.int64),
            )
        ).cpu().tolist()
        stats = RSDecodeStats(
            codewords=received.shape[0],
            clean_codewords=clean_count,
            corrected_codewords=corrected_count,
            uncorrectable_codewords=failed_rows.numel(),
            corrected_symbols=corrected_symbol_count,
        )
        return corrected[:, : self.K].contiguous(), stats

    def _pack_high23_to_bytes(self, high23: torch.Tensor) -> torch.Tensor:
        if high23.ndim != 2 or high23.shape[1] != FLOATS_PER_CODEWORD:
            raise ValueError(
                f"high23 must have shape [batch, {FLOATS_PER_CODEWORD}]"
            )
        padded = torch.nn.functional.pad(high23.to(torch.int64), (0, 1))
        first = padded[:, self._pack_source_indices]
        second = padded[:, self._pack_source_indices + 1]
        packed = (
            (first >> self._pack_first_right_shifts)
            << self._pack_first_left_shifts
        ) | (second >> self._pack_second_right_shifts)
        return packed.to(torch.uint8)

    def _unpack_bytes_to_high23(self, payload: torch.Tensor) -> torch.Tensor:
        if payload.ndim != 2 or payload.shape[1] != K:
            raise ValueError(f"payload must have shape [batch, {K}]")
        padded = torch.nn.functional.pad(payload, (0, 3))
        byte_groups = padded[:, self._unpack_byte_indices].to(torch.int64)
        windows = (byte_groups << self._unpack_byte_shifts).sum(dim=-1)
        return (windows >> self._unpack_right_shifts) & 0x7FFFFF

    @staticmethod
    def _pack_bytes_to_int32(codewords: torch.Tensor) -> torch.Tensor:
        if codewords.ndim != 2 or codewords.shape[1] != N:
            raise ValueError(f"codewords must have shape [batch, {N}]")
        return codewords.contiguous().view(torch.int32).reshape(
            -1, PACKED_INT32_PER_CODEWORD
        )

    @staticmethod
    def _unpack_int32_to_bytes(words: torch.Tensor) -> torch.Tensor:
        if words.ndim != 2 or words.shape[1] != PACKED_INT32_PER_CODEWORD:
            raise ValueError(
                "words must have shape "
                f"[batch, {PACKED_INT32_PER_CODEWORD}]"
            )
        return words.contiguous().view(torch.uint8).reshape(-1, N)

    def encode_float32(
        self,
        values: torch.Tensor,
        *,
        block_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Encode float32 values to a flat int32 codeword container."""

        if values.dtype != torch.float32:
            raise TypeError(f"RS high23 protection requires float32, got {values.dtype}")
        self._require_codec_device(values, "values")
        chunk_size = self._validated_chunk_size(block_chunk_size)
        flat = values.detach().contiguous().view(-1)
        raw = flat.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        high23 = (raw >> 9) & 0x7FFFFF
        padding = (-high23.numel()) % self.FLOATS_PER_CODEWORD
        if padding:
            high23 = torch.cat(
                (
                    high23,
                    torch.zeros(
                        padding, dtype=torch.int64, device=values.device
                    ),
                )
            )
        blocks = high23.reshape(-1, self.FLOATS_PER_CODEWORD)
        encoded = torch.empty(
            (blocks.shape[0], self.PACKED_INT32_PER_CODEWORD),
            dtype=torch.int32,
            device=values.device,
        )
        for start in range(0, blocks.shape[0], chunk_size):
            end = min(start + chunk_size, blocks.shape[0])
            payload = self._pack_high23_to_bytes(blocks[start:end])
            encoded[start:end] = self._pack_bytes_to_int32(
                self.encode_bytes(payload)
            )
        return encoded.reshape(-1)

    def decode_float32(
        self,
        encoded: torch.Tensor,
        *,
        numel: int,
        shape: Optional[Sequence[int]] = None,
        failure_policy: str = "systematic",
        block_chunk_size: int | None = None,
        known_error_mask: torch.Tensor | None = None,
        clear_exponent_msb: bool = True,
    ) -> tuple[torch.Tensor, RSDecodeStats]:
        """Decode an int32 container and reconstruct quantized float32 values.

        ``known_error_mask`` is an optional exact dirty-codeword hint produced
        by :class:`RS4823High23ModelProtector` while injecting faults.  Clean
        systematic codewords can then bypass syndrome calculation.  Omit the
        hint when decoding storage whose mutation history is not tracked.
        """

        if encoded.dtype != torch.int32:
            raise TypeError(f"encoded container must be int32, got {encoded.dtype}")
        self._require_codec_device(encoded, "encoded")
        if numel < 0:
            raise ValueError("numel must be non-negative")
        chunk_size = self._validated_chunk_size(block_chunk_size)
        expected_blocks = (
            numel + self.FLOATS_PER_CODEWORD - 1
        ) // self.FLOATS_PER_CODEWORD
        expected_words = expected_blocks * self.PACKED_INT32_PER_CODEWORD
        if encoded.numel() != expected_words:
            raise ValueError(
                f"expected {expected_words} packed int32 values for {numel} "
                f"floats, got {encoded.numel()}"
            )
        output_shape = tuple(shape) if shape is not None else (numel,)
        if _shape_numel(output_shape) != numel:
            raise ValueError("shape does not match numel")
        if known_error_mask is not None:
            if known_error_mask.device != encoded.device:
                raise ValueError("known_error_mask must be on the encoded device")
            if known_error_mask.numel() != expected_blocks:
                raise ValueError(
                    "known_error_mask must contain one value per RS codeword"
                )
            known_error_mask = known_error_mask.reshape(-1).to(torch.bool)
        if expected_blocks == 0:
            return (
                torch.empty(0, dtype=torch.float32, device=encoded.device).reshape(
                    output_shape
                ),
                RSDecodeStats.empty(),
            )

        words = encoded.reshape(
            expected_blocks, self.PACKED_INT32_PER_CODEWORD
        )
        recovered = torch.empty(numel, dtype=torch.float32, device=encoded.device)
        aggregate = RSDecodeStats.empty()
        for start in range(0, expected_blocks, chunk_size):
            end = min(start + chunk_size, expected_blocks)
            codewords = self._unpack_int32_to_bytes(words[start:end])
            if known_error_mask is None:
                payload, stats = self.decode_bytes(
                    codewords, failure_policy=failure_policy
                )
            else:
                dirty_indices = torch.nonzero(
                    known_error_mask[start:end], as_tuple=False
                ).flatten()
                dirty_count = dirty_indices.numel()
                chunk_codewords = end - start
                if dirty_count == 0:
                    payload = codewords[:, : self.K]
                    stats = RSDecodeStats(
                        codewords=chunk_codewords,
                        clean_codewords=chunk_codewords,
                        corrected_codewords=0,
                        uncorrectable_codewords=0,
                        corrected_symbols=0,
                    )
                elif dirty_count == chunk_codewords:
                    payload, stats = self.decode_bytes(
                        codewords, failure_policy=failure_policy
                    )
                else:
                    payload = codewords[:, : self.K].clone()
                    dirty_payload, dirty_stats = self.decode_bytes(
                        codewords[dirty_indices], failure_policy=failure_policy
                    )
                    payload[dirty_indices] = dirty_payload
                    skipped = chunk_codewords - dirty_count
                    stats = RSDecodeStats(
                        codewords=skipped,
                        clean_codewords=skipped,
                        corrected_codewords=0,
                        uncorrectable_codewords=0,
                        corrected_symbols=0,
                    ) + dirty_stats
            aggregate += stats
            block_values = self._payload_to_float32(
                payload,
                clear_exponent_msb=clear_exponent_msb,
            )
            element_start = start * self.FLOATS_PER_CODEWORD
            element_end = min(end * self.FLOATS_PER_CODEWORD, numel)
            recovered[element_start:element_end] = block_values[
                : element_end - element_start
            ]
        return recovered.reshape(output_shape), aggregate

    def restore_float32_systematic(
        self,
        encoded: torch.Tensor,
        *,
        numel: int,
        shape: Optional[Sequence[int]] = None,
        block_chunk_size: int | None = None,
        clear_exponent_msb: bool = True,
    ) -> torch.Tensor:
        """Restore FP32 values after codeword payload bytes were corrected.

        Low raw bits 8..0 are always zero because only high23 is stored.  Set
        ``clear_exponent_msb=False`` for an ablation that preserves raw bit 30
        from the decoded high23 payload; the default retains the established
        numerical-safety policy.
        """

        if encoded.dtype != torch.int32:
            raise TypeError(f"encoded container must be int32, got {encoded.dtype}")
        self._require_codec_device(encoded, "encoded")
        if numel < 0:
            raise ValueError("numel must be non-negative")
        output_shape = tuple(shape) if shape is not None else (numel,)
        if _shape_numel(output_shape) != numel:
            raise ValueError("shape does not match numel")
        expected_blocks = (
            numel + self.FLOATS_PER_CODEWORD - 1
        ) // self.FLOATS_PER_CODEWORD
        expected_words = expected_blocks * self.PACKED_INT32_PER_CODEWORD
        if encoded.numel() != expected_words:
            raise ValueError(
                f"expected {expected_words} packed int32 values for {numel} "
                f"floats, got {encoded.numel()}"
            )
        if numel == 0:
            return torch.empty(output_shape, dtype=torch.float32, device=self.device)

        chunk_size = self._validated_chunk_size(block_chunk_size)
        words = encoded.reshape(expected_blocks, self.PACKED_INT32_PER_CODEWORD)
        recovered = torch.empty(numel, dtype=torch.float32, device=self.device)
        for start in range(0, expected_blocks, chunk_size):
            end = min(start + chunk_size, expected_blocks)
            codewords = self._unpack_int32_to_bytes(words[start:end])
            block_values = self._payload_to_float32(
                codewords[:, : self.K],
                clear_exponent_msb=clear_exponent_msb,
            )
            element_start = start * self.FLOATS_PER_CODEWORD
            element_end = min(end * self.FLOATS_PER_CODEWORD, numel)
            recovered[element_start:element_end] = block_values[
                : element_end - element_start
            ]
        return recovered.reshape(output_shape)

    def _payload_to_float32(
        self,
        payload: torch.Tensor,
        *,
        clear_exponent_msb: bool = True,
    ) -> torch.Tensor:
        recovered_high = self._unpack_bytes_to_high23(payload)
        raw = recovered_high.reshape(-1) << 9
        if clear_exponent_msb:
            raw = raw & 0xBFFFFFFF
        return raw.to(torch.int32).contiguous().view(torch.float32)

    def _validated_chunk_size(self, value: int | None) -> int:
        chunk_size = self.block_chunk_size if value is None else int(value)
        if chunk_size <= 0:
            raise ValueError("block_chunk_size must be positive")
        return chunk_size


def _shape_numel(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count


__all__ = [
    "DECODER_IMPLEMENTATION",
    "FLOATS_PER_CODEWORD",
    "K",
    "N",
    "NSYM",
    "PACKED_BYTES",
    "PACKED_INT32_PER_CODEWORD",
    "RS4823Codec",
    "RSDecodeStats",
    "T",
]
