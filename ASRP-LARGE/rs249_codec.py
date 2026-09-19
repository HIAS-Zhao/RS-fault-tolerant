"""GF(2^9) RS(24,9) codec with fused and staged GPU decoders."""

from __future__ import annotations

import torch
from pathlib import Path

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only development environments
    triton = None
    tl = None


FIELD_SIZE = 512
FIELD_ORDER = 511
PRIMITIVE_POLYNOMIAL = 0x211  # x^9 + x^4 + 1
N = 24
K = 9
NSYM = N - K
T = NSYM // 2


def _python_gf_tables():
    exp = [0] * (FIELD_ORDER * 2)
    log = [0] * FIELD_SIZE
    value = 1
    for exponent in range(FIELD_ORDER):
        exp[exponent] = value
        log[value] = exponent
        value <<= 1
        if value & FIELD_SIZE:
            value ^= PRIMITIVE_POLYNOMIAL
    if value != 1 or len(set(exp[:FIELD_ORDER])) != FIELD_ORDER:
        raise RuntimeError("0x211 is not primitive for GF(2^9)")
    for exponent in range(FIELD_ORDER, FIELD_ORDER * 2):
        exp[exponent] = exp[exponent - FIELD_ORDER]
    return exp, log


if triton is not None:

    @triton.jit
    def _gf_mul(a, b, multiplication):
        return tl.load(multiplication + a * FIELD_SIZE + b)


    @triton.jit
    def _rs249_decode_kernel(
        received,
        corrected,
        success_out,
        multiplication,
        inverse,
        roots,
        chien,
        scratch,
        count,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid_lane = offsets < count
        recv_base = offsets * 24
        work_base = scratch + offsets * 78
        s_base = work_base
        c_base = work_base + 15
        b_base = work_base + 31
        tmp_base = work_base + 47
        omega_base = work_base + 63

        for index in tl.static_range(0, 78):
            tl.store(work_base + index, 0, mask=valid_lane)

        syndrome_nonzero = tl.zeros((BLOCK,), tl.int1)
        for syndrome_index in tl.static_range(0, 15):
            value = tl.zeros((BLOCK,), tl.int32)
            root = tl.load(roots + syndrome_index)
            for symbol_index in tl.static_range(0, 24):
                symbol = tl.load(
                    received + recv_base + symbol_index, mask=valid_lane, other=0
                ).to(tl.int32)
                value = _gf_mul(value, root, multiplication) ^ symbol
            tl.store(s_base + syndrome_index, value, mask=valid_lane)
            syndrome_nonzero = syndrome_nonzero | (value != 0)

        tl.store(c_base, 1, mask=valid_lane)
        tl.store(b_base, 1, mask=valid_lane)
        degree = tl.zeros((BLOCK,), tl.int32)
        shift = tl.full((BLOCK,), 1, tl.int32)
        previous_discrepancy = tl.full((BLOCK,), 1, tl.int32)

        for syndrome_index in tl.static_range(0, 15):
            discrepancy = tl.load(
                s_base + syndrome_index, mask=valid_lane, other=0
            ).to(tl.int32)
            for locator_index in tl.static_range(1, 16):
                coefficient = tl.load(
                    c_base + locator_index, mask=valid_lane, other=0
                ).to(tl.int32)
                syndrome_value = tl.load(
                    s_base + syndrome_index - locator_index,
                    mask=valid_lane & (locator_index <= syndrome_index),
                    other=0,
                ).to(tl.int32)
                term = _gf_mul(coefficient, syndrome_value, multiplication)
                discrepancy = discrepancy ^ tl.where(
                    locator_index <= degree, term, 0
                )

            nonzero = discrepancy != 0
            inv_previous = tl.load(inverse + previous_discrepancy).to(tl.int32)
            scale = _gf_mul(discrepancy, inv_previous, multiplication)
            for coefficient_index in tl.static_range(0, 16):
                old = tl.load(
                    c_base + coefficient_index, mask=valid_lane, other=0
                ).to(tl.int32)
                tl.store(tmp_base + coefficient_index, old, mask=valid_lane)
                source_index = coefficient_index - shift
                previous_value = tl.load(
                    b_base + source_index,
                    mask=valid_lane & (source_index >= 0) & (source_index <= 15),
                    other=0,
                ).to(tl.int32)
                candidate = old ^ _gf_mul(scale, previous_value, multiplication)
                tl.store(
                    c_base + coefficient_index,
                    tl.where(nonzero, candidate, old),
                    mask=valid_lane,
                )

            major = nonzero & ((2 * degree) <= syndrome_index)
            for coefficient_index in tl.static_range(0, 16):
                old_previous = tl.load(
                    b_base + coefficient_index, mask=valid_lane, other=0
                ).to(tl.int32)
                old_locator = tl.load(
                    tmp_base + coefficient_index, mask=valid_lane, other=0
                ).to(tl.int32)
                tl.store(
                    b_base + coefficient_index,
                    tl.where(major, old_locator, old_previous),
                    mask=valid_lane,
                )
            degree = tl.where(major, syndrome_index + 1 - degree, degree)
            previous_discrepancy = tl.where(
                major, discrepancy, previous_discrepancy
            )
            shift = tl.where(major, 1, shift + 1)

        root_mask = tl.zeros((BLOCK,), tl.int32)
        roots_found = tl.zeros((BLOCK,), tl.int32)
        for position in tl.static_range(0, 24):
            value = tl.load(c_base, mask=valid_lane, other=0).to(tl.int32)
            for locator_degree in tl.static_range(1, 16):
                coefficient = tl.load(
                    c_base + locator_degree, mask=valid_lane, other=0
                ).to(tl.int32)
                power = tl.load(chien + position * 16 + locator_degree)
                term = _gf_mul(coefficient, power, multiplication)
                value = value ^ tl.where(locator_degree <= degree, term, 0)
            found = (value == 0) & syndrome_nonzero
            root_mask = root_mask | tl.where(found, 1 << position, 0)
            roots_found += found.to(tl.int32)

        plausible = (
            syndrome_nonzero
            & (degree > 0)
            & (degree <= 7)
            & (roots_found == degree)
        )

        for output_degree in tl.static_range(0, 15):
            omega = tl.zeros((BLOCK,), tl.int32)
            for locator_degree in tl.static_range(0, 15):
                coefficient = tl.load(
                    c_base + locator_degree, mask=valid_lane, other=0
                ).to(tl.int32)
                syndrome_value = tl.load(
                    s_base + output_degree - locator_degree,
                    mask=valid_lane & (locator_degree <= output_degree),
                    other=0,
                ).to(tl.int32)
                term = _gf_mul(coefficient, syndrome_value, multiplication)
                omega = omega ^ tl.where(
                    (locator_degree <= output_degree)
                    & (locator_degree <= degree),
                    term,
                    0,
                )
            tl.store(omega_base + output_degree, omega, mask=valid_lane)

        zero_denominator = tl.zeros((BLOCK,), tl.int1)
        for position in tl.static_range(0, 24):
            active = plausible & ((root_mask & (1 << position)) != 0)
            root = tl.load(chien + position * 16 + 1)
            omega_value = tl.load(
                omega_base + 14, mask=valid_lane, other=0
            ).to(tl.int32)
            for reverse_index in tl.static_range(0, 14):
                evaluator_degree = 13 - reverse_index
                omega_value = _gf_mul(omega_value, root, multiplication) ^ tl.load(
                    omega_base + evaluator_degree, mask=valid_lane, other=0
                ).to(tl.int32)
            derivative = tl.zeros((BLOCK,), tl.int32)
            for locator_degree in tl.static_range(1, 16, 2):
                coefficient = tl.load(
                    c_base + locator_degree, mask=valid_lane, other=0
                ).to(tl.int32)
                power = tl.load(
                    chien + position * 16 + locator_degree - 1
                )
                term = _gf_mul(coefficient, power, multiplication)
                derivative = derivative ^ tl.where(
                    locator_degree <= degree, term, 0
                )
            zero_denominator = zero_denominator | (active & (derivative == 0))
            inv_derivative = tl.load(inverse + derivative).to(tl.int32)
            magnitude = _gf_mul(omega_value, inv_derivative, multiplication)
            original = tl.load(
                received + recv_base + position, mask=valid_lane, other=0
            ).to(tl.int32)
            tl.store(
                corrected + recv_base + position,
                original ^ tl.where(active, magnitude, 0),
                mask=valid_lane,
            )

        post_zero = tl.full((BLOCK,), True, tl.int1)
        for syndrome_index in tl.static_range(0, 15):
            value = tl.zeros((BLOCK,), tl.int32)
            root = tl.load(roots + syndrome_index)
            for symbol_index in tl.static_range(0, 24):
                symbol = tl.load(
                    corrected + recv_base + symbol_index,
                    mask=valid_lane,
                    other=0,
                ).to(tl.int32)
                value = _gf_mul(value, root, multiplication) ^ symbol
            post_zero = post_zero & (value == 0)

        success = (~syndrome_nonzero) | (
            plausible & (~zero_denominator) & post_zero
        )
        for position in tl.static_range(0, 24):
            candidate = tl.load(
                corrected + recv_base + position, mask=valid_lane, other=0
            )
            original = tl.load(
                received + recv_base + position, mask=valid_lane, other=0
            )
            tl.store(
                corrected + recv_base + position,
                tl.where(success, candidate, original),
                mask=valid_lane,
            )
        tl.store(success_out + offsets, success, mask=valid_lane)


class RS249Codec:
    n = N
    k = K
    nsym = NSYM
    t = T
    symbol_bits = 9
    packed_bytes = 27

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        exp, log = _python_gf_tables()
        multiplication = torch.zeros((FIELD_SIZE, FIELD_SIZE), dtype=torch.int32)
        for left in range(1, FIELD_SIZE):
            left_log = log[left]
            for right in range(1, FIELD_SIZE):
                multiplication[left, right] = exp[left_log + log[right]]
        inverse = torch.zeros(FIELD_SIZE, dtype=torch.int32)
        for value in range(1, FIELD_SIZE):
            inverse[value] = exp[FIELD_ORDER - log[value]]
        self.mul_table = multiplication.to(self.device)
        self.inv_table = inverse.to(self.device)
        self.gf_exp = torch.tensor(exp, dtype=torch.int32, device=self.device)
        self.gf_log = torch.tensor(log, dtype=torch.int32, device=self.device)
        self._cuda_extension = None
        self._init_generator()
        self._init_decoder_tables()

    def _mul(self, left, right):
        return self.mul_table[left.long(), right.long()]

    def _div(self, numerator, denominator):
        return self._mul(numerator, self.inv_table[denominator.long()])

    def _alpha(self, exponent):
        return int(self.gf_exp[exponent % FIELD_ORDER].item())

    def _init_generator(self):
        generator = [1]
        for root_index in range(1, self.nsym + 1):
            root = self._alpha(root_index)
            product = [0] * (len(generator) + 1)
            for index, coefficient in enumerate(generator):
                product[index] ^= coefficient
                if coefficient:
                    product[index + 1] ^= int(
                        self.mul_table[coefficient, root].item()
                    )
            generator = product
        self.generator = torch.tensor(generator, dtype=torch.int32, device=self.device)
        unit = torch.zeros((self.k, self.k), dtype=torch.int32, device=self.device)
        unit[torch.arange(self.k), torch.arange(self.k)] = 1
        unit_parity = self._encode_polynomial(unit)[:, self.k :]
        values = torch.arange(FIELD_SIZE, device=self.device)
        lut = torch.empty(
            (self.k, FIELD_SIZE, self.nsym), dtype=torch.int32, device=self.device
        )
        for position in range(self.k):
            for parity in range(self.nsym):
                lut[position, :, parity] = self.mul_table[
                    values, unit_parity[position, parity]
                ]
        self.encode_lut = lut

    def _encode_polynomial(self, data):
        work = torch.zeros(
            (data.shape[0], self.n), dtype=torch.int32, device=data.device
        )
        work[:, : self.k] = data
        for data_index in range(self.k):
            coefficient = work[:, data_index]
            for generator_index in range(1, self.nsym + 1):
                work[:, data_index + generator_index] ^= self._mul(
                    coefficient, self.generator[generator_index]
                )
        return torch.cat((data, work[:, self.k :]), dim=1)

    def _init_decoder_tables(self):
        roots = [self._alpha(index) for index in range(1, self.nsym + 1)]
        self.roots = torch.tensor(roots, dtype=torch.int32, device=self.device)
        chien = torch.ones(
            (self.n, self.nsym + 1), dtype=torch.int32, device=self.device
        )
        for position in range(self.n):
            value = self._alpha(-(self.n - 1 - position))
            for degree in range(1, self.nsym + 1):
                chien[position, degree] = self._alpha(
                    -(self.n - 1 - position) * degree
                )
        self.chien = chien
        values = torch.arange(FIELD_SIZE, device=self.device)
        syndrome_lut = torch.empty(
            (self.n, FIELD_SIZE, self.nsym), dtype=torch.int32, device=self.device
        )
        for position in range(self.n):
            location_exponent = self.n - 1 - position
            for syndrome_index in range(self.nsym):
                coefficient = self._alpha((syndrome_index + 1) * location_exponent)
                syndrome_lut[position, :, syndrome_index] = self.mul_table[
                    values, coefficient
                ]
        self.syndrome_lut = syndrome_lut

    def encode_blocks(self, data):
        if data.ndim != 2 or data.shape[1] != self.k:
            raise ValueError("RS(24,9) data must have shape (B,9)")
        data = data.to(torch.int32)
        parity = torch.zeros(
            (data.shape[0], self.nsym), dtype=torch.int32, device=data.device
        )
        for position in range(self.k):
            parity ^= self.encode_lut[position, data[:, position].long()]
        return torch.cat((data, parity), dim=1)

    def syndromes(self, received):
        result = torch.zeros((received.shape[0], self.nsym), dtype=torch.int32, device=received.device)
        for position in range(self.n):
            result ^= self.syndrome_lut[position, received[:, position].long()]
        return result

    def berlekamp_massey(self, syndrome):
        batch = syndrome.shape[0]
        locator = torch.zeros((batch, self.nsym + 1), dtype=torch.int32, device=self.device)
        previous = torch.zeros_like(locator)
        locator[:, 0] = 1
        previous[:, 0] = 1
        degree = torch.zeros(batch, dtype=torch.long, device=self.device)
        shift = torch.ones(batch, dtype=torch.long, device=self.device)
        previous_discrepancy = torch.ones(batch, dtype=torch.int32, device=self.device)
        coefficient_indices = torch.arange(
            self.nsym + 1, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        for syndrome_index in range(self.nsym):
            discrepancy = syndrome[:, syndrome_index].clone()
            for locator_index in range(1, syndrome_index + 1):
                term = self._mul(
                    locator[:, locator_index], syndrome[:, syndrome_index - locator_index]
                )
                discrepancy ^= torch.where(
                    locator_index <= degree, term, torch.zeros_like(term)
                )
            nonzero = discrepancy != 0
            old_locator = locator.clone()
            scale = self._div(discrepancy, previous_discrepancy)
            source_indices = coefficient_indices - shift.unsqueeze(1)
            source_valid = source_indices >= 0
            shifted_previous = previous.gather(1, source_indices.clamp_min(0))
            shifted_previous = torch.where(
                source_valid, shifted_previous, torch.zeros_like(shifted_previous)
            )
            candidate = locator ^ self._mul(shifted_previous, scale.unsqueeze(1))
            locator = torch.where(nonzero.unsqueeze(1), candidate, locator)
            major = nonzero & ((2 * degree) <= syndrome_index)
            degree = torch.where(major, syndrome_index + 1 - degree, degree)
            previous = torch.where(major.unsqueeze(1), old_locator, previous)
            previous_discrepancy = torch.where(major, discrepancy, previous_discrepancy)
            shift = torch.where(major, torch.ones_like(shift), shift + 1)
        valid = coefficient_indices <= degree.unsqueeze(1)
        return torch.where(valid, locator, torch.zeros_like(locator)), degree

    def chien_search(self, locator):
        roots = torch.zeros((locator.shape[0], self.n), dtype=torch.bool, device=self.device)
        for position in range(self.n):
            value = locator[:, 0].clone()
            for degree in range(1, self.nsym + 1):
                value ^= self._mul(locator[:, degree], self.chien[position, degree])
            roots[:, position] = value == 0
        return roots

    def error_evaluator(self, syndrome, locator):
        evaluator = torch.zeros_like(syndrome)
        for output_degree in range(self.nsym):
            value = torch.zeros(syndrome.shape[0], dtype=torch.int32, device=self.device)
            for locator_degree in range(output_degree + 1):
                value ^= self._mul(
                    locator[:, locator_degree], syndrome[:, output_degree - locator_degree]
                )
            evaluator[:, output_degree] = value
        return evaluator

    def forney_magnitudes(self, syndrome, locator, error_positions):
        evaluator = self.error_evaluator(syndrome, locator)
        magnitudes = torch.zeros_like(error_positions, dtype=torch.int32)
        zero_denominator = torch.zeros(error_positions.shape[0], dtype=torch.bool, device=self.device)
        for position in range(self.n):
            root = self.chien[position, 1]
            omega = evaluator[:, self.nsym - 1].clone()
            multiply_by_root = self.mul_table[:, root.long()]
            for degree in range(self.nsym - 2, -1, -1):
                omega = multiply_by_root[omega.long()] ^ evaluator[:, degree]
            derivative = torch.zeros(locator.shape[0], dtype=torch.int32, device=self.device)
            for degree in range(1, self.nsym + 1, 2):
                derivative ^= self._mul(locator[:, degree], self.chien[position, degree - 1])
            active = error_positions[:, position]
            zero_denominator |= active & (derivative == 0)
            magnitude = self._div(omega, derivative)
            magnitudes[:, position] = torch.where(active, magnitude, torch.zeros_like(magnitude))
        return magnitudes, zero_denominator

    def decode_blocks_batched(self, received):
        received = received.contiguous().to(torch.int32)
        syndrome = self.syndromes(received)
        syndrome_zero = (syndrome == 0).all(dim=1)
        corrected = received.clone()
        failing_indices = (~syndrome_zero).nonzero().flatten()
        if failing_indices.numel() == 0:
            return corrected, syndrome_zero
        failing_received = received[failing_indices]
        failing_syndrome = syndrome[failing_indices]
        locator, degree = self.berlekamp_massey(failing_syndrome)
        error_positions = self.chien_search(locator)
        roots_found = error_positions.sum(dim=1)
        valid = (degree > 0) & (degree <= self.t) & (roots_found == degree)
        plausible = valid.nonzero().flatten()
        failing_success = torch.zeros_like(valid)
        corrected_failing = failing_received.clone()
        if plausible.numel():
            magnitudes, zero_denominator = self.forney_magnitudes(
                failing_syndrome[plausible], locator[plausible], error_positions[plausible]
            )
            candidate = failing_received[plausible] ^ magnitudes
            post_zero = (self.syndromes(candidate) == 0).all(dim=1)
            plausible_success = (~zero_denominator) & post_zero
            corrected_failing[plausible] = torch.where(
                plausible_success.unsqueeze(1), candidate, failing_received[plausible]
            )
            failing_success[plausible] = plausible_success
        corrected[failing_indices] = corrected_failing
        success = syndrome_zero.clone()
        success[failing_indices] = failing_success
        return corrected, success

    def decode_blocks_fused(self, received):
        if received.device.type != "cuda":
            raise RuntimeError("The fused RS(24,9) decoder requires CUDA")
        received = received.contiguous().to(torch.int32)
        if self._cuda_extension is None:
            from torch.utils.cpp_extension import load

            source_dir = Path(__file__).resolve().parent
            build_dir = source_dir / ".rs249_cuda_build"
            build_dir.mkdir(exist_ok=True)
            self._cuda_extension = load(
                name="rs249_cuda_ext",
                sources=[
                    str(source_dir / "rs249_cuda.cpp"),
                    str(source_dir / "rs249_cuda_kernel.cu"),
                ],
                build_directory=str(build_dir),
                extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
                verbose=False,
            )
        return self._cuda_extension.decode(received, self.gf_exp, self.gf_log)

    def _extension(self):
        if self._cuda_extension is None:
            # Trigger the lazily compiled extension with an empty decode.
            clean = torch.zeros((1, 24), dtype=torch.int32, device=self.device)
            self.decode_blocks_fused(clean)
        return self._cuda_extension

    def _use_cuda_extension(self) -> bool:
        """Prefer the CUDA extension only on CUDA with its sources present.

        This folder ships without ``rs249_cuda.cpp`` / ``rs249_cuda_kernel.cu``;
        everywhere else the pure-torch pack/unpack/restore fallbacks below
        produce bit-identical results.
        """
        if self.device.type != "cuda":
            return False
        source_dir = Path(__file__).resolve().parent
        return (source_dir / "rs249_cuda.cpp").is_file() and (
            source_dir / "rs249_cuda_kernel.cu"
        ).is_file()

    def unpack_blocks(self, packed):
        if self._use_cuda_extension():
            return self._extension().unpack(packed.contiguous())
        return unpack_9bit_symbols(packed)

    def pack_blocks(self, symbols):
        if self._use_cuda_extension():
            return self._extension().pack(symbols.contiguous().to(torch.int32))
        return pack_9bit_symbols(symbols)

    def restore_high9(self, packed, elements):
        if self._use_cuda_extension():
            return self._extension().restore(packed.contiguous(), int(elements))
        high9 = unpack_9bit_symbols(packed.contiguous()).reshape(-1)[: int(elements)]
        return (high9 << 7).to(torch.int16).view(torch.float16)


def pack_9bit_symbols(symbols: torch.Tensor) -> torch.Tensor:
    """Pack ``(B,24)`` 9-bit symbols into exactly 27 bytes/codeword."""
    symbols = symbols.to(torch.int32)
    packed = torch.zeros(
        (symbols.shape[0], 27), dtype=torch.uint8, device=symbols.device
    )
    for index in range(24):
        bit_offset = index * 9
        byte_index = bit_offset // 8
        shift = bit_offset % 8
        value = symbols[:, index]
        packed[:, byte_index] |= ((value << shift) & 0xFF).to(torch.uint8)
        packed[:, byte_index + 1] |= (value >> (8 - shift)).to(torch.uint8)
    return packed


def unpack_9bit_symbols(packed: torch.Tensor) -> torch.Tensor:
    """Unpack ``(B,27)`` bytes into ``(B,24)`` int32 symbols."""
    packed = packed.to(torch.int32)
    symbols = torch.empty(
        (packed.shape[0], 24), dtype=torch.int32, device=packed.device
    )
    for index in range(24):
        bit_offset = index * 9
        byte_index = bit_offset // 8
        shift = bit_offset % 8
        symbols[:, index] = (
            (packed[:, byte_index] >> shift)
            | (packed[:, byte_index + 1] << (8 - shift))
        ) & 0x1FF
    return symbols
