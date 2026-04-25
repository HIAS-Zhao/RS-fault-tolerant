import torch
import torch.nn as nn
class ZMORP:

    @staticmethod
    def _calculate_parity_vectorized(x: torch.Tensor, num_bits=5) -> torch.Tensor:

        parity = torch.zeros_like(x)
        for i in range(num_bits):
            parity ^= ((x >> i) & 1)
        return parity

    @staticmethod
    def protect_model(model: nn.Module) -> None:

        count = 0
        for name, param in model.named_parameters():
            if param.dtype == torch.float16:
                param.data = ZMORP._add_protection_to_tensor(param.data)
                count += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
  
    @staticmethod
    def recover_model(model: nn.Module) -> None:
       
        count = 0
        for param in model.parameters():
            if param.dtype == torch.float16:
                param.data = ZMORP._recover_tensor(param.data)
                count += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
       
    @staticmethod
    def _add_protection_to_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.float16:
            return tensor

        original_shape = tensor.shape
        flat_tensor = tensor.view(-1)
        bits = flat_tensor.view(torch.int16)  # float16 → int16

        sign = (bits >> 15) & 0x1
        exponent = (bits >> 10) & 0x1F        # 5 bits
        mantissa = bits & 0x3FF               # 10 bits

        redundant_exp = exponent
        exp_parity = ZMORP._calculate_parity_vectorized(exponent, num_bits=5)
        red_parity = ZMORP._calculate_parity_vectorized(redundant_exp, num_bits=5)

        protection_bits = (red_parity << 6) | (redundant_exp << 1) | exp_parity  

        mantissa_high = mantissa & 0x380
        new_mantissa = mantissa_high | (protection_bits & 0x7F) 

        protected_bits = (sign << 15) | (exponent << 10) | new_mantissa

        del bits, sign, exponent, mantissa, redundant_exp, exp_parity, red_parity, protection_bits

        return protected_bits.view(torch.float16).view(original_shape)

    @staticmethod
    def _recover_tensor(tensor: torch.Tensor) -> torch.Tensor:

        if tensor.dtype != torch.float16:
            return tensor

        original_shape = tensor.shape
        flat_tensor = tensor.view(-1)
        bits = flat_tensor.view(torch.int16)

        sign = (bits >> 15) & 0x1
        exponent = (bits >> 10) & 0x1F
        mantissa = bits & 0x3FF


        protected_part = mantissa  
        redundant_exp = (protected_part >> 1) & 0x1F      
        exp_parity = protected_part & 0x1                 # LSB
        red_parity = (protected_part >> 6) & 0x1          # MSB of mantissa

        expected_exp_parity = ZMORP._calculate_parity_vectorized(exponent, num_bits=5)
        expected_red_parity = ZMORP._calculate_parity_vectorized(redundant_exp, num_bits=5)

        exp_ok = (exp_parity == expected_exp_parity)
        red_ok = (red_parity == expected_red_parity)

        corrected_exp = exponent.clone()


        mask2 = (~exp_ok) & red_ok
        if mask2.any():
            corrected_exp[mask2] = redundant_exp[mask2]

   
        mask3 = (~exp_ok) & (~red_ok)
        if mask3.any():
            exp_vals = exponent[mask3]
            red_vals = redundant_exp[mask3]
            corrected_vals = torch.zeros_like(exp_vals)

            for bit in range(5):  
                exp_bit = (exp_vals >> bit) & 1
                red_bit = (red_vals >> bit) & 1
                agreed_bit = torch.where(exp_bit == red_bit, exp_bit,
                                         torch.tensor(0, dtype=exp_vals.dtype, device=exp_vals.device))
                corrected_vals += (agreed_bit << bit)

            corrected_exp[mask3] = corrected_vals


        clean_mantissa = mantissa & 0x380  

        recovered_bits = (sign << 15) | (corrected_exp << 10) | clean_mantissa

        del bits, sign, exponent, mantissa, redundant_exp, exp_parity, red_parity, \
            expected_exp_parity, expected_red_parity, exp_ok, red_ok, mask2, mask3, \
            corrected_exp, clean_mantissa

        return recovered_bits.view(torch.float16).view(original_shape)