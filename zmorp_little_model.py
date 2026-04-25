import torch
import torch.nn as nn
class ZMORP:

    @staticmethod
    def _calculate_parity_vectorized(x: torch.Tensor, num_bits: int = 7) -> torch.Tensor:

        parity = torch.zeros_like(x)
        for i in range(num_bits):
            parity ^= (x >> i) & 1
        return parity

    @staticmethod
    def protect_model(model: torch.nn.Module) -> None:

        count = 0
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = ZMORP._add_protection_to_tensor(param.data)
                count += 1


    @staticmethod
    def recover_model(model: torch.nn.Module) -> None:

        count = 0
        for param in model.parameters():
            if param.dtype == torch.float32:
                param.data = ZMORP._recover_tensor(param.data)
                count += 1


    @staticmethod
    def _add_protection_to_tensor(tensor: torch.Tensor) -> torch.Tensor:

        if tensor.dtype != torch.float32:
            return tensor

        original_shape = tensor.shape
        flat = tensor.view(-1)
        bits = flat.view(torch.int32)

        sign = (bits >> 31) & 0x1
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF


        exp_low7 = exponent & 0x7F  # 0b01111111

  
        exp_parity = ZMORP._calculate_parity_vectorized(exp_low7, num_bits=7)

        red_parity = exp_parity.clone()


        new_mantissa = (mantissa & 0xFFFFFF80) | (exp_low7 & 0x7F)


        new_mantissa = (new_mantissa & 0xFFFFFF7F) | (red_parity << 7)


        new_exponent = (exp_parity << 7) | exp_low7

        protected_bits = (sign << 31) | (new_exponent << 23) | new_mantissa

        return protected_bits.view(torch.float32).view(original_shape)

    @staticmethod
    def _recover_tensor(tensor: torch.Tensor) -> torch.Tensor:

        if tensor.dtype != torch.float32:
            return tensor

        original_shape = tensor.shape
        flat = tensor.view(-1)
        bits = flat.view(torch.int32)

        sign = (bits >> 31) & 0x1
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF


        redundant_low7 = mantissa & 0x7F
        

        red_parity_stored = (mantissa >> 7) & 1

        stored_exp_low7 = exponent & 0x7F          
        exp_parity_stored = (exponent >> 7) & 1    

        expected_exp_parity = ZMORP._calculate_parity_vectorized(stored_exp_low7, num_bits=7)
        expected_red_parity = ZMORP._calculate_parity_vectorized(redundant_low7, num_bits=7)


        exp_ok = (exp_parity_stored == expected_exp_parity)
        red_ok = (red_parity_stored == expected_red_parity)

        corrected_low7 = stored_exp_low7.clone()
        
       
        mask2 = (~exp_ok) & red_ok
        if mask2.any():
            corrected_low7[mask2] = redundant_low7[mask2]


        mask3 = (~exp_ok) & (~red_ok)
        if mask3.any():
            exp_vals = stored_exp_low7[mask3]
            red_vals = redundant_low7[mask3]
            corrected_vals = torch.zeros_like(exp_vals)

            for bit in range(7):
                e_bit = (exp_vals >> bit) & 1
                r_bit = (red_vals >> bit) & 1

                agreed_bit = torch.where(e_bit == r_bit, e_bit, torch.tensor(1, dtype=exp_vals.dtype, device=exp_vals.device))
                corrected_vals |= (agreed_bit << bit)

            corrected_low7[mask3] = corrected_vals

       
        recovered_exponent = (0 << 7) | (corrected_low7 & 0x7F)

       
        clean_mantissa = mantissa & 0xFFFFFF00

        recovered_bits = (sign << 31) | (recovered_exponent << 23) | clean_mantissa

        return recovered_bits.view(torch.float32).view(original_shape)