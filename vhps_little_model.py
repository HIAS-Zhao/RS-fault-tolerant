
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import torch
layers=[]
class ExponentClamper:
   
    @staticmethod
    def clamp_exponent_to_0(model: torch.nn.Module) -> None:
        count = 0
        total_elements = 0
        for name, param in model.named_parameters():
            if any(name.startswith(layer) for layer in layers):
                if param.dtype != torch.float32:
                    continue

                original_shape = param.shape
                flat = param.data.view(-1)
                bits = flat.view(torch.int32)

                
                sign = bits >> 31                          # 1 bit
                exp = (bits >> 23) & 0xFF                  # 8 bits
                mantissa = bits & 0x7FFFFF                 # 23 bits

                
                exp_low7 = exp & 0b01111111  # 即 exp & 0x1F

                 
                new_exp = (0b0 << 7) | exp_low7 

                
                new_bits = (sign << 31) | (new_exp << 23) | mantissa

                # 写回
                param.data = new_bits.view(torch.float32).view(original_shape)

                count += 1
                total_elements += flat.numel()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


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
            if any(name.startswith(layer) for layer in layers):
                    continue
            if param.dtype == torch.float32:
                param.data = ZMORP._add_protection_to_tensor(param.data)
                count += 1


    @staticmethod
    def recover_model(model: torch.nn.Module) -> None:

        count = 0
        for name, param in model.named_parameters():
            if any(name.startswith(layer) for layer in layers):
                    continue
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
class FRP:

    def __init__(self, device='cuda'):
        self.device = device
        self._init_galois_field()
        self._init_bch_matrices()

        self.block_size = 2048*2048 

    
    def _init_galois_field(self):

        primitive_poly = 0x43
        size = 64
        gf_antilog = torch.zeros(size, dtype=torch.uint8, device=self.device)
        gf_log = torch.zeros(size, dtype=torch.uint8, device=self.device)

        gf_antilog[0] = 1
        gf_log[1] = 0
        for i in range(1, 63):
            prev = int(gf_antilog[i-1].item())
            x = prev << 1

            if x & 0x40:
                x ^= primitive_poly
            x &= 0x3F
            gf_antilog[i] = x
            gf_log[x] = i

        self.gf_log = gf_log
        self.gf_antilog = gf_antilog
    
    def _init_bch_matrices(self):

        self.k = 45
        self.n = 63

        g_coeffs = [1,1,1,1,0,0,1,1,0,1,0,0,0,0,0,1,1,1,1]

        g = torch.tensor(g_coeffs, dtype=torch.uint8, device=self.device)
        self.G_poly = g

        g_int = 0
        for i, b in enumerate(g_coeffs):
            if b & 1:
                g_int |= (1 << i)
        self.g_int = g_int

        rows = []
        for i in range(self.k):
            m = 1 << i
            shifted = m << (self.n - self.k)
            remainder = self._poly_mod(shifted, self.g_int)
            codeword = shifted ^ remainder

            row = [(codeword >> j) & 1 for j in range(self.n)]
            rows.append(row)
        self.G = torch.tensor(rows, dtype=torch.uint8, device=self.device)

        P = self.G[:, self.k:]
        H = torch.cat([P.t(), torch.eye(self.n - self.k, dtype=torch.uint8, device=self.device)], dim=1)
        self.H = H

    def _poly_deg(self, poly_int):
        if poly_int == 0:
            return -1
        return poly_int.bit_length() - 1

    def _poly_mod(self, dividend, divisor):

        dv_deg = self._poly_deg(divisor)
        tmp = int(dividend)
        while True:
            tdeg = self._poly_deg(tmp)
            if tdeg < dv_deg:
                break
            shift = tdeg - dv_deg
            tmp ^= (divisor << shift)
        return tmp

  
    def _gf_mul_tensor(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: long tensors on self.device containing GF element integers (0..63)
        res = torch.zeros_like(a, dtype=torch.long, device=self.device)
        mask = (a != 0) & (b != 0)
        if mask.any():
            la = self.gf_log[a[mask]].long()
            lb = self.gf_log[b[mask]].long()
            exp = (la + lb) % 63
            res_mask = self.gf_antilog[exp].long()
            res[mask] = res_mask
        return res

    def _gf_mul_3_tensor(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        res = torch.zeros_like(a, dtype=torch.long, device=self.device)
        mask = (a != 0) & (b != 0)& (c != 0)
        if mask.any():
            la = self.gf_log[a[mask]].long()
            lb = self.gf_log[b[mask]].long()
            lc = self.gf_log[c[mask]].long()
            exp = (la + lb + lc) % 63
            res_mask = self.gf_antilog[exp].long()
            res[mask] = res_mask
        return res
    
    def _gf_mul_4_tensor(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # a,b,c,d: long tensors on self.device containing GF element integers (0..63)
        res = torch.zeros_like(a, dtype=torch.long, device=self.device)
        mask = (a != 0) & (b != 0)& (c != 0)& (d != 0)
        if mask.any():
            la = self.gf_log[a[mask]].long()
            lb = self.gf_log[b[mask]].long()
            lc = self.gf_log[c[mask]].long()
            ld = self.gf_log[d[mask]].long()
            exp = (la + lb + lc + ld) % 63
            res_mask = self.gf_antilog[exp].long()
            res[mask] = res_mask
        return res
        
    def _gf_mul_5_tensor(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        res = torch.zeros_like(a, dtype=torch.long, device=self.device)
        mask = (a != 0) & (b != 0)& (c != 0)& (d != 0)& (e != 0)
        if mask.any():
            la = self.gf_log[a[mask]].long()
            lb = self.gf_log[b[mask]].long()
            lc = self.gf_log[c[mask]].long()
            ld = self.gf_log[d[mask]].long()
            le = self.gf_log[e[mask]].long()
            exp = (la + lb + lc + ld + le) % 63
            res_mask = self.gf_antilog[exp].long()
            res[mask] = res_mask
        return res
    def _gf_inv_tensor(self, a: torch.Tensor) -> torch.Tensor:
        res = torch.zeros_like(a, dtype=torch.long, device=self.device)
        mask = (a != 0)
        if mask.any():
            la = self.gf_log[a[mask]].long()
            exp = (-la) % 63
            res[mask] = self.gf_antilog[exp].long()
        return res

    
    def _compute_syndromes_tensor(self, codewords: torch.Tensor) -> torch.Tensor:
        # codewords: (B,) int64 on device, returns syndromes (B, 2t) as long on device
        device = self.device
        B = codewords.size(0)
        t = 3
        n = self.n
        bits = ((codewords.unsqueeze(1) >> torch.arange(n, device=device)) & 1).to(torch.uint8)
        synd = torch.zeros((B, 2 * t), dtype=torch.long, device=device)
        # precompute alpha_pows for positions 0..n-1 for each j
        for j in range(1, 2 * t + 1):
            # alpha_pows: length n, element = alpha^{j*i}
            exps = (j * torch.arange(n, device=device)) % 63
            alpha_pows = self.gf_antilog[exps].long()  # (n,)
            # masked selection: for each codeword, XOR alpha_pows where bit==1
            sel = (bits.long() * alpha_pows.unsqueeze(0))  # (B,n)
            # XOR reduce across n
            s = torch.zeros(B, dtype=torch.long, device=device)
            for idx in range(n):
                s ^= sel[:, idx]
            synd[:, j - 1] = s
        return synd

    
    
    def _peterson_tensor(self, synd: torch.Tensor) -> torch.Tensor:
        device = self.device
        B = synd.size(0)
        t=3
        sigma = torch.zeros((B, t + 1), dtype=torch.long, device=device)
        sigma[:, 0] = 1  # σ0 = 1


        S1 = synd[:, 0]
        S2 = synd[:, 1]
        S3 = synd[:, 2]
        S4 = synd[:, 3]
        S5 = synd[:, 4]
        S6 = synd[:, 5]
        
        S1S3 = self._gf_mul_tensor(S1, S3)
        det_three=(S1S3^S2)
        #five_mask=(det_five != 0)
        
        # four_mask=(det_four != 0)  
        
        three_mask=(det_three != 0)  
        #three_mask=(det_three != 0) 
        
        two_one_mask=(S1 != 0) & (det_three == 0)
        
        
        sigma[two_one_mask, 1] = S1[two_one_mask]
        S1_inv = self._gf_inv_tensor(S1)
        sigma[two_one_mask, 2] = (self._gf_mul_tensor(S3, S1_inv)^S2)[two_one_mask]
        
        

        if three_mask.any():
            sigma[three_mask, 1] = S1[three_mask]
            S1_sq = self._gf_mul_tensor(S1, S1)
            S1_cube = self._gf_mul_tensor(S1_sq, S1)
            result1 = (self._gf_mul_tensor(S1_sq, S3) ^ S5)
            result2 = (self._gf_inv_tensor(S1_cube^S3)) 
            result=self._gf_mul_tensor(result1, result2)
            sigma[three_mask, 2] = result[three_mask]
            sigma[three_mask, 3] = (S1_cube^S3^self._gf_mul_tensor(S1, result))[three_mask]

        

        return sigma

    def _chien_search_tensor(self, sigma: torch.Tensor) -> list:
        # sigma: (B, t+1) tensor of GF elements; returns error mask (B, n) bool on device
        device = self.device
        B = sigma.size(0)
        n = self.n
        error_mask = torch.zeros((B, n), dtype=torch.bool, device=device)
        for i in range(n):
            val = torch.zeros(B, dtype=torch.long, device=device)
            for j in range(sigma.size(1)):
                coeff = sigma[:, j]
                if j == 0:
                    term = coeff
                else:
                    exp = (-i * j) % 63
                    alpha_val = self.gf_antilog[exp].long()
                    term = self._gf_mul_tensor(coeff, torch.full_like(coeff, alpha_val))
                val ^= term
            error_mask[:, i] = (val == 0)
        return error_mask
    
    
    

    def encode(self, model):

        device = self.device
        positions45 = torch.arange(0, self.k, device=device, dtype=torch.int64)
        positions63 = torch.arange(0, self.n, device=device, dtype=torch.int64)
        for name, param in tqdm(model.named_parameters(), desc="BCH Encoding Inplace"):
            if any(name.startswith(layer) for layer in layers):
                if "classwise" not in name and "mask" not in name:
                    flat = param.data.view(-1).to(device)
                    total = flat.numel()
     
                    encoded_flat = torch.empty((total,), dtype=torch.long, device=device)
                    for i in range(0, total, self.block_size):
                        j = min(i + self.block_size, total)
                        chunk = flat[i:j]
   
                        u32 = chunk.view(torch.int32).to(device).to(torch.int64) & 0xFFFFFFFF
  
                        m45 = (u32 << 13) & ((1 << self.k) - 1)
                        bits45 = ((m45.unsqueeze(1) >> positions45) & 1).to(torch.uint8)
                        enc_bits = (bits45.float() @ self.G.float()).remainder(2).to(torch.int64)
                        codewords = (enc_bits.to(torch.int64) << positions63).sum(dim=1)
                        encoded_flat[i:j] = codewords
  
                        del u32, m45, bits45, enc_bits, codewords
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()

                    encoded_tensor = encoded_flat.view(param.shape).to(param.device)

                    try:
                        orig_req = param.requires_grad
                        param._bch_orig_requires_grad = orig_req
                        param.requires_grad = False
                    except Exception:
                        orig_req = None
                    param.data = encoded_tensor

                    try:
                        param._bch_encoded = True
                    except Exception:
                        pass

    def decode(self, model):
  
        device = self.device
        shift = self.n - self.k
        mask45 = (1 << self.k) - 1
        positions = torch.arange(0, self.n, device=device, dtype=torch.int64)
        
        for name, param in tqdm(model.named_parameters(), desc="BCH Decoding Inplace"):
            if any(name.startswith(layer) for layer in layers):

                if "mask" in name or "classwise" in name:
                    continue

   
                encoded_flat = param.data.view(-1)

                if not encoded_flat.dtype.is_floating_point and encoded_flat.dtype == torch.long:
                    encoded_flat = encoded_flat.to(device)
                else:
                    continue

                total = encoded_flat.numel()
                decoded_flat = torch.empty((total,), dtype=torch.float32, device=device)
                for i in range(0, total, self.block_size):
                    j = min(i + self.block_size, total)
                    chunk = encoded_flat[i:j].to(device)
                    synd = self._compute_syndromes_tensor(chunk)

                    sigma = self._peterson_tensor(synd)
                    err_mask = self._chien_search_tensor(sigma)
  
                    errors_count = err_mask.long().sum(dim=1)
  
                    uncorrectable_mask = errors_count > 3
                    
                    pos_weights = (1 << positions).to(device).long()
                    flips = (err_mask.long() * pos_weights.unsqueeze(0)).sum(dim=1)

                    if uncorrectable_mask.any():
                        flips[uncorrectable_mask] = 0
                    corrected = chunk ^ flips
                    message = (corrected >> shift) & mask45
                    u32 = (message >> 13) & 0xFFFFFFFF

                    
                    u32_masked = (u32 & 0xFFFFFFFF).to(torch.int32)
                    recovered = u32_masked.view(torch.float32)
                    decoded_flat[i:j] = recovered
                    del synd, sigma, err_mask, flips, corrected, message, u32
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                decoded_tensor = decoded_flat.view(param.shape).to(param.device)

                param.data = decoded_tensor
                

                try:
                    if hasattr(param, '_bch_orig_requires_grad'):
                        param.requires_grad = param._bch_orig_requires_grad
                        delattr(param, '_bch_orig_requires_grad')
                except Exception:
                    pass
                try:
                    delattr(param, '_bch_encoded')
                except Exception:
                    pass
        ExponentClamper.clamp_exponent_to_0(model)



def protect(model,layer,device="cuda"):
    layers.append(layer)
    protector1 = ZMORP()
    protector2=FRP(device=device)
    protector1.protect_model(model)
    protector2.encode(model)

def recover(model,layer,device="cuda"):
    layers.append(layer)
    protector1 = ZMORP()
    protector2=FRP(device=device)    
    protector1.recover_model(model)
    protector2.decode(model)
