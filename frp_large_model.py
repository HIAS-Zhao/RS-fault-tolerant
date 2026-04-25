import torch
from tqdm import tqdm
class FRP:
    def __init__(self, device='cuda'):
        self.device = device
        self._init_galois_field()
        self._init_bch_matrices()
        self._init_error_tables()
        self.block_size = 2048*2048 
    def _init_galois_field(self):
        primitive_poly = 0x25
        size = 32
        gf_antilog = torch.zeros(size, dtype=torch.uint8, device=self.device)
        gf_log = torch.zeros(size, dtype=torch.uint8, device=self.device)
        gf_antilog[0] = 1
        gf_log[1] = 0
        for i in range(1, 31):
            prev = int(gf_antilog[i-1].item())
            x = prev << 1

            if x & 0x20:
                x ^= primitive_poly
            x &= 0x1F
            gf_antilog[i] = x
            gf_log[x] = i

        self.gf_log = gf_log
        self.gf_antilog = gf_antilog
    
    def _init_bch_matrices(self):
       
        self.k = 16
        self.n = 31
        g_coeffs = [1,1,1,1,0,1,0,1,1,1,1,1,0,0,0,1]

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
    
    def _init_error_tables(self):

        self.error_tables = {}

    


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
        res = torch.zeros_like(a, dtype=torch.long, device=self.device)
        mask = (a != 0) & (b != 0)
        if mask.any():
            la = self.gf_log[a[mask]].long()
            lb = self.gf_log[b[mask]].long()
            exp = (la + lb) % 31
            res_mask = self.gf_antilog[exp].long()
            res[mask] = res_mask
        return res

    def _gf_inv_tensor(self, a: torch.Tensor) -> torch.Tensor:
        res = torch.zeros_like(a, dtype=torch.long, device=self.device)
        mask = (a != 0)
        if mask.any():
            la = self.gf_log[a[mask]].long()
            exp = (-la) % 31
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
        for j in range(1, 2 * t + 1):
            exps = (j * torch.arange(n, device=device)) % 31
            alpha_pows = self.gf_antilog[exps].long()  # (n,)
            sel = (bits.long() * alpha_pows.unsqueeze(0))  # (B,n)
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
        three_error_masks = ((S1S3 ^ S2) != 0)
        two_one_error_masks = (S1 != 0) & ((S1S3 ^ S2) == 0)
        #one_error_masks = (S1 == 0) & ((S1S3 ^ S2) == 0)
        
        #sigma[one_error_masks, 1] = S1[one_error_masks]
        
        sigma[two_one_error_masks, 1] = S1[two_one_error_masks]
        S1_inv = self._gf_inv_tensor(S1)
        sigma[two_one_error_masks, 2] = (self._gf_mul_tensor(S3, S1_inv)^S2)[two_one_error_masks]
        
        sigma[three_error_masks, 1] = S1[three_error_masks]
        S1_sq = self._gf_mul_tensor(S1, S1)
        S1_cube = self._gf_mul_tensor(S1_sq, S1)
        result1 = (self._gf_mul_tensor(S1_sq, S3) ^ S5)
        result2 = (self._gf_inv_tensor(S1_cube^S3)) 
        result=self._gf_mul_tensor(result1, result2)
        sigma[three_error_masks, 2] = result[three_error_masks]
        sigma[three_error_masks, 3] = (S1_cube^S3^self._gf_mul_tensor(S1, result))[three_error_masks]

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
                    exp = (-i * j) % 31
                    #alpha_val = int(self.gf_antilog[exp].item())
                    alpha_val = self.gf_antilog[exp].long()
                    term = self._gf_mul_tensor(coeff, torch.full_like(coeff, alpha_val))
                val ^= term
            error_mask[:, i] = (val == 0)
        return error_mask
    
    
    

    def encode(self, model):
        device = self.device
        positions16 = torch.arange(0, self.k, device=device, dtype=torch.int32)
        positions31 = torch.arange(0, self.n, device=device, dtype=torch.int32)
        for name, param in tqdm(model.named_parameters(), desc="BCH Encoding Inplace"):
            if "classwise" not in name and "mask" not in name:
                flat = param.data.view(-1).to(device)
                total = flat.numel()
                encoded_flat = torch.empty((total,), dtype=torch.int32, device=device)
                for i in range(0, total, self.block_size):
                    j = min(i + self.block_size, total)
                    chunk = flat[i:j]
                    u16 = chunk.view(torch.int16).to(device).to(torch.int32) & 0xFFFF
                    bits16 = ((u16.unsqueeze(1) >> positions16) & 1).to(torch.uint8)
                    enc_bits = (bits16.float() @ self.G.float()).remainder(2).to(torch.int32)
                    codewords = (enc_bits.to(torch.int32) << positions31).sum(dim=1).to(torch.int32)
                    encoded_flat[i:j] = codewords
                    # free temporaries
                    del u16, bits16, enc_bits, codewords
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
        mask16 = (1 << self.k) - 1
        positions = torch.arange(0, self.n, device=device, dtype=torch.int32)
        
       
        for name, param in tqdm(model.named_parameters(), desc="BCH Decoding Inplace"):

            if "mask" in name or "classwise" in name:
                continue

       
            encoded_flat = param.data.view(-1)
           
            if not encoded_flat.dtype.is_floating_point and encoded_flat.dtype == torch.int32:
                encoded_flat = encoded_flat.to(device)
            else:
                
                continue

            total = encoded_flat.numel()
            decoded_flat = torch.empty((total,), dtype=torch.float16, device=device)
            for i in range(0, total, self.block_size):
                j = min(i + self.block_size, total)
                chunk = encoded_flat[i:j].to(device)
                synd = self._compute_syndromes_tensor(chunk)

                sigma = self._peterson_tensor(synd)
                err_mask = self._chien_search_tensor(sigma)

                errors_count = err_mask.long().sum(dim=1)

                uncorrectable_mask = errors_count > 3
                
                if uncorrectable_mask.any():
                    count+=uncorrectable_mask.sum().item()
                
                pos_weights = (1 << positions).to(device).long()
                flips = (err_mask.long() * pos_weights.unsqueeze(0)).sum(dim=1)

                if uncorrectable_mask.any():
                    flips[uncorrectable_mask] = 0
                corrected = chunk ^ flips
                corrected[uncorrectable_mask]=corrected[uncorrectable_mask]&0xbfff
                message = (corrected >> shift) & mask16
                u16=message

               
                u16_masked = (u16 & 0xFFFF).to(torch.int16)
                recovered = u16_masked.view(torch.float16)
                decoded_flat[i:j] = recovered
                del synd, sigma, err_mask, flips, corrected, message, u16
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