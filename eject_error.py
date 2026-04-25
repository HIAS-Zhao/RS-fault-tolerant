import torch
from tqdm import tqdm
def inject_error_to_tensor(tensor, error_rate=1e-6, seed=None, chunk_size=2048*2048):

    if seed is not None:
        torch.manual_seed(seed)
        if tensor.is_cuda:
            torch.cuda.manual_seed(seed)

    device = tensor.device
    dtype_orig = tensor.dtype
    
    if dtype_orig == torch.float16:
        bit_width = 16
        int_dtype = torch.int16
        working_dtype = torch.float16  
    elif dtype_orig == torch.bfloat16:
        bit_width = 16
        int_dtype = torch.int16
        working_dtype = torch.bfloat16  
    elif dtype_orig == torch.float32:
        bit_width = 32
        int_dtype = torch.int32
        working_dtype = torch.float32
    elif dtype_orig == torch.float64:
        bit_width = 64
        int_dtype = torch.int64
        working_dtype = torch.float64
    elif dtype_orig == torch.int64:
        bit_width = 64
        int_dtype = torch.int64
        working_dtype = torch.int64
    else:
        
        raise NotImplementedError(f"Unsupported dtype: {dtype_orig}")

    if tensor.dtype != working_dtype:
        tensor = tensor.to(working_dtype)

    original_shape = tensor.shape
    tensor_flat = tensor.view(-1)
    num_elements = tensor_flat.numel()
    corrupted_flat = tensor_flat.clone()
 
    for i in range(0, num_elements, chunk_size):
        end = min(i + chunk_size, num_elements)
        chunk = tensor_flat[i:end]  # shape: [C]
        C = chunk.numel()
        total_bits = C * bit_width

        rand_bits = torch.rand(total_bits, device=device, dtype=torch.float64)
        flip_mask_bits = rand_bits < error_rate  # [C * bit_width]
        if flip_mask_bits.any():
            flip_mask_bits = flip_mask_bits.view(C, bit_width)
            #flip_copy=flip_mask_bits.clone()

            
            bit_positions = torch.arange(bit_width, dtype=torch.int64, device=device)
            # Use int64 for shift to avoid overflow in shift (PyTorch shift requires scalar or same dtype?)
            flip_mask_int = (flip_mask_bits.to(torch.int64) << bit_positions).sum(dim=1).to(int_dtype)

            chunk_int = chunk.view(int_dtype)  # reinterpret bits as integer
            corrupted_chunk_int = chunk_int ^ flip_mask_int

            corrupted_flat[i:end] = corrupted_chunk_int.view(working_dtype)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    corrupted = corrupted_flat.view(original_shape).to(dtype_orig)
    return corrupted

def inject_error_to_model(model, ber=1e-6, seed=1024, chunk_size=2048*2048):
    for name, module in tqdm(model.named_modules(), desc=f"BER={ber} ejecting error"):
                if hasattr(module, 'weight') and module.weight is not None:
                    corrupted_weight = inject_error_to_tensor(module.weight.data.clone(), error_rate=ber, seed=seed)
                    module.weight.data.copy_(corrupted_weight)