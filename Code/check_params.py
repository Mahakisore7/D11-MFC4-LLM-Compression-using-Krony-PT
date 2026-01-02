import torch

# Load the compressed data
data = torch.load('compressed_gpt2.pt')

total_original_params = 0
total_compressed_params = 0

print(f"{'LAYER':<15} | {'ORIGINAL':<10} | {'COMPRESSED':<10} | {'RATIO':<10}")
print("-" * 55)

for layer_name, layer_data in data.items():
    # 1. Calculate Original Size (768 * 3072)
    orig = 768 * 3072
    
    # 2. Calculate Compressed Size
    # Rank of Low-Rank parts
    U, V = layer_data['U'], layer_data['V']
    rank = U.shape[1]
    
    # Low Rank Params: size(U) + size(V) + rank (singular values)
    lr_params = U.numel() + V.numel() + rank
    
    # Sparse Params: Number of non-zero values
    sparse_vals = layer_data['Sparse_Values'].numel()
    
    # Note: We count indices as parameters here for fairness, 
    # though in hardware they are integers (smaller).
    # Sparse storage cost approx = 2 * values (1 index + 1 val)
    comp = lr_params + (2 * sparse_vals)
    
    total_original_params += orig
    total_compressed_params += comp
    
    ratio = 100 * (1 - comp/orig)
    print(f"{layer_name:<15} | {orig:<10} | {comp:<10} | {ratio:.2f}%")

print("-" * 55)
print(f"TOTAL ORIGINAL:   {total_original_params:,}")
print(f"TOTAL COMPRESSED: {total_compressed_params:,}")
print(f"FINAL COMPRESSION RATIO: {100 * (1 - total_compressed_params/total_original_params):.2f}%")