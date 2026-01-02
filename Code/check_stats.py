import torch
import os

# 1. Get size of your compressed file
compressed_size = os.path.getsize('compressed_gpt2.pt') / (1024 * 1024) # In MB

# 2. Calculate Original Size of those 7 layers
# Each layer is [768, 3072] floats (4 bytes each)
# 7 layers * 768 * 3072 * 4 bytes
original_params = 7 * 768 * 3072
original_size = (original_params * 4) / (1024 * 1024) # In MB

print(f"--- FINAL STATS ---")
print(f"Original Size (7 Layers): {original_size:.2f} MB")
print(f"Compressed Size (7 Layers): {compressed_size:.2f} MB")
print(f"Space Saved: {original_size - compressed_size:.2f} MB")
print(f"Compression Ratio: {100 * (1 - compressed_size/original_size):.2f}%")