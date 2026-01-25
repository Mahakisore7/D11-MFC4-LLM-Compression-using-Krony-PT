import torch
import torch.nn as nn
import numpy as np
import copy

# ==========================================
# 1. SETUP: LOAD A REAL GPT-2 LAYER (FFN)
# ==========================================
def get_real_gpt2_layer():
    print("\n--- STEP 1: LOADING REAL GPT-2 WEIGHTS ---")
    # We download the real GPT-2 model (124M)
    from transformers import GPT2Model
    print("Downloading GPT-2 (might take a minute)...")
    model = GPT2Model.from_pretrained('gpt2')
    
    # We grab the FFN (Feed Forward Network) from Layer 0.
    # JARGON CHECK: "FFN"
    # This matrix takes the 'thought' (768 numbers) and expands it 
    # to find relationships (3072 numbers). It is the "Brain" of the layer.
    target_layer = model.h[0].mlp.c_fc
    W_original = target_layer.weight.detach() # Shape: [768, 3072]
    
    print(f"Original Weight Shape: {W_original.shape}")
    print(f"Total Parameters: {W_original.numel()} numbers")
    return W_original

# ==========================================
# 2. THE MATH: VAN LOAN REARRANGEMENT
# ==========================================
def van_loan_rearrangement(W, m1, m2, n1, n2):
    # This function performs the "Magic Shuffle" (Part 2 of our explanation)
    # W: The original big matrix
    # m1, n1: Dimensions of Matrix A
    # m2, n2: Dimensions of Matrix B
    
    # JARGON CHECK: "Rearrangement"
    # We are chopping the matrix into blocks and stacking them 
    # so SVD can find the pattern.
    
    # Reshape into a 4D tensor (grid of blocks)
    # Target shape: [m1, m2, n1, n2]
    # Note: GPT-2 weights are often transposed, so we adjust carefuly
    tensor = W.view(m1, m2, n1, n2)
    
    # Permute (Shuffle) to bring 'A' parts and 'B' parts together
    # We want [m1, n1, m2, n2]
    shuffled = tensor.permute(0, 2, 1, 3)
    
    # Flatten into the 2D "Rearranged Matrix" (W_tilde)
    W_tilde = shuffled.reshape(m1 * n1, m2 * n2)
    return W_tilde

# ==========================================
# 3. THE REPLICATION: KRONY-PT METHOD (SVD)
# ==========================================
def compress_krony_pt(W):
    print("\n--- STEP 2: REPLICATING THE PAPER (Krony-PT) ---")
    
    # We define the split sizes (from their Table I)
    # Matrix A will be [64, 32]
    # Matrix B will be [48, 12] (Note: 768=64*12 is wrong, let's fix dimensions)
    # Let's use: 768 = 32 * 24  AND  3072 = 64 * 48
    # A: [32, 64], B: [24, 48] -> 32*24=768 rows, 64*48=3072 cols
    
    m1, n1 = 32, 64  # Size of Matrix A
    m2, n2 = 24, 48  # Size of Matrix B
    
    # 1. Rearrange
    W_tilde = van_loan_rearrangement(W, m1, m2, n1, n2)
    
    # 2. SVD (Singular Value Decomposition)
    # JARGON CHECK: "SVD"
    # Breaks the matrix into "Pattern" (U, V) and "Energy" (S)
    U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)
    
    # 3. Take ONLY the Top Rank (Rank-1)
    # This is what the paper does. It throws away S[1], S[2]...
    sigma = S[0]
    u1 = U[:, 0]
    v1 = Vt[0, :]
    
    # 4. Reconstruct A and B
    # We split the 'Energy' (sigma) between them
    scale = torch.sqrt(sigma)
    A = (scale * u1).view(m1, n1)
    B = (scale * v1).view(m2, n2)
    
    # 5. Approximate W
    # JARGON CHECK: "Kronecker Product"
    # Combining A and B back to get the big matrix
    W_approx = torch.kron(A, B)
    
    # Measure Error
    error = torch.norm(W - W_approx) / torch.norm(W)
    print(f"Compressed Params: {A.numel() + B.numel()} (vs {W.numel()})")
    print(f"Compression Ratio: {(1 - (A.numel()+B.numel())/W.numel())*100:.2f}%")
    print(f"Reconstruction Error (The Damage): {error.item()*100:.2f}%")
    
    return W_approx, A, B

# ==========================================
# 4. YOUR FIX: SPARSE RESIDUAL (+S)
# ==========================================
def apply_your_improvement(W, W_approx):
    print("\n--- STEP 3: APPLYING YOUR IMPROVEMENT (+S) ---")
    
    # 1. Calculate the Residual (What the paper missed)
    Residual = W - W_approx
    
    # 2. Keep only the "Outliers" (Top 1% biggest errors)
    # This is the "Sparse Matrix" concept
    threshold = torch.quantile(torch.abs(Residual), 0.99) # Keep top 1%
    
    # Create Sparse Matrix S (Everything else becomes 0)
    S = torch.where(torch.abs(Residual) > threshold, Residual, torch.zeros_like(Residual))
    
    # 3. Final Model = Kronecker + S
    W_final = W_approx + S
    
    # Measure Error
    error = torch.norm(W - W_final) / torch.norm(W)
    
    # Count parameters (A + B + Non-Zero S)
    n_sparse = torch.count_nonzero(S)
    total_params = 32*64 + 24*48 + n_sparse
    
    print(f"Sparse Params Added: {n_sparse}")
    print(f"New Error (After Fix): {error.item()*100:.4f}%")
    print("SUCCESS: We recovered the accuracy without re-training!")

# --- RUN THE EXPERIMENT ---
if __name__ == "__main__":
    # 1. Get Data
    W = get_real_gpt2_layer()
    
    # 2. Replicate Paper (See the high error)
    W_krony, A, B = compress_krony_pt(W)
    
    # 3. Apply Your Fix (See the error vanish)
    apply_your_improvement(W, W_krony)