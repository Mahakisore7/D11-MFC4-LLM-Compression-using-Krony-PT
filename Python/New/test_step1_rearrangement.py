import torch

# ==========================================
# 1. SETUP: CREATE THE SAME 4x4 MATRIX
# ==========================================
# We use the exact same numbers as MATLAB
W_original = torch.tensor([
    [10., 20., 30., 40.],
    [20., 40., 60., 80.],
    [5.,  10., 15., 20.],
    [10., 20., 30., 40.]
])

print("--- 1. ORIGINAL MATRIX (W) ---")
print(W_original)

# Calculate Energy (Frobenius Norm)
# torch.norm calculates the square root of sum of squares
energy_original = torch.norm(W_original)
print(f"Original Energy: {energy_original:.4f}\n")


# ==========================================
# 2. VAN LOAN REARRANGEMENT (The Trick)
# ==========================================
# In MATLAB, we stacked columns.
# In Python, we use 'permute' to swap dimensions.
# This aligns the blocks so SVD can find the pattern.

rows, cols = W_original.shape
block_rows, block_cols = 2, 2  # We want 2x2 blocks

# 1. View: Cut matrix into a grid of blocks (2x2 grid of 2x2 blocks)
# 2. Permute: Swap axes to group the blocks together
# 3. Reshape: Flatten into the rearrangement matrix
W_view = W_original.view(rows // block_rows, block_rows, cols // block_cols, block_cols)
W_permuted = W_view.permute(0, 2, 1, 3) 
W_tilde = W_permuted.reshape(-1, block_rows * block_cols)

print("--- 2. REARRANGED MATRIX ---")
print(W_tilde)
# Note: Python prints rows. 
# Row 1 here [10, 20, 20, 40] matches your MATLAB Block 1 vector!


# ==========================================
# 3. SVD DECOMPOSITION
# ==========================================
# We run SVD on the rearranged matrix
U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)

# Extract Rank-1 Approximation
sigma = S[0]       # Top energy value
u1 = U[:, 0]       # Left vector (Grid pattern)
v1 = Vt[0, :]      # Right vector (Block pattern)

# Reshape back into A and B
scale = torch.sqrt(sigma)
A = (scale * u1).reshape(rows // block_rows, cols // block_cols)
B = (scale * v1).reshape(block_rows, block_cols)

# ==========================================
# 4. NAIVE RECONSTRUCTION (No Alpha)
# ==========================================
W_naive = torch.kron(A, B)

error_naive = torch.norm(W_original - W_naive)
print(f"\n--- 3. NAIVE ERROR: {error_naive:.4f} ---")


# ==========================================
# 5. PAPER METHOD: ALPHA SCALING
# ==========================================
# Formula: dot(Original, Naive) / dot(Naive, Naive)
numerator = torch.sum(W_original * W_naive)
denominator = torch.sum(W_naive * W_naive)
alpha = numerator / denominator

print(f"\n--- 4. CALCULATED ALPHA: {alpha:.4f} ---")

# Apply Alpha
W_paper = alpha * W_naive
error_paper = torch.norm(W_original - W_paper)

print("Reconstructed Matrix (Paper):")
print(W_paper)
print(f"Paper Method Error: {error_paper:.4f}")

# ==========================================
# CHECK
# ==========================================
if error_paper < error_naive:
    print("\nSUCCESS: Alpha reduced the error (Paper Logic Verified).")
else:
    print("\nCheck code.")