import torch

# ==========================================
# 1. SETUP: SAME 4x4 MATRIX
# ==========================================
W_original = torch.tensor([
    [10., 20., 30., 40.],
    [20., 40., 60., 80.],
    [5.,  10., 15., 20.],
    [10., 20., 30., 40.]
])

print("--- 1. ORIGINAL MATRIX ---")
print(W_original)

# ==========================================
# 2. PAPER METHOD (RECAP)
# ==========================================
# Rearrange (Van Loan)
rows, cols = W_original.shape
W_view = W_original.view(rows // 2, 2, cols // 2, 2)
W_tilde = W_view.permute(0, 2, 1, 3).reshape(-1, 4)

# SVD
U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)
sigma = S[0]
u1 = U[:, 0]
v1 = Vt[0, :]

# Reconstruct Base
scale = torch.sqrt(sigma)
A = (scale * u1).reshape(2, 2)
B = (scale * v1).reshape(2, 2)
W_base = torch.kron(A, B)

# Alpha Scaling
alpha = torch.sum(W_original * W_base) / torch.sum(W_base * W_base)
W_paper = alpha * W_base
error_paper = torch.norm(W_original - W_paper)

print(f"\nPaper Method Error: {error_paper:.4f}")

# ==========================================
# 3. YOUR METHOD: SPARSE RESIDUAL
# ==========================================
print("\n--- 2. APPLYING SPARSE RESIDUAL (YOUR METHOD) ---")

# A. Calculate the "Trash" (Residual)
Residual = W_original - W_base  # Note: Using W_base (no alpha yet) is standard, or W_paper. 
                                # Let's use W_base to match MATLAB logic exactly.

# B. Find the Threshold (Keep top 25% biggest errors)
# In MATLAB: sorted_errors = sort(abs(Residual(:)), 'descend')
# In Python:
flat_errors = torch.abs(Residual).flatten()
sorted_errors, indices = torch.sort(flat_errors, descending=True)

# We want to keep the top 4 numbers (25% of 16 pixels)
k = 4
threshold = sorted_errors[k-1] # Python index starts at 0, so index 3 is the 4th number
print(f"Threshold Value: {threshold:.4f} (Errors smaller than this get deleted)")

# C. Create Sparse Matrix S
# "Where error >= threshold, keep error. Else, put 0."
S = torch.where(torch.abs(Residual) >= threshold, Residual, torch.zeros_like(Residual))

print("\nSparse Matrix S (Only big errors kept):")
print(S)

# D. Final Reconstruction
W_yours = W_base + S
error_yours = torch.norm(W_original - W_yours)

print("\n--- 3. FINAL RESULTS ---")
print(f"Paper Error: {error_paper:.4f}")
print(f"Your Error:  {error_yours:.4f}")

if error_yours < error_paper:
    print("\nSUCCESS: Python proves your method wins!")