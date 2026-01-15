import numpy as np

def rearrange_matrix(W, m1, n1, m2, n2):
    # This is the "Van Loan Pitsianis" rearrangement
    # W is (m1*m2) x (n1*n2)
    # We want to reshape it to (m1*n1) x (m2*n2)
    
    # 1. Break into blocks
    blocks = W.reshape(m1, m2, n1, n2)
    
    # 2. Permute dimensions (The "Shuffle")
    # We want (m1, n1, m2, n2)
    shuffled = blocks.transpose(0, 2, 1, 3)
    
    # 3. Flatten to 2D
    W_tilde = shuffled.reshape(m1 * n1, m2 * n2)
    return W_tilde

# Create a random 'perfect' Kronecker matrix to test
A_true = np.random.rand(2, 2)
B_true = np.random.rand(2, 2)
W = np.kron(A_true, B_true) # The big 4x4 matrix

print("Original W (4x4):")
print(W)

# --- YOUR MATH STARTS HERE ---

# 1. Rearrange
W_tilde = rearrange_matrix(W, 2, 2, 2, 2)

# 2. SVD
U, S, Vt = np.linalg.svd(W_tilde)

# 3. Extract A and B (Rank-1 approximation)
sigma = S[0]
u1 = U[:, 0]
v1 = Vt[0, :]

A_pred = u1.reshape(2, 2) * np.sqrt(sigma)
B_pred = v1.reshape(2, 2) * np.sqrt(sigma)

# --- VERIFY ---
W_pred = np.kron(A_pred, B_pred)
print("\nReconstructed W:")
print(W_pred)
print("\nError:", np.linalg.norm(W - W_pred))