import torch
import torch.nn as nn
import numpy as np

class KroneckerCompressionAgent:
    def __init__(self, model, pruning_factor=0.1):
        """
        pruning_factor (float): The percentage of "big errors" to keep.
                                0.1 means we keep the top 10% biggest errors.
                                This is the 'k' from your test script.
        """
        self.model = model
        self.pruning_factor = pruning_factor
        self.device = next(model.parameters()).device

    def van_loan_rearrangement(self, W, block_size):
        """
        Rearranges the matrix W for Kronecker SVD.
        This matches the logic we verified in 'step1_paper_replication.py'.
        """
        rows, cols = W.shape
        m1, m2 = block_size
        n1, n2 = rows // m1, cols // m2
        
        # 1. View as grid of blocks
        # 2. Permute to group block positions (The 'Van Loan' trick)
        # 3. Reshape to 2D matrix
        return W.view(n1, m1, n2, m2).permute(0, 2, 1, 3).reshape(n1 * n2, m1 * m2)

    def compress_layer(self, layer_weight):
        """
        The 'Brain Surgery' function.
        Performs: SVD -> Alpha Scaling -> Sparse Residual
        """
        W = layer_weight.detach()
        rows, cols = W.shape
        
        # -------------------------------------------------------
        # STEP A: DEFINE BLOCK SIZES
        # -------------------------------------------------------
        # GPT-2 Small layers are usually 768x768 or 768x3072.
        # We need factors that divide these numbers cleanly.
        # A good generic split for 768 is 32x24 blocks.
        
        # We find block sizes (r2, c2) such that matrix splits nicely.
        # If the dimension is small (<100), we skip compression.
        if rows < 64 or cols < 64:
            return None, None, None, None

        # Hardcoded block sizes that work well for GPT-2 dimensions
        # A matrix size of (Rows, Cols) approx A(r1, c1) kron B(r2, c2)
        r2, c2 = 32, 32 
        
        # Check if divisible
        if rows % r2 != 0 or cols % c2 != 0:
            # If 32 doesn't fit, try smaller blocks
            r2, c2 = 16, 16
            if rows % r2 != 0 or cols % c2 != 0:
                print(f"Skipping layer (Size {rows}x{cols} not divisible by {r2})")
                return None, None, None, None

        # -------------------------------------------------------
        # STEP B: REARRANGEMENT & SVD (Paper Method)
        # -------------------------------------------------------
        # 1. Rearrange
        W_tilde = self.van_loan_rearrangement(W, (r2, c2))
        
        # 2. SVD
        # We use low-rank SVD (just 1 singular value)
        U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)
        
        # 3. Extract A and B
        sigma = S[0]
        u1 = U[:, 0]
        v1 = Vt[0, :]
        
        scale = torch.sqrt(sigma)
        A = (scale * u1).reshape(rows // r2, cols // c2)
        B = (scale * v1).reshape(r2, c2)
        
        # 4. Base Approximation
        W_base = torch.kron(A, B)
        
        # 5. Alpha Scaling
        numerator = torch.sum(W * W_base)
        denominator = torch.sum(W_base * W_base)
        alpha = numerator / denominator
        W_approx = alpha * W_base
        
        # -------------------------------------------------------
        # STEP C: SPARSE RESIDUAL (Your Method)
        # -------------------------------------------------------
        # 1. Calculate the Trash (Residual)
        Residual = W - W_approx
        
        # 2. Find Threshold (Top k% biggest errors)
        # We assume pruning_factor is like 0.1 (10%)
        # quantile requires value between 0 and 1. If we want top 10%, we ask for 0.9 quantile.
        threshold = torch.quantile(Residual.abs(), 1.0 - self.pruning_factor)
        
        # 3. Create Sparse Mask
        # "Where error >= threshold, keep it. Else 0."
        S_values = torch.where(Residual.abs() >= threshold, Residual, torch.tensor(0.0, device=self.device))
        
        # Convert to sparse format to simulate saving space
        S_sparse = S_values.to_sparse()
        
        return A, B, alpha, S_sparse

    def apply_compression(self):
        """
        Main loop: Goes through GPT-2 and compresses valid Linear layers.
        """
        print(f"Starting Compression (Sparse Factor: {self.pruning_factor})")
        compressed_count = 0
        
        for name, module in self.model.named_modules():
            # We compress Linear layers, but skip the final 'lm_head' to keep output stable
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                print(f"Compressing: {name} | Size: {module.weight.shape}...", end=" ")
                
                # Run the math
                A, B, alpha, S_sparse = self.compress_layer(module.weight)
                
                if A is not None:
                    # ---------------------------------------------------
                    # RECONSTRUCTION (The Stitching)
                    # ---------------------------------------------------
                    # W_final = alpha * (A kron B) + S
                    W_reconstructed = alpha * torch.kron(A, B) + S_sparse.to_dense()
                    
                    # Update the model with new weights
                    module.weight = nn.Parameter(W_reconstructed)
                    
                    print(f"Done! (Alpha: {alpha:.4f})")
                    compressed_count += 1
                else:
                    print("Skipped.")
                
        print(f"\nCompression Complete. {compressed_count} layers optimized.")