import torch
import torch.nn as nn
import numpy as np

class KroneckerCompressionAgent:
    def __init__(self, model, pruning_factor=0.1):
        """
        pruning_factor (float): How many "outliers" to keep in the Sparse Matrix.
                                0.1 means we keep the top 10% biggest errors.
        """
        self.model = model
        self.pruning_factor = pruning_factor
        self.device = next(model.parameters()).device

    def van_loan_rearrangement(self, W, block_size):
        """
        Rearranges the matrix W for Kronecker SVD.
        Uses Standard PyTorch Row-Major (Efficient for GPU).
        """
        rows, cols = W.shape
        m1, m2 = block_size
        n1, n2 = rows // m1, cols // m2
        
        # 1. View as blocks
        # 2. Permute to group blocks together
        # 3. Reshape to 2D
        return W.view(n1, m1, n2, m2).permute(0, 2, 1, 3).reshape(n1 * n2, m1 * m2)

    def compress_layer(self, layer_weight):
        """
        Performs: SVD + Alpha + Sparse Residual
        """
        W = layer_weight.detach()
        rows, cols = W.shape
        
        # Define block sizes (Assuming GPT-2 standard sizes)
        # We try to find factors close to sqrt(rows) and sqrt(cols)
        # For GPT-2 (768), typical split is (32, 24) or similar.
        # Let's simplify and fix a 2x2 split concept for the big matrix.
        # Actually, for the code to be generic, let's use fixed common factors:
        # GPT-2 Small: 768 -> 32x24 blocks results in A(32,24) and B(24,32) approx.
        # For safety, let's hardcode a known good split for 768 dimensions:
        r1, r2 = 32, rows // 32
        c1, c2 = 32, cols // 32
        
        # 1. Rearrange (Van Loan)
        W_tilde = self.van_loan_rearrangement(W, (r2, c2))
        
        # 2. SVD
        U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)
        
        # 3. Rank-1 Approximation (A and B)
        sigma = S[0]
        u1 = U[:, 0]
        v1 = Vt[0, :]
        
        scale = torch.sqrt(sigma)
        A = (scale * u1).reshape(rows // r2, cols // c2)
        B = (scale * v1).reshape(r2, c2)
        
        # 4. Base Approximation
        W_base = torch.kron(A, B)
        
        # 5. Alpha Scaling (The Paper's Method)
        # dot(W, W_base) / dot(W_base, W_base)
        alpha = torch.sum(W * W_base) / torch.sum(W_base * W_base)
        W_approx = alpha * W_base
        
        # ====================================================
        # THE NEW PART: SPARSE RESIDUAL (YOUR METHOD)
        # ====================================================
        
        # 6. Calculate Residual (The Trash)
        Residual = W - W_approx
        
        # 7. Thresholding (Keep top k% biggest errors)
        # We use quantile to find the cutoff value efficiently
        threshold = torch.quantile(Residual.abs(), 1.0 - self.pruning_factor)
        
        # 8. Create Sparse Mask
        # We only keep values where error >= threshold
        S_values = torch.where(Residual.abs() >= threshold, Residual, torch.tensor(0.0, device=self.device))
        
        # Convert to actual Sparse Tensor to save memory
        S_sparse = S_values.to_sparse()
        
        # Return the compressed components
        return A, B, alpha, S_sparse

    def apply_compression(self):
        """
        Iterates through the GPT-2 model and compresses Linear layers.
        """
        print(f"Starting Compression with Sparse Factor: {self.pruning_factor}")
        
        compressed_count = 0
        total_params_saved = 0
        
        # Loop through all layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                # We skip the very last head usually to be safe, but can compress it too.
                print(f"Compressing layer: {name} | Size: {module.weight.shape}")
                
                # Run the compression math
                A, B, alpha, S_sparse = self.compress_layer(module.weight)
                
                # Calculate Parameter Savings
                original_params = module.weight.numel()
                new_params = A.numel() + B.numel() + S_sparse._nnz() + 1 # +1 for alpha
                
                print(f"  -> Original: {original_params} | New: {new_params}")
                print(f"  -> Ratio: {new_params/original_params:.4f}")
                
                # REPLACE WEIGHTS (Ideally, we would replace the Linear layer with a custom layer)
                # For this script, we will just reconstruct the weight to verify accuracy.
                # In a real deployment, you'd write a 'KroneckerLinear' class.
                
                # Reconstruct: W_final = alpha * (A kron B) + S
                W_reconstructed = alpha * torch.kron(A, B) + S_sparse.to_dense()
                
                # Update the model weights
                module.weight = nn.Parameter(W_reconstructed)
                compressed_count += 1
                
        print(f"Compression Complete. {compressed_count} layers processed.")