import torch
import torch.nn as nn
import os
from transformers import GPT2Model, GPT2Config

# =================================================================
# HELPER: THE VAN LOAN REARRANGEMENT (Our Core Math Tool)
# =================================================================
def van_loan_rearrangement(W, m1, m2, n1, n2):
    # This reshuffles the matrix pixels so SVD can find the patterns
    # W: [Rows, Cols] -> [m1*n1, m2*n2]
    return W.view(m1, m2, n1, n2).permute(0, 2, 1, 3).reshape(m1 * n1, m2 * n2)

# =================================================================
# CLASS: THE COMPRESSION AGENT
# =================================================================
class CompressionAgent:
    def __init__(self, model_name='gpt2', energy_threshold=0.90):
        print(f"🤖 AGENT: Waking up... Loading {model_name}...")
        self.model = GPT2Model.from_pretrained(model_name)
        self.config = self.model.config
        
        # Hyperparameter: How much "Brain Energy" must we keep?
        # 0.90 means we promise to keep 90% of the layer's information.
        self.threshold = energy_threshold 
        
        # We will store the new compressed weights here
        self.compressed_layers = {} 

    def analyze_and_compress_layer(self, layer_idx, layer_name, W):
        print(f"\nScanning Layer {layer_idx} ({layer_name})...")
        
        # 1. Determine Dimensions (Factorization)
        # GPT-2 FFN is [768, 3072]. We factor this into:
        # A: [32, 64], B: [24, 48]
        m1, n1 = 32, 64
        m2, n2 = 24, 48
        
        # 2. Rearrange (Van Loan)
        W_tilde = van_loan_rearrangement(W, m1, m2, n1, n2)
        
        # 3. SVD Analysis (The "Doctor Checkup")
        # We look at the Singular Values (S) to decide the Rank
        U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)
        
        # --- ADAPTIVE RANK SELECTION ---
        # Calculate total energy (sum of squared singular values)
        total_energy = torch.sum(S**2)
        current_energy = 0
        rank = 0
        
        # Add ranks one by one until we hit the 90% threshold
        for i, sigma in enumerate(S):
            current_energy += sigma**2
            if (current_energy / total_energy) >= self.threshold:
                rank = i + 1
                break
        
        print(f"   > Diagnosis: This layer needs Rank-{rank} to keep {self.threshold*100}% energy.")
        
        # 4. Decompose (The Surgery)
        # We keep the top 'rank' singular vectors
        U_k = U[:, :rank]
        S_k = torch.diag(S[:rank])
        V_k = Vt[:rank, :]
        
        # Reconstruct the Low-Rank Approximation
        # W_approx = U * S * Vt
        W_tilde_approx = U_k @ S_k @ V_k
        
        # Reshape back to [768, 3072]
        # (This effectively creates sum(A_i (x) B_i))
        W_approx = W_tilde_approx.view(m1, n1, m2, n2).permute(0, 2, 1, 3).reshape(768, 3072)
        
        # 5. Sparse Residual (The Fix)
        # Capture the remaining 10% error + outliers
        Residual = W - W_approx
        sparse_thresh = torch.quantile(torch.abs(Residual), 0.90) # Keep top 20% outliers
        S_sparse = torch.where(torch.abs(Residual) > sparse_thresh, Residual, torch.zeros_like(Residual))
        
        # 6. Save the Components
        # We store U, S, V (Factors) and the Sparse Indices
        layer_data = {
            'U': U_k, # Stores Matrix A information
            'S': S[:rank], # Stores the Energy
            'V': V_k, # Stores Matrix B information
            'Sparse_Indices': S_sparse.to_sparse().indices(),
            'Sparse_Values': S_sparse.to_sparse().values(),
            'Original_Shape': W.shape
        }
        
        return layer_data

    def run(self):
        print("🚀 STARTING COMPRESSION (THE SANDWICH STRATEGY)...")
        
        # SENSITIVE LAYERS: 0, 1, 2 (Start) and 10, 11 (End)
        # ROBUST LAYERS: 3, 4, 5, 6, 7, 8, 9 (Middle)
        
        for i in range(len(self.model.h)):
            # SKIP sensitive layers
            if i < 3 or i > 9:
                print(f"Skipping Layer {i} (Keeping it Dense for Accuracy)...")
                continue

            # COMPRESS middle layers
            layer = self.model.h[i].mlp.c_fc
            W = layer.weight.detach()
            
            compressed_data = self.analyze_and_compress_layer(i, "MLP.c_fc", W)
            self.compressed_layers[f"layer_{i}_mlp"] = compressed_data
            
        print("\n✅ COMPRESSION COMPLETE!")
        print(f"Saving compressed model to 'compressed_gpt2.pt'...")
        torch.save(self.compressed_layers, 'compressed_gpt2.pt')

# =================================================================
# MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    agent = CompressionAgent(energy_threshold=0.60) # Keep 85% energy
    agent.run()