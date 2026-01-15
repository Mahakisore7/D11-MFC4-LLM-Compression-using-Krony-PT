import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# =================================================================
# 1. THE CUSTOM LAYER (The "Kronecker Engine")
# =================================================================
class KroneckerSparseLinear(nn.Module):
    def __init__(self, original_layer, compressed_data):
        super().__init__()
        
        # Extract the compressed components
        # A_factors = U * sqrt(S)
        # B_factors = V * sqrt(S)
        U, S, V = compressed_data['U'], compressed_data['S'], compressed_data['V']
        rank = S.shape[0]
        
        # Distribute energy (sqrt) to both sides so A and B are balanced
        sigma_sqrt = torch.diag(torch.sqrt(S))
        self.A_factors = nn.Parameter(U @ sigma_sqrt) # Shape: [2048, Rank]
        self.B_factors = nn.Parameter(sigma_sqrt @ V) # Shape: [Rank, 1152]
        
        # Dimensions for reshaping during forward pass
        # We assume the standard GPT-2 split we used earlier
        self.m1, self.n1 = 32, 64
        self.m2, self.n2 = 24, 48
        
        # Reconstruct the Sparse Matrix from indices/values
        indices = compressed_data['Sparse_Indices']
        values = compressed_data['Sparse_Values']
        shape = compressed_data['Original_Shape']
        self.Sparse = torch.sparse_coo_tensor(indices, values, shape).to_dense() # Keep dense for speed on CPU
        
        # Bias (we keep the original bias uncompressed)
        self.bias = nn.Parameter(original_layer.bias)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, 768]
        
        # --- PATH 1: LOW RANK (Kronecker) ---
        # The Math: Y = X @ (A (x) B)^T
        # But (A (x) B) is huge. We use the identity:
        # (A (x) B) X_reshaped = A X_reshaped B^T
        # This part is tricky to implement efficiently in PyTorch 
        # without custom CUDA kernels, so for this demo, 
        # we will reconstruct the Low Rank matrix on the fly.
        # It costs memory but proves the math works.
        
        # Reconstruct Low Rank W: W_low = U * S * V
        # W_tilde = self.A_factors @ self.B_factors
        # W_low = W_tilde.view(self.m1, self.n1, self.m2, self.n2).permute(0, 2, 1, 3).reshape(768, 3072)
        
        # --- PATH 2: SPARSE + LOW RANK ---
        # W_final = W_low + S
        # Since we want to run fast, let's pre-calculate W_final 
        # (In a real C++ deployment, you would compute them separately to save RAM)
        
        W_tilde = self.A_factors @ self.B_factors
        W_low = W_tilde.view(self.m1, self.m2, self.n1, self.n2).permute(0, 2, 1, 3).reshape(768, 3072)
        
        W_reconstructed = W_low.t() + self.Sparse.t() # Transpose because Linear expects [Out, In]
        
        return x @ W_reconstructed.t() + self.bias

# =================================================================
# 2. THE CHATBOT
# =================================================================
def start_chat():
    print("⏳ LOADING COMPRESSED MODEL...")
    
    # 1. Load standard GPT-2 structure
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 2. Load our compressed weights
    compressed_data = torch.load('compressed_gpt2.pt')
    print(f"   Loaded {len(compressed_data)} compressed layers.")

    # 3. SURGERY: Replace the layers
    print("   Performing Brain Surgery...")
    for i in range(len(model.transformer.h)):
        # Get the original layer
        original_layer = model.transformer.h[i].mlp.c_fc
        
        # Get our compressed data
        layer_data = compressed_data[f"layer_{i}_mlp"]
        
        # Create the custom Kronecker layer
        new_layer = KroneckerSparseLinear(original_layer, layer_data)
        
        # Replace it!
        model.transformer.h[i].mlp.c_fc = new_layer
        
    print("✅ MODEL READY! (Running on CPU)")
    
    # 4. Chat Loop
    print("\n" + "="*40)
    print("      TALK TO KRONECKER-GPT-2")
    print("="*40)
    
    while True:
        text = input("\nYOU: ")
        if text.lower() in ['exit', 'quit']: break
        
        inputs = tokenizer.encode(text, return_tensors='pt')
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=50, 
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, # Creativity
                temperature=0.7
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"AI:  {response}")

if __name__ == "__main__":
    start_chat()