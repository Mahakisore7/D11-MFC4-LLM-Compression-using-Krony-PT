import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# =================================================================
# 1. THE CUSTOM LAYER (FIXED DIMENSIONS)
# =================================================================
class KroneckerSparseLinear(nn.Module):
    def __init__(self, original_layer, compressed_data):
        super().__init__()
        
        # Extract the compressed components
        U, S, V = compressed_data['U'], compressed_data['S'], compressed_data['V']
        rank = S.shape[0]
        
        # Distribute energy
        sigma_sqrt = torch.diag(torch.sqrt(S))
        self.A_factors = nn.Parameter(U @ sigma_sqrt) 
        self.B_factors = nn.Parameter(sigma_sqrt @ V) 
        
        # Dimensions (Must match compression_agent.py exactly!)
        self.m1, self.n1 = 32, 64
        self.m2, self.n2 = 24, 48
        
        # Sparse Matrix
        indices = compressed_data['Sparse_Indices']
        values = compressed_data['Sparse_Values']
        shape = compressed_data['Original_Shape']
        self.Sparse = torch.sparse_coo_tensor(indices, values, shape).to_dense()
        
        # Bias
        self.bias = nn.Parameter(original_layer.bias)

    def forward(self, x):
        # 1. Reconstruct Low Rank Part
        # W_tilde shape: [2048, 1152]
        W_tilde = self.A_factors @ self.B_factors
        
        # CRITICAL FIX: The view order must be (m1, n1, m2, n2)
        # This unscrambles the matrix blocks correctly.
        W_low = W_tilde.view(self.m1, self.n1, self.m2, self.n2).permute(0, 2, 1, 3).reshape(768, 3072)
        
        # 2. Add Sparse Residual
        W_reconstructed = W_low + self.Sparse
        
        # 3. Forward Pass (GPT-2 Conv1D uses x @ W + b)
        return x @ W_reconstructed + self.bias

# =================================================================
# 2. THE CHATBOT
# =================================================================
def start_chat():
    print("⏳ LOADING COMPRESSED MODEL...")
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Load compressed weights
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    else:
        map_location = None
        
    compressed_data = torch.load('compressed_gpt2.pt', map_location=map_location)
    print(f"   Loaded {len(compressed_data)} compressed layers.")

    # Brain Surgery
    print("   Performing Brain Surgery (Middle Layers Only)...")
    for i in range(len(model.transformer.h)):
        layer_key = f"layer_{i}_mlp"
        
        # Sandwich Strategy: Skip layers that weren't compressed
        if layer_key not in compressed_data:
            print(f"   Layer {i}: Keeping Original (Dense)")
            continue
            
        print(f"   Layer {i}: Injecting Kronecker Matrix")
        
        original_layer = model.transformer.h[i].mlp.c_fc
        layer_data = compressed_data[layer_key]
        
        new_layer = KroneckerSparseLinear(original_layer, layer_data)
        model.transformer.h[i].mlp.c_fc = new_layer
        
    print("✅ MODEL READY! (Running on CPU)")
    
    print("\n" + "="*40)
    print("      TALK TO KRONECKER-GPT-2")
    print("="*40)
    
    while True:
        text = input("\nYOU: ")
        if text.lower() in ['exit', 'quit']: break
        
        inputs = tokenizer.encode(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=50, 
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, 
                temperature=0.7,
                top_k=50,
                repetition_penalty=1.3
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"AI:  {response}")

if __name__ == "__main__":
    start_chat()