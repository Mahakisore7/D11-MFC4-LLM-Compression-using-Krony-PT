import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from compression_agent import KroneckerCompressionAgent
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_NAME = "gpt2"  # You can use 'gpt2-medium' if your PC is fast enough
SPARSE_FACTOR = 0.1  # Keep top 10% errors (The 'k' from your tests)

def load_model():
    """
    Loads the pre-trained GPT-2 model and tokenizer from Hugging Face.
    """
    print(f"\nLoading {MODEL_NAME} model... (This might take a minute)")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        
        # Ensure model is in eval mode (standard for inference)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def chat_with_model(model, tokenizer):
    """
    Runs a simple terminal-based chat loop with the model.
    """
    print("\n" + "="*40)
    print("      TALK TO KRONECKER-GPT-2")
    print("="*40)
    print("Type 'exit' or 'quit' to stop.\n")

    # Set pad token to eos token to avoid warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    while True:
        try:
            user_input = input("YOU: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Encode input
            # We add the attention mask logic to silence standard warnings
            inputs = tokenizer(user_input, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Generate response
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=100,         # Limit response length
                    num_return_sequences=1, # One answer per question
                    do_sample=True,         # Add variety/creativity
                    top_k=50,               # Limit to top 50 likely words
                    top_p=0.95,             # Nucleus sampling
                    temperature=0.7,        # Creativity factor
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode and print
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Remove the user's prompt from the start of the response if needed
            # (GPT-2 often repeats the prompt)
            print(f"AI:  {response[len(user_input):].strip()}\n")
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    # 1. Load the Original "Brain"
    model, tokenizer = load_model()
    if model is None:
        return

    # 2. Apply "Brain Surgery" (Your Method)
    print("\n" + "-"*40)
    print(" PHASE 1: COMPRESSION INITIATED")
    print("-"*40)
    
    # Initialize your agent with the model
    agent = KroneckerCompressionAgent(model, pruning_factor=SPARSE_FACTOR)
    
    # Run the compression math (SVD + Alpha + Sparse)
    agent.apply_compression()

    # 3. Test the Result
    print("\n" + "-"*40)
    print(" PHASE 2: VERIFICATION (CHAT)")
    print("-"*40)
    chat_with_model(model, tokenizer)

if __name__ == "__main__":
    main()