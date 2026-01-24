# LLM Compression using Kronecker Product Decomposition & Sparse Residuals

## 📌 Project Overview

This project focuses on compressing Large Language Models (specifically GPT-2) by decomposing their weight matrices using Kronecker Products.

Standard Low-Rank Approximation (like LoRA) uses matrix multiplication ($A \times B$). Our approach uses the Kronecker Product ($A \otimes B$), which allows for significantly higher compression ratios while preserving structural information.

We have successfully replicated the foundational paper by Van Loan & Pitsianis (1993) and improved upon it by introducing a novel Sparse Residual Correction method that significantly reduces reconstruction error.

## 🎯 Objectives

- Replicate the Van Loan & Pitsianis Kronecker decomposition method (SVD + Alpha Scaling).
- Innovate by adding a Sparse Residual layer to capture "outlier" errors lost by the approximation.
- Verify the mathematics using a controlled 4x4 simulation in both MATLAB and Python.
- Implement the compression agent for the GPT-2 Small architecture.

## 🧠 Methodology

### 1. The Core Algorithm (Van Loan Rearrangement)

To approximate a weight matrix $W$ as $A \otimes B$, we perform the Van Loan Rearrangement $\mathcal{R}(W)$. This operation reshapes the matrix so that the Singular Value Decomposition (SVD) can extract the optimal Kronecker factors.

- Rearrangement: $W \rightarrow \tilde{W}$ (Permuting blocks into vectors)
- SVD: $\tilde{W} \approx \sigma_1 u_1 v_1^T$
- Factors: $A = \sqrt{\sigma_1} \text{reshape}(u_1)$, $B = \sqrt{\sigma_1} \text{reshape}(v_1)$

### 2. Method A: The Paper's Approach (Baseline)

The original 1993 paper improves the standard SVD approximation by applying a scaling factor $\alpha$ to minimize the Frobenius norm energy loss.

$$W_{paper} = \alpha (A \otimes B)$$

**Status:** Successfully Replicated.

**Observation:** While effective, it discards high-frequency details (outliers), leading to higher error.

### 3. Method B: Our Approach (Sparse Residual Correction)

We identified that the error $R = W - W_{paper}$ is not random noise but contains critical "outlier" weights. Instead of discarding them, we:

1. Calculate the Residual $R$.
2. Apply Thresholding to keep the top $k\%$ magnitude errors.
3. Store these as a Sparse Matrix $S$.

Final Reconstruction:

$$W_{ours} = \alpha (A \otimes B) + S$$

## 📊 Verification & Results (The 4x4 "Brain" Simulation)

We validated our hypothesis on a 4x4 Rank-1 toy matrix to prove mathematical correctness before applying it to GPT-2.

| Method | Description | Error (Frobenius Norm) | Result |
|--------|-------------|------------------------|--------|
| Pruning | Baseline heuristic (forcing Identity matrix) | 136.62 |  High Error |
| Paper (Van Loan) | SVD + Alpha Scaling | 9.1492 |  Good Baseline |
| Our Method | SVD + Alpha + Sparse Residual | 4.8922 |  47% Improvement |

**Note:** Results verified in both MATLAB and Python (PyTorch).

## 📂 Repository Structure

### 1. Proof of Concept (Math Verification)

- **step1_paper_replication.py**: Python script replicating the Van Loan rearrangement and SVD logic. Verifies the result matches MATLAB.
- **step2_sparse_residual.py**: Demonstrates the "Sparse Residual" logic, proving the 47% error reduction on the toy matrix.

### 2. The Compression Agent

- **compression_agent.py**: The core "Surgeon" class.
  - Implements van_loan_rearrangement (PyTorch optimized).
  - Performs compress_layer (SVD + Alpha + Sparse logic).
  - Iterates through GPT-2 layers to apply compression.

- **chat_with_gpt2.py**: The "Driver" script.
  - Loads gpt2-small.
  - Runs the Compression Agent.
  - Opens a chat interface to test post-compression coherence.

### 3. Analysis

- **GPT2_Analysis.ipynb** (In Progress): A step-by-step Jupyter Notebook explaining the GPT-2 architecture, visualizing the layers, and documenting the compression flow for production readiness.

## 🚀 How to Run

### Install Dependencies:

```bash
pip install torch transformers numpy
```

### Verify the Math (Simulation):

```bash
python step2_sparse_residual.py
```

Look for "SUCCESS: Python proves your method wins!"

### Run the Real Model:

```bash
python chat_with_gpt2.py
```

This will download GPT-2, compress it layer-by-layer, and let you chat with the result.
