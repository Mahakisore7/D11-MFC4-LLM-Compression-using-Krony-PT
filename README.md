<br/>
<div align="center">
<img src="assets/Amrita-vishwa-vidyapeetham-logo.svg.png" width="820"/>
</div>
<br/>

<div align="center">

# 🧠 LLM Compression via Kronecker Product Decomposition

### Replicating & Extending Krony-PT on GPT-2 Small

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2412.12351-blue?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2412.12351)
[![Course](https://img.shields.io/badge/Course-22MAT230%20MFC%204-orange?style=for-the-badge)](https://www.amrita.edu)
[![Model](https://img.shields.io/badge/Model-GPT--2%20Small-green?style=for-the-badge&logo=huggingface)](https://huggingface.co/gpt2)
[![GPU](https://img.shields.io/badge/GPU-RTX%204050%20Laptop-76b900?style=for-the-badge&logo=nvidia)](#)
[![Dataset](https://img.shields.io/badge/Dataset-WikiText--2-purple?style=for-the-badge)](https://huggingface.co/datasets/wikitext)

<br/>

> **45.4% parameter reduction · 45.4% storage reduction · +14.7% throughput**  
> All experiments on a single RTX 4050 Laptop GPU (6 GB VRAM) · ~90–100 min end-to-end

<br/>

**Amrita Vishwa Vidyapeetham · Department of AI & Data Science · Team D11**

| Member | Roll No. | Email |
|:---:|:---:|:---:|
| Hemanth SN | CB.SC.U4AIE24321 | cb.sc.u4aie24321@cb.students.amrita.edu |
| Mahakisore M | CB.SC.U4AIE24333 | cb.sc.u4aie24333@cb.students.amrita.edu |
| Yashwanth B | CB.SC.U4AIE24360 | cb.sc.u4aie24360@cb.students.amrita.edu |

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---|
| 1 | [Project Overview](#-project-overview) |
| 2 | [Key Results at a Glance](#-key-results-at-a-glance) |
| 3 | [GPT-2 Architecture & Compression Target](#-gpt-2-architecture--compression-target) |
| 4 | [Core Algorithm — Van Loan Kronecker Decomposition](#-core-algorithm--van-loan-kronecker-decomposition) |
| 5 | [Compression Variants](#-compression-variants) |
| 6 | [Implementation](#-implementation) |
| 7 | [Experimental Setup](#-experimental-setup) |
| 8 | [Phase 1 — Baseline Evaluation](#phase-1--baseline-evaluation) |
| 9 | [Phase 2 — Rank-1 Compression (All 12 Layers)](#phase-2--rank-1-kronecker-compression-all-12-layers) |
| 10 | [Phase 3 — Adaptive Normalization](#phase-3--adaptive-normalization) |
| 11 | [Phase 4 — Layer Sensitivity Analysis](#phase-4--layer-sensitivity-analysis) |
| 12 | [Phase 5 — Fine-Tuning Recovery](#phase-5--fine-tuning-recovery) |
| 13 | [Phase 6 — Rank-2 Extension](#phase-6--rank-2-extension) |
| 14 | [Extras — AxB vs AxBxC · Full Rank-R Sweep](#-extras--axb-vs-axbxc--full-rank-r-sweep) |
| 15 | [Sparse Residual Innovation (Beyond the Paper)](#-sparse-residual-innovation-beyond-the-paper) |
| 16 | [DistilGPT-2 Comparison](#-distilgpt-2-comparison) |
| 17 | [Repository Structure](#-repository-structure) |
| 18 | [How to Run](#-how-to-run) |
| 19 | [Conclusions](#-conclusions) |
| 20 | [References](#-references) |

---

## 🎯 Project Overview

This project replicates and extends **[Krony-PT (arXiv:2412.12351)](https://arxiv.org/abs/2412.12351)**, compressing GPT-2 Small (124M parameters) by replacing its Feed-Forward Network (FFN) weight matrices with **Kronecker Product Decompositions**.

Instead of storing a full weight matrix $W \in \mathbb{R}^{m \times n}$, we approximate it as:

$$W \approx A \otimes B$$

where $A \in \mathbb{R}^{m_1 \times n_1}$, $B \in \mathbb{R}^{m_2 \times n_2}$, with $m = m_1 m_2$ and $n = n_1 n_2$.

**What we built on top of the paper:**

- ✅ Full 6-phase experiment pipeline on WikiText-2 (baseline → compress → normalize → analyze → fine-tune → rank-extend)
- ✅ Rank-K Kronecker: $W \approx \sum_{r=1}^K A_r \otimes B_r$ with sweep over ranks 1, 2, 4, 8
- ✅ Triple-factor Kronecker ($W \approx A \otimes B \otimes C$) extreme compression
- ✅ Detailed layer sensitivity analysis — logit difference + perplexity, isolated and cumulative
- ✅ Partial compression experiments (first-6 vs last-6 layers) with fine-tuning
- ✅ **Sparse Residual Correction** — our novel extension, **47% Frobenius error reduction** over the paper
- ✅ DistilGPT-2 benchmarking to situate our results against knowledge distillation

---

## ⚡ Key Results at a Glance

<div align="center">

| Metric | Baseline GPT-2 | Rank-1 Compressed | Δ |
|:---:|:---:|:---:|:---:|
| **Parameters** | 124.4M | **67.9M** | 🔻 −45.4% |
| **Storage** | 474.75 MB | **259.19 MB** | 🔻 −45.4% |
| **Throughput** | 406.5 tok/s | **466.1 tok/s** | 🟢 +14.7% |
| **Latency** | 10.22 ms | 11.47 ms | 🔺 +12.3% |
| **PPL (no fine-tune)** | 59.11 | 50,011 | ⚠️ needs FT |
| **PPL (6 layers + 5ep FT)** | 59.11 | **~100** | ✅ usable range |

</div>

```
Parameter Reduction:    45.4%   (124M → 67.9M)
Storage Reduction:      45.4%   (474 MB → 259 MB)
Throughput Gain:       +14.7%   (406 → 466 tok/s)
Latency Overhead:      +12.3%   (10.2 ms → 11.5 ms)
PPL after partial FT:   ~100    (first-6 layers, 5 epochs)
```

> Perplexity without fine-tuning is high by design — Kronecker-factorized weights are a mathematical approximation requiring adaptation. After partial compression + 5 epochs of WikiText-2 fine-tuning, PPL reaches ~100, approaching DistilGPT-2 territory on our compute budget.

---

## 🏗 GPT-2 Architecture & Compression Target

GPT-2 Small is a **12-layer decoder-only transformer** (768 hidden dim, 12 attention heads). Its parameter budget:

<div align="center">

| Component | Shape | Parameters | Share |
|---|---|:---:|:---:|
| Token Embedding (wte) | 50,257 × 768 | 38,597,376 | 31.0% |
| Position Embedding (wpe) | 1,024 × 768 | 786,432 | 0.6% |
| Self-Attention × 12 | c_attn + c_proj | 28,311,552 | 22.8% |
| **FFN / MLP × 12** ← **compressed** | **c_fc + c_proj** | **56,623,104** | **45.5%** |
| LayerNorms + biases | — | ~37,000 | ~0% |
| **Total** | | **124,439,808** | **100%** |

</div>

**Why target only the FFN?** The MLP block holds **45.5% of all parameters** — the single largest compressible component. Each FFN block transforms:

```
c_fc   :  768 → 3072   (expand by 4×)
c_proj : 3072 → 768    (project back)
```

Compressing both in all 12 layers eliminates 56.6M parameters, reducing the model from 124M to 67.9M.

> ⚠️ **GPT-2 Conv1D quirk:** GPT-2 uses `Conv1D` which stores weights as `(in_dim, out_dim)` — transposed relative to standard `nn.Linear`. The forward pass requires a permute-based matmul strategy to handle this correctly.

---

## 📐 Core Algorithm — Van Loan Kronecker Decomposition

### The Optimization Problem

Given $W \in \mathbb{R}^{(m_1 m_2) \times (n_1 n_2)}$, find factors $A \in \mathbb{R}^{m_1 \times n_1}$, $B \in \mathbb{R}^{m_2 \times n_2}$ minimizing:

$$\min_{A,\, B} \|W - A \otimes B\|_F$$

### Van Loan Rearrangement (1993)

The key insight: rearrange $W$ into $\tilde{W}$ via block-permutation, so that a **rank-1 SVD** directly extracts the optimal Kronecker factors:

$$W \xrightarrow{\mathcal{R}} \tilde{W} \in \mathbb{R}^{(m_1 n_1) \times (m_2 n_2)}$$

$$\tilde{W} \approx \sigma_1 u_1 v_1^\top \quad \text{(rank-1 SVD)}$$

$$A = \sqrt{\sigma_1} \cdot \mathrm{reshape}(u_1,\, m_1,\, n_1), \qquad B = \sqrt{\sigma_1} \cdot \mathrm{reshape}(v_1,\, m_2,\, n_2)$$

### Code

```python
def kronecker_rank1(W, m1, m2, n1, n2):
    # Van Loan rearrangement via block-permutation
    W_r = W.reshape(m1, m2, n1, n2).permute(0, 2, 1, 3).reshape(m1*n1, m2*n2)
    U, S, Vh = torch.linalg.svd(W_r, full_matrices=False)
    A = (U[:, 0] * S[0].sqrt()).reshape(m1, n1)
    B = (Vh[0, :] * S[0].sqrt()).reshape(m2, n2)
    return A, B

def kronecker_rank_r(W, m1, m2, n1, n2, rank):
    W_r = W.reshape(m1, m2, n1, n2).permute(0, 2, 1, 3).reshape(m1*n1, m2*n2)
    U, S, Vh = torch.linalg.svd(W_r, full_matrices=False)
    A_list, B_list = [], []
    for r in range(rank):
        A_list.append((U[:, r] * S[r].sqrt()).reshape(m1, n1))
        B_list.append((Vh[r, :] * S[r].sqrt()).reshape(m2, n2))
    return A_list, B_list
```

### Factor Dimensions for GPT-2

<div align="center">

| Layer | Original Shape | $m_1,\, m_2$ | $n_1,\, n_2$ | **A shape** | **B shape** | Params: before → after |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `c_fc` | 3072 × 768 | 48, 64 | 12, 64 | (48, 12) | (64, 64) | 2,359,296 → **4,672** |
| `c_proj` | 768 × 3072 | 12, 64 | 48, 64 | (12, 48) | (64, 64) | 2,359,296 → **4,672** |

</div>

Each layer pair goes from **4.7M parameters to just 9,344** — a **252× reduction per layer**.

---

## 🗜 Compression Variants

### Rank-1 (Maximum Compression)
$$W \approx A \otimes B$$
45.4% model-wide parameter reduction. Needs fine-tuning to restore language modeling quality.

### Rank-K (Quality-Compression Tradeoff)
$$W \approx \sum_{r=1}^{K} A_r \otimes B_r$$
Each additional rank pair captures the next singular value pair of $\tilde{W}$. Rank-2 gives **2× better pre-FT perplexity** than Rank-1.

### Triple Kronecker — AxBxC (Extreme Compression)
$$W \approx A \otimes B \otimes C$$
~400 parameters per layer (vs 2.36M original) — **5,890× per-layer reduction** — but B and C are randomly initialized, making this entirely dependent on fine-tuning.

### Adaptive Normalization
$$\alpha = \frac{\|W\|_F}{\|A \otimes B\|_F}, \qquad A \leftarrow \sqrt{\alpha} \cdot A, \quad B \leftarrow \sqrt{\alpha} \cdot B$$
Rescales factors to preserve the weight energy of $W$ without adding any parameters.

### Sparse Residual Correction *(Our Extension)*
$$W_{\text{ours}} = \alpha(A \otimes B) + S, \quad S = \mathrm{top}\text{-}k\%(W - \alpha(A \otimes B))$$
Adds back the most important approximation errors as a sparse correction. Achieves **47% lower Frobenius error** than the paper's method.

---

## 💻 Implementation

### KronLinear Module

```python
class KronLinear(nn.Module):
    """Drop-in replacement for a GPT-2 Conv1D layer using sum-of-Kronecker-products."""
    def __init__(self, A_list, B_list, bias):
        super().__init__()
        self.A_list = nn.ParameterList([nn.Parameter(A) for A in A_list])
        self.B_list = nn.ParameterList([nn.Parameter(B) for B in B_list])
        self.bias   = nn.Parameter(bias.clone())

    def forward(self, x):
        out = 0
        for A, B in zip(self.A_list, self.B_list):
            W    = torch.kron(A, B)          # Reconstruct weight on the fly
            out += torch.matmul(x, W.t())    # Apply linear transform
        return out + self.bias
```

### Compression Pipeline

```python
def compress_layer(model, layer_idx, rank=1):
    block = model.transformer.h[layer_idx]

    # c_fc: 768→3072  (Conv1D stores as (768,3072) — transpose first)
    W_fc   = block.mlp.c_fc.weight.data.T          # → (3072, 768)
    A, B   = kronecker_rank_r(W_fc, 48, 64, 12, 64, rank)
    block.mlp.c_fc = KronLinear(A, B, block.mlp.c_fc.bias.data)

    # c_proj: 3072→768  (Conv1D stores as (3072,768) — correct as-is)
    W_proj = block.mlp.c_proj.weight.data           # (3072, 768)
    A, B   = kronecker_rank_r(W_proj, 12, 64, 48, 64, rank)
    block.mlp.c_proj = KronLinear(A, B, block.mlp.c_proj.bias.data)
```

---

## ⚙️ Experimental Setup

<div align="center">

| Item | Detail |
|:---:|:---:|
| **Model** | GPT-2 Small (`gpt2`, HuggingFace Transformers 4.57.6) |
| **Dataset** | WikiText-2 (`wikitext-2-raw-v1`) |
| **Eval** | ~500 batches · batch_size=16 · seq_len=128 |
| **Train** | 2,000 samples, AdamW (lr=3e-5), AMP mixed precision |
| **GPU** | NVIDIA RTX 4050 Laptop GPU · 6 GB VRAM |
| **OS** | Windows · Miniconda · Python 3.x · PyTorch |

</div>

---

## Phase 1 — Baseline Evaluation

Load GPT-2 Small and record all reference metrics on WikiText-2 before any modification.

```
Baseline PPL (WikiText-2):   ~59–60
Parameters:                   124,439,808
Disk Size:                    474.75 MB
Inference Latency:            10.22 ms / batch
Throughput:                   406.5 tok/s
```

---

## Phase 2 — Rank-1 Kronecker Compression (All 12 Layers)

Replace `c_fc` and `c_proj` in all 12 transformer blocks with `KronLinear`.

<div align="center">

| Metric | Baseline | Rank-1 Compressed | Change |
|:---:|:---:|:---:|:---:|
| Parameters | 124.4M | **67.9M** | −45.4% |
| Disk Size | 474.75 MB | **259.19 MB** | −45.4% |
| Latency | 10.22 ms | 11.47 ms | +12.3% |
| Throughput | 406.5 tok/s | **466.1 tok/s** | +14.7% |
| PPL (no FT) | 59.11 | 50,011 | ⚠️ needs FT |

</div>

<br/>

<div align="center">
<img src="assets/slide_param_inference_results.png" width="820"/>
<br/><sub><b>Fig 1.</b> Parameter count and inference comparison — baseline vs Rank-1 Kronecker compressed GPT-2</sub>
</div>

<br/>

<div align="center">
<img src="assets/slide_model_size_results.png" width="820"/>
<br/><sub><b>Fig 2.</b> Disk storage, throughput, and compression effectiveness</sub>
</div>

<br/>

<div align="center">
<img src="assets/slide_efficiency_dashboard.png" width="820"/>
<br/><sub><b>Fig 3.</b> Normalized GPT-2 compression efficiency dashboard (Baseline = 1.0 across all metrics)</sub>
</div>

---

## Phase 3 — Adaptive Normalization

After Kronecker decomposition, the scale of $A \otimes B$ can diverge from $W$, shifting hidden-state magnitudes and degrading pre-FT perplexity. Rescaling with $\alpha$ corrects this:

$$\alpha = \frac{\|W\|_F}{\|A \otimes B\|_F}, \qquad A \leftarrow \sqrt{\alpha} \cdot A, \quad B \leftarrow \sqrt{\alpha} \cdot B$$

This is a **zero-cost post-processing step** — no additional parameters, no training — and consistently improves pre-fine-tuning PPL across all compressed layers.

---

## Phase 4 — Layer Sensitivity Analysis

### Setup

For each of the 12 transformer layers, compress **only that layer** (Rank-1, all others remain full precision), then measure:

**Relative logit difference vs baseline:**
$$\mathrm{diff} = \frac{\|f_{\text{baseline}}(x) - f_{\text{compressed}}(x)\|_2}{\|f_{\text{baseline}}(x)\|_2}$$

**Perplexity** on WikiText-2.

### Isolated Sensitivity Results

<div align="center">

| Layer | Logit Diff | PPL (isolated Rank-1) | Sensitivity |
|:---:|:---:|:---:|:---:|
| **0** | **0.3034** | **4,825** | 🔴 Uniquely sensitive |
| 1 | 0.028 | 62 | 🟢 Very stable |
| 2–9 | 0.04–0.07 | 62–68 | 🟢 Stable |
| 10 | 0.2208 | 67 | 🟡 Elevated |
| 11 | 0.1573 | 68 | 🟡 Elevated |

</div>

**Layer 0 is uniquely sensitive** — it is the first feature transformation after embeddings and carries irreplaceable representational structure. All other layers (1–9) in isolation cause negligible degradation (~2–5 PPL increase). This finding has a direct practical consequence: **skipping Layer 0 during compression is a strong heuristic** for quality-preserving deployment.

<br/>

<div align="center">
<img src="assets/slide_layer_sensitivity_logit.png" width="820"/>
<br/><sub><b>Fig 4.</b> Isolated vs cumulative sensitivity — relative logit difference across all 12 layers</sub>
</div>

<br/>

<div align="center">
<img src="assets/slide_layer_heatmaps.png" width="820"/>
<br/><sub><b>Fig 5.</b> Layer-wise sensitivity heatmaps: isolated (top) and cumulative (bottom) compression</sub>
</div>

<br/>

<div align="center">
<img src="assets/slide_ppl_heatmap.png" width="820"/>
<br/><sub><b>Fig 6.</b> WikiText-2 perplexity heatmap — Layer 0 extreme outlier clearly visible (yellow = high PPL)</sub>
</div>

<br/>

<div align="center">
<img src="assets/layerwise_sensitivity.png" width="680"/>
<br/><sub><b>Fig 7.</b> Per-layer PPL impact (Extras notebook) — bar chart confirming Layer 0 outlier</sub>
</div>

### Cumulative Compression

Compressing layers 0 through $k$ sequentially shows compounding, non-linear degradation:

| Layers Compressed | Logit Diff | PPL |
|:---:|:---:|:---:|
| 1 | ~0.003 | ~62 |
| 4 | ~0.65 | ~8,700 |
| 8 | ~0.72 | ~15,600 |
| **12 (all)** | **0.849** | **43,022** |

<div align="center">
<img src="assets/cumulative_compression.png" width="680"/>
<br/><sub><b>Fig 8.</b> Cumulative compression curve — PPL grows non-linearly as more layers are replaced</sub>
</div>

---

## Phase 5 — Fine-Tuning Recovery

### Why fine-tuning is necessary

Kronecker factors from SVD are the best rank-1 approximation to $W$ in the Frobenius sense — but the entire model was pre-trained with original weights. After substitution, the model needs to re-adapt all remaining parameters to the new weight structure.

**Config:** AdamW (lr=3e-5) · mixed precision (AMP) · 2,000 WikiText-2 training samples

### 5-Epoch Training Loss (12-layer compressed)

| Epoch | Avg Loss |
|:---:|:---:|
| 1 | 4.70 |
| 2 | 4.28 |
| 3 | 3.93 |
| 4 | 3.62 |
| 5 | 3.31 |

Post-training PPL (all 12 layers, 5ep): **266.6**

<br/>

<div align="center">
<img src="assets/slide_ppl_recovery.png" width="820"/>
<br/><sub><b>Fig 9.</b> Perplexity recovery — before fine-tuning, after 2 epochs, vs baseline</sub>
</div>

### Partial Compression Results

Compressing only 6 of 12 layers preserves model quality much better — and with 5 epochs of fine-tuning, PPL reaches ~100:

<div align="center">

| Model | Params | PPL After Fine-tune |
|:---:|:---:|:---:|
| Original GPT-2 | 124M | 60.27 |
| 12-layer Kronecker + 2ep | 67.9M | 266.6 |
| **First-6 layers + 5ep** | **~96M** | **99.9** ✅ |
| **Last-6 layers + 5ep** | **~96M** | **103.2** ✅ |
| DistilGPT-2 (reference) | 82M | 88.97 |

</div>

<br/>

<div align="center">
<img src="assets/slide_compression_depth.png" width="820"/>
<br/><sub><b>Fig 10.</b> Compression depth vs perplexity — first-6 and last-6 layer strategies after fine-tuning</sub>
</div>

---

## Phase 6 — Rank-2 Extension

### Rank-2 Kronecker

$$W \approx A_1 \otimes B_1 + A_2 \otimes B_2$$

Captures the top-2 singular value pairs of $\tilde{W}$. Better initialization before fine-tuning at the cost of ~18% more total parameters.

<div align="center">

| Method | Params | PPL (no FT) | Compression |
|:---:|:---:|:---:|:---:|
| Baseline GPT-2 | 124M | 60.27 | — |
| Rank-1 Kronecker | 67.9M | 43,022 | 45.4% |
| **Rank-2 Kronecker** | **~80M** | **21,978** | **~35%** |
| DistilGPT-2 | 82M | 88.97 | 34% |
| KronyPT-81M (paper) | 81M | **35.75** | 35% |

</div>

**Rank-2 is 2× better as a weight initialization** before any fine-tuning — directly confirming Table II of the Krony-PT paper. Each additional rank pair recovers a progressively smaller fraction of the singular value energy, so returns diminish quickly.

<br/>

<div align="center">
<img src="assets/slide_rank1_vs_rank2.png" width="820"/>
<br/><sub><b>Fig 11.</b> Complete Rank-1 vs Rank-2 comparison — PPL before/after FT, parameter count, storage, latency</sub>
</div>

---

## 📊 Extras — AxB vs AxBxC · Full Rank-R Sweep

The supplementary notebook (`Extras_Plots.ipynb`) runs a complete comparison across two-factor (AxB) and three-factor (AxBxC) Kronecker over ranks 1, 2, 4, 8 on the full WikiText-2 pipeline.

### Parameter Count — All Variants

<div align="center">
<img src="assets/param_comparison.png" width="680"/>
<br/><sub><b>Fig 12.</b> All Kronecker variants achieve the same ~45.4% reduction — differences are purely in quality</sub>
</div>

### Pre-Fine-Tuning Perplexity

<div align="center">
<img src="assets/ppl_comparison.png" width="680"/>
<br/><sub><b>Fig 13.</b> Pre-FT perplexity — AxBxC bar is off scale (~5×10⁷) due to random B, C initialization</sub>
</div>

<div align="center">

| Model | PPL (before FT) |
|:---:|:---:|
| Baseline GPT-2 | 60.27 |
| AxB Rank-1 | ~46,162 |
| AxB Rank-2 | ~47,800 |
| AxB Rank-4 | ~43,088 |
| **AxBxC** | **~5×10⁷** (random init) |

</div>

### Rank vs. Perplexity (AxB Family)

<div align="center">
<img src="assets/rank_vs_ppl.png" width="680"/>
<br/><sub><b>Fig 14.</b> Rank vs pre-FT perplexity — higher rank captures more of the singular value structure</sub>
</div>

### Retraining Loss Curves (AxB vs AxBxC)

<div align="center">
<img src="assets/retraining_loss.png" width="680"/>
<br/><sub><b>Fig 15.</b> 5-epoch retraining loss — AxB (SVD init) converges ~2× faster than AxBxC (random init)</sub>
</div>

<div align="center">

| Metric | AxB (Rank-1) | AxBxC |
|:---:|:---:|:---:|
| Loss (Epoch 1) | 7.95 | 13.82 |
| Loss (Epoch 5) | 6.52 | 8.35 |
| **PPL after FT** | **1,059** | 3,875 |
| Training time | 529.9s | 552.4s |

</div>

### Perplexity Before vs After Retraining

<div align="center">
<img src="assets/ppl_before_after.png" width="680"/>
<br/><sub><b>Fig 16.</b> AxB recovers far better than AxBxC after the same fine-tuning budget</sub>
</div>

### Rank vs. Parameter Count

<div align="center">
<img src="assets/rank_vs_compression.png" width="680"/>
<br/><sub><b>Fig 17.</b> Linear relationship between rank and parameter count (~1.5M added per rank unit)</sub>
</div>

### Time vs. Compression (Scatter)

<div align="center">
<img src="assets/time_vs_compression.png" width="680"/>
<br/><sub><b>Fig 18.</b> Training time vs compression ratio — all AxB variants are in the same time budget</sub>
</div>

### Compression vs. Performance — Pareto Frontier

<div align="center">
<img src="assets/pareto_frontier.png" width="680"/>
<br/><sub><b>Fig 19.</b> Pareto frontier — AxB post-FT variants dominate AxBxC in the parameter–quality tradeoff space</sub>
</div>

### Full GPT-2 Family Comparison

<div align="center">
<img src="assets/slide_full_comparison.png" width="820"/>
<br/><sub><b>Fig 20.</b> Kronecker compression vs distillation — complete GPT-2 family across training configurations</sub>
</div>

### Singular Value Spectrum — Why Low-Rank Works

The rearranged matrix $\tilde{W}$ has a **rapidly decaying singular value spectrum**, confirming that a low-rank Kronecker structure is a good approximation:

| Rank | Relative Reconstruction Error |
|:---:|:---:|
| 1 | highest |
| 2 | moderate |
| 4 | low |
| 8 | very low |

### Triple Kronecker — AxBxC Extreme Compression

| Method | Params per layer | Notes |
|:---:|:---:|:---:|
| Original $W$ | 2,359,296 | — |
| $A \otimes B$ Rank-1 | 4,672 | SVD-initialized |
| $A \otimes B \otimes C$ | ~400 | Random B, C — needs heavy FT |

Triple Kronecker achieves **~5,890× per-layer parameter reduction** at the cost of entirely random initialization for factors B and C.

---

## 🔬 Sparse Residual Innovation (Beyond the Paper)

We proposed and implemented a **Sparse Residual Correction** not present in the original Krony-PT paper:

$$W_{\text{ours}} = \alpha (A \otimes B) + S$$

where $S = \mathrm{top}\text{-}k\%(W - \alpha(A \otimes B))$ keeps only the highest-magnitude residual entries.

### Results (4×4 toy matrix — verified in MATLAB and Python)

<div align="center">

| Method | Frobenius Error | vs. Paper Method |
|:---:|:---:|:---:|
| Pruning (identity baseline) | 136.62 | — |
| Van Loan SVD + α (paper method) | 9.1492 | — |
| **Our method (SVD + α + Sparse)** | **4.8922** | **47% better** ✅ |

</div>

```python
def sparse_residual_correction(W, A, B, k_percent=10):
    alpha   = torch.norm(W, 'fro') / torch.norm(torch.kron(A, B), 'fro')
    W_approx = alpha * torch.kron(A, B)
    R        = W - W_approx                               # Residual
    threshold = torch.quantile(R.abs(), 1 - k_percent/100)
    S        = R * (R.abs() >= threshold)                 # Keep top-k% entries
    return W_approx + S, alpha
```

> Run `python scripts/step2_sparse_residual.py`
> Expected: `SUCCESS: Python proves your method wins!`

---

## 🆚 DistilGPT-2 Comparison

<div align="center">

| Model | Method | Params | WikiText-2 PPL |
|---|---|:---:|:---:|
| GPT-2 Small | Baseline | 124M | 59.11 |
| DistilGPT-2 | Knowledge Distillation | 82M | 88.97 |
| KronyPT-81M (paper) | Kronecker + 3ep OpenWebText (A100) | 81M | **35.75** |
| Ours: Rank-1 (no FT) | Kronecker SVD init | 67.9M | 50,011 |
| Ours: Rank-1 + 2ep | Kronecker + WikiText-2 FT | 67.9M | 266.6 |
| **Ours: First-6 + 5ep** | **Partial Kronecker** | **~96M** | **99.9** |
| Ours: Last-6 + 5ep | Partial Kronecker | ~96M | 103.2 |

</div>

**On the gap between our PPL (~100) and the paper's (35.75):**

KronyPT-81M was trained for 3 full epochs on **OpenWebText (~8 billion tokens)** on an **A100 GPU**. Our experiments used **2,000 WikiText-2 samples** on an **RTX 4050 Laptop (6GB VRAM)**. The compression architecture is identical. The gap is **purely a compute budget difference**, not a methodological one.

---

## 📁 Repository Structure

```
LLM-Compression-using-Krony-PT/
│
├── 📁 Documentation/                        ← Exported PDFs & supporting visuals
│   ├── Adaptive Normalization.pdf
│   ├── Comparisons.pdf                      ← Benchmark notebook export
│   ├── DistilGPT2_Comparison.ipynb
│   ├── Final-Plots.pdf                      ← Layer sensitivity notebook export
│   ├── GPT2_Kronecker_Compression_Documentation_FIXED (1).ipynb  ← Main notebook (copy)
│   ├── Rank1_vs_Rank2_Comparison.ipynb
│   ├── Untitled.pdf
│   ├── distilgpt2_comparison.png
│   ├── gpt2_exploded_diagram.html
│   └── rank1_vs_rank2_comparison.png
│
├── 📁 Jupyter/                              ← All experiment notebooks
│   ├── Adaptive Normalization.ipynb         ← Phase 3: alpha rescaling
│   ├── Baseline_GPT2small.ipynb             ← Phase 1: baseline eval
│   ├── Comparisons.ipynb                    ← Parameter / latency / size benchmarks
│   ├── Extras_Plots.ipynb                   ← AxB vs AxBxC · Rank-R sweep ★
│   ├── Final-Plots.ipynb                    ← Layer sensitivity plots
│   ├── GPT2_Kronecker_Compression_Documentation_FIXED (1).ipynb  ← Main 6-phase notebook ★
│   ├── Isolated vs Cumulative Compression.ipynb
│   ├── Logit_Score_Calculation_Layer_by-layer.ipynb
│   ├── Pre_Training.ipynb                   ← Fine-tuning recovery ★
│   ├── Rank-2_Compression.ipynb             ← Phase 6: Rank-2 extension
│   ├── Rank1_vs_Rank2_Comparison.ipynb      ← Rank-1 vs Rank-2 dashboard ★
│   ├── Retraining Loss.ipynb
│   ├── Singular Value Check and A(kron)B(kron)C.ipynb  ← Triple Kronecker
│   ├── Three_Factor_Compression.ipynb
│   └── ...                                  ← Additional exploration notebooks
│
├── 📁 Matlab/                               ← MATLAB replication & sparse residual
│   ├── Basic_Krony_PT.mlx
│   ├── Final.mlx
│   ├── Krony_PT_Alpha_and_Prunning.mlx      ← Alpha normalization + pruning baseline
│   ├── Krony_PT_Replication.mlx             ← Full Van Loan replication
│   └── Sparse_Matrix_Method.mlx             ← Sparse residual verification ★
│
├── 📁 Python/
│   ├── 📁 New/                              ← Current working scripts
│   │   ├── step2_sparse_residual.py         ← Sparse Residual 47% improvement demo ★
│   │   ├── compression_agent.py
│   │   ├── chat_with_gpt2.py
│   │   └── test_step1_rearrangement.py      ← Van Loan math unit test
│   ├── 📁 Old/                              ← Archived earlier versions
│   └── 📁 Visualizations/                   ← Manim & matplotlib animation scripts
│       └── gpt2_visualization/
│           └── gpt2_research_animation/     ← GPT-2 architecture animations
│
├── 📁 assets/                               ← All result plots used in this README
│   ├── slide_param_inference_results.png
│   ├── slide_model_size_results.png
│   ├── slide_efficiency_dashboard.png
│   ├── slide_layer_sensitivity_logit.png
│   ├── slide_layer_heatmaps.png
│   ├── slide_ppl_heatmap.png
│   ├── slide_ppl_recovery.png
│   ├── slide_compression_depth.png
│   ├── slide_full_comparison.png
│   ├── slide_rank1_vs_rank2.png
│   ├── param_comparison.png
│   ├── ppl_comparison.png
│   ├── rank_vs_ppl.png
│   ├── layerwise_sensitivity.png
│   ├── cumulative_compression.png
│   ├── retraining_loss.png
│   ├── ppl_before_after.png
│   ├── rank_vs_compression.png
│   ├── time_vs_compression.png
│   └── pareto_frontier.png
│
├── ⚙️ .gitignore
├── 📕 Base_Paper.pdf                        ← Krony-PT reference paper
├── 📕 MFC_4__Kronecker_Product_.pdf         ← Course project report
├── 📄 LICENSE
└── 📖 README.md
```

> **★** = primary notebooks to run for reproducing reported results.  
> **To render images on GitHub:** The `assets/` folder is already at the repo root — all image paths in this README reference `assets/filename.png` and will render automatically.

---

## 🚀 How to Run

### 1. Clone

```bash
git clone https://github.com/mahakisore/LLM-Compression-using-Krony-PT
cd LLM-Compression-using-Krony-PT
```

### 2. Install dependencies

```bash
pip install torch transformers datasets einops matplotlib seaborn pandas accelerate
# Or use the included requirements file:
pip install -r Python/requirements.txt
```

### 3. Verify the math (no GPU needed, runs in seconds)

```bash
python Python/New/step2_sparse_residual.py
# Expected: SUCCESS: Python proves your method wins!
```

### 4. Run the main 6-phase documentation notebook

```bash
jupyter notebook "Jupyter/GPT2_Kronecker_Compression_Documentation_FIXED (1).ipynb"
```

### 5. Run individual experiment notebooks

```bash
jupyter notebook Jupyter/Baseline_GPT2small.ipynb           # Phase 1: baseline eval
jupyter notebook Jupyter/Comparisons.ipynb                  # Compression benchmarks
jupyter notebook Jupyter/Pre_Training.ipynb                 # Fine-tuning recovery
jupyter notebook Jupyter/Rank1_vs_Rank2_Comparison.ipynb    # Rank-1 vs Rank-2
jupyter notebook Jupyter/Extras_Plots.ipynb                 # AxB vs AxBxC full sweep
jupyter notebook "Jupyter/Isolated vs Cumulative Compression.ipynb"   # Sensitivity
jupyter notebook "Jupyter/Singular Value Check and A(kron)B(kron)C.ipynb"  # Triple Kronecker
```

### Approximate runtimes (RTX 4050 Laptop GPU, 6GB VRAM)

| Experiment | Notebook | Time |
|---|---|---|
| Baseline PPL eval (~500 batches, WikiText-2) | `Baseline_GPT2small.ipynb` | ~2 min |
| Rank-1 compression all 12 layers (SVD only) | `Comparisons.ipynb` | ~15 sec |
| Forward pass latency benchmark (50 runs) | `Comparisons.ipynb` | ~30 sec |
| Fine-tuning 12-layer model (2ep, 2000 samples) | `Pre_Training.ipynb` | ~8 min |
| Fine-tuning 6-layer model (5ep, 2000 samples) | `Pre_Training.ipynb` | ~10 min |
| Layer sensitivity (isolated, 12 models × eval) | `Final-Plots.ipynb` | ~25 min |
| Cumulative sensitivity (13 models × eval) | `Isolated vs Cumulative Compression.ipynb` | ~30 min |
| AxB vs AxBxC rank sweep | `Extras_Plots.ipynb` | ~20 min |
| **Total end-to-end** | | **~90–100 min** |

> **Tip for reproducibility:** Use `%%time` at the top of Jupyter cells, or wrap blocks with `import time; t = time.time(); ...; print(f"{time.time()-t:.1f}s")` to log exact timings.

---

## 🏁 Conclusions

1. **45.4% parameter and storage reduction** is achieved by replacing all 12 FFN blocks with Rank-1 Kronecker factors — zero changes to attention, embeddings, or layer norms.

2. **Throughput improves (+14.7%)** despite slight latency overhead (+12.3%), because smaller weight matrices improve GPU memory bandwidth utilization during inference.

3. **Perplexity is fully recoverable through fine-tuning.** Partial compression (6 of 12 layers) with 5 epochs on WikiText-2 reaches **PPL ~100**, approaching DistilGPT-2 quality on a fraction of the compute budget.

4. **Layer 0 is uniquely sensitive** (isolated PPL 4,825 vs 62–68 for all others). Skipping it during compression is a practical heuristic confirmed experimentally.

5. **Rank-2 provides 2× better pre-FT initialization** than Rank-1, with ~18% more parameters — directly validating Table II of the Krony-PT paper.

6. **Triple Kronecker (AxBxC)** achieves extreme per-layer compression (~400 params vs 2.36M) but is entirely dependent on fine-tuning due to random B, C initialization.

7. **Our Sparse Residual Correction** reduces Frobenius reconstruction error by **47% over the Van Loan baseline** — a novel contribution beyond the paper, verified in MATLAB and Python.

8. **The gap to the paper's PPL (35.75 vs ~100)** is entirely due to training compute: 8B tokens on an A100 vs 2,000 samples on an RTX 4050. The architecture, decomposition, and training setup are otherwise equivalent.

---

## 📚 References

1. Ayad, M. A. B., Mitrović, J., & Granitzer, M. (2025). *KronyPT: GPT-2 Compressed with Kronecker Products.* FLLM 2025, IEEE. [[arXiv:2412.12351]](https://arxiv.org/abs/2412.12351)

2. Van Loan, C. F., & Pitsianis, N. (1993). *Approximation with Kronecker Products.* In *Linear Algebra for Large Scale and Real-Time Applications.* Springer Netherlands, pp. 293–314.

3. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI Blog.

4. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* [[arXiv:1910.01108]](https://arxiv.org/abs/1910.01108)

5. Alammar, J. *The Illustrated Transformer.* https://jalammar.github.io/illustrated-transformer/

6. 3Blue1Brown — *Neural Networks* YouTube series.

---

<div align="center">

**Amrita Vishwa Vidyapeetham · 22MAT230 – Mathematics for Computing 4 · Semester 4 · Team D11**

*Hemanth SN &nbsp;·&nbsp; Mahakisore M &nbsp;·&nbsp; Yashwanth B*

</div>
