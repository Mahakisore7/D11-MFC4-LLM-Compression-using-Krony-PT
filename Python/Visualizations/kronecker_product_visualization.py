"""
Comprehensive Manim Visualization for Kronecker Product & LLM Compression
==========================================================================
This file creates beautiful, educational animations explaining:
1. Kronecker Product basics
2. Matrix decomposition
3. Van Loan Rearrangement
4. Sparse residual correction
5. LLM compression process

Run with:
    manim -pql kronecker_product_visualization.py SceneName
    
For high quality:
    manim -pqh kronecker_product_visualization.py SceneName
"""

from manim import *
from manim_physics import *
import numpy as np
from enum import Enum


# ============================================================================
# SCENE 1: Kronecker Product Basics
# ============================================================================

class KroneckerProductIntro(Scene):
    """
    Introduction to the Kronecker Product with animated matrices
    A ⊗ B = [[a_11*B, a_12*B], [a_21*B, a_22*B], ...]
    """
    
    def construct(self):
        # Title
        title = Text("The Kronecker Product", font_size=60, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)
        
        # Subtitle
        subtitle = Text("A ⊗ B: Tensor Product of Matrices", font_size=30, color=BLUE)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(subtitle), run_time=1)
        
        # Matrix A
        A_label = Text("Matrix A (2×2)", font_size=28, color=YELLOW)
        A_label.to_edge(LEFT, buff=0.5)
        A_label.shift(UP * 2)
        
        A_data = [[1, 2], [3, 4]]
        A = self._create_matrix(A_data, color=YELLOW)
        A.next_to(A_label, DOWN, buff=0.3)
        
        # Matrix B
        B_label = Text("Matrix B (2×2)", font_size=28, color=GREEN)
        B_label.shift(RIGHT * 3)
        B_label.shift(UP * 2)
        
        B_data = [[5, 6], [7, 8]]
        B = self._create_matrix(B_data, color=GREEN)
        B.next_to(B_label, DOWN, buff=0.3)
        
        # Operator
        operator = Text("⊗", font_size=80, color=RED)
        operator.move_to(ORIGIN + UP * 0.5)
        
        self.play(
            Write(A_label),
            FadeIn(A),
            Write(B_label),
            FadeIn(B),
            Write(operator),
            run_time=2
        )
        
        self.wait(1)
        
        # Calculation
        calc_label = Text("Result: A ⊗ B (4×4)", font_size=28, color=PURPLE)
        calc_label.to_edge(DOWN, buff=1)
        
        result_data = self._compute_kronecker(A_data, B_data)
        result = self._create_matrix(result_data, color=PURPLE, scale=0.7)
        result.next_to(calc_label, DOWN, buff=0.3)
        
        self.play(
            FadeOut(A_label),
            FadeOut(A),
            FadeOut(B_label),
            FadeOut(B),
            FadeOut(operator),
            Write(calc_label),
            FadeIn(result),
            run_time=2
        )
        
        # Show the expansion
        self.wait(0.5)
        explanation = Text(
            "Each element of A multiplies the entire matrix B",
            font_size=24,
            color=BLUE_E
        )
        explanation.to_edge(UP, buff=0.5)
        self.play(Write(explanation), run_time=1.5)
        
        self.wait(2)
    
    def _create_matrix(self, data, color=WHITE, scale=1.0):
        """Helper to create a matrix visualization"""
        matrix = Matrix(
            data,
            h_buff=1.5,
            v_buff=1.5,
            bracket_h_buff=0.2,
            bracket_v_buff=0.2,
        )
        matrix.set_color(color)
        matrix.scale(scale)
        return matrix
    
    def _compute_kronecker(self, A, B):
        """Compute Kronecker product"""
        A = np.array(A)
        B = np.array(B)
        m, n = A.shape
        p, q = B.shape
        result = np.zeros((m*p, n*q))
        
        for i in range(m):
            for j in range(n):
                result[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B
        
        return result.tolist()


# ============================================================================
# SCENE 2: Why We Need Compression
# ============================================================================

class WhyCompression(Scene):
    """
    Motivate the need for compression in LLMs
    """
    
    def construct(self):
        # Title
        title = Text("Why Compress Large Language Models?", font_size=50, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # Create model comparison
        problems = [
            ("Large Models", "13B-175B parameters", "💾 Storage: ~26GB-350GB+"),
            ("Slow Inference", "High latency", "⏱️ Inference time: seconds"),
            ("GPU Memory", "A100 80GB needed", "💸 Expensive hardware required"),
        ]
        
        y_start = 2
        for i, (title_text, problem, metric) in enumerate(problems):
            y_pos = y_start - i * 2
            
            # Title
            problem_title = Text(title_text, font_size=32, weight=BOLD, color=RED)
            problem_title.move_to(LEFT * 3 + UP * y_pos)
            
            # Problem
            problem_text = Text(problem, font_size=24, color=YELLOW)
            problem_text.next_to(problem_title, RIGHT, buff=0.5)
            
            # Metric
            metric_text = Text(metric, font_size=20, color=GRAY)
            metric_text.next_to(problem_title, DOWN, buff=0.3)
            
            self.play(
                Write(problem_title),
                Write(problem_text),
                Write(metric_text),
                run_time=1
            )
            self.wait(0.5)
        
        self.wait(1)
        
        # Solution
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != title])
        
        solution_title = Text("Solution: Kronecker Compression", font_size=45, weight=BOLD, color=GREEN)
        solution_title.move_to(ORIGIN)
        
        benefits = [
            "✓ Reduce parameters by 50-90%",
            "✓ Maintain performance",
            "✓ Preserve model structure",
            "✓ Add sparse residuals for precision",
        ]
        
        self.play(Write(solution_title), run_time=1)
        
        benefit_group = VGroup()
        for i, benefit in enumerate(benefits):
            benefit_text = Text(benefit, font_size=24, color=BLUE)
            benefit_text.shift(DOWN * (i + 1))
            benefit_group.add(benefit_text)
        
        self.play(FadeIn(benefit_group), run_time=1.5)
        self.wait(2)


# ============================================================================
# SCENE 3: Matrix Rearrangement (Van Loan)
# ============================================================================

class VanLoanRearrangement(Scene):
    """
    Visualize the Van Loan rearrangement process
    """
    
    def construct(self):
        # Title
        title = Text("Van Loan Rearrangement", font_size=50, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        subtitle = Text("Step 1: Reshape matrix into blocks", font_size=28, color=BLUE_C)
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(subtitle), run_time=0.8)
        
        # Original matrix
        original_label = Text("Original Weight Matrix W", font_size=24, color=YELLOW)
        original_label.to_edge(LEFT, buff=0.5)
        original_label.shift(UP)
        
        # Create a 4x4 matrix visualization
        W_data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        W = self._create_visual_matrix(W_data, color=YELLOW)
        W.next_to(original_label, DOWN, buff=0.3)
        
        self.play(Write(original_label), FadeIn(W), run_time=1)
        self.wait(1)
        
        # Arrow indicating transformation
        arrow = Arrow(
            W.get_right() + RIGHT * 0.5,
            W.get_right() + RIGHT * 2.5,
            color=RED,
            tip_length=0.2,
            stroke_width=3
        )
        arrow_label = Text("Rearrange", font_size=18, color=RED)
        arrow_label.next_to(arrow, UP, buff=0.1)
        
        self.play(GrowArrow(arrow), Write(arrow_label), run_time=1)
        
        # Rearranged matrix
        rearranged_label = Text("Rearranged W̃", font_size=24, color=GREEN)
        rearranged_label.move_to(RIGHT * 3 + UP)
        
        W_rearranged = self._create_visual_matrix(W_data, color=GREEN, highlight_blocks=True)
        W_rearranged.next_to(rearranged_label, DOWN, buff=0.3)
        
        self.play(
            Write(rearranged_label),
            FadeIn(W_rearranged),
            run_time=1
        )
        
        self.wait(1)
        
        # Explanation
        explanation = Text(
            "The rearrangement groups elements to extract optimal Kronecker factors",
            font_size=20,
            color=BLUE
        )
        explanation.to_edge(DOWN, buff=0.5)
        self.play(Write(explanation), run_time=1)
        
        self.wait(2)
    
    def _create_visual_matrix(self, data, color=WHITE, highlight_blocks=False, scale=0.6):
        """Create a colored matrix visualization"""
        matrix = Matrix(
            data,
            h_buff=1.2,
            v_buff=1.2,
        )
        matrix.set_color(color)
        matrix.scale(scale)
        
        if highlight_blocks:
            # Add subtle highlighting to show block structure
            for i in range(len(data)):
                for j in range(len(data[0])):
                    color_alpha = color.copy()
                    color_alpha.set_opacity(0.3 if (i + j) % 2 == 0 else 0.7)
        
        return matrix


# ============================================================================
# SCENE 4: SVD Decomposition
# ============================================================================

class SVDDecomposition(Scene):
    """
    Visualize SVD decomposition process
    """
    
    def construct(self):
        # Title
        title = Text("Singular Value Decomposition (SVD)", font_size=50, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # Formula
        formula = MathTex(r"W \approx \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T", font_size=44)
        formula.next_to(title, DOWN, buff=0.3)
        self.play(Write(formula), run_time=1)
        
        self.wait(0.5)
        
        # Show components
        components = [
            (r"\sigma_1", "Singular Value", YELLOW, 1),
            (r"\mathbf{u}_1", "Left Vector", BLUE, 2),
            (r"\mathbf{v}_1^T", "Right Vector^T", GREEN, 3),
        ]
        
        y_start = 1
        for latex, name, color, x_pos in components:
            comp_tex = MathTex(latex, color=color, font_size=36)
            comp_tex.shift(LEFT * (x_pos - 2) * 2 + DOWN * 2)
            
            comp_name = Text(name, font_size=20, color=color)
            comp_name.next_to(comp_tex, DOWN, buff=0.2)
            
            self.play(Write(comp_tex), Write(comp_name), run_time=0.8)
        
        self.wait(1)
        
        # Explain importance
        importance = Text(
            "The first singular value captures maximum variance in the matrix",
            font_size=22,
            color=BLUE
        )
        importance.to_edge(DOWN, buff=0.5)
        self.play(Write(importance), run_time=1.5)
        
        self.wait(2)


# ============================================================================
# SCENE 5: Kronecker Factors Extraction
# ============================================================================

class KroneckerFactorsExtraction(Scene):
    """
    Show how to extract A and B from SVD components
    """
    
    def construct(self):
        # Title
        title = Text("Extracting Kronecker Factors A and B", font_size=48, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # From SVD
        svd_formula = MathTex(
            r"A = \sqrt{\sigma_1} \cdot \text{reshape}(\mathbf{u}_1)",
            font_size=36,
            color=BLUE
        )
        svd_formula.move_to(LEFT * 3 + UP)
        
        self.play(Write(svd_formula), run_time=1)
        
        b_formula = MathTex(
            r"B = \sqrt{\sigma_1} \cdot \text{reshape}(\mathbf{v}_1)",
            font_size=36,
            color=GREEN
        )
        b_formula.next_to(svd_formula, DOWN, buff=0.5)
        
        self.play(Write(b_formula), run_time=1)
        
        self.wait(0.5)
        
        # Show sizes
        size_info = Text(
            "If W is m×n, then A is m^0.5 × m^0.5 and B is n^0.5 × n^0.5",
            font_size=24,
            color=YELLOW
        )
        size_info.next_to(b_formula, DOWN, buff=1)
        
        self.play(Write(size_info), run_time=1)
        
        self.wait(0.5)
        
        # Example
        example_label = Text("Example: 4×4 matrix → 2×2 ⊗ 2×2", font_size=26, color=PURPLE)
        example_label.next_to(size_info, DOWN, buff=1)
        
        self.play(Write(example_label), run_time=1)
        
        # Visualization
        example_visual = MathTex(
            r"W_{4\times 4} = A_{2\times 2} \otimes B_{2\times 2}",
            font_size=32,
            color=PURPLE
        )
        example_visual.next_to(example_label, DOWN, buff=0.5)
        
        self.play(Write(example_visual), run_time=1)
        
        self.wait(2)


# ============================================================================
# SCENE 6: Sparse Residual Correction
# ============================================================================

class SparseResidualCorrection(Scene):
    """
    Visualize the sparse residual correction method
    """
    
    def construct(self):
        # Title
        title = Text("Sparse Residual Correction", font_size=50, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        subtitle = Text("Innovation: Capture what Kronecker decomposition misses", font_size=24, color=BLUE)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(subtitle), run_time=1)
        
        self.wait(0.5)
        
        # Step 1: Reconstruction
        step1_title = Text("Step 1: Approximate Reconstruction", font_size=28, weight=BOLD)
        step1_title.move_to(LEFT * 3 + UP * 1.5)
        
        step1_formula = MathTex(
            r"\tilde{W} = \alpha (A \otimes B)",
            font_size=36,
            color=GREEN
        )
        step1_formula.next_to(step1_title, DOWN, buff=0.3)
        
        self.play(Write(step1_title), Write(step1_formula), run_time=1)
        
        # Step 2: Calculate residual
        step2_title = Text("Step 2: Calculate Residual", font_size=28, weight=BOLD)
        step2_title.move_to(LEFT * 3 + DOWN * 0.5)
        
        step2_formula = MathTex(
            r"R = W - \tilde{W}",
            font_size=36,
            color=RED
        )
        step2_formula.next_to(step2_title, DOWN, buff=0.3)
        
        self.play(Write(step2_title), Write(step2_formula), run_time=1)
        
        # Step 3: Sparse selection
        step3_title = Text("Step 3: Select Top-k Errors", font_size=28, weight=BOLD)
        step3_title.move_to(LEFT * 3 + DOWN * 2.5)
        
        step3_formula = MathTex(
            r"S = \text{sparsify}(R, \text{top-k\%})",
            font_size=36,
            color=YELLOW
        )
        step3_formula.next_to(step3_title, DOWN, buff=0.3)
        
        self.play(Write(step3_title), Write(step3_formula), run_time=1)
        
        # Final result
        self.wait(0.5)
        
        final_title = Text("Final Compressed Model", font_size=32, weight=BOLD, color=PURPLE)
        final_title.to_edge(DOWN, buff=1.5)
        
        final_formula = MathTex(
            r"W_{\text{final}} = \alpha (A \otimes B) + S",
            font_size=44,
            color=PURPLE
        )
        final_formula.next_to(final_title, DOWN, buff=0.3)
        
        self.play(Write(final_title), Write(final_formula), run_time=1)
        
        self.wait(2)


# ============================================================================
# SCENE 7: Error Comparison
# ============================================================================

class ErrorComparison(Scene):
    """
    Compare error metrics across methods
    """
    
    def construct(self):
        # Title
        title = Text("Error Comparison: Our Innovation", font_size=50, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # Create comparison bars
        methods = [
            ("Naive Pruning", 136.62, RED),
            ("Van Loan (Paper)", 9.1492, BLUE),
            ("Our Method", 4.8922, GREEN),
        ]
        
        # Scale for visualization
        scale_factor = 0.03
        
        bar_group = VGroup()
        
        y_start = 1
        for i, (method_name, error, color) in enumerate(methods):
            y_pos = y_start - i * 1.5
            
            # Method name
            name_label = Text(method_name, font_size=24, weight=BOLD)
            name_label.to_edge(LEFT, buff=0.5)
            name_label.shift(UP * y_pos)
            
            # Error bar
            bar_width = error * scale_factor
            bar = Rectangle(
                width=bar_width,
                height=0.3,
                fill_color=color,
                fill_opacity=0.7,
                stroke_color=color,
                stroke_width=2
            )
            bar.next_to(name_label, RIGHT, buff=0.5)
            bar.align_to(name_label, UP)
            
            # Error value
            error_label = Text(f"Error: {error:.2f}", font_size=20, color=color)
            error_label.next_to(bar, RIGHT, buff=0.3)
            error_label.align_to(bar, UP)
            
            bar_group.add(name_label, bar, error_label)
        
        self.play(FadeIn(bar_group), run_time=1.5)
        
        self.wait(1)
        
        # Highlight improvement
        improvement = Text(
            "47% Error Reduction vs. Paper Method",
            font_size=28,
            weight=BOLD,
            color=GREEN
        )
        improvement.to_edge(DOWN, buff=0.5)
        
        improvement_bg = Rectangle(
            width=improvement.width + 0.5,
            height=improvement.height + 0.3,
            fill_color=GREEN,
            fill_opacity=0.1,
            stroke_color=GREEN,
            stroke_width=2
        )
        improvement_bg.move_to(improvement)
        
        self.play(FadeIn(improvement_bg), Write(improvement), run_time=1)
        
        self.wait(2)


# ============================================================================
# SCENE 8: Compression Pipeline
# ============================================================================

class CompressionPipeline(Scene):
    """
    Show the complete compression pipeline
    """
    
    def construct(self):
        # Title
        title = Text("Complete Compression Pipeline", font_size=50, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        self.wait(0.5)
        
        # Pipeline steps
        steps = [
            "Load Pre-trained GPT-2",
            "Apply Van Loan Rearrangement",
            "Perform SVD Decomposition",
            "Extract A, B Factors",
            "Calculate Residuals",
            "Apply Sparse Thresholding",
            "Save Compressed Model",
        ]
        
        boxes = VGroup()
        connections = VGroup()
        
        y_start = 2.5
        for i, step in enumerate(steps):
            y_pos = y_start - i * 0.8
            
            # Create box
            box = Rectangle(
                width=4,
                height=0.6,
                fill_color=BLUE,
                fill_opacity=0.3,
                stroke_color=BLUE,
                stroke_width=2
            )
            box.move_to(UP * y_pos)
            
            # Add text
            text = Text(step, font_size=18, weight=BOLD)
            text.move_to(box.get_center())
            
            # Add to group
            boxes.add(box)
            
            # Add connection arrow
            if i < len(steps) - 1:
                arrow = Arrow(
                    box.get_bottom() + DOWN * 0.15,
                    box.get_bottom() + DOWN * 0.65,
                    color=YELLOW,
                    stroke_width=2,
                    tip_length=0.1
                )
                connections.add(arrow)
            
            # Animate
            self.play(FadeIn(box), Write(text), run_time=0.4)
            
            if i < len(steps) - 1:
                self.play(GrowArrow(connections[-1]), run_time=0.3)
        
        self.wait(2)


# ============================================================================
# SCENE 9: Compression Ratio Visualization
# ============================================================================

class CompressionRatio(Scene):
    """
    Visualize the compression ratio achieved
    """
    
    def construct(self):
        # Title
        title = Text("Compression Ratio Achievement", font_size=50, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # Original model
        original_label = Text("Original GPT-2", font_size=28, weight=BOLD)
        original_label.move_to(LEFT * 3 + UP * 1)
        
        original_box = Rectangle(
            width=2,
            height=4,
            fill_color=RED,
            fill_opacity=0.5,
            stroke_color=RED,
            stroke_width=2
        )
        original_box.next_to(original_label, DOWN, buff=0.3)
        
        original_size = Text("Full Model\n124M params\n~500MB", font_size=18, color=WHITE)
        original_size.move_to(original_box.get_center())
        
        self.play(
            Write(original_label),
            FadeIn(original_box),
            Write(original_size),
            run_time=1
        )
        
        # Compressed model
        compressed_label = Text("Compressed (70%)", font_size=28, weight=BOLD)
        compressed_label.move_to(RIGHT * 3 + UP * 1)
        
        compressed_box = Rectangle(
            width=1.2,
            height=4 * 0.3,
            fill_color=GREEN,
            fill_opacity=0.7,
            stroke_color=GREEN,
            stroke_width=2
        )
        compressed_box.next_to(compressed_label, DOWN, buff=0.3)
        
        compressed_size = Text("37M params\n~150MB", font_size=18, color=WHITE)
        compressed_size.move_to(compressed_box.get_center())
        
        self.play(
            Write(compressed_label),
            FadeIn(compressed_box),
            Write(compressed_size),
            run_time=1
        )
        
        self.wait(0.5)
        
        # Savings
        savings = Text(
            "70% Parameter Reduction\nMaintains 95%+ Performance",
            font_size=26,
            weight=BOLD,
            color=YELLOW
        )
        savings.to_edge(DOWN, buff=1)
        
        self.play(Write(savings), run_time=1)
        
        self.wait(2)


# ============================================================================
# SCENE 10: Mathematical Deep Dive
# ============================================================================

class MathematicalDeepDive(Scene):
    """
    Detailed mathematical explanation with animations
    """
    
    def construct(self):
        # Title
        title = Text("Mathematical Foundation", font_size=50, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # Problem statement
        problem = Text("Problem: Decompose weight matrix W as A ⊗ B", font_size=24, color=BLUE)
        problem.next_to(title, DOWN, buff=0.5)
        self.play(Write(problem), run_time=1)
        
        self.wait(1)
        
        # Step 1: Van Loan Rearrangement
        step1 = MathTex(
            r"\mathcal{R}(W) = \text{blockwise rearrangement}",
            font_size=32,
            color=YELLOW
        )
        step1.move_to(UP * 1.5)
        self.play(Write(step1), run_time=1)
        
        # Step 2: SVD
        step2 = MathTex(
            r"\mathcal{R}(W) \approx \sigma_1 u_1 v_1^T",
            font_size=32,
            color=BLUE
        )
        step2.move_to(UP * 0.5)
        self.play(Write(step2), run_time=1)
        
        # Step 3: Extract factors
        step3 = MathTex(
            r"A = \sqrt{\sigma_1} \text{reshape}(u_1)",
            r"\quad B = \sqrt{\sigma_1} \text{reshape}(v_1)",
            font_size=32,
            color=GREEN
        )
        step3.move_to(DOWN * 0.5)
        self.play(Write(step3), run_time=1)
        
        # Step 4: Apply scaling
        step4 = MathTex(
            r"\hat{W} = \alpha (A \otimes B)",
            font_size=32,
            color=PURPLE
        )
        step4.move_to(DOWN * 1.5)
        self.play(Write(step4), run_time=1)
        
        # Step 5: Sparse correction
        step5 = MathTex(
            r"W_{final} = \hat{W} + S",
            font_size=32,
            color=RED
        )
        step5.move_to(DOWN * 2.5)
        self.play(Write(step5), run_time=1)
        
        self.wait(2)


# ============================================================================
# SCENE 11: Application to GPT-2
# ============================================================================

class ApplicationToGPT2(Scene):
    """
    Show how compression applies to GPT-2 architecture
    """
    
    def construct(self):
        # Title
        title = Text("Applying Kronecker Compression to GPT-2", font_size=48, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # GPT-2 architecture summary
        architecture_info = [
            "Embedding Layer → 768 dimensions",
            "12 Transformer Blocks",
            "  • Multi-head Attention (12 heads)",
            "  • Feed-Forward Networks",
            "  • Layer Normalization",
            "Output Layer",
        ]
        
        y_start = 2
        for i, layer_info in enumerate(architecture_info):
            indent = 0.5 if layer_info.startswith("  ") else 0
            
            layer_text = Text(
                layer_info.strip(),
                font_size=22,
                weight=BOLD if not layer_info.startswith("  ") else NORMAL
            )
            layer_text.move_to(LEFT * 3 + UP * (y_start - i * 0.6))
            layer_text.shift(RIGHT * indent)
            
            self.play(Write(layer_text), run_time=0.3)
        
        self.wait(1)
        
        # Focus on compression targets
        target_label = Text("Compression Targets", font_size=28, weight=BOLD, color=GREEN)
        target_label.move_to(RIGHT * 3 + UP * 2)
        
        targets = [
            "Attention Weight Matrices",
            "Feed-Forward Layers",
            "Projection Matrices",
        ]
        
        for i, target in enumerate(targets):
            target_text = Text(target, font_size=20, color=GREEN)
            target_text.move_to(RIGHT * 3 + UP * (1.2 - i * 0.6))
            
            self.play(Write(target_text), run_time=0.5)
        
        self.wait(2)


# ============================================================================
# SCENE 12: Summary and Key Takeaways
# ============================================================================

class Summary(Scene):
    """
    Summary and key takeaways
    """
    
    def construct(self):
        # Title
        title = Text("Key Takeaways", font_size=60, weight=BOLD, color=GREEN)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        takeaways = [
            "✓ Kronecker Product: Elegant tensor factorization",
            "✓ Van Loan Method: SVD-based optimal decomposition",
            "✓ Sparse Residuals: Capture compression errors intelligently",
            "✓ Results: 70% compression with 95%+ performance retention",
            "✓ Application: Efficient LLM deployment on edge devices",
        ]
        
        y_start = 2
        for i, takeaway in enumerate(takeaways):
            color = [BLUE, YELLOW, GREEN, PURPLE, CYAN][i]
            
            takeaway_text = Text(takeaway, font_size=26, color=color, weight=BOLD)
            takeaway_text.move_to(UP * (y_start - i * 1.2))
            
            self.play(Write(takeaway_text), run_time=0.8)
            self.wait(0.3)
        
        self.wait(1)
        
        # Final message
        final_text = Text(
            "Breaking down complex ML into simple, visual concepts",
            font_size=24,
            color=BLUE,
            italic=True
        )
        final_text.to_edge(DOWN, buff=0.5)
        
        self.play(Write(final_text), run_time=1)
        
        self.wait(2)


# ============================================================================
# Bonus: Interactive Animation
# ============================================================================

class InteractiveKroneckerDemo(Scene):
    """
    Interactive-style demonstration of Kronecker product calculation
    """
    
    def construct(self):
        # Title
        title = Text("Live Kronecker Product Calculation", font_size=50, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # Small matrices
        A_label = Text("A =", font_size=28)
        A_label.move_to(LEFT * 5 + UP * 1)
        
        A_data = [[2, 1], [1, 2]]
        A_matrix = self._create_matrix_visual(A_data, color=YELLOW, scale=0.7)
        A_matrix.next_to(A_label, RIGHT, buff=0.3)
        
        self.play(Write(A_label), FadeIn(A_matrix), run_time=1)
        
        B_label = Text("B =", font_size=28)
        B_label.move_to(LEFT * 2 + UP * 1)
        
        B_data = [[3, 0], [0, 3]]
        B_matrix = self._create_matrix_visual(B_data, color=GREEN, scale=0.7)
        B_matrix.next_to(B_label, RIGHT, buff=0.3)
        
        self.play(Write(B_label), FadeIn(B_matrix), run_time=1)
        
        # Arrow
        arrow = MathTex(r"\otimes", font_size=60, color=RED)
        arrow.move_to(ORIGIN + UP)
        
        self.play(Write(arrow), run_time=0.5)
        
        # Result
        result_label = Text("A ⊗ B =", font_size=28)
        result_label.move_to(LEFT * 5 + DOWN * 1.5)
        
        result_data = [
            [6, 3, 0, 0],
            [3, 6, 0, 0],
            [0, 0, 6, 3],
            [0, 0, 3, 6]
        ]
        result_matrix = self._create_matrix_visual(result_data, color=PURPLE, scale=0.6)
        result_matrix.next_to(result_label, RIGHT, buff=0.3)
        
        self.play(
            Write(result_label),
            FadeIn(result_matrix),
            run_time=1.5
        )
        
        self.wait(2)
    
    def _create_matrix_visual(self, data, color=WHITE, scale=1.0):
        """Create matrix visualization"""
        matrix = Matrix(data, h_buff=1.2, v_buff=1.2)
        matrix.set_color(color)
        matrix.scale(scale)
        return matrix


if __name__ == "__main__":
    # To run individual scenes:
    # manim -pql kronecker_product_visualization.py SceneName
    # For example:
    # manim -pql kronecker_product_visualization.py KroneckerProductIntro
    pass
