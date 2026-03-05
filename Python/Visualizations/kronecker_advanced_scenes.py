"""
Advanced Manim Visualizations: Enhanced Animations
==================================================

This module contains more sophisticated and visually impressive
animations for the Kronecker Product visualization project.

These scenes showcase:
- Complex mathematical animations
- Data visualizations with smooth transitions
- 3D-like effects and transformations
- Custom mobjects and effects

Run with:
    manim -pqh kronecker_advanced_scenes.py SceneName
"""

from manim import *
import numpy as np
from enum import Enum


# ============================================================================
# SCENE 1: Animated Matrix Multiplication
# ============================================================================

class AnimatedMatrixMultiplication(Scene):
    """
    Animated visualization of matrix multiplication process
    Shows how A ⊗ B is computed element by element
    """
    
    def construct(self):
        title = Text("Matrix Multiplication: A ⊗ B", font_size=54, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)
        
        # Create two small matrices
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        
        # Display matrices
        A_matrix = self._create_colored_matrix(A, color=BLUE)
        A_label = Text("A =", font_size=28)
        A_group = VGroup(A_label, A_matrix)
        A_group.arrange(RIGHT, buff=0.5)
        A_group.move_to(LEFT * 4 + UP * 0.5)
        
        B_matrix = self._create_colored_matrix(B, color=GREEN)
        B_label = Text("B =", font_size=28)
        B_group = VGroup(B_label, B_matrix)
        B_group.arrange(RIGHT, buff=0.5)
        B_group.move_to(LEFT + UP * 0.5)
        
        self.play(FadeIn(A_group), FadeIn(B_group), run_time=1)
        
        # Show the operation
        operation = MathTex(r"A \otimes B", font_size=48, color=RED)
        operation.move_to(RIGHT * 2 + UP * 0.5)
        
        self.play(Write(operation), run_time=0.5)
        
        # Show result with animation
        self.wait(0.5)
        
        result_label = Text("Result (4×4):", font_size=28)
        result_label.to_edge(LEFT, buff=0.5)
        result_label.shift(DOWN * 2)
        
        self.play(Write(result_label), run_time=0.5)
        
        # Compute Kronecker product
        result_data = self._compute_kronecker(A, B)
        
        # Visualize computation step by step
        result_matrix_parts = []
        for i in range(2):
            for j in range(2):
                part = self._create_colored_matrix(
                    [[result_data[i*2+k][j*2+l] for l in range(2)] for k in range(2)],
                    color=YELLOW,
                    scale=0.5
                )
                part.move_to(
                    LEFT * 3.5 + DOWN * (2.5 + i * 1.2) + RIGHT * (j * 1.2)
                )
                result_matrix_parts.append(part)
                
                self.play(FadeIn(part), run_time=0.3)
        
        # Show complete matrix
        self.wait(1)
        
        # Add highlights showing the structure
        for i, part in enumerate(result_matrix_parts):
            self.play(ScaleInPlace(part, 1.2), run_time=0.2)
            self.play(ScaleInPlace(part, 1/1.2), run_time=0.2)
        
        self.wait(2)
    
    def _create_colored_matrix(self, data, color=WHITE, scale=1.0):
        """Create matrix with colored entries"""
        matrix = Matrix(data, h_buff=1.2, v_buff=1.2)
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
# SCENE 2: 3D Matrix Transformation
# ============================================================================

class Matrix3DTransformation(ThreeDScene):
    """
    3D visualization of matrix rearrangement
    Shows how a 2D matrix can be visualized as a 3D tensor
    """
    
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        
        title = Text("3D Matrix Transformation", font_size=48, weight=BOLD)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        
        # Create 3D representation of matrix
        matrix_data = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        # Create cubes for each element
        cubes = []
        for i in range(3):
            for j in range(3):
                height = matrix_data[i][j] / 9.0  # Normalize height
                cube = Cube(
                    side_length=0.8,
                    fill_color=interpolate_color(BLUE, RED, height),
                    fill_opacity=0.7,
                    stroke_width=2
                )
                cube.scale([1, 1, height])
                cube.move_to(np.array([i - 1, j - 1, height/2]))
                cubes.append(cube)
                
                self.play(FadeIn(cube), run_time=0.1)
        
        self.wait(1)
        
        # Rotate view
        self.move_camera(phi=45 * DEGREES, theta=180 * DEGREES, run_time=3)
        self.move_camera(phi=75 * DEGREES, theta=45 * DEGREES, run_time=3)
        
        # Highlight rearrangement
        explanation = Text(
            "Rearrangement groups elements for optimal factorization",
            font_size=20,
            color=YELLOW
        )
        explanation.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(explanation)
        
        self.wait(2)


# ============================================================================
# SCENE 3: SVD Spectrum Analysis
# ============================================================================

class SVDSpectrumAnalysis(Scene):
    """
    Visualize the singular values spectrum
    Shows importance of different singular values
    """
    
    def construct(self):
        title = Text("SVD Singular Value Spectrum", font_size=50, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)
        
        # Simulate singular values (decreasing exponentially)
        n_values = 10
        singular_values = np.exp(-np.arange(n_values) * 0.5)
        singular_values = singular_values / singular_values[0]  # Normalize
        
        # Create axes
        axes = Axes(
            x_range=[0, n_values, 1],
            y_range=[0, 1.2, 0.2],
            axis_config={"color": GREY_B},
            tips=False,
        )
        axes.move_to(ORIGIN)
        
        # Create bars
        bars = []
        colors = color_gradient([BLUE, GREEN, YELLOW, RED], n_values)
        
        for i, (sv, color) in enumerate(zip(singular_values, colors)):
            bar = Rectangle(
                width=0.6,
                height=sv,
                fill_color=color,
                fill_opacity=0.7,
                stroke_color=color,
                stroke_width=2
            )
            bar.move_to(axes.c2p(i + 0.5, sv/2))
            bars.append(bar)
        
        # Add value labels
        self.play(
            Write(axes),
            *[FadeIn(bar) for bar in bars],
            run_time=1.5
        )
        
        # Add labels
        x_label = Text("Singular Value Index", font_size=20)
        x_label.move_to(axes.c2p(n_values/2, -0.35))
        
        y_label = Text("Normalized Value", font_size=20)
        y_label.move_to(axes.c2p(-2, 0.6))
        
        self.play(Write(x_label), Write(y_label), run_time=1)
        
        # Highlight first singular value
        highlight = SurroundingRectangle(bars[0], color=GREEN, buff=0.1)
        highlight_label = Text(
            "σ₁: Captures most variance",
            font_size=22,
            color=GREEN
        )
        highlight_label.next_to(highlight, UP, buff=0.5)
        
        self.play(Create(highlight), Write(highlight_label), run_time=1)
        
        # Show decay
        decay_info = Text(
            "Higher indexed values capture decreasing details",
            font_size=20,
            color=BLUE
        )
        decay_info.to_edge(DOWN, buff=0.5)
        
        self.play(Write(decay_info), run_time=1)
        
        self.wait(2)


# ============================================================================
# SCENE 4: Compression Rate Animation
# ============================================================================

class CompressionRateAnimation(Scene):
    """
    Animated visualization of compression progress
    """
    
    def construct(self):
        title = Text("Compression Rate Progress", font_size=50, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)
        
        # Initial size
        initial_size = 500  # MB
        target_size = 150   # MB
        
        self.wait(0.5)
        
        # Progress bar with animation
        progress_box = Rectangle(
            width=6,
            height=0.8,
            stroke_color=BLUE,
            stroke_width=2,
            fill_opacity=0.1
        )
        progress_box.move_to(ORIGIN + UP)
        
        progress_fill = Rectangle(
            width=0,
            height=0.8,
            fill_color=GREEN,
            fill_opacity=0.7
        )
        progress_fill.align_to(progress_box, LEFT)
        
        # Size labels
        initial_label = Text(f"{initial_size}MB", font_size=24, color=RED)
        initial_label.next_to(progress_box, LEFT, buff=0.5)
        
        final_label = Text(f"{target_size}MB", font_size=24, color=GREEN)
        final_label.next_to(progress_box, RIGHT, buff=0.5)
        
        # Percentage indicator
        percentage_text = DecimalNumber(0, num_decimal_places=1, font_size=28, color=YELLOW)
        percentage_text.next_to(progress_box, DOWN, buff=0.3)
        percentage_symbol = Text("%", font_size=28, color=YELLOW)
        percentage_symbol.next_to(percentage_text, RIGHT, buff=0.1)
        
        # Display components
        self.play(
            FadeIn(progress_box),
            Write(initial_label),
            Write(final_label),
            run_time=1
        )
        
        self.add(percentage_text, percentage_symbol)
        
        # Animate compression
        compression_ratio = (initial_size - target_size) / initial_size
        final_width = 6 * (1 - compression_ratio)
        
        def update_percentage(m):
            current_width = progress_fill.get_width()
            current_percentage = (current_width / 6) * 100 * (1 - compression_ratio) + \
                               compression_ratio * 100
            percentage_text.set_value(current_percentage)
        
        self.play(
            FadeIn(progress_fill),
            UpdateFromAlphaFunc(
                progress_fill,
                lambda m, alpha: m.set_width(final_width * alpha),
                rate_func=smooth
            ),
            UpdateFromFunc(percentage_text, update_percentage),
            run_time=3
        )
        
        # Result summary
        self.wait(0.5)
        
        summary_lines = [
            f"Initial Size: {initial_size}MB",
            f"Compressed Size: {target_size}MB",
            f"Compression: {compression_ratio*100:.1f}%",
            f"Parameters Reduced: ~70%",
            f"Performance Retained: 95%+"
        ]
        
        summary_group = VGroup()
        for i, line in enumerate(summary_lines):
            line_text = Text(line, font_size=20)
            line_text.shift(DOWN * (1.5 + i * 0.6))
            summary_group.add(line_text)
        
        self.play(FadeIn(summary_group), run_time=1.5)
        
        self.wait(2)


# ============================================================================
# SCENE 5: Error Reduction Animation
# ============================================================================

class ErrorReductionVisualization(Scene):
    """
    Show error reduction from different methods
    """
    
    def construct(self):
        title = Text("Error Reduction Analysis", font_size=50, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)
        
        # Method comparison data
        methods = {
            "Pruning": {"error": 136.62, "color": RED},
            "Van Loan": {"error": 9.1492, "color": BLUE},
            "Ours": {"error": 4.8922, "color": GREEN}
        }
        
        # Create axes
        axes = Axes(
            x_range=[0, 150, 25],
            y_range=[0, 3, 1],
            axis_config={"color": GREY_B},
            tips=False,
            height=4,
            width=8
        )
        axes.move_to(ORIGIN + UP * 0.2)
        
        self.play(Write(axes), run_time=1)
        
        # Add bars with animation
        bars_group = VGroup()
        for i, (method, data) in enumerate(methods.items()):
            error = data["error"]
            x_pos = i + 0.5
            
            # Bar
            bar = BarChart(
                values=[error],
                bar_colors=[data["color"]],
                bar_width=0.6,
                y_max=150
            )
            
            # Simpler approach: create rectangle
            bar_rect = Rectangle(
                width=0.6,
                height=axes.y_axis.p2n(error),
                fill_color=data["color"],
                fill_opacity=0.7,
                stroke_color=data["color"],
                stroke_width=2
            )
            bar_rect.move_to(axes.c2p(x_pos, axes.y_axis.p2n(error)/2))
            
            # Label
            label = Text(method, font_size=20, weight=BOLD)
            label.next_to(bar_rect, DOWN, buff=0.3)
            
            # Error value
            value = DecimalNumber(0, num_decimal_places=2, font_size=18, color=data["color"])
            value.next_to(bar_rect, UP, buff=0.2)
            
            # Animate bar growth
            self.play(
                GrowFromPoint(bar_rect, bar_rect.get_bottom()),
                Write(label),
                run_time=0.8
            )
            
            # Animate value counting up
            self.play(
                ChangeDecimalToValue(value, error, run_time=0.8),
                run_time=0.8
            )
        
        # Show improvement
        improvement = Text(
            "47% Error Reduction (Ours vs Van Loan)",
            font_size=26,
            weight=BOLD,
            color=GREEN
        )
        improvement.to_edge(DOWN, buff=1)
        
        improvement_box = SurroundingRectangle(improvement, color=GREEN, buff=0.3)
        
        self.play(
            Create(improvement_box),
            Write(improvement),
            run_time=1
        )
        
        self.wait(2)


# ============================================================================
# SCENE 6: Layer-by-Layer Compression
# ============================================================================

class LayerByLayerCompression(Scene):
    """
    Visualize compressing each layer of a neural network
    """
    
    def construct(self):
        title = Text("Layer-by-Layer Compression Process", font_size=48, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)
        
        subtitle = Text("Applying Kronecker decomposition to each transformer block", font_size=20)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(subtitle), run_time=0.8)
        
        # Create layer blocks
        n_layers = 6  # Show 6 layers (representing 12 in GPT-2)
        layer_blocks = []
        
        y_start = 1.5
        for i in range(n_layers):
            y_pos = y_start - i * 0.7
            
            # Original layer
            orig_block = Rectangle(
                width=1.5,
                height=0.5,
                fill_color=RED,
                fill_opacity=0.3,
                stroke_color=RED,
                stroke_width=2
            )
            orig_block.move_to(LEFT * 3 + UP * y_pos)
            
            orig_label = Text(f"Layer {i+1}", font_size=14)
            orig_label.move_to(orig_block.get_center())
            
            # Arrow
            arrow = Arrow(
                orig_block.get_right() + RIGHT * 0.2,
                orig_block.get_right() + RIGHT * 1.2,
                color=YELLOW,
                stroke_width=2,
                tip_length=0.1
            )
            
            # Compressed layer
            comp_block = Rectangle(
                width=0.8,
                height=0.5,
                fill_color=GREEN,
                fill_opacity=0.5,
                stroke_color=GREEN,
                stroke_width=2
            )
            comp_block.move_to(RIGHT * 1 + UP * y_pos)
            
            comp_label = Text(f"A⊗B+S", font_size=12)
            comp_label.move_to(comp_block.get_center())
            
            # Compression ratio
            ratio_text = Text("70%", font_size=12, color=GREEN)
            ratio_text.next_to(comp_block, RIGHT, buff=0.2)
            
            # Animate
            self.play(
                FadeIn(orig_block),
                Write(orig_label),
                run_time=0.2
            )
            
            self.play(
                GrowArrow(arrow),
                run_time=0.1
            )
            
            self.play(
                FadeIn(comp_block),
                Write(comp_label),
                Write(ratio_text),
                run_time=0.2
            )
            
            layer_blocks.append((orig_block, comp_block))
        
        # Summary
        self.wait(0.5)
        
        summary = VGroup()
        summary_texts = [
            "Total Parameters: 124M → 37M (70% reduction)",
            "Computation: Reduced proportionally",
            "Performance: 95%+ maintained"
        ]
        
        for i, text in enumerate(summary_texts):
            summary_text = Text(text, font_size=18, color=BLUE)
            summary_text.shift(DOWN * (2.5 + i * 0.4))
            summary.add(summary_text)
        
        self.play(FadeIn(summary), run_time=1)
        
        self.wait(2)


# ============================================================================
# SCENE 7: Data Flow Diagram
# ============================================================================

class DataFlowDiagram(Scene):
    """
    Animated data flow through compression pipeline
    """
    
    def construct(self):
        title = Text("Compression Pipeline Data Flow", font_size=50, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)
        
        # Define pipeline stages
        stages = [
            ("Input\nWeight Matrix", BLUE, "W: m×n"),
            ("Van Loan\nRearrangement", YELLOW, "W̃"),
            ("SVD\nDecomposition", ORANGE, "σ, u, v"),
            ("Extract\nFactors", GREEN, "A, B"),
            ("Compute\nResidual", RED, "R=W-Ŵ"),
            ("Sparse\nSelection", PURPLE, "S (sparse)"),
            ("Combine\nFactors", CYAN, "A⊗B + S"),
            ("Output\nCompressed", GREEN, "W_final")
        ]
        
        boxes = []
        x_start = -7
        
        for i, (stage_name, color, info) in enumerate(stages):
            x_pos = x_start + i * 2
            
            # Stage box
            box = Rectangle(
                width=1.6,
                height=1.2,
                fill_color=color,
                fill_opacity=0.3,
                stroke_color=color,
                stroke_width=2
            )
            box.move_to(UP * 0.5 + RIGHT * x_pos)
            
            # Stage label
            stage_label = Text(stage_name, font_size=12, weight=BOLD)
            stage_label.move_to(box.get_center() + UP * 0.2)
            
            # Info label
            info_label = Text(info, font_size=10)
            info_label.move_to(box.get_center() + DOWN * 0.3)
            
            # Animate in
            self.play(
                FadeIn(box),
                Write(stage_label),
                Write(info_label),
                run_time=0.3
            )
            
            # Add arrow to next stage
            if i < len(stages) - 1:
                arrow = Arrow(
                    box.get_right() + RIGHT * 0.1,
                    box.get_right() + RIGHT * 0.8,
                    color=YELLOW,
                    stroke_width=2,
                    tip_length=0.08
                )
                self.play(GrowArrow(arrow), run_time=0.2)
            
            boxes.append((box, stage_label, info_label))
        
        self.wait(1)
        
        # Add timing information
        timing_info = Text(
            "Total time per model: ~5-10 minutes | Output saved as checkpoint",
            font_size=16,
            color=BLUE
        )
        timing_info.to_edge(DOWN, buff=0.5)
        
        self.play(Write(timing_info), run_time=1)
        
        self.wait(2)


if __name__ == "__main__":
    # Run with: manim -pqh kronecker_advanced_scenes.py AnimatedMatrixMultiplication
    pass
