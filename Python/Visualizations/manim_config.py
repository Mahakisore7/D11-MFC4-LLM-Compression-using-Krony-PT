"""
Advanced Manim Configuration for Kronecker Product Visualization
=================================================================

This file provides optimal settings for rendering high-quality videos
similar to 3Blue1Brown's style.

Usage:
    Place this file in the project root and name it 'manim_config.py'
    Or use: manim -c config.py scene_name.py SceneName
"""

from manim import *

# Configure the rendering quality and style
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 60

# Color theme inspired by 3Blue1Brown
# Primary colors
KRONECKER_BLUE = "#2D5C88"
KRONECKER_GREEN = "#31A854"
KRONECKER_RED = "#E74C3C"
KRONECKER_YELLOW = "#F39C12"
KRONECKER_PURPLE = "#9B59B6"

# Background color (dark for better contrast)
config.background_color = "#1a1a1a"

# Text rendering settings
config.tex_template.preamble += r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{xcolor}
"""

# Quality presets
# For development: manim -ql
# For preview: manim -pqm
# For high quality: manim -pqh
# For ultra quality: manim -pqk


class ManimConfig:
    """Extended configuration options"""
    
    # Animation speeds (in seconds)
    FADE_IN_TIME = 0.5
    WRITE_TIME = 0.5
    MOVE_TIME = 0.7
    ROTATE_TIME = 1.0
    
    # Color palette
    COLORS = {
        'primary': KRONECKER_BLUE,
        'success': KRONECKER_GREEN,
        'error': KRONECKER_RED,
        'warning': KRONECKER_YELLOW,
        'info': KRONECKER_PURPLE,
    }
    
    # Font sizes
    TITLE_SIZE = 54
    SUBTITLE_SIZE = 36
    SECTION_SIZE = 32
    NORMAL_SIZE = 24
    SMALL_SIZE = 18
    
    # Stroke widths
    THIN = 1
    NORMAL = 2
    THICK = 3
    VERY_THICK = 4
    
    @staticmethod
    def get_color_gradient(steps=5):
        """Generate color gradients for visualizations"""
        colors = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
        ]
        return colors[:min(steps, len(colors))]


# Scene rendering template
class HighQualityScene(Scene):
    """
    Base scene class with optimal settings for 3Blue1Brown style videos
    """
    
    CONFIG = {
        "camera_config": {
            "background_color": config.background_color,
        }
    }
    
    def setup(self):
        """Initialize scene with custom settings"""
        super().setup()
        # Adjust lighting for better text visibility
        self.camera.background_color = color_to_rgb("#1a1a1a")
    
    def create_title(self, text, size=54):
        """Create a title with consistent styling"""
        return Text(
            text,
            font_size=size,
            weight=BOLD,
            color=WHITE
        )
    
    def create_subtitle(self, text, color=BLUE, size=32):
        """Create a subtitle with consistent styling"""
        return Text(
            text,
            font_size=size,
            color=color,
            weight=NORMAL
        )
    
    def create_equation(self, tex, color=YELLOW, size=44):
        """Create an equation with consistent styling"""
        return MathTex(tex, font_size=size, color=color)
    
    def create_labeled_box(self, label, width=3, height=2, color=BLUE):
        """Create a labeled box visualization"""
        box = Rectangle(
            width=width,
            height=height,
            fill_color=color,
            fill_opacity=0.2,
            stroke_color=color,
            stroke_width=2
        )
        
        label_text = Text(label, font_size=20, color=color)
        label_text.move_to(box.get_center())
        
        return VGroup(box, label_text)
    
    def animate_number_change(
        self,
        initial_value,
        final_value,
        label="",
        color=YELLOW,
        run_time=2
    ):
        """Animate a number changing from initial to final value"""
        number = DecimalNumber(
            initial_value,
            color=color,
            font_size=40
        )
        
        if label:
            label_text = Text(label, font_size=24, color=color)
            label_text.next_to(number, UP)
            group = VGroup(label_text, number)
        else:
            group = number
        
        number.add_updater(
            lambda m: m.set_value(
                initial_value + (final_value - initial_value) * (self.time - 0) / run_time
            )
        )
        
        self.add(group)
        self.wait(run_time)
        self.remove(group)
        
        return group


# Gradient and effect utilities
def create_gradient_box(
    width=3,
    height=2,
    colors=None,
    stroke_width=2
):
    """
    Create a box with gradient coloring
    """
    if colors is None:
        colors = ["#2D5C88", "#31A854"]
    
    box = Rectangle(
        width=width,
        height=height,
        fill_color=colors[0],
        fill_opacity=0.6,
        stroke_color=colors[-1],
        stroke_width=stroke_width
    )
    return box


def create_comparison_bars(
    data_dict,
    title="Comparison",
    colors=None,
    scale=0.02
):
    """
    Create comparison bars for data visualization
    
    Args:
        data_dict: {"label": value, ...}
        title: Title of comparison
        colors: List of colors for bars
        scale: Scaling factor for bar width
    """
    if colors is None:
        colors = ManimConfig.get_color_gradient(len(data_dict))
    
    title_text = Text(title, font_size=32, weight=BOLD)
    
    bars_group = VGroup()
    max_value = max(data_dict.values())
    
    for i, (label, value) in enumerate(data_dict.items()):
        y_pos = -i * 1.2
        
        # Label
        label_text = Text(label, font_size=20)
        label_text.to_edge(LEFT, buff=0.5)
        label_text.shift(UP * y_pos)
        
        # Bar
        bar = Rectangle(
            width=value * scale,
            height=0.4,
            fill_color=colors[i % len(colors)],
            fill_opacity=0.7,
            stroke_width=2
        )
        bar.next_to(label_text, RIGHT, buff=0.5)
        
        # Value
        value_text = Text(f"{value:.2f}", font_size=18)
        value_text.next_to(bar, RIGHT, buff=0.3)
        
        bars_group.add(label_text, bar, value_text)
    
    return VGroup(title_text, bars_group)


# Smooth transitions and effects
class SmoothTransition(Animation):
    """
    Smooth transition between two objects
    """
    
    CONFIG = {
        "rate_func": smooth,
    }
    
    def __init__(self, mobject, target_mobject, **kwargs):
        self.target_mobject = target_mobject
        super().__init__(mobject, **kwargs)
    
    def interpolate_mobject(self, alpha):
        """Interpolate between mobject and target"""
        self.mobject.become(
            interpolate(self.mobject, self.target_mobject, alpha)
        )


if __name__ == "__main__":
    print("Manim Configuration loaded successfully")
    print(f"Resolution: {config.pixel_width}x{config.pixel_height}")
    print(f"Frame rate: {config.frame_rate} fps")
    print(f"Background color: {config.background_color}")
