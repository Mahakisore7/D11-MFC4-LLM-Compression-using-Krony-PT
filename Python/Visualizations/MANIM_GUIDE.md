# Manim Visualization Guide: Kronecker Product & LLM Compression

## Overview

This project contains a comprehensive set of Manim animations that explain the Kronecker Product decomposition method for LLM compression, inspired by the beautiful visualizations of 3Blue1Brown.

**Contents:**
- 13 detailed animated scenes
- High-quality 1080p/4K rendering support
- Interactive rendering manager
- Educational explanations with smooth animations
- Mathematical deep-dives with clear visuals

---

## Installation

### Prerequisites

You need to have the following installed:

1. **Python 3.9+**
   ```bash
   python --version
   ```

2. **FFmpeg** (for video compilation)
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Or use: `choco install ffmpeg` (if using Chocolatey)

3. **LaTeX** (for mathematical rendering)
   - Windows: Install MiKTeX from [miktex.org](https://miktex.org)
   - Or use: `choco install miktex` (if using Chocolatey)

### Step 1: Install Manim

```bash
pip install manim
```

For better rendering with better quality:
```bash
pip install manim[latex]
```

### Step 2: Verify Installation

```bash
manim --version
```

Should show something like: `Manim Community v0.18.x`

### Step 3: Install Additional Dependencies

```bash
pip install numpy scipy matplotlib
```

---

## Usage

### Method 1: Interactive Menu (Recommended for beginners)

```bash
python render_visualizations.py
```

This opens an interactive menu where you can:
- List all available scenes
- Select quality presets
- Render individual scenes
- Render all scenes
- Compile videos

### Method 2: Command Line Arguments

#### List all scenes:
```bash
python render_visualizations.py -l
```

#### Render specific scene:
```bash
python render_visualizations.py -s KroneckerProductIntro -q high
```

#### Render all scenes:
```bash
python render_visualizations.py -a -q high
```

#### Compile videos:
```bash
python render_visualizations.py -c -o my_video.mp4
```

### Method 3: Direct Manim Command

```bash
# Low quality (fast, for testing)
manim -ql kronecker_product_visualization.py KroneckerProductIntro

# Preview quality
manim -pqm kronecker_product_visualization.py KroneckerProductIntro

# High quality
manim -pqh kronecker_product_visualization.py KroneckerProductIntro

# Ultra quality (4K)
manim -pqk kronecker_product_visualization.py KroneckerProductIntro

# With automatic preview after rendering
manim -pql kronecker_product_visualization.py KroneckerProductIntro
```

---

## Scene List

### 1. **KroneckerProductIntro** (20 seconds)
Introduction to the Kronecker Product
- Shows animated matrices A and B
- Visualizes the tensor product operation
- Explains the mathematical notation

### 2. **WhyCompression** (18 seconds)
Motivation for LLM compression
- Storage requirements of large models
- Inference latency concerns
- Hardware cost considerations
- Solution overview

### 3. **VanLoanRearrangement** (22 seconds)
The Van Loan rearrangement process
- Original weight matrix visualization
- Rearrangement transformation
- Block structure explanation

### 4. **SVDDecomposition** (18 seconds)
Singular Value Decomposition
- SVD formula breakdown
- Components explanation
- Importance of singular values

### 5. **KroneckerFactorsExtraction** (20 seconds)
Extracting Kronecker factors A and B
- Reshaping operations
- Size calculations
- Factor reconstruction

### 6. **SparseResidualCorrection** (24 seconds)
Novel sparse residual correction method
- Step 1: Approximate reconstruction
- Step 2: Calculate residuals
- Step 3: Sparse selection
- Final compressed model formula

### 7. **ErrorComparison** (20 seconds)
Comparing error metrics
- Naive pruning baseline
- Van Loan method results
- Our improved method
- 47% error reduction visualization

### 8. **CompressionPipeline** (22 seconds)
Complete compression workflow
- Load pre-trained model
- Apply Van Loan rearrangement
- SVD decomposition
- Factor extraction
- Residual calculation
- Sparse thresholding
- Save compressed model

### 9. **CompressionRatio** (18 seconds)
Compression ratio achievement
- Original model size
- Compressed model size
- Parameter reduction percentage
- Performance retention metrics

### 10. **MathematicalDeepDive** (25 seconds)
Detailed mathematical explanation
- Problem formulation
- Van Loan rearrangement equation
- SVD decomposition
- Factor extraction formulas
- Scaling and correction

### 11. **ApplicationToGPT2** (20 seconds)
Applying compression to GPT-2
- GPT-2 architecture overview
- 12 transformer blocks
- Compression targets
- Layer-by-layer application

### 12. **Summary** (18 seconds)
Key takeaways and conclusions
- Main concepts recap
- Innovation highlights
- Practical applications
- Future directions

### 13. **InteractiveKroneckerDemo** (16 seconds)
Live Kronecker product calculation
- Small 2×2 matrices
- Real-time calculation
- Result visualization

---

## Quality Presets

| Preset | Flag | Resolution | FPS | Use Case |
|--------|------|-----------|-----|----------|
| **Development** | `-ql` | 480p | 15 | Testing, development |
| **Preview** | `-pqm` | 720p | 30 | Quick review |
| **High** | `-pqh` | 1080p | 60 | Production quality |
| **Ultra** | `-pqk` | 4K | 60 | Premium quality |

### Rendering Time Estimates

- **Development**: 1-2 seconds per scene
- **Preview**: 5-10 seconds per scene
- **High**: 15-30 seconds per scene
- **Ultra**: 60-120 seconds per scene

---

## File Structure

```
project/
├── kronecker_product_visualization.py    # Main visualization file (13 scenes)
├── manim_config.py                       # Configuration and styling
├── render_visualizations.py              # Rendering manager script
├── videos/                               # Output directory
│   ├── 480p_15fps/                      # Development quality
│   ├── 720p_30fps/                      # Preview quality
│   ├── 1080p_60fps/                     # High quality (default)
│   └── 2160p_60fps/                     # Ultra quality
└── README.md                             # This file
```

---

## Customization

### Changing Colors

Edit `manim_config.py`:

```python
KRONECKER_BLUE = "#2D5C88"      # Primary blue
KRONECKER_GREEN = "#31A854"     # Success green
KRONECKER_RED = "#E74C3C"       # Error red
KRONECKER_YELLOW = "#F39C12"    # Warning yellow
KRONECKER_PURPLE = "#9B59B6"    # Info purple
```

### Changing Animation Speeds

In `manim_config.py`:

```python
FADE_IN_TIME = 0.5      # Fade in duration
WRITE_TIME = 0.5        # Text write duration
MOVE_TIME = 0.7         # Movement duration
```

### Changing Font Sizes

```python
TITLE_SIZE = 54
SUBTITLE_SIZE = 36
SECTION_SIZE = 32
NORMAL_SIZE = 24
SMALL_SIZE = 18
```

### Adding Custom Scenes

Create a new scene in `kronecker_product_visualization.py`:

```python
class MyCustomScene(Scene):
    def construct(self):
        # Your animation code here
        title = Text("My Scene", font_size=54)
        self.play(Write(title))
        self.wait(2)
```

Then render it:
```bash
manim -pqh kronecker_product_visualization.py MyCustomScene
```

---

## Troubleshooting

### Issue: "manim: command not found"
**Solution:** Make sure Manim is installed:
```bash
pip install manim
```

### Issue: "FFmpeg not found"
**Solution:** Install FFmpeg:
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Or: `choco install ffmpeg` (with Chocolatey)

### Issue: "pdflatex not found"
**Solution:** Install LaTeX:
- Windows: Install MiKTeX from [miktex.org](https://miktex.org)
- Or: `choco install miktex` (with Chocolatey)

### Issue: Rendering is slow
**Solution:** Use lower quality presets for testing:
```bash
manim -ql kronecker_product_visualization.py SceneName
```

### Issue: Out of memory
**Solution:** Reduce quality or render scenes individually:
```bash
manim -ql kronecker_product_visualization.py KroneckerProductIntro
```

### Issue: Animations look blocky
**Solution:** Use higher quality presets:
```bash
manim -pqh kronecker_product_visualization.py SceneName
```

---

## Tips for Best Results

1. **Close other applications** before rendering high-quality videos to free up system resources

2. **Use development quality** (`-ql`) for testing and iteration

3. **Use high quality** (`-pqh`) for final exports to YouTube/presentations

4. **Render overnight** for ultra quality if your system is slow

5. **Check the output** in `videos/` directory after rendering

6. **Edit animations** in the main file and re-render to preview changes

---

## Video Compilation

To create a single video from all scenes:

```bash
python render_visualizations.py -c -o final_video.mp4
```

This will:
1. Find all rendered MP4 files
2. Concatenate them in order
3. Create a final compiled video

---

## Learning Resources

### Understanding Manim
- [Manim Community Documentation](https://docs.manim.community/)
- [3Blue1Brown's Manim Tutorials](https://www.youtube.com/playlist?list=PLjq2Fcg4_cKPuqxu1iRUQ-GS74KFOmL5)

### Mathematical Background
- [Kronecker Product (Wikipedia)](https://en.wikipedia.org/wiki/Kronecker_product)
- [SVD Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- [Your Project's Paper](https://arxiv.org/abs/2412.12351)

---

## Performance Tips

### For Faster Rendering
- Use `-ql` for testing
- Reduce `frame_rate` in config
- Render scenes individually instead of all at once

### For Better Quality
- Use `-pqh` or `-pqk`
- Ensure LaTeX is properly installed
- Use high-quality output format

### System Requirements
- **Minimum**: 4GB RAM, Multi-core processor
- **Recommended**: 8GB RAM, 6+ cores, SSD
- **Optimal**: 16GB+ RAM, 8+ cores, SSD

---

## Advanced Usage

### Batch Rendering on Windows (PowerShell)

```powershell
# Render all scenes in high quality
foreach ($scene in 1..13) {
    python render_visualizations.py -a -q high
}
```

### Custom Resolution

Edit `manim_config.py`:

```python
config.pixel_height = 2160  # 4K height
config.pixel_width = 3840   # 4K width
```

### Add Watermark

Edit `kronecker_product_visualization.py`:

```python
class SceneWithWatermark(Scene):
    def construct(self):
        watermark = Text("© Your Name", font_size=20, opacity=0.3)
        watermark.to_corner(DOWN + RIGHT)
        self.add(watermark)
```

---

## Contributing

Feel free to:
- Add new scenes
- Improve existing animations
- Fix bugs
- Optimize rendering

Simply edit the files and re-render to see changes!

---

## License

This visualization project is part of the LLM Compression using Kronecker Product Decomposition research project.

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in the visualization files
3. Consult the [Manim Documentation](https://docs.manim.community/)
4. Check the [Project README](README.md)

---

## Scene Timing Reference

```
Total Duration: ~4 minutes (260 seconds)
- Introduction: 50 seconds
- Core Methods: 80 seconds  
- Results & Comparison: 38 seconds
- Application: 52 seconds
- Conclusion: 40 seconds
```

Estimated rendering time:
- Low quality: 5-10 minutes
- Preview quality: 20-40 minutes
- High quality: 60-120 minutes
- Ultra quality: 200+ minutes

---

**Happy Animating! 🎬**
