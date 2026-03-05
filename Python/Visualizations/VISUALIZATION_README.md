# 🎬 Kronecker Product Visualization: Complete Package

> **Beautiful Manim animations explaining LLM compression through Kronecker Product decomposition, inspired by 3Blue1Brown**

## 📦 What You Get

This package contains **20+ professional animations** showcasing:

✅ **13 Main Visualization Scenes** (`kronecker_product_visualization.py`)
- Kronecker product basics
- Mathematical decomposition process
- SVD analysis
- Sparse residual correction (our innovation)
- Compression results and comparisons
- Application to GPT-2
- Key takeaways

✅ **7 Advanced Animation Scenes** (`kronecker_advanced_scenes.py`)
- Animated matrix multiplication
- 3D transformations
- Spectrum analysis
- Real-time compression rates
- Error reduction visualizations
- Layer-by-layer compression
- Data flow diagrams

✅ **Rendering Tools**
- Interactive rendering manager
- Command-line interface
- Quick-start script for beginners
- Video compilation support

✅ **Complete Documentation**
- Detailed Manim guide
- Installation instructions
- Usage examples
- Troubleshooting help
- Mathematical explanations

---

## 🚀 Quick Start (2 minutes)

### 1. **Install Requirements**
```bash
# Install Manim
pip install manim

# For Windows users (install these first):
# - MiKTeX: https://miktex.org/download
# - FFmpeg: https://ffmpeg.org/download.html
```

### 2. **Run Quick-Start Menu**
```bash
python quick_start.py
```

### 3. **Select a Scene to Render**
```
Choose from 13 main scenes or 7 advanced scenes
Low quality renders in ~30 seconds
```

That's it! Your first Manim video will be rendered! 🎉

---

## 📊 Scene Overview

### **Main Visualizations** (13 scenes, ~4 minutes total)

| # | Scene | Duration | Topic |
|---|-------|----------|-------|
| 1 | KroneckerProductIntro | 20s | Tensor product basics |
| 2 | WhyCompression | 18s | LLM compression motivation |
| 3 | VanLoanRearrangement | 22s | Matrix rearrangement |
| 4 | SVDDecomposition | 18s | Singular value decomposition |
| 5 | KroneckerFactorsExtraction | 20s | Factor extraction |
| 6 | SparseResidualCorrection | 24s | **Our novel method** |
| 7 | ErrorComparison | 20s | Results visualization |
| 8 | CompressionPipeline | 22s | Complete workflow |
| 9 | CompressionRatio | 18s | Compression metrics |
| 10 | MathematicalDeepDive | 25s | Mathematical details |
| 11 | ApplicationToGPT2 | 20s | GPT-2 application |
| 12 | Summary | 18s | Key takeaways |
| 13 | InteractiveDemo | 16s | Live calculation |

### **Advanced Visualizations** (7 scenes, ~8 minutes total)

| # | Scene | Duration | Topic |
|---|-------|----------|-------|
| 1 | AnimatedMatrixMultiplication | 30s | Matrix ⊗ animation |
| 2 | Matrix3DTransformation | 45s | 3D visualization |
| 3 | SVDSpectrumAnalysis | 40s | Singular values |
| 4 | CompressionRateAnimation | 50s | Progress animation |
| 5 | ErrorReductionVisualization | 50s | Error metrics |
| 6 | LayerByLayerCompression | 45s | Network compression |
| 7 | DataFlowDiagram | 60s | Pipeline flow |

---

## 📁 File Structure

```
project/
├── 📄 kronecker_product_visualization.py     (13 scenes, 850+ lines)
├── 📄 kronecker_advanced_scenes.py           (7 scenes, 700+ lines)
├── 🔧 render_visualizations.py               (Rendering manager)
├── 🎨 manim_config.py                        (Configuration & styling)
├── 🎯 quick_start.py                         (Easy start script)
├── 📖 MANIM_GUIDE.md                         (Complete guide)
├── 📋 README.md                              (This file)
└── 📹 videos/                                (Output directory)
```

---

## 💻 Usage Examples

### **Example 1: Interactive Menu (Easiest)**
```bash
python quick_start.py
```
Menu-driven interface with all options

### **Example 2: Render Specific Scene**
```bash
# Low quality (30 seconds)
manim -ql kronecker_product_visualization.py KroneckerProductIntro

# High quality (5 minutes)
manim -pqh kronecker_product_visualization.py KroneckerProductIntro

# Preview immediately after rendering
manim -pql kronecker_product_visualization.py KroneckerProductIntro
```

### **Example 3: Render All Scenes**
```bash
# Using manager script
python render_visualizations.py -a -q high

# Or interactive mode
python render_visualizations.py

# Then select "4. Render all scenes"
```

### **Example 4: Render Advanced Scenes**
```bash
# Single advanced scene
manim -pqh kronecker_advanced_scenes.py AnimatedMatrixMultiplication

# All advanced scenes via manager
python render_visualizations.py -a -q high  # (renders both main and advanced)
```

### **Example 5: Compile Videos**
```bash
python render_visualizations.py -c -o my_video.mp4
```

---

## 🎯 Quality Options

| Quality | Flag | Resolution | FPS | Render Time | Best For |
|---------|------|-----------|-----|-------------|----------|
| **Low** | `-ql` | 480p | 15 | ~30s/scene | Testing |
| **Medium** | `-pqm` | 720p | 30 | ~5m/scene | Preview |
| **High** | `-pqh` | 1080p | 60 | ~15m/scene | YouTube |
| **Ultra** | `-pqk` | 4K | 60 | ~60m/scene | 4K Export |

### **Choose Based On:**
- 🧪 **Testing**: Use `-ql` (renders in 30 seconds)
- 📺 **YouTube**: Use `-pqh` (crisp 1080p quality)
- 🎬 **Premium**: Use `-pqk` (4K cinematic)

---

## 🎨 Customization

### **Change Colors**
Edit `manim_config.py`:
```python
KRONECKER_BLUE = "#2D5C88"      # Change these hex colors
KRONECKER_GREEN = "#31A854"
KRONECKER_RED = "#E74C3C"
```

### **Change Animation Speed**
Edit `manim_config.py`:
```python
FADE_IN_TIME = 0.5      # Adjust these values
WRITE_TIME = 0.5
MOVE_TIME = 0.7
```

### **Add Custom Scene**
Edit `kronecker_product_visualization.py`:
```python
class MyCustomScene(Scene):
    def construct(self):
        title = Text("My Custom Scene", font_size=54)
        self.play(Write(title))
        self.wait(2)
```

Then render:
```bash
manim -pqh kronecker_product_visualization.py MyCustomScene
```

---

## 🔧 Installation (Detailed)

### **Windows Users**

1. **Install Python** (3.9+)
   - Download from [python.org](https://www.python.org)
   - Enable "Add Python to PATH"

2. **Install MiKTeX** (for LaTeX rendering)
   - Download from [miktex.org](https://miktex.org/download)
   - Run installer
   - Install missing packages on demand when Manim first runs

3. **Install FFmpeg** (for video compilation)
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Extract to a folder (e.g., `C:\ffmpeg`)
   - Add to Windows PATH (Search: Edit environment variables)

4. **Install Manim and Dependencies**
   ```bash
   pip install manim
   pip install numpy scipy matplotlib
   ```

5. **Verify Installation**
   ```bash
   manim --version
   ```

### **macOS Users**

```bash
# Using Homebrew
brew install manim
brew install miktex
brew install ffmpeg

# Or using pip
pip install manim
```

### **Linux Users**

```bash
# Ubuntu/Debian
sudo apt-get install texlive texlive-fonts-recommended texlive-latex-extra ffmpeg
pip install manim

# Fedora
sudo dnf install texlive texlive-latex-extra ffmpeg
pip install manim
```

---

## 📖 Mathematical Background

### **Kronecker Product**
$$A \otimes B = \begin{bmatrix} a_{11}B & a_{12}B \\ a_{21}B & a_{22}B \end{bmatrix}$$

### **Van Loan Decomposition**
$$W \approx \alpha (A \otimes B)$$

Where:
- $A = \sqrt{\sigma_1} \cdot \text{reshape}(u_1)$
- $B = \sqrt{\sigma_1} \cdot \text{reshape}(v_1)$
- $\alpha$ is an optimized scaling factor

### **Our Innovation: Sparse Residual**
$$W_{\text{final}} = \alpha (A \otimes B) + S$$

Where $S$ is a sparse matrix capturing high-frequency errors

---

## 📊 Results Summary

| Method | Error | Parameters | Performance |
|--------|-------|-----------|-------------|
| Original | 0% | 100% | 100% |
| Naive Pruning | 136.62 | 30% | 60% |
| Van Loan | 9.1492 | 30% | 95%+ |
| **Our Method** | **4.8922** | **30%** | **95%+** |

**47% error reduction vs. the paper method!** 🎯

---

## 🎬 Video Compilation

Combine all rendered scenes into one video:

```bash
python render_visualizations.py -c -o kronecker_complete.mp4
```

This creates a single MP4 file with all animations in sequence.

---

## 🐛 Troubleshooting

### **"manim: command not found"**
```bash
pip install manim
```

### **"FFmpeg not found"**
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Mac: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

### **"pdflatex not found"**
- Windows: Install MiKTeX from [miktex.org](https://miktex.org)
- Mac: `brew install miktex`
- Linux: `sudo apt-get install texlive-latex-base`

### **Slow rendering?**
Use low quality for testing:
```bash
manim -ql kronecker_product_visualization.py SceneName
```

### **Out of memory?**
- Close other applications
- Render one scene at a time
- Use lower quality (`-ql` instead of `-pqk`)

### **"MikTeX not properly configured"**
Run MiKTeX console and update all packages:
1. Open MiKTeX Console
2. Select "Updates" tab
3. Click "Update now"

---

## 📚 Learning Resources

- 📖 [Manim Documentation](https://docs.manim.community/)
- 🎥 [3Blue1Brown Manim Tutorials](https://www.youtube.com/playlist?list=PLjq2Fcg4_cKPuqxu1iRUQ-GS74KFOmL5)
- 🔢 [Kronecker Product (Wikipedia)](https://en.wikipedia.org/wiki/Kronecker_product)
- 📄 [Your Paper](https://arxiv.org/abs/2412.12351)

---

## 💡 Tips & Tricks

### **Speed Up Testing**
Always use `-ql` for iteration:
```bash
manim -ql kronecker_product_visualization.py KroneckerProductIntro
```

### **Render Overnight**
For high-quality renders:
```bash
# On Windows, save as batch.bat:
@echo off
manim -pqk kronecker_product_visualization.py KroneckerProductIntro
```

### **Custom Resolution**
Edit `manim_config.py`:
```python
config.pixel_height = 2160  # 4K
config.pixel_width = 3840
```

### **Batch Rename Videos**
After rendering, videos are in `videos/` with timestamps.
Use file explorer or script to rename for easier identification.

---

## 🤝 Contributing

Feel free to:
- ✏️ Add new scenes
- 🎨 Improve existing animations
- 🐛 Report bugs
- 📚 Suggest improvements

Just edit the files and re-render!

---

## 📄 License

This visualization package is part of the LLM Compression using Kronecker Product Decomposition research project at your institution.

---

## 👥 Team

| Name | Role |
|------|------|
| Hemanth SN | Team Member |
| Mahakisore M | Team Member |
| Yashwanth B | Team Member |

**Project**: LLM Compression using Kronecker Product Decomposition & Sparse Residuals
**Based on**: [Paper Reference](https://arxiv.org/abs/2412.12351)

---

## 🎯 Quick Command Reference

```bash
# Interactive menu
python quick_start.py

# List all scenes
python render_visualizations.py -l

# Render one scene (low quality)
manim -ql kronecker_product_visualization.py KroneckerProductIntro

# Render one scene (high quality)
manim -pqh kronecker_product_visualization.py KroneckerProductIntro

# Render all scenes
python render_visualizations.py -a -q high

# Compile videos
python render_visualizations.py -c

# Get help
python render_visualizations.py -h

# Direct Manim help
manim --help
```

---

## 📞 Support

For issues:
1. Check MANIM_GUIDE.md
2. Review code comments
3. Check [Manim Docs](https://docs.manim.community/)
4. See troubleshooting section above

---

## ✨ Highlights

🌟 **20+ Professional Animations**
- Smooth transitions
- Beautiful colors
- Clear explanations

🌟 **Easy to Use**
- Interactive menu included
- One-command rendering
- Beginner-friendly

🌟 **Highly Customizable**
- Change colors easily
- Adjust animation speed
- Add custom scenes

🌟 **Production Quality**
- 1080p/4K output
- 60 FPS smooth animation
- YouTube-ready format

🌟 **Complete Documentation**
- Installation guide
- Usage examples
- Mathematical explanations

---

## 🚀 Get Started Now!

```bash
# 1. Install Manim (if not done)
pip install manim

# 2. Run quick start
python quick_start.py

# 3. Select a scene and watch it render!
```

**Your first Manim video will be ready in under a minute!** 🎬

---

**Happy Animating! 🎨✨**

For more information, see `MANIM_GUIDE.md` or visit [Manim Community](https://www.manim.community)
