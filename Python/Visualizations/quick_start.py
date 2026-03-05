#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK START GUIDE: Kronecker Product Visualization
===================================================

This script helps you get started with rendering visualizations quickly.

Usage:
    python quick_start.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def check_installation():
    """Check if all required tools are installed"""
    print("\n" + "="*70)
    print("Checking Installation...")
    print("="*70 + "\n")
    
    # Check Python version
    print("✓ Python version:", sys.version.split()[0])
    
    # Check Manim
    try:
        result = subprocess.run(
            ["manim", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print("✓ Manim installed:", result.stdout.strip())
    except Exception as e:
        print("✗ Manim NOT found!")
        print("  Install with: pip install manim")
        return False
    
    # Check FFmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print("✓ FFmpeg installed")
    except Exception as e:
        print("⚠ FFmpeg NOT found (needed for video compilation)")
        print("  Install from: https://ffmpeg.org/download.html")
    
    print("\n" + "="*70 + "\n")
    return True


def list_quick_renders():
    """Show quick render options"""
    print("="*70)
    print("Quick Render Options")
    print("="*70 + "\n")
    
    quick_options = [
        ("1", "KroneckerProductIntro", "Quick intro (~5 seconds low quality)"),
        ("2", "WhyCompression", "Motivation for compression"),
        ("3", "VanLoanRearrangement", "Van Loan method visualization"),
        ("4", "SparseResidualCorrection", "Our novel sparse residual method"),
        ("5", "ErrorComparison", "Error comparison with charts"),
        ("6", "Summary", "Key takeaways"),
        ("7", "All Scenes (High Quality)", "Render all 13 scenes (LONG - ~2 hours)"),
        ("8", "All Advanced Scenes", "Render 7 advanced scenes (MEDIUM - ~1 hour)"),
    ]
    
    for option_num, scene, description in quick_options:
        print(f"{option_num}. {scene:<30} - {description}")
    
    print("\n0. Exit")
    print("\n" + "="*70 + "\n")


def render_quick_demo():
    """Render a quick demo"""
    print("Rendering quick demo: KroneckerProductIntro (low quality)...\n")
    
    cmd = [
        "manim",
        "-ql",
        "kronecker_product_visualization.py",
        "KroneckerProductIntro"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Done! Check the 'videos' folder for output.")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Rendering failed. Check error messages above.")
        return False


def show_tips():
    """Show useful tips"""
    print("\n" + "="*70)
    print("Manim Tips & Tricks")
    print("="*70 + "\n")
    
    tips = [
        ("Preview After Rendering", "manim -pql script.py SceneName", "Opens video automatically"),
        ("Different Quality Levels", "-ql (low), -pqm (medium), -pqh (high)", "Higher quality = longer render"),
        ("List Scenes in File", "manim --list_scenes script.py", "See all scenes available"),
        ("Check Render Progress", "Watch the terminal output", "Shows percentage complete"),
        ("Speed Up Testing", "Use -ql flag for testing", "Low quality renders in seconds"),
        ("Get Help", "manim --help", "Full command reference"),
    ]
    
    for tip_title, command, description in tips:
        print(f"💡 {tip_title}")
        print(f"   Command: {command}")
        print(f"   {description}\n")
    
    print("="*70 + "\n")


def show_file_structure():
    """Show file structure"""
    print("\n" + "="*70)
    print("Project File Structure")
    print("="*70 + "\n")
    
    structure = """
├── kronecker_product_visualization.py
│   └── 13 main visualization scenes
│
├── kronecker_advanced_scenes.py
│   └── 7 advanced animation scenes
│
├── render_visualizations.py
│   └── Rendering manager (interactive + CLI)
│
├── manim_config.py
│   └── Color scheme, fonts, animation speeds
│
├── MANIM_GUIDE.md
│   └── Complete guide with examples
│
├── quick_start.py
│   └── This file
│
└── videos/
    └── Output directory for rendered videos
    """
    
    print(structure)
    print("="*70 + "\n")


def main():
    """Main function"""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("\n")
    print("+" + "="*68 + "+")
    print("|" + " "*68 + "|")
    print("|" + "Kronecker Product Visualization - Quick Start Guide".center(68) + "|")
    print("|" + " "*68 + "|")
    print("+" + "="*68 + "+")
    
    # Check installation
    if not check_installation():
        print("\n⚠ Some dependencies are missing. Please install them and try again.")
        input("Press Enter to exit...")
        return 1
    
    while True:
        list_quick_renders()
        choice = input("Select an option (0-8): ").strip()
        
        if choice == "0":
            print("Goodbye! 👋\n")
            break
        
        elif choice == "1":
            print("Rendering: KroneckerProductIntro (low quality, ~30 seconds)\n")
            render_scene("KroneckerProductIntro", quality="-ql")
        
        elif choice == "2":
            print("Rendering: WhyCompression (medium quality, ~2 minutes)\n")
            render_scene("WhyCompression", quality="-pqm")
        
        elif choice == "3":
            print("Rendering: VanLoanRearrangement (medium quality, ~3 minutes)\n")
            render_scene("VanLoanRearrangement", quality="-pqm")
        
        elif choice == "4":
            print("Rendering: SparseResidualCorrection (medium quality, ~3 minutes)\n")
            render_scene("SparseResidualCorrection", quality="-pqm")
        
        elif choice == "5":
            print("Rendering: ErrorComparison (medium quality, ~2 minutes)\n")
            render_scene("ErrorComparison", quality="-pqm")
        
        elif choice == "6":
            print("Rendering: Summary (medium quality, ~2 minutes)\n")
            render_scene("Summary", quality="-pqm")
        
        elif choice == "7":
            print("\n" + "="*70)
            print("RENDERING ALL 13 SCENES IN HIGH QUALITY")
            print("="*70)
            print("\nEstimated time: 1.5 - 2 hours")
            print("Your computer may be slow during rendering")
            print("You can stop anytime with Ctrl+C\n")
            
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                render_all_scenes(quality="-pqh")
        
        elif choice == "8":
            print("\n" + "="*70)
            print("RENDERING ALL 7 ADVANCED SCENES IN HIGH QUALITY")
            print("="*70)
            print("\nEstimated time: 45 minutes - 1.5 hours\n")
            
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                render_advanced_scenes(quality="-pqh")
        
        else:
            print("Invalid option. Please try again.\n")
            continue
        
        # Show menu again
        print("\n" + "="*70)
        again = input("Render another scene? (y/n): ").strip().lower()
        if again != 'y':
            print("\nFor more advanced options, use: python render_visualizations.py")
            show_tips()
            show_file_structure()
            break


def render_scene(scene_name, quality="-pqh"):
    """Render a single scene"""
    cmd = [
        "manim",
        quality,
        "kronecker_product_visualization.py",
        scene_name
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Rendering complete!")
        print(f"✓ Video saved in videos/ folder")
    except KeyboardInterrupt:
        print("\n\nRendering cancelled by user.")
    except subprocess.CalledProcessError:
        print("\n✗ Rendering failed.")


def render_all_scenes(quality="-pqh"):
    """Render all scenes"""
    scenes = [
        "KroneckerProductIntro",
        "WhyCompression",
        "VanLoanRearrangement",
        "SVDDecomposition",
        "KroneckerFactorsExtraction",
        "SparseResidualCorrection",
        "ErrorComparison",
        "CompressionPipeline",
        "CompressionRatio",
        "MathematicalDeepDive",
        "ApplicationToGPT2",
        "Summary",
        "InteractiveKroneckerDemo",
    ]
    
    completed = 0
    failed = 0
    
    for i, scene in enumerate(scenes, 1):
        print(f"\n{'='*70}")
        print(f"Rendering scene {i}/{len(scenes)}: {scene}")
        print(f"{'='*70}\n")
        
        cmd = [
            "manim",
            quality,
            "kronecker_product_visualization.py",
            scene
        ]
        
        try:
            subprocess.run(cmd, check=True)
            completed += 1
            print(f"✓ {scene} complete")
        except KeyboardInterrupt:
            print(f"\n\nRendering stopped by user at scene {i}/{len(scenes)}")
            break
        except subprocess.CalledProcessError:
            failed += 1
            print(f"✗ {scene} failed")
    
    print(f"\n{'='*70}")
    print(f"Rendering Summary: {completed} successful, {failed} failed")
    print(f"{'='*70}\n")


def render_advanced_scenes(quality="-pqh"):
    """Render advanced scenes"""
    scenes = [
        "AnimatedMatrixMultiplication",
        "Matrix3DTransformation",
        "SVDSpectrumAnalysis",
        "CompressionRateAnimation",
        "ErrorReductionVisualization",
        "LayerByLayerCompression",
        "DataFlowDiagram",
    ]
    
    completed = 0
    failed = 0
    
    for i, scene in enumerate(scenes, 1):
        print(f"\n{'='*70}")
        print(f"Rendering advanced scene {i}/{len(scenes)}: {scene}")
        print(f"{'='*70}\n")
        
        cmd = [
            "manim",
            quality,
            "kronecker_advanced_scenes.py",
            scene
        ]
        
        try:
            subprocess.run(cmd, check=True)
            completed += 1
            print(f"✓ {scene} complete")
        except KeyboardInterrupt:
            print(f"\n\nRendering stopped by user at scene {i}/{len(scenes)}")
            break
        except subprocess.CalledProcessError:
            failed += 1
            print(f"✗ {scene} failed")
    
    print(f"\n{'='*70}")
    print(f"Advanced Rendering Summary: {completed} successful, {failed} failed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nExiting... Goodbye! 👋\n")
        sys.exit(0)
