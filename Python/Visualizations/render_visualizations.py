#!/usr/bin/env python3
"""
Master Script: Kronecker Product Visualization Manager
======================================================

This script provides an easy interface to render all visualizations
and compile them into a complete video.

Usage:
    python render_visualizations.py                    # Interactive menu
    python render_visualizations.py -a                 # Render all scenes
    python render_visualizations.py -l                 # List all scenes
    python render_visualizations.py -s SceneName       # Render specific scene
    python render_visualizations.py -h                 # Help
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict
import json


class ManimRenderer:
    """Handle rendering of Manim scenes"""
    
    # All available scenes
    SCENES = [
        {
            "name": "KroneckerProductIntro",
            "title": "Kronecker Product Basics",
            "duration": 20,
            "description": "Introduction to the Kronecker Product with animated matrices"
        },
        {
            "name": "WhyCompression",
            "title": "Why Compression?",
            "duration": 18,
            "description": "Motivate the need for LLM compression"
        },
        {
            "name": "VanLoanRearrangement",
            "title": "Van Loan Rearrangement",
            "duration": 22,
            "description": "Visualize the matrix rearrangement process"
        },
        {
            "name": "SVDDecomposition",
            "title": "SVD Decomposition",
            "duration": 18,
            "description": "Show how SVD extracts the key components"
        },
        {
            "name": "KroneckerFactorsExtraction",
            "title": "Kronecker Factors",
            "duration": 20,
            "description": "Extract A and B from the decomposition"
        },
        {
            "name": "SparseResidualCorrection",
            "title": "Sparse Residual Correction",
            "duration": 24,
            "description": "Our novel improvement to capture missed details"
        },
        {
            "name": "ErrorComparison",
            "title": "Error Comparison",
            "duration": 20,
            "description": "Compare error metrics across methods"
        },
        {
            "name": "CompressionPipeline",
            "title": "Compression Pipeline",
            "duration": 22,
            "description": "Complete compression workflow visualization"
        },
        {
            "name": "CompressionRatio",
            "title": "Compression Ratio",
            "duration": 18,
            "description": "Visualize the compression ratio achieved"
        },
        {
            "name": "MathematicalDeepDive",
            "title": "Mathematical Deep Dive",
            "duration": 25,
            "description": "Detailed mathematical explanation"
        },
        {
            "name": "ApplicationToGPT2",
            "title": "Application to GPT-2",
            "duration": 20,
            "description": "How to apply compression to GPT-2"
        },
        {
            "name": "Summary",
            "title": "Key Takeaways",
            "duration": 18,
            "description": "Summary and key takeaways"
        },
        {
            "name": "InteractiveKroneckerDemo",
            "title": "Interactive Demo",
            "duration": 16,
            "description": "Live Kronecker product calculation"
        },
    ]
    
    # Render quality presets
    QUALITY_PRESETS = {
        "development": {
            "flag": "-ql",
            "description": "Low quality - fast rendering for testing",
            "resolution": "480p",
            "fps": 15
        },
        "preview": {
            "flag": "-pqm",
            "description": "Medium quality - preview quality",
            "resolution": "720p",
            "fps": 30
        },
        "high": {
            "flag": "-pqh",
            "description": "High quality - good quality rendering",
            "resolution": "1080p",
            "fps": 60
        },
        "ultra": {
            "flag": "-pqk",
            "description": "Ultra quality - highest quality rendering",
            "resolution": "4K",
            "fps": 60
        }
    }
    
    def __init__(self, project_root=None):
        """Initialize the renderer"""
        self.project_root = project_root or Path.cwd()
        self.script_file = self.project_root / "kronecker_product_visualization.py"
        self.output_dir = self.project_root / "videos"
        self.output_dir.mkdir(exist_ok=True)
    
    def list_scenes(self):
        """List all available scenes"""
        print("\n" + "="*70)
        print("Available Scenes for Rendering")
        print("="*70 + "\n")
        
        total_duration = 0
        for i, scene in enumerate(self.SCENES, 1):
            print(f"{i:2d}. {scene['name']:<35} ({scene['duration']}s)")
            print(f"    Title: {scene['title']}")
            print(f"    {scene['description']}\n")
            total_duration += scene['duration']
        
        print("="*70)
        print(f"Total estimated duration: {total_duration}s ({total_duration//60}m {total_duration%60}s)")
        print("="*70 + "\n")
    
    def list_quality_presets(self):
        """List quality presets"""
        print("\n" + "="*70)
        print("Quality Presets")
        print("="*70 + "\n")
        
        for preset_name, preset in self.QUALITY_PRESETS.items():
            print(f"{preset_name.upper()}: {preset['flag']}")
            print(f"  {preset['description']}")
            print(f"  Resolution: {preset['resolution']}, FPS: {preset['fps']}\n")
        
        print("="*70 + "\n")
    
    def render_scene(self, scene_name: str, quality: str = "high", preview: bool = False):
        """
        Render a single scene
        
        Args:
            scene_name: Name of the scene to render
            quality: Quality preset (development, preview, high, ultra)
            preview: Whether to open the video after rendering
        """
        # Validate scene name
        scene_names = [s['name'] for s in self.SCENES]
        if scene_name not in scene_names:
            print(f"Error: Scene '{scene_name}' not found.")
            print(f"Available scenes: {', '.join(scene_names)}")
            return False
        
        # Validate quality
        if quality not in self.QUALITY_PRESETS:
            print(f"Error: Quality '{quality}' not recognized.")
            self.list_quality_presets()
            return False
        
        quality_flag = self.QUALITY_PRESETS[quality]['flag']
        preview_flag = "-p" if preview else ""
        
        print(f"\n{'='*70}")
        print(f"Rendering Scene: {scene_name}")
        print(f"Quality: {quality.upper()}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        # Build command
        cmd = [
            "manim",
            quality_flag,
            preview_flag,
            str(self.script_file),
            scene_name
        ]
        
        # Remove empty strings
        cmd = [c for c in cmd if c]
        
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            subprocess.run(cmd, check=True, cwd=str(self.project_root))
            print(f"\n✓ Successfully rendered {scene_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error rendering {scene_name}: {e}")
            return False
    
    def render_all_scenes(self, quality: str = "high", preview: bool = False):
        """
        Render all scenes
        
        Args:
            quality: Quality preset
            preview: Whether to preview after rendering
        """
        print(f"\n{'='*70}")
        print(f"Rendering All Scenes ({len(self.SCENES)} total)")
        print(f"Quality: {quality.upper()}")
        print(f"{'='*70}\n")
        
        successful = 0
        failed = 0
        
        for scene in self.SCENES:
            if self.render_scene(scene['name'], quality, preview=False):
                successful += 1
            else:
                failed += 1
        
        print(f"\n{'='*70}")
        print(f"Rendering Complete: {successful} successful, {failed} failed")
        print(f"{'='*70}\n")
        
        return failed == 0
    
    def compile_video(self, output_file: str = "kronecker_compression.mp4"):
        """
        Compile all rendered scenes into a single video
        
        Args:
            output_file: Name of output video file
        """
        print(f"\n{'='*70}")
        print("Compiling Videos")
        print(f"{'='*70}\n")
        
        # Find all rendered mp4 files
        video_files = sorted(self.output_dir.glob("**/*.mp4"))
        
        if not video_files:
            print("Error: No video files found to compile.")
            return False
        
        print(f"Found {len(video_files)} video files:")
        for vf in video_files:
            print(f"  - {vf.name}")
        
        # Create concat file
        concat_file = self.output_dir / "concat.txt"
        with open(concat_file, 'w') as f:
            for video_file in video_files:
                f.write(f"file '{video_file.absolute()}'\n")
        
        output_path = self.output_dir / output_file
        
        # Use ffmpeg to concatenate
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path)
        ]
        
        print(f"\nCommand: {' '.join(cmd)}\n")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\n✓ Successfully compiled video: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error compiling video: {e}")
            return False
        finally:
            # Cleanup concat file
            if concat_file.exists():
                concat_file.unlink()
    
    def interactive_menu(self):
        """Interactive menu for rendering"""
        while True:
            print("\n" + "="*70)
            print("Kronecker Product Visualization Renderer")
            print("="*70 + "\n")
            
            print("1. List all scenes")
            print("2. List quality presets")
            print("3. Render specific scene")
            print("4. Render all scenes")
            print("5. Compile all videos")
            print("6. Exit\n")
            
            choice = input("Select option (1-6): ").strip()
            
            if choice == "1":
                self.list_scenes()
            
            elif choice == "2":
                self.list_quality_presets()
            
            elif choice == "3":
                self.list_scenes()
                scene_num = input("Enter scene number (or name): ").strip()
                
                # Try to parse as number
                try:
                    scene_idx = int(scene_num) - 1
                    if 0 <= scene_idx < len(self.SCENES):
                        scene_name = self.SCENES[scene_idx]['name']
                    else:
                        scene_name = scene_num
                except ValueError:
                    scene_name = scene_num
                
                quality = input("Quality (development/preview/high/ultra) [high]: ").strip() or "high"
                preview = input("Preview after rendering? (y/n) [n]: ").strip().lower() == 'y'
                
                self.render_scene(scene_name, quality, preview)
            
            elif choice == "4":
                quality = input("Quality (development/preview/high/ultra) [high]: ").strip() or "high"
                confirm = input(f"Render all {len(self.SCENES)} scenes in {quality} quality? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    self.render_all_scenes(quality)
            
            elif choice == "5":
                output_name = input("Output filename (without extension) [kronecker_compression]: ").strip() or "kronecker_compression"
                output_name = output_name + ".mp4" if not output_name.endswith(".mp4") else output_name
                
                self.compile_video(output_name)
            
            elif choice == "6":
                print("Exiting...")
                break
            
            else:
                print("Invalid option. Please try again.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Kronecker Product Visualization Renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python render_visualizations.py                    # Interactive menu
  python render_visualizations.py -a                 # Render all scenes
  python render_visualizations.py -l                 # List all scenes
  python render_visualizations.py -s KroneckerProductIntro -q high  # Render specific scene
        """
    )
    
    parser.add_argument('-a', '--all', action='store_true', help='Render all scenes')
    parser.add_argument('-l', '--list', action='store_true', help='List all scenes')
    parser.add_argument('-s', '--scene', type=str, help='Render specific scene')
    parser.add_argument('-q', '--quality', type=str, default='high',
                       choices=['development', 'preview', 'high', 'ultra'],
                       help='Render quality (default: high)')
    parser.add_argument('-p', '--preview', action='store_true', help='Preview after rendering')
    parser.add_argument('-c', '--compile', action='store_true', help='Compile all videos')
    parser.add_argument('-o', '--output', type=str, default='kronecker_compression.mp4',
                       help='Output filename for compiled video')
    
    args = parser.parse_args()
    
    renderer = ManimRenderer()
    
    # Handle different modes
    if args.list:
        renderer.list_scenes()
        return 0
    
    elif args.all:
        success = renderer.render_all_scenes(args.quality, args.preview)
        return 0 if success else 1
    
    elif args.scene:
        success = renderer.render_scene(args.scene, args.quality, args.preview)
        return 0 if success else 1
    
    elif args.compile:
        success = renderer.compile_video(args.output)
        return 0 if success else 1
    
    else:
        # Interactive mode
        renderer.interactive_menu()
        return 0


if __name__ == "__main__":
    sys.exit(main())
