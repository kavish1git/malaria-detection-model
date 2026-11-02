"""
src/generate_eda_plots.py
-------------------------
Generates comprehensive EDA figures and Model Architecture diagrams.

Output Figures:
- results/class_distribution.png
- results/representative_cells.png
- results/cell_dimension_distribution.png
- results/model_architecture.png
"""

import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "Arial"


def generate_class_distribution_plot(data_dir: str, results_dir: str):
    manifest_path = os.path.join(data_dir, "dataset_manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}")
        return
        
    df = pd.read_csv(manifest_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Overall Balance
    palette = {"Parasitized": "#d62728", "Uninfected": "#1f77b4"}
    sns.countplot(
        data=df,
        x="category",
        palette=palette,
        ax=ax1,
        edgecolor="black"
    )
    ax1.set_title("Overall Dataset Class Balance (N=27,558)", fontsize=13, weight="bold", pad=12)
    ax1.set_xlabel("Clinical Category", fontsize=11)
    ax1.set_ylabel("Image Count", fontsize=11)
    for p in ax1.patches:
        count = int(p.get_height())
        ax1.annotate(f"{count:,} ({count/len(df)*100:.1f}%)",
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha="center", va="center", color="white", weight="bold", fontsize=11)
                     
    # 2. Split Distribution by Class
    split_order = ["train", "val", "test"]
    split_labels = ["Training (70%)", "Validation (15%)", "Testing (15%)"]
    sns.countplot(
        data=df,
        x="split",
        hue="category",
        order=split_order,
        palette=palette,
        ax=ax2,
        edgecolor="black"
    )
    ax2.set_xticklabels(split_labels)
    ax2.set_title("Patient-Grouped Split Distribution (Zero Patient Overlap)", fontsize=13, weight="bold", pad=12)
    ax2.set_xlabel("Dataset Partition", fontsize=11)
    ax2.set_ylabel("Image Count", fontsize=11)
    ax2.legend(title="Category", frameon=True)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, "class_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to: {save_path}")


def generate_representative_cells_plot(data_dir: str, results_dir: str):
    manifest_path = os.path.join(data_dir, "dataset_manifest.csv")
    if not os.path.exists(manifest_path):
        return
    df = pd.read_csv(manifest_path)
    
    parasitized_samples = df[df["category"] == "Parasitized"].sample(5, random_state=42)["filepath"].values
    uninfected_samples = df[df["category"] == "Uninfected"].sample(5, random_state=42)["filepath"].values
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    
    # Row 1: Parasitized
    for i, fpath in enumerate(parasitized_samples):
        img = Image.open(fpath)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Parasitized #{i+1}\n({img.size[0]}x{img.size[1]})", fontsize=10, weight="bold", color="#d62728")
        axes[0, i].axis("off")
        
    # Row 2: Uninfected
    for i, fpath in enumerate(uninfected_samples):
        img = Image.open(fpath)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Uninfected #{i+1}\n({img.size[0]}x{img.size[1]})", fontsize=10, weight="bold", color="#1f77b4")
        axes[1, i].axis("off")
        
    plt.suptitle("Representative Microscopic Blood Cell Smears (Giemsa Stained)", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "representative_cells.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved representative cells plot to: {save_path}")


def generate_architecture_diagram(results_dir: str):
    """
    Renders an academic block diagram of the CNN architecture.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    
    blocks = [
        ("Input", "128x128x3\nRGB Image", "#e2e8f0", 0.5, 3.5),
        ("Block 1", "Conv2D (32, 3x3)\nBatchNorm + ReLU\nMaxPool (2x2)\nDropout (0.20)\n-> 64x64x32", "#bfdbfe", 2.2, 3.5),
        ("Block 2", "Conv2D (64, 3x3)\nBatchNorm + ReLU\nMaxPool (2x2)\nDropout (0.25)\n-> 32x32x64", "#93c5fd", 3.9, 3.5),
        ("Block 3", "Conv2D (128, 3x3)\nBatchNorm + ReLU\nMaxPool (2x2)\nDropout (0.30)\n-> 16x16x128", "#60a5fa", 5.6, 3.5),
        ("Block 4", "Conv2D (256, 3x3)\nBatchNorm + ReLU\nGAP (256-dim)\n-> 256 vector", "#3b82f6", 7.3, 3.5),
        ("Dense Head", "Dense (128, L2)\nBatchNorm + ReLU\nDropout (0.50)\nDense (1, Sigmoid)", "#1d4ed8", 9.0, 3.5),
    ]
    
    for title, text, color, x, y in blocks:
        box = patches.FancyBboxPatch(
            (x - 0.65, y - 1.2), 1.3, 2.4,
            boxstyle="round,pad=0.1",
            fc=color, ec="#1e293b", lw=1.5
        )
        ax.add_patch(box)
        tc = "white" if color in ["#3b82f6", "#1d4ed8"] else "#0f172a"
        ax.text(x, y + 0.7, title, ha="center", va="center", weight="bold", fontsize=11, color=tc)
        ax.text(x, y - 0.2, text, ha="center", va="center", fontsize=8.5, color=tc)
        
    # Draw arrows between blocks
    for i in range(len(blocks) - 1):
        x_start = blocks[i][3] + 0.65
        x_end = blocks[i+1][3] - 0.65
        ax.annotate(
            "", xy=(x_end, 3.5), xytext=(x_start, 3.5),
            arrowprops=dict(arrowstyle="->", lw=2, color="#334155")
        )
        
    ax.set_title("Custom Convolutional Neural Network (CNN) Architecture for Malaria Detection",
                 fontsize=14, weight="bold", pad=20)
                 
    plt.tight_layout()
    save_path = os.path.join(results_dir, "model_architecture.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved architecture diagram to: {save_path}")


if __name__ == "__main__":
    d_dir = os.path.join(PROJECT_ROOT, "data")
    r_dir = os.path.join(PROJECT_ROOT, "results")
    generate_class_distribution_plot(d_dir, r_dir)
    generate_representative_cells_plot(d_dir, r_dir)
    generate_architecture_diagram(r_dir)
