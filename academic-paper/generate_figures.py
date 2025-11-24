"""
Generate figures for thesis paper from trained models.

This script helps generate the required figures for the Results section:
1. Confusion Matrix for Neural Network model
2. Power Consumption Profile (with estimated data)
3. (Optional) System Architecture Diagram

Usage:
    python generate_figures.py

Requirements:
    pip install matplotlib seaborn numpy pandas scikit-learn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create figures directory if it doesn't exist
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Activity labels for HAR task
ACTIVITY_LABELS = [
    'Standing',
    'Still',
    'Walking',
    'Walking\nUpstairs',
    'Walking\nDownstairs',
    'Running'
]


def generate_confusion_matrix_figure():
    """
    Generate confusion matrix figure for Neural Network model.

    TODO: Replace this with actual data from your trained model!

    To get real data:
    1. Load your trained NN model from models/ directory
    2. Run predictions on test set: y_pred = model.predict(X_test)
    3. Generate confusion matrix: cm = confusion_matrix(y_test, y_pred)
    4. Pass cm to this function instead of the estimated matrix below
    """

    # ESTIMATED confusion matrix (96.2% accuracy)
    # Replace with actual confusion_matrix(y_test, y_pred) from your model!
    cm_estimated = np.array([
        [191,   2,   1,   1,   0,   0],  # Standing: 195 total
        [2, 135,   3,   2,   0,   0],  # Still: 142 total
        [1,   3, 171,   2,   1,   0],  # Walking: 178 total
        [1,   1,   3,  91,   2,   0],  # Walking Upstairs: 98 total
        [0,   0,   1,   0,  50,   0],  # Walking Downstairs: 51 total
        [0,   0,   0,   0,   0, 168],  # Running: 168 total
    ])

    # Calculate accuracy from matrix
    accuracy = np.trace(cm_estimated) / np.sum(cm_estimated) * 100

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot heatmap with annotations
    sns.heatmap(cm_estimated,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=ACTIVITY_LABELS,
                yticklabels=ACTIVITY_LABELS,
                cbar_kws={'label': 'Number of Samples'},
                square=True)

    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - Neural Network Model\n(Overall Accuracy: {accuracy:.1f}%)',
              fontsize=14, fontweight='bold', pad=20)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = FIGURES_DIR / "confusion_matrix_nn.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")
    print(f"  Accuracy from matrix: {accuracy:.1f}%")
    print(f"  Total samples: {np.sum(cm_estimated)}")

    plt.close()


def generate_power_consumption_figure():
    """
    Generate power consumption profile bar chart.

    These values are ESTIMATES based on typical ARM Cortex-M4 characteristics.
    For accurate data, measure with:
    - Multimeter in series with VCC line
    - Nordic Power Profiler Kit II
    - Joulescope precision DC energy analyzer
    """

    # Power consumption phases (estimated values in mA @ 3.3V)
    phases = ['Idle\n(Polling)', 'Sensor\nSampling',
              'Feature\nExtraction', 'Prediction\n(Inference)']
    current_ma = [2.1, 5.4, 7.8, 8.2]
    power_mw = [i * 3.3 for i in current_ma]  # Convert to mW at 3.3V

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Current consumption
    bars1 = ax1.bar(phases, current_ma, color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'],
                    edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Current Draw (mA)', fontsize=12, fontweight='bold')
    ax1.set_title('Current Consumption Profile\n(Seeed XIAO nRF52840 @ 3.3V)',
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 10)

    # Add value labels on bars
    for bar, val in zip(bars1, current_ma):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{val:.1f} mA',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Subplot 2: Power consumption
    bars2 = ax2.bar(phases, power_mw, color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'],
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Power Consumption (mW)', fontsize=12, fontweight='bold')
    ax2.set_title('Power Consumption Profile\n(Seeed XIAO nRF52840 @ 3.3V)',
                  fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 30)

    # Add value labels on bars
    for bar, val in zip(bars2, power_mw):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{val:.1f} mW',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add average line
    # Exclude idle, average active phases
    avg_current = np.mean(current_ma[1:])
    avg_power = avg_current * 3.3
    ax1.axhline(y=avg_current, color='red', linestyle='--',
                linewidth=2, label=f'Avg Active: {avg_current:.1f} mA')
    ax2.axhline(y=avg_power, color='red', linestyle='--',
                linewidth=2, label=f'Avg Active: {avg_power:.1f} mW')

    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_path = FIGURES_DIR / "power_consumption_profile.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Power consumption profile saved to: {output_path}")
    print(
        f"  Average active current: {avg_current:.1f} mA ({avg_power:.1f} mW)")
    print(f"  Estimated battery life (600mAh): {600/avg_current:.1f} hours")

    plt.close()


def generate_system_architecture_diagram():
    """
    Generate system architecture diagram (optional).

    This creates a simplified flowchart. For publication-quality diagrams,
    consider using:
    - draw.io (diagrams.net) - export as PNG
    - Microsoft Visio
    - LaTeX TikZ for vector graphics
    """

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Define components as rectangles
    components = [
        {'name': 'Data Upload\n(CSV Files)', 'pos': (
            0.5, 0.9), 'color': '#e8f4f8'},
        {'name': 'Interactive Preprocessing\n(Draggable Windows)', 'pos': (
            0.5, 0.75), 'color': '#b3e5fc'},
        {'name': 'Feature Extraction\n(138 Features)', 'pos': (
            0.5, 0.6), 'color': '#81d4fa'},
        {'name': 'Model Training\n(RF/SVM/NN)',
         'pos': (0.5, 0.45), 'color': '#4fc3f7'},
        {'name': 'Code Generation\n(C++ Export)',
         'pos': (0.5, 0.3), 'color': '#29b6f6'},
        {'name': 'Edge Deployment\n(Arduino/ARM)',
         'pos': (0.5, 0.15), 'color': '#0288d1'},
    ]

    # Draw components
    for comp in components:
        rect = plt.Rectangle((comp['pos'][0] - 0.15, comp['pos'][1] - 0.05),
                             0.3, 0.08,
                             facecolor=comp['color'],
                             edgecolor='black',
                             linewidth=2)
        ax.add_patch(rect)
        ax.text(comp['pos'][0], comp['pos'][1], comp['name'],
                ha='center', va='center', fontsize=11, fontweight='bold')

    # Draw arrows
    for i in range(len(components) - 1):
        ax.annotate('',
                    xy=(components[i+1]['pos'][0],
                        components[i+1]['pos'][1] + 0.05),
                    xytext=(components[i]['pos'][0],
                            components[i]['pos'][1] - 0.05),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Add side annotations
    ax.text(0.15, 0.75, 'User\nInterface', ha='center', va='center',
            fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.15, 0.45, 'ML\nTraining', ha='center', va='center',
            fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(0.85, 0.3, 'Optimization\nModes:\n• Accuracy\n• Speed\n• Power\n• Balanced',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('HAR Framework Architecture Overview',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save figure
    output_path = FIGURES_DIR / "system_architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ System architecture diagram saved to: {output_path}")
    print(f"  Note: For publication quality, consider redrawing in draw.io or TikZ")

    plt.close()


def main():
    """Generate all figures for thesis paper."""

    print("=" * 60)
    print("Generating Thesis Figures")
    print("=" * 60)
    print()

    # Generate confusion matrix
    print("[1/3] Generating confusion matrix...")
    generate_confusion_matrix_figure()
    print()

    # Generate power consumption profile
    print("[2/3] Generating power consumption profile...")
    generate_power_consumption_figure()
    print()

    # Generate system architecture (optional)
    print("[3/3] Generating system architecture diagram...")
    generate_system_architecture_diagram()
    print()

    print("=" * 60)
    print("✓ All figures generated successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Review generated figures in figures/ directory")
    print("2. Replace confusion matrix with ACTUAL model predictions (see code comments)")
    print("3. Measure real power consumption if possible (replace estimates)")
    print("4. Uncomment \\includegraphics lines in sections/results.tex")
    print("5. Compile LaTeX: pdflatex → bibtex → pdflatex × 2")
    print()
    print("Figure locations:")
    print(f"  • {FIGURES_DIR / 'confusion_matrix_nn.png'}")
    print(f"  • {FIGURES_DIR / 'power_consumption_profile.png'}")
    print(f"  • {FIGURES_DIR / 'system_architecture.png'}")


if __name__ == "__main__":
    main()
