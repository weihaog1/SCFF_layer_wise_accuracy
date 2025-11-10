"""
Visualization script for layer-wise accuracy evolution
Creates publication-quality plots from training history
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_history(filename='layer_accuracies_history.json'):
    """Load accuracy history from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def plot_layer_accuracies(history_data, save_path='layer_accuracies_plot.png'):
    """
    Create a publication-quality plot of layer accuracies over epochs

    Args:
        history_data (dict): Dictionary containing epoch numbers and layer accuracies
        save_path (str): Path to save the figure
    """
    epochs = history_data['epochs']
    accuracies = history_data['accuracies']
    num_layers = history_data['num_layers']

    # Set style for publication quality
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    # Create figure with high DPI
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette (professional colors)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Markers for each layer
    markers = ['o', 's', '^', 'D', 'v']

    # Plot each layer
    for i in range(num_layers):
        layer_name = f'layer_{i}'
        if layer_name in accuracies and len(accuracies[layer_name]) > 0:
            ax.plot(epochs, accuracies[layer_name],
                   color=colors[i % len(colors)],
                   linewidth=2.5,
                   marker=markers[i % len(markers)],
                   markersize=8,
                   label=f'Layer {i}',
                   alpha=0.85,
                   markeredgewidth=1.5,
                   markeredgecolor='white')

    # Customize plot
    ax.set_xlabel('Training Epochs', fontsize=16, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Layer-wise Accuracy Evolution in SCFF (CIFAR-10)',
                fontsize=18, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Legend
    ax.legend(fontsize=13, loc='lower right', frameon=True,
             shadow=True, fancybox=True, framealpha=0.95)

    # Axis formatting
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Set y-axis limits with some padding
    all_accs = [acc for layer_accs in accuracies.values() for acc in layer_accs]
    if all_accs:
        min_acc = min(all_accs)
        max_acc = max(all_accs)
        padding = (max_acc - min_acc) * 0.1
        ax.set_ylim(max(0, min_acc - padding), min(100, max_acc + padding))

    # Add metadata text
    metadata_text = (f"Timestamp: {history_data['timestamp']}\n"
                    f"Total Epochs: {history_data['total_epochs']}\n"
                    f"Eval Frequency: {history_data['eval_frequency']}")

    ax.text(0.02, 0.98, metadata_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {save_path}")

    return fig, ax


def plot_comparison(history_data, save_path='layer_comparison.png'):
    """
    Create a comparison plot showing final accuracies and improvement trends

    Args:
        history_data (dict): Dictionary containing epoch numbers and layer accuracies
        save_path (str): Path to save the figure
    """
    epochs = history_data['epochs']
    accuracies = history_data['accuracies']
    num_layers = history_data['num_layers']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot 1: Final accuracies bar chart
    final_accs = []
    layer_labels = []
    for i in range(num_layers):
        layer_name = f'layer_{i}'
        if layer_name in accuracies and len(accuracies[layer_name]) > 0:
            final_accs.append(accuracies[layer_name][-1])
            layer_labels.append(f'Layer {i}')

    x_pos = np.arange(len(layer_labels))
    bars = ax1.bar(x_pos, final_accs, color=colors[:len(final_accs)], alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Final Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Final Accuracy by Layer', fontsize=16, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(layer_labels, fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.tick_params(axis='y', labelsize=11)

    # Plot 2: Accuracy improvement over training
    for i in range(num_layers):
        layer_name = f'layer_{i}'
        if layer_name in accuracies and len(accuracies[layer_name]) > 0:
            layer_accs = accuracies[layer_name]
            if len(layer_accs) > 1:
                initial_acc = layer_accs[0]
                improvements = [(acc - initial_acc) for acc in layer_accs]

                ax2.plot(epochs, improvements,
                        color=colors[i % len(colors)],
                        linewidth=2.5,
                        marker='o',
                        markersize=6,
                        label=f'Layer {i}',
                        alpha=0.85)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax2.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy Improvement (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Accuracy Improvement from Initial', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {save_path}")

    return fig, (ax1, ax2)


def generate_summary_report(history_data, save_path='training_summary.txt'):
    """
    Generate a text summary report of the training

    Args:
        history_data (dict): Dictionary containing training history
        save_path (str): Path to save the report
    """
    epochs = history_data['epochs']
    accuracies = history_data['accuracies']
    num_layers = history_data['num_layers']

    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SCFF CIFAR-10 Training Summary Report\n")
        f.write("="*70 + "\n\n")

        f.write(f"Timestamp: {history_data['timestamp']}\n")
        f.write(f"Number of Layers: {num_layers}\n")
        f.write(f"Total Training Epochs: {history_data['total_epochs']}\n")
        f.write(f"Evaluation Frequency: Every {history_data['eval_frequency']} epochs\n")
        f.write(f"Number of Evaluations: {len(epochs)}\n\n")

        f.write("-"*70 + "\n")
        f.write("Layer-wise Performance Summary\n")
        f.write("-"*70 + "\n\n")

        for i in range(num_layers):
            layer_name = f'layer_{i}'
            if layer_name in accuracies and len(accuracies[layer_name]) > 0:
                layer_accs = accuracies[layer_name]

                f.write(f"Layer {i}:\n")
                f.write(f"  Initial Accuracy: {layer_accs[0]:.2f}%\n")
                f.write(f"  Final Accuracy:   {layer_accs[-1]:.2f}%\n")
                f.write(f"  Best Accuracy:    {max(layer_accs):.2f}%\n")
                f.write(f"  Improvement:      {layer_accs[-1] - layer_accs[0]:+.2f}%\n")
                f.write(f"  Std Dev:          {np.std(layer_accs):.2f}%\n\n")

        f.write("-"*70 + "\n")
        f.write("Detailed Accuracy Timeline\n")
        f.write("-"*70 + "\n\n")

        f.write(f"{'Epoch':<10}")
        for i in range(num_layers):
            f.write(f"Layer {i:<8}")
        f.write("\n")
        f.write("-"*70 + "\n")

        for idx, epoch in enumerate(epochs):
            f.write(f"{epoch:<10}")
            for i in range(num_layers):
                layer_name = f'layer_{i}'
                if layer_name in accuracies and idx < len(accuracies[layer_name]):
                    f.write(f"{accuracies[layer_name][idx]:<13.2f}")
                else:
                    f.write(f"{'N/A':<13}")
            f.write("\n")

        f.write("\n" + "="*70 + "\n")

    print(f"✓ Summary report saved to: {save_path}")


def main():
    """Main function to generate all visualizations"""
    print("\n" + "="*70)
    print("SCFF Layer-wise Accuracy Visualization")
    print("="*70 + "\n")

    # Check if history file exists
    if not Path('layer_accuracies_history.json').exists():
        print("Error: layer_accuracies_history.json not found!")
        print("Please run the training script first.")
        return

    # Load history
    print("Loading training history...")
    history_data = load_history('layer_accuracies_history.json')
    print(f"✓ Loaded {len(history_data['epochs'])} evaluation points")
    print(f"✓ {history_data['num_layers']} layers tracked\n")

    # Generate plots
    print("Generating visualizations...\n")

    # Main accuracy plot
    plot_layer_accuracies(history_data, 'layer_accuracies_plot.png')

    # Comparison plot
    plot_comparison(history_data, 'layer_comparison.png')

    # Summary report
    generate_summary_report(history_data, 'training_summary.txt')

    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. layer_accuracies_plot.png  - Main accuracy evolution plot")
    print("  2. layer_comparison.png       - Comparison and improvement plots")
    print("  3. training_summary.txt       - Detailed text report")
    print("\n")

    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()
