# SCFF Layer-wise Accuracy Research

Research implementation of **Self-Contrastive Forward-Forward (SCFF)** algorithm for tracking layer-wise accuracies during training on CIFAR-10.

## 🎯 Research Objective

Track and visualize how each layer's accuracy evolves independently during parallel training to understand:
- Do shallow layers learn faster than deep layers?
- Is the final layer always the best performer?
- What are the learning dynamics of each layer over time?

## 📊 What This Repo Does

This is a **modified version** of the SCFF CIFAR-10 Parallel training that:
- ✅ **Tracks each layer independently** during training
- ✅ **Generates publication-quality plots** showing all layers on one graph
- ✅ **Saves complete training history** to JSON
- ✅ **Works seamlessly on Google Colab**
- ✅ **Beautiful console output** with progress bars

## 🚀 Quick Start

### Google Colab (Recommended)

```python
# 1. Clone this repo
!git clone https://github.com/YOUR_USERNAME/SCFF-Research.git
%cd SCFF-Research

# 2. Install dependencies
!pip install -q -r requirements.txt

# 3. Quick test (5 minutes)
!python SCFF_CIFAR_Parallel_Research.py --alleps 20 20 20 --eval_frequency 10

# 4. Full training (25 minutes)
!python SCFF_CIFAR_Parallel_Research.py --alleps 100 100 100 --eval_frequency 10 --save_model

# 5. Generate plots
!python plot_layer_accuracies.py

# 6. Display results
from IPython.display import Image, display
display(Image('layer_accuracies_plot.png'))
```

### Local Machine

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/SCFF-Research.git
cd SCFF-Research
pip install -r requirements.txt

# Run training
python SCFF_CIFAR_Parallel_Research.py --alleps 100 100 100 --eval_frequency 10 --save_model

# Generate visualizations
python plot_layer_accuracies.py
```

## 📁 Repository Structure

```
SCFF-Research/
├── SCFF_CIFAR_Parallel_Research.py   # Modified training script
├── plot_layer_accuracies.py          # Visualization script
├── config.json                        # Network hyperparameters
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── .gitignore                         # Git ignore rules
```

## ⚙️ Command Line Arguments

```bash
python SCFF_CIFAR_Parallel_Research.py \
    --alleps 100 100 100        # Epochs per layer (3 layers)
    --eval_frequency 10          # Evaluate every N epochs
    --lr 0.02 0.001 0.0004      # Learning rates per layer
    --th1 1 4 5                  # Positive thresholds
    --th2 2 5 7                  # Negative thresholds
    --device_num 0               # GPU index
    --seed_num 1234              # Random seed
    --save_model                 # Save trained weights
```

### Key Parameters

- **`--alleps`**: Training epochs per layer (default: `100 100 100`)
  - Fast test: `--alleps 20 20 20` (~5 min)
  - Normal: `--alleps 100 100 100` (~25 min)
  - Thorough: `--alleps 150 150 150` (~40 min)

- **`--eval_frequency`**: Evaluation interval (default: `10`)
  - More data points: `--eval_frequency 5` (slower)
  - Fewer data points: `--eval_frequency 20` (faster)

## 📈 Output Files

After training, you'll get:

### 1. `layer_accuracies_history.json`
Complete training history with all layer accuracies:
```json
{
  "epochs": [10, 20, 30, ...],
  "accuracies": {
    "layer_0": [65.2, 68.3, 70.1, ...],
    "layer_1": [70.5, 73.2, 75.8, ...],
    "layer_2": [72.1, 75.4, 78.2, ...]
  },
  "num_layers": 3,
  "total_epochs": 100
}
```

### 2. `layer_accuracies_plot.png`
Main research plot showing all layers' accuracies over time:
```
Accuracy (%)
 85 |                                    Layer 2 ──────
    |                              Layer 1 ────
 80 |                        Layer 0 ──
    |
 75 |            All layers training in parallel
    |
 70 |___________________________________________________
    0        25        50        75        100     Epochs
```

### 3. `layer_comparison.png`
Bar charts and improvement analysis

### 4. `training_summary.txt`
Detailed statistics for each layer

## 📊 Expected Results

Based on the original paper ([Nature Communications 2025](https://doi.org/10.1038/s41467-025-61037-0)):

| Metric | Expected Value |
|--------|----------------|
| Final Combined Accuracy | ~80.8% |
| Layer 0 (Shallow) | ~70-75% |
| Layer 1 (Middle) | ~75-78% |
| Layer 2 (Deep) | ~78-80% |
| Training Time (100 epochs, GPU) | ~20-30 min |

## 🖥️ Console Output

Beautiful real-time progress during training:

```
══════════════════════════════════════════════════════════════════════
Epoch 10/100
──────────────────────────────────────────────────────────────────────
Training: 100%|████████████| 500/500 [00:45<00:00, G+=2.345, G-=-1.234]

Epoch 10 Summary:
  Time: 47.23s
  Avg Goodness (Positive): 2.3456
  Avg Goodness (Negative): -1.2345

══════════════════════════════════════════════════════════════════════
Layer-wise Accuracy Evaluation at Epoch 10
══════════════════════════════════════════════════════════════════════
  Evaluating Layer 0...
    Layer 0 Accuracy: 68.50%
  Evaluating Layer 1...
    Layer 1 Accuracy: 73.20%
  Evaluating Layer 2...
    Layer 2 Accuracy: 76.80%
══════════════════════════════════════════════════════════════════════
```

## 🔬 Research Questions

This implementation helps answer:

1. **Learning Speed**: Which layer converges fastest?
2. **Final Performance**: Is deeper always better?
3. **Stability**: How consistent is learning across epochs?
4. **Efficiency**: What's the best layer for feature extraction?

## 💡 Usage Tips

### Faster Experimentation

```bash
# Ultra-fast test (5 min, 4 data points)
python SCFF_CIFAR_Parallel_Research.py --alleps 20 20 20 --eval_frequency 5

# Moderate test (12 min, 10 data points)
python SCFF_CIFAR_Parallel_Research.py --alleps 50 50 50 --eval_frequency 5
```

### For Reproducibility

Always use the same seed:
```bash
python SCFF_CIFAR_Parallel_Research.py --seed_num 42
```

### Memory Optimization

Already optimized! But if needed:
- Increase `--eval_frequency 20` (evaluate less often)
- Data is automatically downloaded to `./data/`

## 🔧 Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 4GB+ GPU memory

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision tqdm matplotlib scikit-learn
```

## 🐛 Troubleshooting

**Problem**: "sklearn not available"
- Solution: `pip install scikit-learn`
- Note: Will use PyTorch classifier as fallback

**Problem**: No GPU in Colab
- Solution: Runtime → Change runtime type → Hardware accelerator → GPU

**Problem**: Out of memory
- Solution: Code is optimized, but try `--eval_frequency 20`

**Problem**: Slow download
- Solution: CIFAR-10 auto-downloads to `./data/` (170MB)

## 📝 What's Modified from Original SCFF?

### Original Implementation
- ❌ No layer-wise tracking
- ❌ Minimal console output
- ❌ Only final accuracy reported

### This Implementation
- ✅ Tracks each layer independently
- ✅ Beautiful progress bars
- ✅ Saves complete history
- ✅ Publication-quality plots
- ✅ Detailed statistics

## 🎓 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{chen2025scff,
  title={Self-Contrastive Forward-Forward algorithm},
  author={Chen, Xing and Liu, Dongshu and Laydevant, J{\'e}r{\'e}mie and Grollier, Julie},
  journal={Nature Communications},
  volume={16},
  pages={5978},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41467-025-61037-0}
}
```

## 📚 Original Repository

This is a research fork. Original implementation:
- Repository: [neurophysics-cnrsthales/contrastive-forward-forward](https://github.com/neurophysics-cnrsthales/contrastive-forward-forward)
- Paper: [Nature Communications](https://doi.org/10.1038/s41467-025-61037-0)

## 🤝 Contributing

Found a bug or have suggestions? Please open an issue!

## 📄 License

Same license as the original repository (check original repo for details).

## 👥 Authors

**Research Implementation:**
- [Your Name] - Layer-wise tracking modifications

**Original Algorithm:**
- Xing Chen, Dongshu Liu, Jérémie Laydevant, Julie Grollier
- Laboratoire Albert Fert, CNRS, Thales, Université Paris-Saclay

---

**Happy researching!** 🚀

For questions about the original algorithm, see the [original repository](https://github.com/neurophysics-cnrsthales/contrastive-forward-forward).
