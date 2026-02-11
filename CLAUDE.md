# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrAISformer is a PyTorch implementation of a generative transformer for AIS (Automatic Identification System) trajectory prediction. The transformer architecture is adapted from minGPT (Karpathy's implementation).

**Paper**: https://arxiv.org/abs/2109.03958

## Project Structure

```
TrAISformer/
├── src/                     # All Python source code
│   ├── trAISformer.py       # Main training script
│   ├── models.py            # Model architectures
│   ├── trainers.py          # Training logic
│   ├── datasets.py          # Data loading
│   ├── utils.py             # Utility functions
│   ├── config_trAISformer.py # Configuration
│   ├── evaluate_model.py    # Model evaluation
│   ├── compare_models.py    # Model comparison
│   └── compare_full_errors.py # Detailed error analysis
├── docs/                    # Documentation
│   ├── PHASE1_SUMMARY.md    # Phase 1 optimization results
│   └── metrics_tracking.md  # Performance metrics
├── experiments/             # All experimental outputs
│   ├── baseline/            # Baseline experiment
│   └── <config-hash>/       # Experiments (auto-generated names)
├── comparisons/             # Model comparison outputs
│   └── phase1/              # Phase 1 vs baseline
└── data/                    # Datasets
```

## Running the Code

### Training and Evaluation
```bash
cd src
python trAISformer.py
```

**Note**: All scripts must be run from the `src/` directory due to module imports.

The main script (`trAISformer.py`) performs the following:
1. Loads AIS trajectory data from `../data/ct_dma/`
2. Trains the model (if `retrain=True` in config)
3. Loads the best checkpoint from `../experiments/<config-hash>/model.pt`
4. Evaluates on test set and generates prediction error plots

### Model Evaluation
```bash
cd src
python evaluate_model.py --model-path ../experiments/<model-dir>/model.pt --output-json results.json
```

### Model Comparison
```bash
cd src
python compare_models.py --baseline ../experiments/baseline/baseline_results.json --optimized results.json --output-dir ../comparisons/
```

### Configuration

All settings are controlled via `src/config_trAISformer.py`. Key parameters:

- `retrain`: Set to `True` to train, `False` to skip training and load existing model
- `device`: CUDA device or CPU (default: `cuda:0`)
- `max_epochs`: Number of training epochs (default: 50)
- `batch_size`: Training batch size (default: 32)
- `n_samples`: Number of samples for ensemble prediction (default: 16)
- Model architecture: `n_head`, `n_layer`, embedding sizes
- Dropout rates: `embd_pdrop`, `resid_pdrop`, `attn_pdrop` (default: 0.2, increased from 0.1 to reduce overfitting)
- Early stopping: `early_stopping_patience` (default: 5 epochs)
- Learning rate parameters: `learning_rate`, `lr_decay`, `weight_decay`

The configuration generates a unique hash-based directory name in `./results/` for each experiment.

## Architecture

### Core Components

**src/trAISformer.py** (main script)
- Orchestrates data loading, training, evaluation, and visualization
- Uses matplotlib for plotting (must use 'Agg' backend for headless/remote environments)

**src/models.py**
- `TrAISformer`: Main transformer model
- `CausalSelfAttention`: Multi-head self-attention with causal masking
- Based on GPT architecture with custom position/velocity embeddings for AIS data

**trainers.py**
- `Trainer`: Training loop with validation, checkpointing, and learning rate scheduling
- `sample()`: Autoregressive sampling function for trajectory generation
- Saves best model based on validation loss

**datasets.py**
- `AISDataset`: Custom PyTorch dataset for AIS trajectories
- `AISDataset_grad`: Variant for gradient-based (velocity) representations
- Data format: `[lat, lon, sog, cog, timestamp, mmsi]` where lat/lon/sog/cog are normalized to [0,1)

**utils.py**
- `haversine()`: Great circle distance calculation
- `set_seed()`: Deterministic random seeding
- Logging utilities

### Data Pipeline

1. Load pickled trajectory data from `./data/ct_dma/`:
   - `ct_dma_train.pkl`
   - `ct_dma_valid.pkl`
   - `ct_dma_test.pkl`

2. Filter trajectories:
   - Remove tracks with NaN values
   - Remove tracks shorter than `min_seqlen`
   - Trim to moving portion (SOG > threshold)

3. Create PyTorch DataLoaders with max sequence length padding

### Model Checkpointing

- Best model is automatically saved to `./results/<config-hash>/model.pt`
- Checkpointing based on validation loss (monitors for improvement)
- Model is always loaded before evaluation, regardless of whether training occurred

## Important Implementation Details

### Remote/Headless Environment
The code MUST use matplotlib's 'Agg' backend for WSL/remote servers without display:
```python
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
```

This is already implemented in `trAISformer.py`.

### Overfitting Patterns
The model is prone to overfitting:
- Training loss decreases continuously (can even go negative)
- Validation loss typically plateaus around epoch 10-12 then increases
- Best checkpoint is automatically saved when validation loss stops improving

**Overfitting mitigations implemented:**
- ✅ **Early stopping**: Training stops automatically if validation loss doesn't improve for 5 epochs (configurable via `early_stopping_patience`)
- ✅ **Increased dropout**: All dropout rates increased to 0.2 (from 0.1) for better regularization
- ✅ **Best model checkpointing**: Automatically saves the best model based on validation loss

**Additional mitigation strategies:**
- Reduce model size (`n_layer=6` instead of 8, or smaller embeddings)
- Add data augmentation for trajectory data
- Adjust learning rate decay schedule

### Coordinate System
- Input: Normalized lat/lon/sog/cog in [0,1)
- Internal: Model works with discretized/tokenized representations
- Output: Predictions converted back to lat/lon coordinates
- Evaluation: Haversine distance in kilometers

### Sampling Modes
Controlled by `sample_mode` in config:
- `"pos"`: Direct position prediction
- `"pos_vicinity"`: Sample from vicinity of predicted position (uses `r_vicinity` parameter)
- `"velo"`: Velocity-based prediction

## Common Issues

1. **RuntimeError: main thread is not in main loop** (tkinter errors)
   - Already fixed: matplotlib uses 'Agg' backend
   - Caused by GUI backends conflicting with training loop threading

2. **Model overfitting**
   - Monitor validation loss curve
   - Best model automatically saved
   - Consider increasing dropout or reducing model complexity

3. **CUDA out of memory**
   - Reduce `batch_size` in config
   - Reduce `max_seqlen` or embedding dimensions
   - Use CPU by setting `device = torch.device("cpu")`

4. **Missing checkpoint file**
   - Ensure training has completed at least one epoch
   - Check `./results/<config-hash>/model.pt` exists
   - Verify `savedir` path in config

## Development Environment

Originally developed with:
- Python 3.7 (updated to 3.10+ for compatibility)
- PyTorch 1.6 with CUDA 9.2
- See `requirements.yml` for full conda environment
- See `pyproject.toml` for modern pip dependencies

Current minimum requirements:
- Python >= 3.10
- torch >= 2.10.0
- numpy >= 2.2.6
- matplotlib >= 3.10.8
- tqdm >= 4.67.3
