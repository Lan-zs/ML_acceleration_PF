# Scalable Data-Driven Modeling of Microstructure Evolution

This repository contains the implementation for the paper **"Scalable data-driven modeling of microstructure evolution by learning local dependency and spatiotemporal translation invariance rules in phase field simulation"**.

## Abstract

Phase-field (PF) simulation provides a powerful framework for predicting microstructural evolution but suffers from prohibitive computational costs that severely limit accessible spatiotemporal scales in practical applications. While data-driven methods have emerged as promising approaches for accelerating PF simulations, existing methods require extensive training data from numerous evolution trajectories, and their inherent black-box nature raises concerns about long-term prediction reliability. 

This work demonstrates, through examples of grain growth and spinodal decomposition, that a minimalist Convolutional Neural Network (CNN) trained with a remarkably small dataset—even from a single small-scale simulation—can achieve seamless scalability to larger systems and reliable long-term predictions far beyond the temporal range of the training data. The key insight of this work lies in revealing that the success of CNN-based models stems from the alignment between their inductive biases and the physical priors of phase-field simulations—specifically, locality and spatiotemporal translation invariance. 

Through effective receptive field analysis, we verify that the model captures these essential properties during training. Therefore, from a reductionist perspective, the surrogate model essentially establishes a spatiotemporally invariant regression mapping between a grid point's local environment (i.e., the neighboring field information that governs its evolution) and its subsequent state. Further analysis of the model's feature space demonstrates that microstructural evolution effectively represents a continuous redistribution of a finite set of local environments. When the model has already encountered nearly all possible local environments in the early-stage training data, it can reliably generalize to much longer evolution timescales, regardless of the dramatic changes in global microstructural morphology.


## Project Structure

```
DeepGrainGrowth/
├── dataset/                          # Training and test data
│   ├── grain_growth/                 # Isothermal grain growth data
│   ├── grain_growth_temp/            # Non-isothermal grain growth data
│   ├── spinodal_decomposition_10dt/  # 10dt timestep data
│   ├── spinodal_decomposition_100dt/ # 100dt timestep data
│   ├── spinodal_decomposition_500dt/ # 500dt timestep data
│   └── spinodal_decomposition_3stage/# Three-stage rollout evaluation data
├── ideal_grain_growth/               # Isothermal grain growth experiments
│   ├── run_training.py              # Main training script
│   └── model_analysis/              # Analysis tools
├── non-isothermal_grain_growth/      # Temperature-dependent grain growth
│   ├── run_training.py              # Training with temperature field
│   └── model_analysis/              # Temperature-aware analysis
├── spinodal_decomposition/           # Spinodal decomposition experiments
│   ├── run_*dt_training.py          # Training scripts for different timesteps
│   ├── noise_start_rollout_3stage.py# Three-stage rollout analysis
│   └── *dt_model_analysis/          # Analysis modules
└── src/                             # Shared source code
    ├── models/                      # Neural network architectures
    ├── training/                    # Training utilities
    ├── evaluation/                  # Evaluation tools
    ├── analysis/                    # Analysis frameworks
    └── utils/                       # Visualization and utilities
```

## Quick Start

### Training a Model

```python
# For isothermal grain growth
cd ideal_grain_growth
python run_training.py

# For non-isothermal grain growth
cd non-isothermal_grain_growth
python run_training.py

# For spinodal decomposition
cd spinodal_decomposition
python run_100dt_training.py  # or run_10dt_training.py, run_500dt_training.py
```

### Running Analysis

```python
# Feature space analysis
python model_analysis/feature_space_analysis_runner.py

# Receptive field analysis
python model_analysis/receptive_field_statistics.py

# Grain growth kinetics
python model_analysis/grain_kinetics_analysis.py
```

## Key Results

The implementation demonstrates that:
1. **Minimal Training Data**: Models can be trained effectively using data from a single, small-scale simulation trajectory
2. **Long-term Reliability**: Predictions remain accurate far beyond training temporal ranges
3. **Physical Consistency**: Models capture essential physical properties (locality, translation invariance)
4. **Scalability**: Seamless performance on systems larger than training data


## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy
- Matplotlib
- h5py
- scikit-learn

## License

This project is licensed under the MIT License.