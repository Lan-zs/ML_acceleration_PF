### DeepGrainGrowth Project Guide

This guide documents all modules in the project, explains the required execution order for specific analysis scripts, and summarizes each file's role for quick onboarding.

---

## 1) Repository Structure (all modules)

### Dataset Structure
The project includes datasets for different physical phenomena:
- `dataset/grain_growth/`: Isothermal grain growth data (Training_save100dt/, Test/ with 256/, 512/, 1024/ )
- `dataset/grain_growth_temp/`: Non-isothermal grain growth data (Training/, Test/, Evaluation/)
- `dataset/spinodal_decomposition_10dt/`: Spinodal decomposition with 10dt timestep (Training/, Test/, Evaluation/)
- `dataset/spinodal_decomposition_100dt/`: Spinodal decomposition with 100dt timestep (Training/, Test/)
- `dataset/spinodal_decomposition_500dt/`: Spinodal decomposition with 500dt timestep (Training/, Test/)
- `dataset/spinodal_decomposition_3stage/`: Three-stage spinodal decomposition data (.h5 files)

### Core Modules

- `ideal_grain_growth/`
  - `run_training.py`: End-to-end experiment runner that orchestrates model creation, training, evaluation, and result logging. Adjust paths and experiment settings inside this script to match your environment. Outputs include best/final checkpoints, loss curves, and evaluation figures under a variant-specific result directory.
  - `model_analysis/`
    - `grain_kinetics_analysis.py`: Compares grain growth kinetics between the trained model and phase-field (PF) simulations. It:
      - Generates a Voronoi microstructure as the model's initial state and performs a long rollout, periodically saving `.dat` snapshots.
      - Analyzes a PF `.h5` file at matched intervals to compute mean grain radius squared over time.
      - Saves kinetics data (including regression metrics) and evolution visualizations. The saved `.dat` snapshots are required by the distribution analysis.
    - `grain_distribution_analysis.py`: Computes and compares steady-state grain size distributions for ML rollout and PF simulations. It:
      - Loads the model rollout `.dat` files produced by `grain_kinetics_analysis.py` and extracts grain size histograms over selected time windows.
      - Loads PF data from an `.h5` file, extracts histograms over selected frames, and aggregates steady-state distributions.
      - Produces combined comparison data and figures, plus evolution summaries.
    - `feature_space_analysis_runner.py`: Feature-space entrypoint that prepares artifacts for downstream analyses. It:
      - Loads a trained model and an evaluation `.h5` sequence.
      - Samples per-pixel features over specified frame ranges via a forward hook and reduces them (PCA/TSNE).
      - Saves reduced coordinates, the fitted reducer (PCA with `transform`), and a smooth non-convex boundary of the training feature distribution.
    - `feature_space_frame_density_analysis.py`: Uses the reducer and boundary saved by the runner to analyze frame-wise point densities in the reduced feature space. It:
      - Samples points for selected frames, extracts features, projects them using the saved reducer, and renders KDE density maps with consistent global axes and the boundary overlay.
    - `feature_space_rollout_trajectory_analysis.py`: Uses the reducer and boundary saved by the runner to visualize fixed-point rollout trajectories. It:
      - Selects valid fixed points on the initial frame, records their reduced features over the rollout, and overlays their trajectories on the boundary.
    - `receptive_field_imshow.py`: Visualizes receptive fields as heatmaps for selected points or categories. It:
      - Computes input gradients for targeted output pixels to obtain effective receptive field maps.
      - Produces styled visualizations to qualitatively inspect spatial influence patterns.
    - `receptive_field_statistics.py`: Quantifies receptive fields across sampling regimes. It:
      - Aggregates effective receptive field maps over many points, normalizes and averages them, and estimates effective sizes (e.g., pixel-count threshold metrics).
      - Compares effective receptive field statistics with theoretical receptive field derived from the model's architecture.
    - `single_grain_neumann_mullins_law.py`: Validates the Neumann–Mullins law on single-grain behavior. It:
      - Extracts grain boundaries/topology from frames, measures geometric properties, and relates temporal changes to theoretical predictions for curvature-driven growth.

- `non-isothermal_grain_growth/`
  - `run_training.py`: Training script for grain growth models with temperature field input. Uses 2-channel input (phase field + temperature) and specialized dataloader/trainer for temperature-dependent grain growth.
  - `model_analysis/`
    - `feature_space_analysis_temp.py`: Feature space analysis adapted for temperature-dependent models, handling dual-channel inputs and temperature variations.
    - `receptive_field_imshowplot.py`: Receptive field visualization for dual-channel models, showing spatial influence patterns for both phase field and temperature inputs.
    - `receptive_field_statistics.py`: Statistical analysis of receptive fields for temperature-dependent grain growth models.
    - `rollout_with_timevarying_temperature.py`: Long-term rollout predictions with dynamic temperature fields to evaluate model performance under non-isothermal conditions.

- `spinodal_decomposition/`
  - `run_10dt_training.py`: Training script for spinodal decomposition models with 10dt timestep prediction. Uses ResidualConcatNet_Res architecture for delta prediction.
  - `run_100dt_training.py`: Training script for spinodal decomposition models with 100dt timestep prediction. Uses standard ResidualConcatNet architecture.
  - `run_100dt_training_nonsigmoid.py`: Training script for spinodal decomposition models without sigmoid activation in the final layer.
  - `run_500dt_training.py`: Training script for spinodal decomposition models with 500dt timestep prediction.
  - `noise_start_rollout_3stage.py`: Three-stage rollout analysis starting from noise initialization for spinodal decomposition models.
  - `two_point_correlation_comparison.py`: Analysis script comparing two-point correlation functions between model predictions and ground truth for spinodal decomposition.
  - `100dt_model_analysis/`: Analysis modules for 100dt models (feature space, receptive field analysis)
  - `100dt_model_nonsigmoid_analysis/`: Analysis modules for non-sigmoid 100dt models

- `src/`
  - `analysis/`
    - `feature_space_analysis.py`: Reusable utilities for feature extraction via forward hooks, per-pixel sampling, dimensionality reduction (PCA/TSNE), nearest-neighbor distance statistics, farthest point sampling, and smooth non-convex boundary construction. Designed to be model-agnostic; runners supply the model.
    - `grain_growth_quantitative_validation.py`: Tools for quantitative validation of grain growth, including Voronoi microstructure generation, model rollout with periodic saving to `.dat`, grain identification and radius extraction, grain size distribution computation, steady-state averaging, and combined dataset preparation for visualization.
    - `receptive_field_analysis.py`: Utilities for receptive field studies, including theoretical receptive field size calculation for the provided architectures, point-wise gradient-based effective receptive fields, and aggregation/normalization helpers for comparative analysis.
  - `evaluation/`
    - `evaluator.py`: Evaluation utilities for one-step and rollout prediction modes, with comprehensive visualizations, error metrics (total/changed/unchanged regions), broken-axis plots, and error data serialization for later analysis.
  - `models/`
    - `residual_net.py`: Model implementations used throughout the project:
      - `ResidualConcatNet`: Residual blocks with channel concatenation and a scaled output range suitable for phase-field states.
      - `ResidualConcatNet_Res`: A residual variant that predicts deltas and adds them to the input (for ablation/experimentation).
  - `training/`
    - `dataloader.py`: HDF5 data access (random-access and iterable) yielding current/next phase-field pairs (and temperature field). Handles `temp_data`/`tem_data` key variations.
    - `trainer.py`: One-step training loop with optional augmentation and noise, metric logging, visualization snapshots during training, checkpointing, LR scheduling, and loss curve persistence.
  - `utils/`
    - `visualization.py`: Shared plotting utilities for training/evaluation/analysis, including custom colormaps, multi-panel model performance visualizations, error plots, kinetics comparison, steady-state distribution comparison, and grain evolution figure generation.

---

## 2) Required Execution Order for Analyses

- Grain size distribution analysis depends on prior kinetics results:
  1) `ideal_grain_growth/model_analysis/grain_kinetics_analysis.py` (produces model rollout `.dat` files and kinetics data)
  2) `ideal_grain_growth/model_analysis/grain_distribution_analysis.py` (consumes the `.dat` files to compute steady-state distributions)

- Feature-space downstream analyses depend on the feature-space runner:
  1) `ideal_grain_growth/model_analysis/feature_space_analysis_runner.py` (produces reducer and boundary)
  2) `ideal_grain_growth/model_analysis/feature_space_frame_density_analysis.py` (uses reducer/boundary for frame-wise densities)
  3) `ideal_grain_growth/model_analysis/feature_space_rollout_trajectory_analysis.py` (uses reducer/boundary for fixed-point trajectories)

All other analyses in `model_analysis/` can be run independently of each other.

---

## 3) Notes

- Paths inside the analysis and runner scripts are examples and should be updated to match your local dataset and result directories.
- Saved artifacts (e.g., reducer, boundary data, kinetics/distribution joblib files, `.dat` snapshots) are reused across analyses to ensure consistency and avoid recomputation.
