# Hyperparameter Tuning with Keras Tuner

This lab introduces **hyperparameter tuning** using [Keras Tuner](https://keras-team.github.io/keras-tuner/). You train a baseline model on Fashion MNIST with fixed hyperparameters, then use an automated search to find a better set of hyperparameters and compare results.

## Overview

- **Dataset:** [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) (image classification).
- **Baseline:** A small DNN (Flatten → Dense(512) → Dropout(0.2) → Dense(10)) trained for a fixed number of epochs.
- **Tuning:** A *hypermodel* is defined with a search space; a tuner (here **RandomSearch**) runs multiple trials to minimize validation loss; the best configuration is then retrained and evaluated.

## Setup

1. Create a virtual environment and activate it.
2. Install dependencies:
   ```bash
   pip install tensorflow keras-tuner
   ```

## Running the lab

- **Notebook:** Open `Keras_Tuner.ipynb` in Jupyter or Colab and run all cells.
- **Script:** From this folder run:
  ```bash
  python keras_tuner.py
  ```
  The script runs baseline training, hyperparameter search, and final comparison. TensorBoard logs are written to `./tb_logs` and `./keras_tuner` if you want to inspect training.

## Project structure

```
Hyper_Parameters_Tuning/
├── Keras_Tuner.ipynb   # Notebook version (with markdown and outputs)
├── keras_tuner.py      # Python script version (same logic)
└── README.md           # This file
```

After a run, `kt_dir/` will contain Keras Tuner trial logs and checkpoints.

---

## Changes made (Lab submission)

This lab was modified so it is not identical to the original repo. Summary of changes:

### 1. Different tuner and objective

- **Tuner:** Replaced **Hyperband** with **RandomSearch** (`kt.RandomSearch` with `max_trials=20`, `executions_per_trial=1`).
- **Why:** RandomSearch samples hyperparameter combinations at random; it is simple and parallelizable. Hyperband, by contrast, uses adaptive resource allocation (training many configs for a few epochs and continuing only the best). The notebook markdown explains this difference.
- **Objective:** Optimize **validation loss** (minimize) using `kt.Objective('val_loss', direction='min')` instead of maximizing validation accuracy.

### 2. Wider search space

- **Dropout:** Tunable via `hp.Choice('dropout', values=[0.1, 0.2, 0.3, 0.4])` and applied to the Dropout layer.
- **Activation:** First Dense layer activation is tunable: `hp.Choice('activation', values=['relu', 'tanh'])`.
- **Learning rate:** Replaced a fixed set of choices with a continuous log-scale range: `hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')`.

### 3. Second tunable Dense layer

- Added a second Dense layer with tunable units: `hp.Int('units_2', min_value=32, max_value=256, step=32)` (name `tuned_dense_2`).
- Architecture is now: Flatten → Dense(units, activation) → Dense(units_2, relu) → Dropout → Dense(10, softmax), so both width and “depth” are tuned.

### 4. Training and early stopping

- **NUM_EPOCHS** increased from 10 to **15** for both baseline and tuning.
- **EarlyStopping:** Now monitors **validation accuracy** with **patience=3** (instead of validation loss with patience=5), so training stops when val accuracy stops improving for 3 epochs.

### Summary table

| Item | Original | Modified |
|------|----------|----------|
| Tuner | Hyperband | RandomSearch (20 trials) |
| Objective | val_accuracy (max) | val_loss (min) |
| Dense layers tuned | 1 (units) | 2 (units, units_2) |
| Dropout | Fixed 0.2 | Choice [0.1, 0.2, 0.3, 0.4] |
| Activation (1st Dense) | relu | Choice [relu, tanh] |
| Learning rate | Choice [1e-2, 1e-3, 1e-4] | Float 1e-4–1e-2 (log) |
| NUM_EPOCHS | 10 | 15 |
| EarlyStopping | val_loss, patience=5 | val_accuracy, patience=3 |

These updates are reflected in both `Keras_Tuner.ipynb` and `keras_tuner.py`.
