# Using GitHub Actions for Model Training and Versioning

This repository demonstrates how to use GitHub Actions to automate the process of training a machine learning model, storing the model, and versioning it. This allows you to easily update and improve your model in a collaborative environment.

Watch the tutorial video for this lab at [Github action Lab2](https://youtu.be/cj5sXIMZUjQ)

## Prerequisites

- [GitHub](https://github.com) account
- Basic knowledge of Python and machine learning
- Git command-line tool (optional)

## Getting Started

1. **Fork this Repository**: Click the "Fork" button at the top right of this [repository](https://github.com/raminmohammadi/MLOps/) to create your own copy.
2. **Clone Your Repository**:
  ```bash
   git clone https://github.com/your-username/your-forked-repo.git
   cd your-forked-repo

  ```
3. GitHub account
4. Basic knowledge of Python and machine learning
5. Git command-line tool (optional)

# Running the Workflow

## Customize Model Training

1. Modify the `train_model.py` script in the `src/` directory according to your dataset and model requirements. This script generates synthetic data for demonstration purposes.

## Push Your Changes:

1. Commit your changes and push them to your forked repository.

## GitHub Actions Workflow:

1. Once you push changes to the main branch, the GitHub Actions workflow will be triggered automatically.

## View Workflow Progress:

1. You can track the progress of the workflow by going to the "Actions" tab in your GitHub repository.

## Retrieve the Trained Model:

1. After the workflow completes successfully, the trained model will be stored in the `models/` directory.

# Model Evaluation

The model evaluation is performed automatically within the GitHub Actions workflow. The evaluation results (e.g., F1 Score) are stored in the `metrics/` directory.

# Versioning the Model

Each time you run the workflow, a new version of the model is created and stored. You can access and use these models for your projects.

# GitHub Actions Workflow Details

The workflow consists of the following steps:

- Generate and Store Timestamp: A timestamp is generated and stored in a file for versioning.
- Model Training: The `train_model.py` script is executed, which trains a random forest classifier on synthetic data and stores the model in the `models/` directory.
- Model Evaluation: The `evaluate_model.py` script is executed to evaluate the model's F1 Score on synthetic data, and the results are stored in the `metrics/` directory.
- Store and Version the New Model: The trained model is moved to the `models/` directory with a timestamp-based version.
- Commit and Push Changes: The metrics and updated model are committed to the repository, allowing you to track changes.

# Model Calibration Workflow

## Overview

The `model_calibration_on_push.yml` workflow is a part of the automation process for machine learning model calibration within this repository. It is essential for ensuring that the model's predictions are accurate and well-calibrated, a critical step in machine learning applications.

## Workflow Purpose

This workflow's primary purpose is to calibrate a trained machine learning model after each push to the main branch of the repository. Calibration is a crucial step to align model predictions with reality, particularly when dealing with classification tasks. In simple terms, calibration ensures that a model's predicted probabilities match the actual likelihood of an event happening.

## Workflow Execution

Let's break down how this workflow operates step by step:

### Step 1: Trigger on Push to Main Branch

- This workflow is automatically initiated when changes are pushed to the main branch of the repository. It ensures that the model remains calibrated and up-to-date with the latest data and adjustments.

### Step 2: Prepare Environment

- The workflow begins by setting up a Python environment and installing the necessary Python libraries and dependencies. This is crucial to ensure that the model calibration process can be executed without any issues.

### Step 3: Load Trained Model

- The trained machine learning model, which has been previously saved in the `models/` directory, is loaded into memory. This model should be the most recent version, as trained by the `train_model.py` script.

### Step 4: Calibrate Model Probabilities

- In this step, the model's predicted probabilities are calibrated. Calibration methods, such as Platt scaling or isotonic regression, are applied. These methods adjust the model's predicted probabilities to match the actual likelihood of an event occurring. This calibration step is critical for reliable decision-making based on the model's predictions.

### Step 5: Save Calibrated Model

- The calibrated model is saved back to the `models/` directory. It is given a distinct identifier to differentiate it from the original, uncalibrated models. This ensures that both the original model and the calibrated model are available for comparison and use.

### Step 6: Commit and Push Changes

- This final step involves committing the calibrated model and any other relevant files to the repository. It is essential to keep track of the changes made during the calibration process and to store the calibrated model in the repository for future applications and reference.

# Customization

The `model_calibration_on_push.yml` workflow can be customized to align with your specific project requirements. You can modify calibration methods, the directory where the calibrated model is saved, or any other aspects of the calibration process to meet your project's unique needs.

# Integration with Model Training

This workflow is designed to work seamlessly with the main model training workflow, `model_retraining_on_push.yml`. In the initial workflow, the model is trained, and in this workflow, the calibrated model is generated. The calibrated model can then be used in applications where precise, well-calibrated probabilities are essential.

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments

- This project uses GitHub Actions for continuous integration and deployment.
- Model training and evaluation are powered by Python and scikit-learn.

# Questions or Issues

If you have any questions or encounter issues while using this GitHub Actions workflow, please open an issue in the Issues section of your repository.

---

## Changes made (Lab-5 submission)

### GitHub Actions workflow updates

- Updated `workflows/model_retraining_on_push.yml` to be more robust to paths by setting `defaults.run.working-directory: Labs/Github_Labs/Lab2`, and by running `python src/train_model.py ...` and `python src/evaluate_model.py ...` from within that directory.
- Updated `workflows/model_calibration_on_push.yml` to match the new script output locations by staging `metrics/<timestamp>_metrics.json` and `models/model_<timestamp>_dt_model.joblib` directly (instead of using fragile `mv`/absolute paths).
- Updated `workflows/model_calibration_on_push.yml` to include `permissions: contents: write` so the workflow can successfully `git push` back to the repository.
- Updated the scheduled calibration workflow `workflows/model_calibration.yml` to fix a bug where `metrics_filename` was referenced without being defined, and to stage/move the correct `metrics/` and `models/` artifacts.
- Updated both workflow YAMLs to use modern GitHub actions versions (`actions/checkout@v4` and `actions/setup-python@v5`) and to install dependencies from the lab-local `requirements.txt`.

### Script changes (training and evaluation)

- Refactored `src/train_model.py` to generate a deterministic synthetic dataset based on `--timestamp` and to write artifacts directly to the lab folders:
- dataset to `data/data.pickle` and `data/target.pickle`
- model to `models/model_<timestamp>_dt_model.joblib`
- Refactored `src/evaluate_model.py` to load the model from `models/model_<timestamp>_dt_model.joblib`, recreate the same deterministic synthetic dataset from `--timestamp`, and write metrics directly to `metrics/<timestamp>_metrics.json`.
- Made the `mlflow` dependency import lazy (only imported when `use_mlflow` is enabled) so the training logic can be imported without requiring `mlflow` in every environment.

