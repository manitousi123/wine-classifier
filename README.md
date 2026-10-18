# Wine Classifier: From Weak Baseline to Defensible Model

A compact machine learning portfolio project that shows why preprocessing choices matter for distance-based models.

## Problem

Classify samples in scikit-learn's copy of the UCI Wine dataset into one of three classes. The project deliberately starts with an unscaled K-nearest neighbors baseline, then fixes its scale sensitivity with a pipeline.

## What the project demonstrates

- Data inspection with pandas-backed scikit-learn data
- A stratified train/test split
- Baseline-first experimentation
- Feature standardization with `StandardScaler`
- A leakage-safe `make_pipeline` workflow
- Holdout accuracy and five-fold cross-validation
- Per-class precision, recall, and F1 scores
- Confusion-matrix error analysis
- Reproducible dependencies and Git history

## Approach

1. Load 178 samples with 13 numeric features and three target classes.
2. Reserve 30 percent of the data as an untouched stratified test set.
3. Fit a five-neighbor KNN baseline on the raw feature values.
4. Fit a second five-neighbor KNN model behind `StandardScaler` in one pipeline.
5. Compare holdout accuracy, inspect five-fold cross-validation, and review per-class errors.

## Run locally on Windows

Create and select the `.venv` environment described by the project guide, then run:

```powershell
python train.py
```

The script prints the observed metrics and opens a confusion matrix. The scores are computed at runtime rather than hard-coded into this README.

## Why the scaled pipeline is the defensible model

KNN uses distances between samples. When feature ranges differ substantially, large-magnitude features can dominate those distances. `StandardScaler` puts features on comparable scales, while the pipeline keeps scaling attached to each model fit instead of preprocessing the entire dataset in advance.

## Interview talking points

- Why a baseline should come before optimization
- Why the test set stays untouched until final evaluation
- Why preprocessing belongs inside a pipeline
- What accuracy hides and what the classification report reveals
- Why cross-validation is more persuasive than one split alone

## Next experiment

Tune the neighbor count with cross-validation on the training split, then evaluate the selected pipeline once on the untouched test set.