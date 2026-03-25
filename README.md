# Batch_13_final_year_project
A Workload-Aware Machine Learning Framework for Cloud Cost Waste Detection and Predictive Optimization in Hyperscale Clusters
# Waste Detection Module & Workload-Aware Prediction Layer

## Overview
This project implements a comprehensive **Waste Detection Module and Workload-Aware Prediction Layer** for analyzing and predicting resource utilization inefficiencies in cloud computing environments. Designed to work with the **Google Borg Traces Dataset**, it specifically quantifies and categorizes resource waste into actionable metrics (Idle, Over-Provision, and Failure waste). It also incorporates machine learning pipelines to predict future resource states and waste behavior, enabling administrators to dynamically optimize cloud infrastructure investments.

## Core Components
- **Data Ingestion & Parsing (`load_borg_traces`)**: Automatically loads, sanitizes, and parses raw Google Borg traces. It handles complex dict-encoded string attributes and relative microsecond timestamps, synthesizing them into structured feature vectors.
- **Waste Score Evaluation (`calculate_waste_score`)**: Computes a continuous, composite WasteScore bounded between [0, 1] factoring three primary inefficiency profiles:
  - *Idle Waste*: Identification of jobs possessing negligible usage relative to requested allocation.
  - *Over-Provision Waste*: Capturing the gap between requested computational allocation and actual peak limits.
  - *Failure Waste*: Accounting for resources successfully exhausted by jobs that ultimately failed.
- **XGBoost Prediction Pipeline (`train_xgboost_predictor`)**: Deploys `XGBRegressor` to forecast future continuous representations of CPU Utilization and WasteScores. Also implements a composite Random Forest and dual XGBoost soft-weighted classifier ensemble (`train_precision_recall_ensemble`) for binary classification boundaries utilizing SMOTETomek class-imbalance corrections.
- **Workload-Aware Clustering (`cluster_workloads`)**: Unsupervised mapping via $k$-Means clustering over multi-dimensional feature sets (CPU usages, lengths, allocations) mapping job structures to specific cluster profiles.
- **Statistical Analytics Engine (`bootstrap_ci`)**: Computes non-parametric Bootstrap Confidence Intervals (CI) to offer robust confidence bounds.
- **Automated Visualization Sequence (`generate_visualizations`)**: Programmatically renders 9 professional-grade statistical charts (PDFs). Key plots include:
  1. Waste Category Breakdowns
  2. WasteScore KDE & Distribution
  3. Model Performance & CIs
  4. Actual vs Predicted Scatterplots (CPU Utilization and WasteScore)
  5. Residual Error Density Summaries
  6. Timeseries Comparative Forecasts
  7. Cross-Model Top-5 Feature Importance Bars

## Dependencies
Ensure that your Python ecosystem includes the vital prerequisites arrayed below prior to execution:
```txt
pandas
numpy
xgboost
scikit-learn
scipy
matplotlib
seaborn
imbalanced-learn (optional, used for SMOTE and Tomek link strategies)
```

## Directory Structure
- `new_waste_2.py`: The root execution script containing pipeline configurations, dataset logic, ML training logic, and plotting definitions.
- `borg_traces_data.csv/borg_traces_data.csv`: The targeted input data (expected Google Borg traces format representation).
- `images/` / `plots/`: Output directories structured for local artifact storage (metrics mapping, pdf outputs).

## Usage
Simply launch the core execution script through the command line or compatible interactive Python environment:

```bash
python new_waste_2.py
```

The application's console loop outputs detailed logging surrounding sampling fractions, data filters, generated category thresholds, evaluation boundaries (e.g., MAE, RMSE, $R^2$), and dataset geometries. Upon execution completion, plots are autonomously exported identically scaling standard Windows default viewer applications for rapid analytic interpretations.

