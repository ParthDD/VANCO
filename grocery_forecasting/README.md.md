#  Grocery Sales Forecasting Pipeline

An enterprise-grade, modular machine learning pipeline built to predict retail sales for a multi-category grocery retailer. This architecture is designed to prevent data leakage over a strict multi-day inference horizon while dynamically benchmarking multiple gradient boosting frameworks (**LightGBM** and **XGBoost**).

The project is driven by a decoupled configuration pattern via `config.yaml`, enabling seamless hyperparameter experimentation and model framework switching without altering core data engineering logic.

---

##  Architecture & Component Design

The repository is structured to separate exploratory sandboxing, data engineering pipelines, validation testbeds, and production deployment scripts.

```
D:\Parth\vanco-solution-architecture\grocery_forecasting\
├── config.yaml                          # Centralized configuration orchestrator (paths, validation splits, params)
├── data_processor.py                    # In-memory ETL, domain-feature engineering, & leakage-proof lag engine
├── train_comparison.py                  # Laboratory script evaluating metric optimization trade-offs
├── generate_submission.py               # Production deployment engine generating final test inference
├── notebooks/grocery_forecasting.ipynb  # Interactive exploratory data analysis (EDA) and visualization sandbox
└── requirements.txt                     # Explicit tracking file for system dependencies
```

---

##  Execution Sequence & Data Flow

Data processing flows through a hierarchical, bottom-up sequence. The core pipelines are completely modular; top-level drivers automatically orchestrate downstream scripts.

### Workflow A: The Experimentation Loop (When refining features or tuning parameters)
```
[1. grocery_forecasting.ipynb] ──> [2. data_processor.py] ──> [3. train_comparison.py]
  (Explore raw distributions)       (Codify clean features)     (Evaluate cross-metric matrix)
```

### Workflow B: The Production Pipeline (When deploying for submission output)
```
[1. config.yaml] ───────────────> [2. generate_submission.py] ────────────> [3. submission.csv]
  (Lock hyperparameters & paths)    (Orchestrates data pipeline,             (Final 28,512-row 
                                     trains optimal model, inverts logs)       inference asset)
```

---

##  Core Engineering Highlights

### 1. Leakage-Proof Lag Engine
Because the test horizon covers a **16-day future window**, standard `lag_1` features create immediate data leakage during inference (the true sales for day 1 of the test set are unknown when predicting day 2). 

To ensure complete stability across the entire forecasting profile, our feature engine utilizes a **16-day blind spot gap**. All historical anchors (`lag_16`, `lag_17`, `lag_18`, `lag_21`, `lag_28`) and rolling statistics (7-day moving average and standard deviation) are locked behind this protective boundary.

### 2. Domain-Specific Feature Extraction
The dataset is enriched with explicit retail domain indicators derived from macroeconomic and regional constraints:
* **Bi-Weekly Payday Flags:** Captures institutional liquidity shocks by signaling the 15th and the final day of each calendar month.
* **Macroeconomic Oil Imputation:** Bridges physical oil market gaps (weekend/holiday closures) by propagating prices forward across a continuous calendar grid using customized `.ffill().bfill()` tracking.
* **Refined Holiday Profile:** Mitigates training noise by dynamically stripping out officially transferred holidays that behaved as typical retail workdays.
* **Earthquake Recovery Shock Window:** Maps consumer demand spikes using a localized binary window following the April 16, 2016 seismic event.

---

##  Empirical Benchmarking Results

Our offline laboratory cross-evaluated models trained directly on **Raw Unit Scales (minimizing RMSE)** against models trained on a **Log-Transformed Target Space (minimizing RMSLE)** across a 16-day validation split (`2017-08-01` to `2017-08-15`).

| Evaluation Metric Tracked | Model A (Raw Scale Optimizer) | Model B (Log Space Optimizer) |
| :--- | :---: | :---: |
| **Validation RMSLE** | `1.36378` | **`0.42024` (Champion)** |
| **Validation MAE (Units)** | `84.33 units` | **`81.52 units`** |
| **Validation RMSE (Units)** | **`275.30 units`** | `288.08 units` |

### Architectural Insights:
1. **Neutralization of Big Box Bias:** Model A (Raw Optimizer) became deeply obsessed with reducing errors at flagship, high-volume supermarkets, making massive percentage errors across smaller product families (resulting in a broken RMSLE of `1.36`). Model B treats percentage errors symmetrically across all 1,782 distinct time-series combinations.
2. **Mitigation of Target Noise:** Raw sales values are highly volatile. Model A experienced severe variance disruption and early-stopped at **iteration 69**. Log-transforming the target ($\ln(	ext{sales} + 1)$) stabilized variance, allowing LightGBM to train deeper to **iteration 998**, discovering fine-grained weekly trends and lowering the average unit error (**MAE**) by nearly 3 whole units globally.

---

## Quick Start & Deployment Guide

### 1. Environment Initialization
Install all required operational dependencies using the centralized package manager tracking file:
```powershell
pip install -r requirements.txt
```

### 2. Run the Diagnostic Lab Comparison
To evaluate metric optimization trade-offs and display validation profiles on your local holdout slice:
```powershell
python train_comparison.py
```

### 3. Generate Production Inferences
To execute the production pipeline, train the champion log-space architecture, run inference across the blind test matrix, and write your final Kaggle-ready output:
```powershell
python generate_submission.py
```

The pipeline will automatically write a verified, formatted **28,512-row** file to the path specified in your config:
`D:\Parth\vanco-solution-architecture\grocery_forecasting\submission.csv`
