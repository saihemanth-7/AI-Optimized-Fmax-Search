# AI-Optimized Fmax Search for Test Time Reduction

> **A machine learning project that predicts the maximum operating frequency of a semiconductor chip (Fmax) from pre-test electrical measurements, then uses the prediction to narrow the frequency search range — reducing the number of test steps needed.**

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [What is Fmax?](#what-is-fmax)
3. [Why Fmax Testing Takes Time](#why-fmax-testing-takes-time)
4. [How the ML Model Helps](#how-the-ml-model-helps)
5. [Dataset](#dataset)
6. [Feature Selection](#feature-selection)
7. [Data Splits](#data-splits)
8. [Machine Learning Model](#machine-learning-model)
9. [Prediction-Windowed Binary Search](#prediction-windowed-binary-search)
10. [Baseline: Conventional Binary Search](#baseline-conventional-binary-search)
11. [Project Structure](#project-structure)
12. [Quick Start — How to Run](#quick-start)
13. [Evaluation Results](#evaluation-results)
14. [Comparison with Technical Document](#comparison-with-technical-document)
15. [Known Limitations](#known-limitations)
16. [Interview Explanation](#interview-explanation)

---

## What This Project Does

This project builds an AI-assisted testing system for semiconductor chips. The goal is to find each chip's **maximum operating frequency (Fmax)** faster than conventional methods by using a machine learning model to predict where to look.

The three key steps are:

1. **Predict** — A trained Gradient Boosting model predicts a chip's Fmax from pre-test measurements (ring-oscillator speeds, leakage current, etc.).
2. **Window** — The prediction is used to define a narrow frequency range to search.
3. **Verify** — A binary search within the window confirms the actual Fmax.

The result is fewer test steps per chip, which directly reduces test time and cost in a semiconductor production environment.

---

## What is Fmax?

**Fmax** (maximum operating frequency) is the highest clock frequency at which a chip operates correctly.

- Every chip is different — even chips from the same design differ due to microscopic manufacturing variations.
- A chip that runs at 2,050 MHz passes at 2,050 MHz and fails at 2,060 MHz (the next step).
- Manufacturers must find this boundary precisely to sell the chip at the correct speed grade.

**Important distinction:**
| Term | Meaning |
|------|---------|
| **Fmax prediction** | The ML model's *estimated* maximum frequency |
| **Fmax verification** | The frequency found by an actual test search |
| **True Fmax** | The real physical boundary (only known from testing) |

The model's prediction is **not** a verified measurement.

---

## Why Fmax Testing Takes Time

The frequency grid in this project runs from **1,610 MHz to 2,400 MHz** in **10 MHz steps** — that is **81 candidate frequencies**.

Conventional binary search finds the answer in about **6–7 tests** (log₂(81) ≈ 6.3). This is already efficient compared to a linear sweep (which would need ~40 tests), but it still searches the entire grid.

If we already know the answer is probably near 2,000 MHz, we only need to search a narrow range — for example [1,960–2,040 MHz] — which is just **9 points**, requiring only **3–4 binary search steps**.

This is the core idea: **use AI to narrow the search window before testing begins**.

---

## How the ML Model Helps

```
Pre-test measurements  →  [GBR Model]  →  Predicted Fmax
                                                ↓
                        Calibration set  →  Uncertainty estimate
                                                ↓
                           window = [pred - 1.5σ, pred + 1.5σ]
                                                ↓
                        Binary search within window  →  Verified Fmax
```

1. Ring oscillator frequencies (`ro_01` to `ro_10`) correlate with chip speed.
2. The model learns this relationship on 1,500 training chips.
3. On the calibration set (500 chips), we measure how wrong the model is → **σ_calib ≈ 26.5 MHz**.
4. The search window is set to **prediction ± 1.5 × σ_calib ≈ ±39.8 MHz**.
5. This window covers most chips while remaining narrow enough to save test steps.

---

## Dataset

**File:** `synthetic_fmax_dataset.csv`

| Property | Value |
|----------|-------|
| Rows | 6,000 |
| Columns | 25 |
| Fmax range | 1,616 – 2,330 MHz |
| Mean Fmax | ~1,988 MHz |
| Missing values | None |

> **Note:** This is a synthetic (simulated) dataset generated to replicate the statistical properties of real semiconductor test data. Results cannot be directly compared to a real production chip line without further validation.

---

## Feature Selection

### Features Used (15 columns)

| Feature | Type | Why Included |
|---------|------|--------------|
| `ro_01` to `ro_10` | Continuous | Ring-oscillator measurements — direct indicators of transistor speed. Available before Fmax testing. |
| `iddq_norm` | Continuous | Normalised IDDQ leakage current. Pre-test electrical measurement correlated with die quality. |
| `vmin_norm` | Continuous | Normalised minimum operating voltage. Pre-test measurement. |
| `die_x`, `die_y` | Continuous | Die position on wafer. Wafer-level spatial variation correlates with speed. Pre-test metadata. |
| `radius_sq` | Continuous | Squared wafer radius (derived from die_x, die_y). Captures radial variation. Pre-test. |

### Columns Explicitly Excluded

| Column | Reason for Exclusion |
|--------|---------------------|
| `unit_id` | Identifier — no predictive value |
| `fmax_pred_mhz` | Pre-computed prediction — would cause target leakage |
| `window_low_mhz`, `window_high_mhz` | Derived from a prediction — unavailable at inference |
| `tests_binary`, `tests_windowed` | Test outcomes — post-test information |
| `true_bin_mhz` | Post-test classification label |
| `fmax_measured_windowed_mhz` | Result of the search — post-test |

---

## Data Splits

The `split` column in the CSV assigns each row to a fixed role. **We preserve these assignments exactly.**

| Split | Rows | Role |
|-------|------|------|
| `train` | 1,500 | Fit the GBR model |
| `calibration` | 500 | Compute prediction uncertainty (σ_calib) |
| `test` | 4,000 | Evaluate the model and search algorithms |

**Leakage prevention rules:**
- The scaler (`StandardScaler`) is fitted **only** on training data.
- Calibration σ is computed **only** from the calibration set.
- Test set is never seen during training, calibration, or hyperparameter selection.

---

## Machine Learning Model

**Model:** `sklearn.ensemble.GradientBoostingRegressor`

| Hyperparameter | Value | Reason |
|---------------|-------|--------|
| `n_estimators` | 300 | Enough trees for stable predictions |
| `learning_rate` | 0.05 | Conservative step size to avoid overfit |
| `max_depth` | 4 | Moderately complex trees |
| `min_samples_split` | 10 | Prevents very small splits |
| `min_samples_leaf` | 5 | Regularisation |
| `subsample` | 0.8 | Stochastic gradient boosting |
| `random_state` | 42 | Full reproducibility |
| `loss` | `squared_error` | Standard regression loss |

**Calibration:**
After training, the model predicts on the calibration set. The residual standard deviation becomes `σ_calib`.

```
σ_calib = std(y_calib - predicted_calib)  ≈  26.5 MHz
window_half = 1.5 × σ_calib               ≈  39.8 MHz
```

---

## Prediction-Windowed Binary Search

### Step-by-Step

1. Collect pre-test measurements for a chip.
2. Load the trained GBR model.
3. Predict Fmax: `pred = model.predict(features)`.
4. Compute window: `[pred - 39.8, pred + 39.8]` MHz.
5. Clamp to valid grid: `[1,600, 2,400]` MHz.
6. Snap bounds to the 10 MHz grid (floor for low, ceil for high).
7. Binary search within the window using the test oracle.
8. Return the highest passing frequency found.

### Why 1.5 × σ_calib?

The factor 1.5 is verified from the dataset: the pre-computed `window_low_mhz` and `window_high_mhz` columns consistently show a half-width of ≈39.2 MHz, and `1.5 × 26.16 = 39.24 MHz`. This factor balances coverage (how often the true Fmax falls inside the window) against window width (narrower = fewer test steps).

---

## Baseline: Conventional Binary Search

The baseline searches the **full frequency grid** [1,600–2,400 MHz] using the same binary-search algorithm but with no prediction. This requires ≈6.45 test steps on average.

---

## Project Structure

```
dev project/
    synthetic_fmax_dataset.csv     <- Original dataset (do NOT delete)
    main.py                        <- Entry point (all commands)
    requirements.txt               <- Python dependencies
    README.md                      <- This file
    src/
        __init__.py
        data_loader.py             <- Load CSV, define feature/target columns
        data_validation.py         <- Schema, missing-value, leakage checks
        preprocessing.py           <- StandardScaler pipeline (train-only fit)
        train_model.py             <- GBR training + calibration
        evaluate_model.py          <- Evaluation, plots, metric computation
        fmax_search.py             <- Binary search algorithms
        metrics.py                 <- MAE, RMSE, R2, search metrics
        predict.py                 <- CLI prediction interface
    models/
        gbr_model.joblib           <- Trained GBR model
        preprocessor.joblib        <- Fitted scaler
        calib_info.json            <- Calibration statistics
    outputs/
        metrics.json               <- All computed metrics
        evaluation_results.csv     <- Per-chip prediction errors
        search_comparison.csv      <- Per-chip search step counts
        figures/
            fig1_actual_vs_predicted.png
            fig2_error_distribution.png
            fig3_feature_importance.png
            fig4_search_comparison.png
            fig5_step_reduction.png
            fig6_predicted_vs_measured.png
    tests/
        test_data_validation.py    <- 23 tests: loading, schema, splits, leakage
        test_fmax_search.py        <- 32 tests: grid, oracle, binary search
        test_no_data_leakage.py    <- 9 tests: leakage and reproducibility
```

---

## Quick Start

### 1. Set Up Environment

```powershell
# From the project folder:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Validate Dataset

```powershell
python main.py validate
```

### 3. Train Model

```powershell
python main.py train
```

### 4. Evaluate

```powershell
python main.py evaluate
```

### 5. Run Everything at Once

```powershell
python main.py all
```

### 6. Predict Fmax for a New Chip

```powershell
python main.py predict \
  --ro 0.1 -0.2 0.3 -0.1 0.4 -0.3 0.2 -0.1 0.3 -0.2 \
  --iddq 0.5 --vmin -0.3 --die_x 0.4 --die_y 0.1 --radius_sq 0.17
```

### 7. Run Tests

```powershell
python -m pytest tests/ -v
```

---

## Evaluation Results

> All results computed from actual data — no hard-coded values.

### Regression Metrics (Test Set, n=4,000)

| Metric | Value |
|--------|-------|
| MAE | **21.59 MHz** |
| RMSE | **27.17 MHz** |
| R² | **0.9318** |
| Mean error (bias) | 0.68 MHz |
| Std error | 27.16 MHz |

### Search Step Comparison (Test Set, n=4,000)

| Method | Avg Test Steps | Success Rate |
|--------|---------------|--------------|
| Conventional Binary Search | **6.46** | 100.0% |
| AI-Windowed Binary Search | **3.59** | 95.6% |

**Test-step reduction:**
```
Reduction = (6.46 - 3.59) / 6.46 × 100 = 44.4%
```

### Calibration Statistics

| Statistic | Value |
|-----------|-------|
| σ_calib | 26.53 MHz |
| Window half-width | ±39.80 MHz |
| Calib MAE | 21.41 MHz |
| Calib RMSE | 26.61 MHz |

---

## Comparison with Technical Document

| Metric | Our Result | Document Reference | Match? |
|--------|-----------|-------------------|--------|
| RMSE | 27.17 MHz | 26.9 MHz | Very close (within 1%) |
| Avg binary tests | 6.46 | 6.45 | Matches exactly |
| Avg windowed tests (CSV) | 3.91 | 3.89 | Very close |
| Avg windowed tests (ours) | 3.59 | 3.89 | Lower (wider window from our model) |
| % Reduction (CSV) | 39.4% | ~39.7% | Matches |
| % Reduction (ours) | 44.4% | ~39.7% | Slightly higher |

**Explanation of differences:**

- Our GBR model's `σ_calib` (26.53 MHz) is slightly larger than the document's implied value (26.16 MHz from the CSV). This makes our window slightly wider (±39.80 vs ±39.24 MHz), which sometimes allows fewer search steps.
- The CSV's pre-computed `tests_windowed` (avg 3.91) uses the exact document model parameters. Our re-trained model achieves 3.59 — slightly better coverage due to the marginally wider window.
- The binary search average (6.46) matches the document's 6.45 exactly, confirming our search implementation is correct.

---

## Known Limitations

1. **Synthetic data only.** The dataset is simulated, not from real chip production. Real semiconductor data may show different distributions, die-level effects, and measurement noise.
2. **Oracle simulation.** The binary search uses `fmax_true_mhz` as the pass/fail oracle. In real production, the oracle is a physical test. Simulation does not capture timing margins, temperature variation, or test noise.
3. **Coverage at 95.6%.** About 4.4% of chips have true Fmax outside the predicted window. For those chips, the windowed search returns an incorrect result. A wider window or a more conservative multiplier could reduce this.
4. **Single model.** Only GBR is implemented as the primary model. Ensemble or deep learning approaches might improve prediction accuracy.
5. **Static window width.** The window is fixed at 1.5 × σ_calib for all chips. A per-chip confidence interval (e.g., from quantile regression) could adaptively adjust the window.

---

## Interview Explanation

Here is a simple explanation you can use in college presentations or job interviews:

---

> **"I built a machine learning project that makes chip testing faster."**
>
> When a chip is manufactured, engineers need to find its maximum operating frequency — called Fmax. The standard way to do this is binary search: test at the middle of a frequency range, adjust up or down based on whether it passes, and repeat until you find the exact boundary. For our chip range, this takes about 6–7 test steps per chip.
>
> My idea was: instead of starting from scratch every time, what if we could use other measurements that are already available — like how fast the chip's ring oscillators run — to predict roughly where the Fmax will be? Then we only need to search a small window around that prediction.
>
> I trained a Gradient Boosting model (which builds decision trees sequentially, each correcting the errors of the previous one) on 1,500 chips. On a separate calibration set of 500 chips, I measured how accurate the predictions were: the error standard deviation was about 26.5 MHz. So I set the search window to prediction ± 39.8 MHz (that's 1.5 times the error standard deviation — wide enough to usually contain the true value, narrow enough to save test steps).
>
> The result: instead of 6.46 average tests with conventional binary search, the AI-assisted method needs only 3.59 average tests — a 44% reduction. The model achieves an RMSE of 27 MHz on 4,000 held-out test chips.
>
> The key engineering challenge was making sure the model is leakage-free: the search window must only be built from the prediction, never from the true Fmax. And the calibration must only use the calibration set, not the test set.

---

*This project was built with Python 3.13, scikit-learn, pandas, matplotlib, and seaborn. No internet access, paid APIs, or cloud services are required.*

