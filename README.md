# House Price Prediction (Ames Housing)

Predicting house prices in Ames, Iowa with regression models, plus a full EDA, model comparison, and a simple buyer-suggestion tool.

## Problem Statement

1. **Task 1:** Complete data analysis of the dataset
2. **Task 2a:** Build an ML model to predict house prices
3. **Task 2b:** Find which features drive price
4. **Task 3:** Suggest houses to buyers based on budget, area, and requirements

## Dataset

Ames Housing: 1460 houses, 79 features, target = `SalePrice`.

## Approach

- **Missing values:** Most NaNs meant "feature not present" (no pool, no alley, no garage), so I filled them with `'None'` or `0` instead of dropping. `LotFrontage` was filled with the median of each neighborhood.
- **Target:** `SalePrice` was right-skewed (1.88). A `log1p` transform brought it to 0.12. Predictions are converted back with `expm1`.
- **Outliers:** Removed 2 huge-but-cheap houses (`GrLivArea > 4000`, `SalePrice < 300000`).
- **Encoding:** Ordinal mapping for quality ratings, one-hot for the rest.
- **Feature engineering:** `TotalSF`, `TotalBath`, `HouseAge`, `Remodeled`.
- **Modelling:** 80/20 split, Linear Regression, Random Forest, Gradient Boosting, tuned with GridSearchCV. Target columns were dropped from the inputs to avoid data leakage.

## Results

| Model | RMSE (log) | R2 | Avg error ($) |
|---|---|---|---|
| Gradient Boosting | 0.1224 | 0.911 | 14,883 |
| Linear Regression | 0.1305 | 0.899 | 15,570 |
| Random Forest | 0.1454 | 0.875 | 16,595 |
| Gradient Boosting (tuned + new features) | 0.1227 | 0.911 | 15,264 |

**Best model: Gradient Boosting**, with an average error of about $15k per house. Tuning and the new features gave no meaningful improvement.

## Key Findings

- **Most correlated with price:** TotalSF (0.83), OverallQual (0.80), GrLivArea (0.73)
- **Top feature importances:** OverallQual (~37%), GrLivArea (~17%), TotalBsmtSF (~5%)
- Build quality and living area matter most.

## Buyer Suggestions

Compares each house's actual price with the model's predicted price. Houses priced well below prediction are flagged as good deals.

```python
suggest(budget=200000, neighborhood='NAmes', min_bed=3)
```

Returns the 5 best deals within the budget.

> The model saw most of these houses in training, so the savings are approximate.

## Challenges

| Challenge | Technique | Why |
|---|---|---|
| NaNs meaning "no feature" | Fill with `'None'` / 0 | Absence carries information |
| Right-skewed price | `log1p` / `expm1` | Skew 1.88 to 0.12 |
| Extreme outliers | Removed 2 rows | Unusual sales distorted the fit |
| 79 mixed-type features | Ordinal map + one-hot | Models need numbers |
| Data leakage risk | Dropped target columns from X | Model must not see the answer |
| Small gains from tuning | Kept the simpler setup | Gain was negligible |

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook

## How to Run

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Open the notebook and run **Kernel > Restart & Run All**.

## Author

Mayank, Bengaluru
