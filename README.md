House Price Prediction (Ames Housing)

Predicting residential house prices in Ames, Iowa using regression models, with a full EDA, model comparison, and a simple buyer-suggestion tool.

Project ID: PRCP-1020-HousePricePred

Problem Statement
Task 1: Complete data analysis report on the dataset.
Task 2a: Build a robust ML model to predict house prices.
Task 2b: Find which features drive price and how.
Task 3: Suggest houses to buyers based on area, budget, and requirements.

Also included: a Model Comparison Report and a Challenges Report (both inside the notebook).

Dataset
Ames Housing dataset: 1460 houses, 79 features, target = SalePrice
Covers lot, building, quality, basement, garage, porch, sale details, and more
Approach
1. Data cleaning
Most missing values (PoolQC, Alley, Fence, MiscFeature, FireplaceQu, Garage*, Bsmt*) meant the house doesn't have that feature, so I filled them with 'None' (categorical) or 0 (numeric) instead of dropping them.
Electrical (1 row) filled with the mode.
LotFrontage filled with the median of each Neighborhood.
2. Target transformation
SalePrice was right-skewed (skew = 1.88). Applied np.log1p, which reduced skew to 0.12.
Predictions are converted back to dollars using np.expm1.
3. Outliers
Removed 2 houses with GrLivArea > 4000 and SalePrice < 300000 (unusual sales that distort the fit). 1460 rows became 1458.
4. Encoding
Ordered ratings (Po, Fa, TA, Gd, Ex) mapped to 0 to 5.
Remaining categorical columns one-hot encoded.
5. Feature engineering
TotalSF, TotalBath, HouseAge, Remodeled
6. Modelling
80/20 train/test split (random_state=42)
Dropped both SalePrice and SalePrice_log from the inputs to avoid data leakage
Models: Linear Regression, Random Forest, Gradient Boosting
Hyperparameter tuning of Gradient Boosting with GridSearchCV
Results
Model	RMSE (log)	R2	Avg error ($)
Gradient Boosting	0.1224	0.911	14,883
Linear Regression	0.1305	0.899	15,570
Random Forest	0.1454	0.875	16,595
Gradient Boosting (tuned + new features)	0.1227	0.911	15,264

Recommended model: Gradient Boosting. It has the lowest error and predicts price within about $15k on average. Tuning and engineered features gave no meaningful improvement, so the model had already captured most of the signal.

Key Findings
Most correlated with price: TotalSF (0.83), OverallQual (0.80), GrLivArea (0.73)
Top feature importances (Gradient Boosting): OverallQual (~37%), GrLivArea (~17%), TotalBsmtSF (~5%)
Build quality and living area matter most. Basement size, garage area, and kitchen quality come next.
Buyer Suggestions (Task 3)

Compares each house's actual price with the model's predicted price. Houses priced well below their prediction are flagged as good deals.

python
suggest(budget=200000, neighborhood='NAmes', min_bed=3)

Returns the top 5 houses within budget, ranked by the gap between predicted and actual price.

Note: the model was trained on most of these houses, so the estimated savings are approximate.

Challenges
Challenge	Technique	Why
NaNs that meant "no feature"	Fill with 'None' / 0	Absence carries information
LotFrontage truly missing	Neighborhood-wise median	Nearby houses have similar frontage
Right-skewed price	log1p / expm1	Skew 1.88 to 0.12
Extreme outliers	Removed 2 rows	Unusual sales distorted the fit
79 mixed-type features	Ordinal map + one-hot	Models need numbers
Data leakage risk	Dropped target columns from X	Model must not see the answer
Small gains from model changes	Feature engineering + tuning	Gain was negligible, so keep it simple
Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook

How to Run
bash
git clone <your-repo-url>
cd <repo-folder>
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook

Open the notebook and run Kernel > Restart & Run All. Place train.csv in the same folder.

Repository Structure
.
├── House_Price_Prediction.ipynb   # full analysis, models, reports
├── train.csv                      # dataset
└── README.md

Author
Mayank
