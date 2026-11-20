# Unicorn Companies Data Analysis

An end-to-end data science pipeline on 1,037 privately held startups valued at $1B+ ("unicorns"): data cleaning, statistical testing, multi-level exploratory analysis, and machine learning, all in Python.

## Key findings

- **Typical valuation:** geometric mean about $2.0B (95% CI $1.96B to $2.15B). The arithmetic mean ($3.29B) is inflated by a few giants (skew 11.6).
- **Industry does not explain valuation:** ANOVA p = 0.79, eta-squared = 0.009, and 0 of 105 Tukey pairs differ.
- **US vs non-US:** statistically significant (p = 0.03) but a tiny effect (Cohen's d = 0.14).
- **Exit rates differ by industry:** permutation test p = 0.023, Cramer's V = 0.16. Mobile & telecom is highest (13%), and Health, Cybersecurity, Data management and Auto have no exits.
- **Funding is the main valuation driver:** Spearman 0.57 with Total Raised, and 61% of Random Forest feature importance.
- **Valuation model (log scale, 5-fold CV R2):** Linear 0.36, Random Forest 0.45, XGBoost 0.47. Random Forest and XGBoost are within noise of each other.
- **Exit classifier (49 exits out of 1,037):** CV AUC 0.75 to 0.83 and CV average precision 0.23 to 0.29 (chance = 0.047). At a CV-tuned threshold, Random Forest reaches test recall 0.50 and precision 0.24.

## Data cleaning notes

- 16 rows had shifted columns (City held the industry, Industry held the investors). They were repaired, not dropped.
- Industry spelling variants were merged (15 industries).
- Valuation and Total Raised were parsed from text (`$1.2B`, `$143M`) into numbers.
- Impossible founding years were set to missing, and Total Raised under $5M was treated as unreliable in the ML notebook.
- `Financial Stage` is 95% empty and holds only exit events, so the 3-class stage model was replaced by a binary **Exited vs Private** target.

## Project structure

```
data/                 raw and cleaned CSVs
notebooks/
  01_EDA.ipynb
  02_Statistical_Analysis.ipynb
  03_Univariate_Bivariate_Multivariate.ipynb
  04_ML_Analysis.ipynb
results/plots/        saved figures
results/models/       saved models (not tracked by Git)
requirements.txt
```

## Run it

```bash
python -m venv venv
venv\Scripts\activate        # Windows (use source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
```

Open the notebooks in VS Code or Jupyter and run them in order, starting with `01_EDA.ipynb`, which produces the cleaned dataset.

## Limitations

- **Survivorship bias:** only companies that reached $1B are included.
- **Snapshot data:** valuations are from a single point in time.
- **Class imbalance:** only 49 exits, so classifier metrics are noisy and the test set has just 10 exits.
- **Possible leakage:** valuation is used to predict exit, and exited companies may carry their exit-time valuation.
- Results show association, not causation.

## Tech stack

pandas, numpy, matplotlib, seaborn, scipy, statsmodels, scikit-learn, xgboost, joblib

## License

MIT. The dataset follows its original source terms (Kaggle: Unicorn Companies Dataset).