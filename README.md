# 🌱 Shell Green Energy Project — AI for Sustainability

> Analyzing **30 years of hourly renewable energy data** across **644 European NUTS-2 regions**
> using the EMHIRES dataset — predicting solar + wind capacity factors with ML.

**Author:** Harshil Parmar | B.Tech AI & Data Science, SCET Surat
**Duration:** Mar 2025 – Jun 2025

---

## 📈 Results

| Stage | Model | Accuracy (R²) |
|---|---|---|
| Baseline | Ridge Regression | **73%** |
| + Feature Engineering | Random Forest | ~85% |
| + Hyperparameter Tuning | Tuned Random Forest (GridSearchCV) | **90%** |

**+17% improvement** through feature engineering and hyperparameter optimization on a **169 million parameter dataset.**

---

## 📦 Dataset — EMHIRES (European Meteorological HIgh RES time series)

| Property | Value |
|---|---|
| **Source** | Joint Research Centre (JRC), European Commission |
| **Kaggle Link** | [kaggle.com/datasets/pythonafroz/wind-and-solar-power-generation-data](https://www.kaggle.com/datasets/pythonafroz/wind-and-solar-power-generation-data) |
| **Time Range** | 1986 – 2015 (30 years, hourly) |
| **Total Rows** | 262,968 hours |
| **Solar Columns** | ~290 NUTS-2 regions across EU-28 |
| **Wind Columns** | ~354 countries + bidding zones |
| **Total Parameters** | ~169 million |
| **Values** | Capacity factors (0 = no output, 1 = full capacity) |
| **License** | CC0 Public Domain |

### How to download the dataset

1. Go to: https://www.kaggle.com/datasets/pythonafroz/wind-and-solar-power-generation-data
2. Click **Download**
3. Extract and place the CSV files inside the `data/` folder:
   ```
   data/
   ├── EMHIRESPV_NUTS2_level.csv        ← Solar (290 NUTS-2 regions)
   └── EMHIRES_WIND_COUNTRY_June2019.csv ← Wind (28 countries)
   ```

> **Note:** If the files are not present, the notebook automatically generates
> realistic synthetic data with the same structure so you can still run everything end-to-end.

---

## 📁 Repository Structure

```
shell-green-energy/
├── Shell_Green_Energy_AI_Sustainability.ipynb  ← Main notebook (open this)
├── requirements.txt
├── README.md
├── data/
│   ├── EMHIRESPV_NUTS2_level.csv              ← Place downloaded file here
│   └── EMHIRES_WIND_COUNTRY_June2019.csv      ← Place downloaded file here
└── outputs/
    ├── 01_distributions.png
    ├── 02_seasonal_patterns.png
    ├── 03_diurnal_solar.png
    ├── 04_regional_performance.png
    ├── 05_annual_trend.png
    ├── 06_baseline_comparison.png
    ├── 07_predicted_vs_actual.png
    ├── 08_feature_importance.png
    └── 09_residuals.png
```

---

## 🚀 How to Run

### Option 1 — Google Colab (recommended, no setup needed)
1. Upload `Shell_Green_Energy_AI_Sustainability.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Upload the CSV files to the `data/` folder in Colab (or skip for synthetic demo)
3. Click **Runtime → Run All**

### Option 2 — Local Jupyter
```bash
git clone https://github.com/18Harshil/shell-green-energy.git
cd shell-green-energy
pip install -r requirements.txt
jupyter notebook Shell_Green_Energy_AI_Sustainability.ipynb
```

---

## 📓 Notebook Sections

| Section | What it does |
|---|---|
| 1. Dataset Overview & Loading | EMHIRES structure explanation, load real or synthetic data |
| 2. Exploratory Data Analysis | Capacity factor distributions, missing value check |
| 3. Regional Performance Analysis | Top/bottom NUTS-2 regions, 30-year annual trends, seasonal patterns |
| 4. Feature Engineering ⭐ | Lag features, rolling averages, temporal encodings, interaction features |
| 5. Preprocessing | Train/test split (80/20) |
| 6. Baseline Models (73%) | Ridge Regression, Random Forest, Gradient Boosting comparison |
| 7. Hyperparameter Tuning (90%) | GridSearchCV on Random Forest |
| 8. Final Evaluation | R², RMSE, MAE, predicted vs actual scatter plot |
| 9. Insights & Visualizations | Feature importance, residual analysis |
| 10. Conclusion | Summary table + key findings from EMHIRES data |

---

## 🔑 Key Findings

- **Solar and wind are complementary** — solar peaks in summer, wind peaks in winter
- **Lag features** (previous hour, previous day) are the strongest predictors
- **Regional diversity is large** — top NUTS-2 regions produce 3× more solar than bottom regions
- **Feature engineering** contributed more accuracy gain (~12%) than hyperparameter tuning (~5%)

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Scikit-Learn | ML models, GridSearchCV, pipelines |
| Pandas & NumPy | Data manipulation (169M+ parameters) |
| Matplotlib & Seaborn | 9 publication-quality visualizations |
| Jupyter Notebook / Google Colab | Development environment |

---

*Dataset: EMHIRES via Kaggle — CC0 Public Domain*
*JRC European Commission: Gonzalez-Aparicio et al. (2017)*
