# 🌱 Shell Green Energy Project — AI for Sustainability

> Machine learning pipeline to analyze renewable energy datasets, identify performance trends,  
> and predict energy output — built entirely in **Jupyter Notebook**.

**Author:** Harshil Parmar | B.Tech AI & Data Science, SCET Surat  
**Period:** Mar 2025 – Jun 2025

---

## 📈 Results

| Stage | Model | Accuracy (R²) |
|---|---|---|
| Baseline | Ridge Regression | 73% |
| + Feature Engineering | Random Forest | ~85% |
| + Hyperparameter Tuning | Tuned Random Forest | **90%** |

**+17% improvement** through feature engineering on 1M+ parameter datasets.

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Scikit-Learn | ML models, GridSearchCV, pipelines |
| Pandas & NumPy | Data manipulation (1M+ parameters) |
| Matplotlib & Seaborn | Visualizations |
| Jupyter Notebook / Google Colab | Development environment |

---

## 📁 Project Structure

```
shell-green-energy/
├── Shell_Green_Energy_AI_Sustainability.ipynb   ← Main notebook (open this)
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Option 1 — Google Colab (recommended, no setup needed)
1. Upload `Shell_Green_Energy_AI_Sustainability.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Click **Runtime → Run All**

### Option 2 — Local Jupyter
```bash
git clone https://github.com/18Harshil/shell-green-energy.git
cd shell-green-energy
pip install -r requirements.txt
jupyter notebook Shell_Green_Energy_AI_Sustainability.ipynb
```

---

## 📓 Notebook Contents

| Section | What it does |
|---|---|
| 1. Imports & Setup | Libraries and environment check |
| 2. Data Loading & Exploration | Load dataset, shape, missing values |
| 3. EDA | Distribution plots, correlation heatmap, regional analysis |
| 4. Feature Engineering | Temporal features, interaction features — **biggest accuracy jump here** |
| 5. Preprocessing | Train/test split |
| 6. Baseline Models | Ridge, Random Forest, Gradient Boosting comparison |
| 7. Hyperparameter Tuning | GridSearchCV on Random Forest |
| 8. Final Evaluation | R², RMSE, MAE + predicted vs actual plot |
| 9. Feature Importance | Which features matter most |
| 10. Conclusion | Summary table + key takeaways |
