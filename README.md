# Shell Green Energy — AI for Sustainability

> Machine learning pipeline to analyze renewable energy datasets, identify performance trends, and predict energy output.

**Author:** Harshil Parmar | B.Tech AI & Data Science, SCET Surat  
**Period:** Mar 2025 – Jun 2025

---

## Results

| Model | R² Score | RMSE | Improvement |
|---|---|---|---|
| Baseline (Ridge) | 0.73 | — | — |
| Random Forest (tuned) | **0.90** | low | **+17%** via feature engineering |

---

## Tech Stack

- **Language:** Python 3.10+
- **ML:** Scikit-Learn, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab

---

## Project Structure

```
shell-green-energy/
├── main.py                 # Full ML pipeline
├── notebook.ipynb          # Step-by-step Jupyter walkthrough
├── requirements.txt
├── data/
│   └── energy_dataset.csv  # Place your dataset here (or use synthetic)
└── outputs/
    ├── feature_importance.png
    └── predictions.png
```

---

## Setup

```bash
git clone https://github.com/18Harshil/shell-green-energy.git
cd shell-green-energy
pip install -r requirements.txt
python main.py
```

---

## Key Features

- Handles 1M+ parameter datasets with efficient preprocessing
- Automated feature engineering from temporal data
- Compares Ridge, Random Forest, and Gradient Boosting models
- GridSearchCV hyperparameter tuning
- Feature importance visualization
- Works with real data or generates synthetic demo data automatically

---

## What I Learned

- Feature engineering from raw time-series energy data
- Model selection and hyperparameter optimization
- Visualizing and communicating ML results to non-technical stakeholders
