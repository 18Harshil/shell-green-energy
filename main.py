"""
Shell Green Energy Project — AI for Sustainability
Author: Harshil Parmar
Description: ML pipeline to analyze renewable energy datasets,
             identify performance trends, and predict energy output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ─── 1. DATA LOADING ──────────────────────────────────────────────────────────

def load_data(filepath: str = "data/energy_dataset.csv") -> pd.DataFrame:
    """Load and return the renewable energy dataset."""
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except FileNotFoundError:
        print("[INFO] Dataset not found. Generating synthetic dataset for demo...")
        df = generate_synthetic_data()
    return df


def generate_synthetic_data(n: int = 100_000) -> pd.DataFrame:
    """Generate a realistic synthetic renewable energy dataset."""
    np.random.seed(42)
    df = pd.DataFrame({
        "timestamp":          pd.date_range("2020-01-01", periods=n, freq="h"),
        "solar_irradiance":   np.random.uniform(0, 1200, n),       # W/m²
        "wind_speed":         np.random.uniform(0, 25, n),          # m/s
        "temperature":        np.random.uniform(-5, 45, n),         # °C
        "humidity":           np.random.uniform(10, 100, n),        # %
        "cloud_cover":        np.random.uniform(0, 100, n),         # %
        "panel_efficiency":   np.random.uniform(0.15, 0.22, n),     # ratio
        "turbine_capacity":   np.random.choice([1.5, 2.0, 2.5, 3.0], n),  # MW
        "grid_demand":        np.random.uniform(50, 500, n),        # MWh
        "maintenance_flag":   np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "region":             np.random.choice(["North","South","East","West"], n),
    })

    # Synthetic target: energy output (MWh)
    df["energy_output"] = (
        0.4 * df["solar_irradiance"] * df["panel_efficiency"] * 0.01 +
        0.3 * df["wind_speed"] ** 2 * df["turbine_capacity"] * 0.002 +
        np.random.normal(0, 10, n)
    ).clip(0)

    print(f"[INFO] Generated synthetic dataset: {df.shape}")
    return df


# ─── 2. PREPROCESSING ─────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """Feature engineering, encoding, and train/test split."""
    df = df.copy()

    # Temporal features
    if "timestamp" in df.columns:
        df["hour"]  = pd.to_datetime(df["timestamp"]).dt.hour
        df["month"] = pd.to_datetime(df["timestamp"]).dt.month
        df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
        df.drop(columns=["timestamp"], inplace=True)

    # Encode categoricals
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Drop target
    X = df.drop(columns=["energy_output"])
    y = df["energy_output"]

    print(f"[INFO] Features: {X.shape[1]}   |   Target: energy_output")
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ─── 3. MODEL TRAINING ────────────────────────────────────────────────────────

def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """Train multiple models and return results."""
    models = {
        "Ridge Regression":          Ridge(alpha=1.0),
        "Random Forest":             RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":         GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    }

    results = {}
    for name, m in models.items():
        pipe = build_pipeline(m)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2  = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae  = mean_absolute_error(y_test, preds)
        results[name] = {"model": pipe, "r2": r2, "rmse": rmse, "mae": mae, "preds": preds}
        print(f"  [{name}]  R²={r2:.4f}   RMSE={rmse:.2f}   MAE={mae:.2f}")

    return results


# ─── 4. FEATURE IMPORTANCE ────────────────────────────────────────────────────

def plot_feature_importance(results: dict, feature_names):
    rf = results["Random Forest"]["model"]["model"]
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    importances.head(10).plot(kind="bar", color="#1A237E")
    plt.title("Top 10 Feature Importances — Random Forest", fontsize=14, fontweight="bold")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("[INFO] Saved: outputs/feature_importance.png")


def plot_predictions(results: dict, y_test):
    best = max(results, key=lambda k: results[k]["r2"])
    preds = results[best]["preds"]

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test[:2000], preds[:2000], alpha=0.3, s=10, color="#1565C0")
    lim = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lim, lim, "r--", linewidth=1.5)
    plt.xlabel("Actual Energy Output (MWh)")
    plt.ylabel("Predicted Energy Output (MWh)")
    plt.title(f"Predicted vs Actual — {best}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/predictions.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved: outputs/predictions.png  (Best model: {best})")


# ─── 5. HYPERPARAMETER TUNING ─────────────────────────────────────────────────

def tune_random_forest(X_train, y_train):
    """Grid search for best Random Forest params."""
    print("\n[INFO] Running hyperparameter tuning (this may take a moment)...")
    param_grid = {
        "model__n_estimators":  [100, 200],
        "model__max_depth":     [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }
    pipe = build_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1))
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    print(f"[INFO] Best params: {gs.best_params_}   |   Best CV R²: {gs.best_score_:.4f}")
    return gs.best_estimator_


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("  Shell Green Energy — AI for Sustainability")
    print("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    print("\n[STEP 1] Training baseline models...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n[STEP 2] Plotting results...")
    plot_feature_importance(results, X_train.columns.tolist())
    plot_predictions(results, y_test)

    print("\n[STEP 3] Hyperparameter tuning on best model...")
    best_model = tune_random_forest(X_train, y_train)
    final_preds = best_model.predict(X_test)
    final_r2 = r2_score(y_test, final_preds)
    print(f"\n[RESULT] Tuned model R²: {final_r2:.4f}  ({final_r2*100:.1f}% accuracy)")
    print("\n[DONE] All outputs saved to /outputs/")
