from pathlib import Path
import joblib
import pandas as pd

base = Path(__file__).parent
print("Project folder:", base)

# 1) files exist?
needed = [
    base / "data" / "Coffee Shop Sales.xlsx",
    base / "data" / "product_catalog.xlsx",
    base / "models" / "hourly_models.pkl",
    base / "models" / "daily_model.pkl",
    base / "models" / "features_hourly.pkl",
    base / "models" / "features_daily.pkl",
]
for f in needed:
    print("found  " if f.exists() else "MISSING", f.name)

# 2) load models and feature tables
hourly_models = joblib.load(base / "models" / "hourly_models.pkl")
daily_model = joblib.load(base / "models" / "daily_model.pkl")
hourly = pd.read_pickle(base / "models" / "features_hourly.pkl")
daily = pd.read_pickle(base / "models" / "features_daily.pkl")
catalog = pd.read_excel(base / "data" / "product_catalog.xlsx")
print("hourly rows:", len(hourly), "| daily rows:", len(daily), "| products in catalog:", len(catalog))

# 3) one test prediction
feats = hourly_models["target_2h"].feature_name_
row = hourly.dropna(subset=feats).tail(1)
pred = hourly_models["target_2h"].predict(row[feats])[0]
print("test prediction (2h):", round(float(pred), 2))
print("ALL GOOD")
