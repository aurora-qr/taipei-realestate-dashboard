"""房價預測模型訓練模組。

訓練 Linear Regression 與 Random Forest 兩個模型，
比較 R²/RMSE/MAE，將表現較佳者存到 models/best_model.pkl。
"""

from __future__ import annotations

import json
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data.db")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")

CATEGORICAL_FEATURES = ["行政區", "建物型態"]
NUMERIC_FEATURES = ["總坪數", "屋齡", "樓層"]
TARGET = "單價(萬/坪)"


def load_dataset(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """從 SQLite 載入清整後資料。"""
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql("SELECT * FROM transactions", con)
    return df


def build_preprocessor() -> ColumnTransformer:
    """OneHot 編碼類別欄位，數值欄位 passthrough。"""
    # sklearn 1.2+ 用 sparse_output；較舊版用 sparse。這裡用相容寫法。
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("cat", ohe, CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def evaluate(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """計算 R²、RMSE、MAE，回傳指標 dict。"""
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "name": name,
        "r2": float(r2_score(y_test, pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_test, pred)),
    }


def train(db_path: str = DEFAULT_DB_PATH) -> dict:
    """訓練兩個模型，存表現較佳者，回傳完整 metadata。"""
    df = load_dataset(db_path)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=18, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
    }

    results = []
    pipelines = {}
    for name, est in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        pipe.fit(X_train, y_train)
        metric = evaluate(name, pipe, X_test, y_test)
        results.append(metric)
        pipelines[name] = pipe

    # 印出對照表
    table = pd.DataFrame(results).set_index("name")
    print("=== 模型表現對照 ===")
    print(table.round(4).to_string())

    best = max(results, key=lambda r: r["r2"])
    best_name = best["name"]
    best_pipe = pipelines[best_name]
    print(f"\n最佳模型：{best_name} (R²={best['r2']:.4f})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)

    # 取出 OneHot 後完整特徵名（給前端參考）
    ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = cat_names + NUMERIC_FEATURES

    metadata = {
        "best_model": best_name,
        "metrics": {r["name"]: {k: v for k, v in r.items() if k != "name"} for r in results},
        "selected_metrics": {k: v for k, v in best.items() if k != "name"},
        "feature_names": feature_names,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "districts": sorted(df["行政區"].dropna().unique().tolist()),
        "building_types": sorted(df["建物型態"].dropna().unique().tolist()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"已儲存：{MODEL_PATH}")
    print(f"Metadata：{META_PATH}")
    return metadata


def load_model() -> tuple[Pipeline, dict]:
    """載入已存模型與 metadata。"""
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return pipe, meta


if __name__ == "__main__":
    train()
