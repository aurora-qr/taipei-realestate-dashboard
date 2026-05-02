"""台北市實價登錄資料清整模組。

讀取內政部實價登錄原始 CSV，清整後存入 SQLite。
資料特性：
- 第二列是英文欄名，需跳過
- 日期是民國年制 (e.g. 1140604 表示民國 114 年 6 月 4 日)
- 行政區資訊散落於「鄉鎮市區」與「土地位置建物門牌」
"""

from __future__ import annotations

import glob
import os
import re
import sqlite3

import numpy as np
import pandas as pd

# 平方公尺 → 坪 的換算係數
SQM_TO_PING = 0.3025
# 元/平方公尺 → 萬/坪
UNIT_PRICE_FACTOR = 3.305785 / 10000

# 視為住宅相關的「主要用途」
RESIDENTIAL_USES = {"住家用", "住商用", "見其他登記事項"}

# 行政區萃取（台北市共 12 區）
TAIPEI_DISTRICTS = [
    "中正區", "大同區", "中山區", "松山區", "大安區", "萬華區",
    "信義區", "士林區", "北投區", "內湖區", "南港區", "文山區",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data.db")


def load_raw_csvs(pattern: str = "raw_*.csv", base_dir: str | None = None) -> pd.DataFrame:
    """讀取所有原始 CSV，跳過英文欄名列後合併。"""
    base_dir = base_dir or PROJECT_ROOT
    files = sorted(glob.glob(os.path.join(base_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"找不到符合 {pattern} 的檔案於 {base_dir}")

    frames = []
    for path in files:
        # header=0 取第一列作欄名，skiprows=[1] 跳過第二列(英文欄名)
        df = pd.read_csv(path, header=0, skiprows=[1], dtype=str, encoding="utf-8")
        df["__source_file"] = os.path.basename(path)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    # 處理 BOM：第一個欄名可能帶
    raw.columns = [c.lstrip("﻿").strip() for c in raw.columns]
    return raw


def filter_residential(df: pd.DataFrame) -> pd.DataFrame:
    """只保留主要用途為住宅相關的紀錄。"""
    if "主要用途" not in df.columns:
        return df
    mask = df["主要用途"].fillna("").isin(RESIDENTIAL_USES)
    return df[mask].copy()


def to_numeric(series: pd.Series) -> pd.Series:
    """字串轉數值，無法解析者轉 NaN。"""
    return pd.to_numeric(series, errors="coerce")


def parse_roc_date(value: str) -> int | float:
    """民國年月日字串 → 西元年份 (int)；無法解析則 NaN。

    例：'1140604' → 2025；'0811110' → 1992。
    """
    if not isinstance(value, str):
        return np.nan
    s = value.strip()
    if not s.isdigit() or len(s) < 5:
        return np.nan
    # 取前 3 碼為民國年（不足 3 碼則前 2 碼）
    if len(s) >= 7:
        roc_year = int(s[:3])
    elif len(s) == 6:
        roc_year = int(s[:2])
    else:
        return np.nan
    if roc_year <= 0 or roc_year > 200:
        return np.nan
    return roc_year + 1911


def extract_district(row: pd.Series) -> str | float:
    """從「鄉鎮市區」或門牌字串中萃取行政區。"""
    raw_district = str(row.get("鄉鎮市區", "") or "").strip()
    if raw_district in TAIPEI_DISTRICTS:
        return raw_district
    address = str(row.get("土地位置建物門牌", "") or "")
    for d in TAIPEI_DISTRICTS:
        if d in address:
            return d
    return np.nan


# 樓層中文 → 數字對照
_FLOOR_CN = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def parse_floor(value: str) -> float:
    """從「移轉層次」字串解析出主要樓層數字。

    例：'五層' → 5；'地下一層' → -1；'十二層' → 12；'十五層，十六層' → 15。
    """
    if not isinstance(value, str) or not value:
        return np.nan
    s = value.split("，")[0].split(",")[0].strip()
    if not s:
        return np.nan

    # 阿拉伯數字直接抓
    m = re.search(r"-?\d+", s)
    if m:
        n = int(m.group())
        if "地下" in s:
            n = -abs(n)
        return float(n)

    is_basement = "地下" in s
    # 移除非數字字元 (層、地上、地下、全) 以做中文數字解析
    core = re.sub(r"[層地下上全]", "", s)
    if not core:
        return np.nan

    # 中文數字解析：支援 一~九十九
    n: int | None = None
    if core in _FLOOR_CN:
        n = _FLOOR_CN[core]
    elif core.startswith("十") and len(core) == 2 and core[1] in _FLOOR_CN:
        n = 10 + _FLOOR_CN[core[1]]
    elif core.endswith("十") and len(core) == 2 and core[0] in _FLOOR_CN:
        n = _FLOOR_CN[core[0]] * 10
    elif len(core) == 3 and core[1] == "十" and core[0] in _FLOOR_CN and core[2] in _FLOOR_CN:
        n = _FLOOR_CN[core[0]] * 10 + _FLOOR_CN[core[2]]

    if n is None:
        return np.nan
    return float(-n if is_basement else n)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """主清整流程：型別轉換、欄位衍生、極端值過濾。"""
    df = filter_residential(df)

    df["總價元"] = to_numeric(df.get("總價元"))
    df["建物移轉總面積平方公尺"] = to_numeric(df.get("建物移轉總面積平方公尺"))
    df["單價元平方公尺"] = to_numeric(df.get("單價元平方公尺"))

    df["單價(萬/坪)"] = df["單價元平方公尺"] * UNIT_PRICE_FACTOR
    df["總坪數"] = df["建物移轉總面積平方公尺"] * SQM_TO_PING

    df["行政區"] = df.apply(extract_district, axis=1)

    df["交易年"] = df["交易年月日"].apply(parse_roc_date)
    df["建築完成年"] = df["建築完成年月"].apply(parse_roc_date)
    df["屋齡"] = df["交易年"] - df["建築完成年"]

    df["樓層"] = df["移轉層次"].apply(parse_floor)
    df["建物型態"] = df["建物型態"].fillna("其他")

    # 必要欄位非空
    df = df.dropna(subset=["單價(萬/坪)", "總坪數", "行政區", "屋齡"])

    # 過濾極端值：單價以分位數、坪數與屋齡用硬性區間
    lo, hi = df["單價(萬/坪)"].quantile([0.05, 0.95])
    df = df[(df["單價(萬/坪)"] >= lo) & (df["單價(萬/坪)"] <= hi)]
    df = df[(df["總坪數"] >= 5) & (df["總坪數"] <= 200)]
    df = df[(df["屋齡"] >= 0) & (df["屋齡"] <= 80)]

    keep_cols = [
        "行政區", "建物型態", "主要用途",
        "總價元", "單價(萬/坪)", "總坪數",
        "屋齡", "樓層", "交易年", "建築完成年",
        "土地位置建物門牌", "__source_file",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)
    df["樓層"] = df["樓層"].fillna(1.0)
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH, table: str = "transactions") -> str:
    """將清整後資料寫入 SQLite。"""
    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)
    return db_path


def run_pipeline(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """完整 ETL：讀檔 → 清整 → 寫 DB。"""
    raw = load_raw_csvs()
    cleaned = clean(raw)
    save_to_sqlite(cleaned, db_path)
    return cleaned


if __name__ == "__main__":
    df = run_pipeline()
    print(f"原始檔已清整，輸出 {len(df):,} 筆")
    print(f"行政區分佈：\n{df['行政區'].value_counts().to_string()}")
    print(f"\n單價(萬/坪) 描述：\n{df['單價(萬/坪)'].describe().to_string()}")
    print(f"\n總坪數描述：\n{df['總坪數'].describe().to_string()}")
    print(f"\n屋齡描述：\n{df['屋齡'].describe().to_string()}")
    print(f"\n建物型態：\n{df['建物型態'].value_counts().to_string()}")
    print(f"\n資料庫：{DEFAULT_DB_PATH}")
