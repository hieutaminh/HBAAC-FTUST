"""
HBAAC 2026 - Daily SKU Sales Forecasting
Forecast horizon: F1 (2025-09-06) to F56 (2025-10-31)
Public eval: _validation rows (F1..F28)
Private eval: _evaluation rows (F1..F28, corresponding to days 29..56)
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/tuanduongphan/Code/HBAAC2026"
FORECAST_START = date(2025, 9, 6)
N_FORECAST = 56  # total forecast days
BACKTEST_DAYS = 28  # holdout window for backtest

# Shrinkage thresholds (by # positive-sale days in full history)
SPARSE_ZERO_THRESH = 3    # predict pure 0
SPARSE_SHRINK_THRESH = 30  # heavy shrinkage below this

# Recency windows for active SKUs
RECENT_WINDOWS = [7, 14, 28, 56, 90]
RECENCY_WEIGHT = [0.40, 0.25, 0.20, 0.10, 0.05]  # weights sum to 1.0


# ── 1. Load & clean ───────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def make_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Net returns, sum to daily SKU qty, fill missing days with 0."""
    daily = (
        df.groupby(["Date", "ItemCode"])["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "qty"})
    )
    # Clip at 0 — negative net means full-return day, treat as 0 demand
    daily["qty"] = daily["qty"].clip(lower=0)

    # Full cross-join: every SKU × every date
    all_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    all_skus = df["ItemCode"].unique()
    idx = pd.MultiIndex.from_product([all_dates, all_skus], names=["Date", "ItemCode"])
    panel = daily.set_index(["Date", "ItemCode"]).reindex(idx, fill_value=0).reset_index()
    return panel


# ── 2. Feature extraction per SKU ─────────────────────────────────────────────
def sku_features(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Compute per-SKU forecasting features from data up to `cutoff`.
    Returns a dataframe indexed by ItemCode.
    """
    hist = panel[panel["Date"] <= cutoff].copy()

    # Total history length (days)
    total_days = (cutoff - panel["Date"].min()).days + 1

    feats = []
    for sku, grp in hist.groupby("ItemCode"):
        grp = grp.sort_values("Date").set_index("Date")["qty"]

        n_sale_days = (grp > 0).sum()
        sale_freq = n_sale_days / total_days  # fraction of days with any sale

        if n_sale_days == 0:
            feats.append({
                "ItemCode": sku,
                "n_sale_days": 0,
                "sale_freq": 0.0,
                "mean_pos_qty": 0.0,
                "recent_blend": 0.0,
                **{f"dow_freq_{d}": 0.0 for d in range(7)},
                **{f"dow_mean_{d}": 0.0 for d in range(7)},
            })
            continue

        mean_pos_qty = grp[grp > 0].mean()

        # Recency-weighted mean (over all days including zeros)
        recent_blend = 0.0
        for w, wt in zip(RECENT_WINDOWS, RECENCY_WEIGHT):
            window_data = grp.iloc[-w:] if len(grp) >= w else grp
            recent_blend += wt * window_data.mean()

        # Day-of-week: sale frequency and mean positive qty
        dow_freq = {}
        dow_mean = {}
        for d in range(7):
            mask = grp.index.dayofweek == d
            if mask.sum() == 0:
                dow_freq[f"dow_freq_{d}"] = sale_freq
                dow_mean[f"dow_mean_{d}"] = mean_pos_qty
            else:
                dow_data = grp[mask]
                dow_freq[f"dow_freq_{d}"] = (dow_data > 0).mean()
                dow_mean[f"dow_mean_{d}"] = dow_data[dow_data > 0].mean() if (dow_data > 0).any() else 0.0

        feats.append({
            "ItemCode": sku,
            "n_sale_days": n_sale_days,
            "sale_freq": sale_freq,
            "mean_pos_qty": mean_pos_qty,
            "recent_blend": recent_blend,
            **dow_freq,
            **dow_mean,
        })

    return pd.DataFrame(feats).set_index("ItemCode")


# ── 3. Predict for a single forecast horizon ──────────────────────────────────
def predict(feats: pd.DataFrame, start_date: date, n_days: int) -> pd.DataFrame:
    """
    Given per-SKU features, predict qty for each day in [start_date, start_date+n_days).
    Returns wide DataFrame: rows=SKU, cols=day 1..n_days.
    """
    all_days = [start_date + timedelta(days=i) for i in range(n_days)]
    preds = {}

    for sku, row in feats.iterrows():
        n_sd = row["n_sale_days"]

        if n_sd <= SPARSE_ZERO_THRESH:
            preds[sku] = [0.0] * n_days
            continue

        day_preds = []
        for d in all_days:
            dow = d.weekday()  # 0=Mon, 6=Sun
            p_sale = row[f"dow_freq_{dow}"]
            cond_qty = row[f"dow_mean_{dow}"]
            if pd.isna(cond_qty) or cond_qty == 0:
                cond_qty = row["mean_pos_qty"]

            raw = p_sale * cond_qty

            # Blend with recency signal — weight recency more for active SKUs
            if n_sd < SPARSE_SHRINK_THRESH:
                # heavy shrinkage: mostly rely on dow-adjusted historical mean
                alpha = 0.2  # recency weight
            else:
                alpha = 0.5

            blended = alpha * row["recent_blend"] * (p_sale / max(row["sale_freq"], 1e-6)) + (1 - alpha) * raw

            # Further shrink sparse SKUs toward 0
            if n_sd < SPARSE_SHRINK_THRESH:
                shrink = n_sd / SPARSE_SHRINK_THRESH
                blended = shrink * blended

            day_preds.append(max(0.0, blended))

        preds[sku] = day_preds

    result = pd.DataFrame(preds, index=[f"F{i+1}" for i in range(n_days)]).T
    result.index.name = "ItemCode"
    return result


# ── 4. Backtest ───────────────────────────────────────────────────────────────
def backtest(panel: pd.DataFrame) -> dict:
    """
    Hold out last BACKTEST_DAYS days, train on prior data, measure MAE.
    """
    last_date = panel["Date"].max()
    cutoff = last_date - pd.Timedelta(days=BACKTEST_DAYS)
    bt_start = cutoff + pd.Timedelta(days=1)

    print(f"Backtest: train up to {cutoff.date()}, validate {bt_start.date()} – {last_date.date()}")

    feats = sku_features(panel, cutoff)
    pred_df = predict(feats, bt_start.date(), BACKTEST_DAYS)

    # Actual values
    actual = (
        panel[(panel["Date"] > cutoff)]
        .pivot(index="ItemCode", columns="Date", values="qty")
        .fillna(0)
    )
    actual.columns = [f"F{i+1}" for i in range(len(actual.columns))]

    # Align
    common_skus = pred_df.index.intersection(actual.index)
    pred_vals = pred_df.loc[common_skus].values
    true_vals = actual.loc[common_skus].values

    mae = np.mean(np.abs(pred_vals - true_vals))
    rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
    zero_pct = (true_vals == 0).mean()

    # MAE by sparsity tier
    n_sale = feats.loc[common_skus, "n_sale_days"]
    tier_mae = {}
    for label, lo, hi in [("sparse(<=3)", 0, 3), ("low(4-30)", 4, 30),
                           ("medium(31-100)", 31, 100), ("active(>100)", 101, 1e9)]:
        mask = (n_sale > lo) & (n_sale <= hi)
        if mask.sum() > 0:
            tier_mae[label] = np.mean(np.abs(
                pred_df.loc[common_skus[mask]].values - actual.loc[common_skus[mask]].values
            ))

    return {"mae": mae, "rmse": rmse, "zero_pct": zero_pct, "tier_mae": tier_mae}


# ── 5. Build submission ───────────────────────────────────────────────────────
def build_submission(panel: pd.DataFrame, output_path: str):
    cutoff = panel["Date"].max()
    feats = sku_features(panel, cutoff)

    # validation: days 1-28
    val_pred = predict(feats, FORECAST_START, 28)
    # evaluation: days 29-56 (reuse same feats; starts from day 29)
    eval_start = FORECAST_START + timedelta(days=28)
    eval_pred = predict(feats, eval_start, 28)

    sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    sub_skus = sub["id"].str.replace("_validation|_evaluation", "", regex=True)

    rows = []
    for _, row in sub.iterrows():
        sku = row["id"].replace("_validation", "").replace("_evaluation", "")
        suffix = "validation" if "_validation" in row["id"] else "evaluation"
        src = val_pred if suffix == "validation" else eval_pred
        if sku in src.index:
            vals = src.loc[sku].values.tolist()
        else:
            vals = [0.0] * 28
        rows.append([row["id"]] + [round(max(0.0, v), 4) for v in vals])

    out = pd.DataFrame(rows, columns=["id"] + [f"F{i+1}" for i in range(28)])
    out.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path}  ({len(out)} rows)")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data(f"{DATA_DIR}/train.csv")
    print(f"  Rows: {len(df):,}  |  SKUs: {df['ItemCode'].nunique():,}  |  "
          f"Dates: {df['Date'].min().date()} – {df['Date'].max().date()}")

    print("\nBuilding daily panel...")
    panel = make_daily_panel(df)
    print(f"  Panel shape: {panel.shape}")

    print("\nRunning backtest...")
    bt = backtest(panel)
    print(f"  Overall MAE : {bt['mae']:.4f}")
    print(f"  Overall RMSE: {bt['rmse']:.4f}")
    print(f"  Zero-rate in actuals: {bt['zero_pct']:.1%}")
    print("  MAE by sparsity tier:")
    for k, v in bt["tier_mae"].items():
        print(f"    {k}: {v:.4f}")

    print("\nBuilding submission...")
    sub = build_submission(panel, f"{DATA_DIR}/submission_v1.csv")
    print("\nDone. Preview:")
    print(sub.head(5).to_string())
