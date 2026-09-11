"""
NHL Fantasy Points Prediction Pipeline
=======================================
Predicts next-season FantasyPoints_PG (and full-season GP) using a strict
2-season lookback window, walk-forward validation, and two candidate models
(Ridge regression, LightGBM) compared against a naive "same as last season"
baseline.

Usage:
    python fantasy_pipeline.py

Outputs:
    validation_results.csv   -- MAE by fold / position / model vs. naive baseline
    projections_2026_27.csv  -- final 26-27 fantasy point projections
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import lightgbm as lgb

DATA_PATH = "nhl_fantasy_dataset.xlsx"
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
LOOKBACK = 2  # <-- the parameter this whole workflow hinges on

# Season indices: 0=21-22, 1=22-23, 2=23-24, 3=24-25, 4=25-26
# 25-26 is a partial season -> usable as feature history, never as a training target.
VALID_TARGETS = [1, 2, 3]        # 22-23, 23-24, 24-25 have complete outcomes to learn from
PROJECTION_TARGET = 5            # 26-27 (doesn't exist in the sheet -- we're extrapolating)

RATE_FEATURES = [
    "Goals_PG", "Assists_PG", "Shots_PG", "Hits_PG", "Blocks_PG",
    "TOI_PG", "iCF_PG", "MP_xGoals_PG", "GameScore_PG", "SH%",
]
PCT_COLS_TO_CLEAN = ["IPP%", "SH%", "xGoalsPct", "CorsiPct", "FenwickPct"]


# ---------------------------------------------------------------------------
# 1. Load & reshape wide (one sheet per season) -> long (one row per player-season)
# ---------------------------------------------------------------------------
def load_long(path: str) -> pd.DataFrame:
    frames = []
    for i, s in enumerate(SEASONS):
        df = pd.read_excel(path, sheet_name=s, skiprows=1)
        # skiprows=1 leaves the repeated "DEFENSEMEN" sub-header row (Player=="Player")
        # embedded mid-sheet -- drop it along with the blank divider row.
        df = df[df["Player"].notna() & (df["Player"] != "Player") & df["playerId"].notna()].copy()
        df["season"] = s
        df["season_idx"] = i
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)

    for col in PCT_COLS_TO_CLEAN:
        long_df[col] = pd.to_numeric(long_df[col].replace("-", np.nan), errors="coerce")

    long_df["GP"] = pd.to_numeric(long_df["GP"], errors="coerce")
    for col in RATE_FEATURES + ["Age", "FantasyPoints_PG"]:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")
    long_df["is_defenseman"] = (long_df["Position"] == "D")
    return long_df


# ---------------------------------------------------------------------------
# 2. Leakage-safe feature construction: strictly seasons < target_idx,
#    capped at the last `lookback` seasons, recency-weighted.
# ---------------------------------------------------------------------------
def build_features(long_df: pd.DataFrame, target_idx: int, lookback: int = LOOKBACK) -> pd.DataFrame:
    hist = long_df[
        (long_df["season_idx"] < target_idx)
        & (long_df["season_idx"] >= target_idx - lookback)
        & (long_df["GP"] > 0)
    ].copy()

    target = long_df[
        (long_df["season_idx"] == target_idx) & (long_df["GP"] > 0)
    ][["playerId", "Player", "Position", "Age", "GP", "FantasyPoints_PG"]].copy()
    target = target.rename(columns={"Age": "target_age", "GP": "target_GP"})

    if hist.empty or target.empty:
        return pd.DataFrame()

    hist = hist.sort_values(["playerId", "season_idx"])
    hist["recency_rank"] = hist.groupby("playerId")["season_idx"].rank(method="first")
    hist["weight"] = 2 ** hist["recency_rank"]  # most recent prior season weighted ~2x the one before

    def weighted_avg(g):
        w = g["weight"]
        out = {c: (g[c] * w).sum() / w.sum() for c in RATE_FEATURES}
        out["last_season_FantasyPoints_PG"] = g.loc[g["season_idx"].idxmax(), "FantasyPoints_PG"]
        return pd.Series(out)

    agg = hist.groupby("playerId").apply(weighted_avg, include_groups=False).reset_index()
    counts = hist.groupby("playerId").agg(
        n_hist_seasons=("season_idx", "nunique"),
        hist_gp_total=("GP", "sum"),
    ).reset_index()
    agg = agg.merge(counts, on="playerId", how="left")

    out = target.merge(agg, on="playerId", how="left")
    out["age"] = pd.to_numeric(out["target_age"], errors="coerce")
    out["age_sq"] = out["age"] ** 2
    out["target_idx"] = target_idx
    return out


# ---------------------------------------------------------------------------
# 3. Empirical-Bayes shrinkage on thin-sample players (applied per position group)
# ---------------------------------------------------------------------------
def shrink_rate_features(df: pd.DataFrame, cols: list, k: int = 50) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        pos_mean = df[col].mean()
        df[col] = (
            df[col] * df["hist_gp_total"] + pos_mean * k
        ) / (df["hist_gp_total"] + k)
    return df


def build_projection_features(long_df: pd.DataFrame, target_idx: int, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Like build_features, but for a season that doesn't exist in the sheet yet (e.g. 26-27).
    There's no target row to anchor on, so the player universe and their Position/Age come
    from their most recent actual season instead."""
    hist = long_df[
        (long_df["season_idx"] < target_idx)
        & (long_df["season_idx"] >= target_idx - lookback)
        & (long_df["GP"] > 0)
    ].copy()
    if hist.empty:
        return pd.DataFrame()

    hist = hist.sort_values(["playerId", "season_idx"])
    hist["recency_rank"] = hist.groupby("playerId")["season_idx"].rank(method="first")
    hist["weight"] = 2 ** hist["recency_rank"]

    def weighted_avg(g):
        w = g["weight"]
        out = {c: (g[c] * w).sum() / w.sum() for c in RATE_FEATURES}
        latest = g.loc[g["season_idx"].idxmax()]
        out["last_season_FantasyPoints_PG"] = latest["FantasyPoints_PG"]
        out["Player"] = latest["Player"]
        out["Position"] = latest["Position"]
        out["target_age"] = latest["Age"] + (target_idx - latest["season_idx"])  # age up to target season
        return pd.Series(out)

    agg = hist.groupby("playerId").apply(weighted_avg, include_groups=False).reset_index()
    counts = hist.groupby("playerId").agg(
        n_hist_seasons=("season_idx", "nunique"),
        hist_gp_total=("GP", "sum"),
    ).reset_index()
    out = agg.merge(counts, on="playerId", how="left")
    out["age"] = pd.to_numeric(out["target_age"], errors="coerce")
    out["age_sq"] = out["age"] ** 2
    out["target_idx"] = target_idx
    return out


FEATURE_COLS = RATE_FEATURES + ["last_season_FantasyPoints_PG", "age", "age_sq",
                                 "n_hist_seasons", "hist_gp_total"]
TARGET_COL = "FantasyPoints_PG"


# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------
def make_ridge():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=5.0)),
    ])


def make_lgbm():
    return lgb.LGBMRegressor(
        max_depth=3,
        num_leaves=7,
        n_estimators=150,
        learning_rate=0.05,
        min_child_samples=10,
        reg_alpha=1.0,
        reg_lambda=1.0,
        verbosity=-1,
    )


MODELS = {"ridge": make_ridge, "lgbm": make_lgbm}


# ---------------------------------------------------------------------------
# 5. Walk-forward validation
# ---------------------------------------------------------------------------
def walk_forward_validate(long_df: pd.DataFrame) -> pd.DataFrame:
    all_rows = pd.concat([build_features(long_df, t) for t in VALID_TARGETS], ignore_index=True)

    folds = [
        {"train": [1], "val": 2},
        {"train": [1, 2], "val": 3},
    ]

    results = []
    for position_label, pos_mask_fn in [
        ("Forwards", lambda d: d["Position"] != "D"),
        ("Defensemen", lambda d: d["Position"] == "D"),
    ]:
        pos_rows = all_rows[pos_mask_fn(all_rows)]
        pos_rows = shrink_rate_features(pos_rows, RATE_FEATURES)

        for fold in folds:
            # Rows with no prior-season history at all (true rookies) can't be scored by
            # a lag-based model -- they belong in a separate cold-start model, not here.
            required = [TARGET_COL, "last_season_FantasyPoints_PG"]
            train = pos_rows[pos_rows["target_idx"].isin(fold["train"])].dropna(subset=required)
            val = pos_rows[pos_rows["target_idx"] == fold["val"]].dropna(subset=required)
            if len(train) < 15 or len(val) < 5:
                continue

            naive_mae = mean_absolute_error(val[TARGET_COL], val["last_season_FantasyPoints_PG"])

            row = {
                "position": position_label,
                "train_targets": fold["train"],
                "val_target": fold["val"],
                "n_train": len(train),
                "n_val": len(val),
                "naive_mae": round(naive_mae, 3),
            }
            for name, factory in MODELS.items():
                model = factory()
                model.fit(train[FEATURE_COLS], train[TARGET_COL])
                pred = model.predict(val[FEATURE_COLS])
                row[f"{name}_mae"] = round(mean_absolute_error(val[TARGET_COL], pred), 3)
            results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 6. Final fit (all valid targets) + 26-27 projection
# ---------------------------------------------------------------------------
def project_next_season(long_df: pd.DataFrame, best_model_name: str) -> pd.DataFrame:
    train_rows = pd.concat([build_features(long_df, t) for t in VALID_TARGETS], ignore_index=True)
    proj_rows = build_projection_features(long_df, PROJECTION_TARGET)  # hist = 24-25, 25-26

    projections = []
    for position_label, pos_mask_fn in [
        ("Forwards", lambda d: d["Position"] != "D"),
        ("Defensemen", lambda d: d["Position"] == "D"),
    ]:
        required = [TARGET_COL, "last_season_FantasyPoints_PG"]
        tr = shrink_rate_features(train_rows[pos_mask_fn(train_rows)], RATE_FEATURES).dropna(subset=required)
        pj_all = shrink_rate_features(proj_rows[pos_mask_fn(proj_rows)], RATE_FEATURES)
        no_history = pj_all[pj_all["last_season_FantasyPoints_PG"].isna()]
        pj = pj_all.dropna(subset=["last_season_FantasyPoints_PG"])
        if not no_history.empty:
            print(f"  [{position_label}] {len(no_history)} player(s) skipped (no season history "
                  f"in the {LOOKBACK}-season lookback window -- need a cold-start model): "
                  f"{', '.join(no_history['Player'].tolist())}")
        if tr.empty or pj.empty:
            continue

        model = MODELS[best_model_name]()
        model.fit(tr[FEATURE_COLS], tr[TARGET_COL])
        pj = pj.copy()
        pj["proj_FantasyPoints_PG"] = model.predict(pj[FEATURE_COLS])

        # simple GP projection: recency-weighted historical GP, capped at 82
        gp_model = MODELS["ridge"]()
        gp_model.fit(tr[FEATURE_COLS], tr["target_GP"])
        pj["proj_GP"] = np.clip(gp_model.predict(pj[FEATURE_COLS]), 0, 82)

        pj["proj_total_FantasyPoints"] = pj["proj_FantasyPoints_PG"] * pj["proj_GP"]
        pj["position_group"] = position_label
        projections.append(pj)

    result = pd.concat(projections, ignore_index=True)
    return result[[
        "Player", "Position", "position_group", "age",
        "last_season_FantasyPoints_PG", "proj_FantasyPoints_PG",
        "proj_GP", "proj_total_FantasyPoints",
    ]].sort_values("proj_total_FantasyPoints", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    long_df = load_long(DATA_PATH)

    val_results = walk_forward_validate(long_df)
    val_results.to_csv("validation_results.csv", index=False)
    print("=== Walk-forward validation (lower MAE = better) ===")
    print(val_results.to_string(index=False))

    # Pick whichever model wins more folds on average MAE across both positions
    avg_ridge = val_results["ridge_mae"].mean()
    avg_lgbm = val_results["lgbm_mae"].mean()
    avg_naive = val_results["naive_mae"].mean()
    print(f"\nAvg MAE -> naive: {avg_naive:.3f} | ridge: {avg_ridge:.3f} | lgbm: {avg_lgbm:.3f}")
    best_model_name = "ridge" if avg_ridge <= avg_lgbm else "lgbm"
    print(f"Selected model for final projection: {best_model_name}")

    projections = project_next_season(long_df, best_model_name)
    projections.to_csv("projections_2026_27.csv", index=False)
    print("\n=== Top 15 projected 26-27 fantasy scorers ===")
    print(projections.head(15).to_string(index=False))
