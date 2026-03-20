"""
categorization.py
=================
Maps the 14 DataGenome metrics (output of derive_all_metrics) to per-sample
category labels.

Categories (not mutually exclusive — a sample can belong to multiple):
    is_noisy        — mislabeled / contradictory signal
    is_boundary     — legitimately ambiguous, near decision boundary
    is_rare         — under-represented sub-population
    is_redundant    — highly similar to many easy samples
    is_easy         — learned early, prototypical, never forgotten
    is_mildly_hard  — learned in first 20% of training
    is_hard         — genuinely complex, learned mid-training
    is_very_hard    — learned late (60–80% of training)

Each category is a boolean column. A sample can have multiple True flags.
Prediction depth (metric ⑬) is excluded from this version.

Entry point
-----------
categorize(df) → pd.DataFrame
    Accepts the DataFrame returned by derive_all_metrics().
    Returns the same DataFrame with 8 boolean category columns appended,
    plus a 'primary_category' column (first match in priority order).
"""

import os
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Threshold helpers — percentile-based where HTML says "high / low / top N%"
# ---------------------------------------------------------------------------

def _percentile_mask_low(series: pd.Series, pct: float) -> pd.Series:
    """True for samples in the bottom `pct`% of the distribution."""
    threshold = np.nanpercentile(series.values, pct)
    return series <= threshold


def _percentile_mask_high(series: pd.Series, pct: float) -> pd.Series:
    """True for samples in the top `pct`% of the distribution."""
    threshold = np.nanpercentile(series.values, 100 - pct)
    return series >= threshold


# ---------------------------------------------------------------------------
# Per-category boolean masks
# ---------------------------------------------------------------------------

def _mask_noisy(df: pd.DataFrame) -> pd.Series:
    """
    NOISY / MISLABELED
    ------------------
    rho = NaN (margin never achieves a stable positive crossing)
    AND negative AUM
    AND high forgetting events (top 20%)
    AND high ensemble disagreement (top 20%), or NaN (single-seed run)

    For single-seed runs ensemble_dis is NaN → relax that condition.
    """
    rho_nan      = df["rho"].isna()
    neg_aum      = df["aum"] < 0
    high_fe      = _percentile_mask_high(df["fe"].astype(float), pct=20)

    if df["ensemble_dis"].isna().all():
        # No multi-seed data — rely on the three training-signal conditions
        return rho_nan & neg_aum & high_fe
    else:
        high_ens = df["ensemble_dis"].fillna(0) >= np.nanpercentile(df["ensemble_dis"].values, 80)
        return rho_nan & neg_aum & high_fe & high_ens


def _mask_boundary(df: pd.DataFrame) -> pd.Series:
    """
    BOUNDARY / AMBIGUOUS
    --------------------
    rho finite AND > 0.50 (late but not absent crossover)
    AND high margin variance (top 20%)
    AND high forgetting events (top 20%)
    AND AUM >= 0 (distinguishes from NOISY which has negative AUM)
    AND moderate-to-high ensemble disagreement (top 30%), or NaN → relax
    """
    rho_late      = df["rho"].notna() & (df["rho"] > 0.50)
    high_mvar     = _percentile_mask_high(df["margin_var"], pct=20)
    high_fe       = _percentile_mask_high(df["fe"].astype(float), pct=20)
    nonneg_aum    = df["aum"] >= 0

    if df["ensemble_dis"].isna().all():
        return rho_late & high_mvar & high_fe & nonneg_aum
    else:
        mod_ens = df["ensemble_dis"].fillna(0) >= np.nanpercentile(df["ensemble_dis"].values, 70)
        return rho_late & high_mvar & high_fe & nonneg_aum & mod_ens


def _mask_rare(df: pd.DataFrame) -> pd.Series:
    """
    RARE / UNDER-REPRESENTED
    ------------------------
    rho > 0.50 (late learning due to scarcity, not complexity)
    AND knn_density in bottom 10% (isolated in embedding space)
    AND low prototypicality (high intra-class distance to centroid, bottom 30%)
    """
    rho_late    = df["rho"].notna() & (df["rho"] > 0.50)
    low_density = _percentile_mask_high(df["knn_density"], pct=10)     # top 10% of normalised dist = bottom 10% density = most isolated
    low_proto   = _percentile_mask_low(df["prototypicality"], pct=30)
    return rho_late & low_density & low_proto


def _mask_redundant(df: pd.DataFrame) -> pd.Series:
    """
    REDUNDANT
    ---------
    rho < 0.05 (learned extremely early)
    AND knn_density in top 10% (dense cluster)
    AND near-zero grand (gradient norm, bottom 20%)
    AND high prototypicality (top 30%)
    """
    rho_early    = df["rho"].notna() & (df["rho"] < 0.05)
    high_density = _percentile_mask_low(df["knn_density"], pct=10)     # bottom 10% of normalised dist = most crowded
    low_grand    = _percentile_mask_low(df["grand"], pct=20)
    high_proto   = _percentile_mask_high(df["prototypicality"], pct=30)
    return rho_early & high_density & low_grand & high_proto


def _mask_easy(df: pd.DataFrame) -> pd.Series:
    """
    EASY / PROTOTYPICAL
    -------------------
    rho < 0.10 (learned in first 10% of training)
    AND high AUM (top 30%)
    AND zero forgetting events
    AND high prototypicality (top 30%)
    """
    rho_early  = df["rho"].notna() & (df["rho"] < 0.10)
    high_aum   = _percentile_mask_high(df["aum"], pct=30)
    zero_fe    = df["fe"] == 0
    high_proto = _percentile_mask_high(df["prototypicality"], pct=30)
    return rho_early & high_aum & zero_fe & high_proto


def _mask_mildly_hard(df: pd.DataFrame) -> pd.Series:
    """
    MILDLY HARD
    -----------
    rho in [0.10, 0.20)
    AND AUM >= median(AUM)  (moderate positive — not struggling)
    AND low forgetting (fe <= 1)
    """
    rho_range   = df["rho"].notna() & (df["rho"] >= 0.10) & (df["rho"] < 0.20)
    mod_aum     = df["aum"] >= float(df["aum"].median())
    low_fe      = df["fe"] <= 1
    return rho_range & mod_aum & low_fe


def _mask_hard(df: pd.DataFrame) -> pd.Series:
    """
    HARD / COMPLEX
    --------------
    rho in [0.20, 0.60)
    AND low AUM (bottom 40%)
    AND moderate forgetting (fe >= 1)
    AND knn_density NOT in bottom 10% (not isolated — distinguishes from RARE)
    """
    rho_range     = df["rho"].notna() & (df["rho"] >= 0.20) & (df["rho"] < 0.60)
    low_aum       = _percentile_mask_low(df["aum"], pct=40)
    some_fe       = df["fe"] >= 1
    not_isolated  = ~_percentile_mask_high(df["knn_density"], pct=10)   # NOT top 10% of normalised dist (not isolated)
    return rho_range & low_aum & some_fe & not_isolated


def _mask_very_hard(df: pd.DataFrame) -> pd.Series:
    """
    VERY HARD / NEAR-BOUNDARY
    -------------------------
    rho in [0.60, 0.80]
    AND near-zero AUM: |AUM| in bottom 20% of |AUM| distribution
    AND high forgetting (top 20%)
    """
    rho_range    = df["rho"].notna() & (df["rho"] >= 0.60) & (df["rho"] <= 0.80)
    near_zero_aum = pd.Series(np.abs(df["aum"].values), index=df.index)
    small_aum    = _percentile_mask_low(near_zero_aum, pct=20)
    high_fe      = _percentile_mask_high(df["fe"].astype(float), pct=20)
    return rho_range & small_aum & high_fe


# ---------------------------------------------------------------------------
# Primary category assignment (priority order, first match wins)
# ---------------------------------------------------------------------------

_PRIORITY_ORDER = [
    "is_noisy",
    "is_boundary",
    "is_rare",
    "is_redundant",
    "is_easy",
    "is_mildly_hard",
    "is_hard",
    "is_very_hard",
]


def _assign_primary(df: pd.DataFrame) -> pd.Series:
    """
    Walk the priority list and assign the first True category as primary.
    Samples that match no category get 'unassigned'.
    """
    primary = pd.Series("unassigned", index=df.index, dtype=object)
    for col in reversed(_PRIORITY_ORDER):   # reversed so highest priority overwrites last
        primary[df[col]] = col.replace("is_", "")
    return primary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def categorize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append 8 boolean category columns and a primary_category column to the
    DataFrame returned by derive_all_metrics().

    Parameters
    ----------
    df : pd.DataFrame
        Output of derive_all_metrics(). Must contain columns:
            rho, aum, fe, margin_var, knn_density, prototypicality,
            grand, ensemble_dis

    Returns
    -------
    pd.DataFrame
        Same rows as input with additional columns:
            is_noisy, is_boundary, is_rare, is_redundant,
            is_easy, is_mildly_hard, is_hard, is_very_hard,
            primary_category
    """
    out = df.copy()

    out["is_noisy"]       = _mask_noisy(df).astype(bool)
    out["is_boundary"]    = _mask_boundary(df).astype(bool)
    out["is_rare"]        = _mask_rare(df).astype(bool)
    out["is_redundant"]   = _mask_redundant(df).astype(bool)
    out["is_easy"]        = _mask_easy(df).astype(bool)
    out["is_mildly_hard"] = _mask_mildly_hard(df).astype(bool)
    out["is_hard"]        = _mask_hard(df).astype(bool)
    out["is_very_hard"]   = _mask_very_hard(df).astype(bool)

    out["primary_category"] = _assign_primary(out)

    return out


# ---------------------------------------------------------------------------
# Save to disk
# ---------------------------------------------------------------------------

def save_categorized_report(
    df: pd.DataFrame,
    output_dir: str,
) -> str:
    """
    Save the categorized DataFrame and associated arrays to disk.

    Writes:
        <output_dir>/sample_map.parquet    — the full DataFrame (metrics + categories)
        <output_dir>/knn_raw_distances.npy — [N] raw mean kNN distances
        <output_dir>/knn_normalised_density.npy — [N] global-mean-normalised density
        <output_dir>/knn_metadata.json     — global_mean_distance, k, N

    Parameters
    ----------
    df : pd.DataFrame
        Output of categorize(). Must have knn attrs from derive_all_metrics().
    output_dir : str
        Directory to write files into (created if it doesn't exist).

    Returns
    -------
    str : path to the saved parquet file
    """
    os.makedirs(output_dir, exist_ok=True)

    parquet_path = os.path.join(output_dir, "sample_map.parquet")
    df.to_parquet(parquet_path)

    # Save kNN intermediate arrays if available in attrs
    if "knn_raw_distances" in df.attrs:
        np.save(
            os.path.join(output_dir, "knn_raw_distances.npy"),
            df.attrs["knn_raw_distances"],
        )
    if "knn_global_mean" in df.attrs:
        knn_meta = {
            "global_mean_distance": df.attrs["knn_global_mean"],
            "num_samples": len(df),
        }
        with open(os.path.join(output_dir, "knn_metadata.json"), "w") as f:
            json.dump(knn_meta, f, indent=2)

    # knn_density column in the DataFrame IS the normalised density, but also
    # save it as a standalone .npy for quick loading without parquet
    if "knn_density" in df.columns:
        np.save(
            os.path.join(output_dir, "knn_normalised_density.npy"),
            df["knn_density"].values,
        )

    return parquet_path
