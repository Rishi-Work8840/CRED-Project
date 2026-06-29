"""
CRED Accelerator — Scoring Engine
===================================
Target: Microsoft Fabric Notebook (PySpark / Python)
Purpose: Score reports from a standardised metadata template
         and classify each into Consolidate / Retain / Eliminate / Develop

Input:  cred_metadata_input table (Lakehouse Bronze layer)
Output: cred_scored_reports table (Lakehouse Gold layer)

Author: Migration Accelerator Team
"""

import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime, date
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION
# ─────────────────────────────────────────────────────────────

CONFIG = {
    # Recency thresholds (days since last accessed)
    "ELIMINATE_RECENCY_DAYS": 90,
    "DEVELOP_RECENCY_DAYS": 60,

    # Usage frequency thresholds (views in last 90 days)
    "LOW_USAGE_THRESHOLD": 5,
    "HIGH_USAGE_THRESHOLD": 30,

    # Similarity thresholds for consolidation (Jaccard index 0–1)
    "CONSOLIDATE_SIMILARITY_HIGH": 0.75,   # Strong consolidation candidate
    "CONSOLIDATE_SIMILARITY_LOW": 0.50,    # Possible consolidation candidate

    # Weights for composite similarity score
    "WEIGHT_TABLES": 0.40,
    "WEIGHT_METRICS": 0.35,
    "WEIGHT_DIMENSIONS": 0.25,

    # CRED decision priority order (higher = overrides lower)
    # Eliminate > Consolidate > Retain > Develop
}


# ─────────────────────────────────────────────────────────────
# SECTION 2 — DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "report_id", "report_name", "report_description",
    "source_platform", "business_domain", "owner_email",
    "last_accessed_date", "view_count_90d",
    "source_tables",   # pipe-delimited string e.g. "sales_fact|customer_dim|date_dim"
    "metrics",         # pipe-delimited string e.g. "total_revenue|avg_order_value"
    "dimensions",      # pipe-delimited string e.g. "region|product_category|month"
    "report_type",     # e.g. "operational", "executive", "ad-hoc", "scheduled"
    "tags",            # pipe-delimited free tags e.g. "finance|monthly|legacy"
]

def load_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate template columns and coerce types."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalise date column
    df["last_accessed_date"] = pd.to_datetime(df["last_accessed_date"], errors="coerce")
    df["view_count_90d"] = pd.to_numeric(df["view_count_90d"], errors="coerce").fillna(0).astype(int)

    # Lowercase & strip all string columns
    str_cols = ["source_tables", "metrics", "dimensions", "tags",
                "report_name", "report_description", "business_domain",
                "source_platform", "report_type"]
    for col in str_cols:
        df[col] = df[col].fillna("").str.lower().str.strip()

    # Flag rows with no last_accessed_date as stale (treat as > threshold)
    df["_has_access_date"] = df["last_accessed_date"].notna()

    return df


# ─────────────────────────────────────────────────────────────
# SECTION 3 — FINGERPRINTING
# ─────────────────────────────────────────────────────────────

def parse_pipe(value: str) -> set:
    """Parse a pipe-delimited string into a normalised set."""
    if not value or pd.isna(value):
        return set()
    return {v.strip() for v in str(value).split("|") if v.strip()}

def build_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    """Build structural fingerprint sets for each report."""
    df["_fp_tables"]     = df["source_tables"].apply(parse_pipe)
    df["_fp_metrics"]    = df["metrics"].apply(parse_pipe)
    df["_fp_dimensions"] = df["dimensions"].apply(parse_pipe)
    df["_fp_tags"]       = df["tags"].apply(parse_pipe)

    # Combined fingerprint (all structural elements)
    df["_fp_combined"] = df.apply(
        lambda r: r["_fp_tables"] | r["_fp_metrics"] | r["_fp_dimensions"], axis=1
    )

    # Uniqueness flag — reports that are the sole user of their source tables
    all_tables = pd.Series(
        [t for fp in df["_fp_tables"] for t in fp]
    ).value_counts()
    df["_unique_tables"] = df["_fp_tables"].apply(
        lambda fp: all(all_tables.get(t, 0) <= 1 for t in fp) if fp else False
    )

    return df


# ─────────────────────────────────────────────────────────────
# SECTION 4 — RECENCY & USAGE SCORING
# ─────────────────────────────────────────────────────────────

def score_recency_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Compute recency_days and usage_band for each report."""
    today = pd.Timestamp(date.today())

    df["recency_days"] = (today - df["last_accessed_date"]).dt.days
    # Treat missing dates as 999 (very stale)
    df["recency_days"] = df["recency_days"].fillna(999).astype(int)

    # Usage band
    def usage_band(views):
        if views == 0:        return "none"
        elif views <= 5:      return "low"
        elif views <= 30:     return "medium"
        else:                 return "high"

    df["usage_band"] = df["view_count_90d"].apply(usage_band)

    return df


# ─────────────────────────────────────────────────────────────
# SECTION 5 — SIMILARITY SCORING (Weighted Jaccard)
# ─────────────────────────────────────────────────────────────

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0  # Both empty = identical
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def compute_pairwise_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted Jaccard similarity for every report pair.
    Returns a similarity matrix and cluster assignments.
    """
    n = len(df)
    ids = df["report_id"].tolist()
    idx_map = {rid: i for i, rid in enumerate(ids)}

    # Similarity matrix
    sim_matrix = np.zeros((n, n))

    w_t = CONFIG["WEIGHT_TABLES"]
    w_m = CONFIG["WEIGHT_METRICS"]
    w_d = CONFIG["WEIGHT_DIMENSIONS"]

    for i, j in combinations(range(n), 2):
        row_i = df.iloc[i]
        row_j = df.iloc[j]

        sim = (
            w_t * jaccard(row_i["_fp_tables"],     row_j["_fp_tables"])     +
            w_m * jaccard(row_i["_fp_metrics"],    row_j["_fp_metrics"])    +
            w_d * jaccard(row_i["_fp_dimensions"], row_j["_fp_dimensions"])
        )
        sim_matrix[i][j] = sim
        sim_matrix[j][i] = sim

    # For each report: find its highest-similarity partner
    best_match_idx  = np.argmax(sim_matrix, axis=1)
    best_match_score = sim_matrix[np.arange(n), best_match_idx]

    df["_sim_score"]        = best_match_score.round(4)
    df["_sim_partner_id"]   = [ids[i] for i in best_match_idx]
    df["_sim_partner_name"] = [
        df.iloc[i]["report_name"] for i in best_match_idx
    ]

    # Assign consolidation clusters (greedy: seed from highest sim pairs)
    threshold = CONFIG["CONSOLIDATE_SIMILARITY_HIGH"]
    cluster_id = [-1] * n
    next_cluster = 0

    for i in range(n):
        if cluster_id[i] != -1:
            continue
        # Check all partners above threshold
        partners = [j for j in range(n) if i != j and sim_matrix[i][j] >= threshold]
        if partners:
            cid = next_cluster
            next_cluster += 1
            cluster_id[i] = cid
            for p in partners:
                if cluster_id[p] == -1:
                    cluster_id[p] = cid

    df["_cluster_id"] = cluster_id  # -1 = no cluster (unique)

    # Within each cluster, mark the anchor (highest total view count)
    df["_is_cluster_anchor"] = False
    for cid in set(c for c in cluster_id if c >= 0):
        members = df[df["_cluster_id"] == cid].copy()
        anchor_idx = members["view_count_90d"].idxmax()
        df.at[anchor_idx, "_is_cluster_anchor"] = True

    return df


# ─────────────────────────────────────────────────────────────
# SECTION 6 — CRED CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify_cred(row) -> tuple:
    """
    Apply CRED decision tree.
    Returns (cred_label, confidence, rationale)
    """
    elim_days  = CONFIG["ELIMINATE_RECENCY_DAYS"]
    dev_days   = CONFIG["DEVELOP_RECENCY_DAYS"]
    low_use    = CONFIG["LOW_USAGE_THRESHOLD"]
    high_use   = CONFIG["HIGH_USAGE_THRESHOLD"]
    sim_high   = CONFIG["CONSOLIDATE_SIMILARITY_HIGH"]
    sim_low    = CONFIG["CONSOLIDATE_SIMILARITY_LOW"]

    recency    = row["recency_days"]
    usage      = row["view_count_90d"]
    sim_score  = row["_sim_score"]
    in_cluster = row["_cluster_id"] >= 0
    is_anchor  = row["_is_cluster_anchor"]
    unique_src = row["_unique_tables"]

    # ── ELIMINATE ──────────────────────────────────────────
    if recency > elim_days and usage <= low_use:
        confidence = "HIGH" if recency > 180 and usage == 0 else "MEDIUM"
        return (
            "Eliminate",
            confidence,
            f"Not accessed in {recency} days and only {usage} views in last 90 days. "
            f"No evidence of active use."
        )

    if recency > elim_days and not in_cluster:
        return (
            "Eliminate",
            "MEDIUM",
            f"Not accessed in {recency} days. No similar reports suggest this is not "
            f"a consolidation target — low priority for migration."
        )

    # ── CONSOLIDATE ────────────────────────────────────────
    if in_cluster and not is_anchor:
        return (
            "Consolidate",
            "HIGH" if sim_score >= sim_high else "MEDIUM",
            f"Structural similarity of {sim_score:.0%} with '{row['_sim_partner_name']}'. "
            f"Recommend merging into the anchor report for this cluster."
        )

    # ── RETAIN ─────────────────────────────────────────────
    if recency <= 30 and usage >= high_use:
        return (
            "Retain",
            "HIGH",
            f"Actively used: {usage} views in 90 days, last accessed {recency} days ago. "
            f"{'Unique data source — no duplication risk.' if unique_src else 'High-value report.'}"
        )

    if is_anchor and usage >= high_use:
        return (
            "Retain",
            "HIGH",
            f"Anchor of a consolidation cluster with {usage} views. "
            f"Other similar reports should be merged into this one."
        )

    if unique_src and usage >= low_use and recency <= elim_days:
        return (
            "Retain",
            "MEDIUM",
            f"Only report using its source tables. {usage} views in 90 days. "
            f"Unique data asset — migrate with priority."
        )

    # ── DEVELOP ────────────────────────────────────────────
    if recency <= dev_days and usage > 0:
        return (
            "Develop",
            "MEDIUM",
            f"Active but underutilised ({usage} views, {recency} days since last access). "
            f"Consider redesigning for the new platform."
        )

    # Default: Develop (needs review)
    return (
        "Develop",
        "LOW",
        f"Does not clearly fit other categories. Manual review recommended. "
        f"({recency} days since access, {usage} views)"
    )


def apply_cred_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Apply CRED classification to every row."""
    results = df.apply(classify_cred, axis=1)
    df["cred_label"]      = [r[0] for r in results]
    df["cred_confidence"] = [r[1] for r in results]
    df["cred_rationale"]  = [r[2] for r in results]
    return df


# ─────────────────────────────────────────────────────────────
# SECTION 7 — OUTPUT PREPARATION
# ─────────────────────────────────────────────────────────────

TECHNICAL_COLUMNS = [
    "report_id", "report_name", "source_platform", "business_domain",
    "owner_email", "report_type", "last_accessed_date", "recency_days",
    "view_count_90d", "usage_band", "source_tables", "metrics", "dimensions",
    "_sim_score", "_sim_partner_id", "_sim_partner_name",
    "_cluster_id", "_is_cluster_anchor", "_unique_tables",
    "cred_label", "cred_confidence", "cred_rationale", "tags"
]

EXEC_COLUMNS = [
    "report_id", "report_name", "business_domain", "owner_email",
    "last_accessed_date", "view_count_90d",
    "cred_label", "cred_confidence", "cred_rationale"
]

def prepare_outputs(df: pd.DataFrame) -> tuple:
    """Return (technical_df, executive_df, summary_dict)."""
    tech_df = df[[c for c in TECHNICAL_COLUMNS if c in df.columns]].copy()
    exec_df = df[[c for c in EXEC_COLUMNS if c in df.columns]].copy()

    # Summary stats
    total = len(df)
    by_label = df["cred_label"].value_counts().to_dict()
    summary = {
        "total_reports": total,
        "consolidate":   by_label.get("Consolidate", 0),
        "retain":        by_label.get("Retain", 0),
        "eliminate":     by_label.get("Eliminate", 0),
        "develop":       by_label.get("Develop", 0),
        "estimated_migration_reduction_pct": round(
            (by_label.get("Eliminate", 0) + by_label.get("Consolidate", 0)) / total * 100, 1
        ) if total > 0 else 0,
        "consolidation_clusters": int(df["_cluster_id"].max()) + 1 if df["_cluster_id"].max() >= 0 else 0,
        "run_timestamp": datetime.now().isoformat()
    }

    return tech_df, exec_df, summary


# ─────────────────────────────────────────────────────────────
# SECTION 8 — MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_cred_pipeline(input_df: pd.DataFrame) -> tuple:
    """
    Full CRED pipeline.

    Args:
        input_df: DataFrame matching the CRED metadata template

    Returns:
        (technical_df, executive_df, summary_dict)

    Usage in Fabric Notebook:
        df = spark.read.format("delta").load("Tables/cred_metadata_input").toPandas()
        tech, exec, summary = run_cred_pipeline(df)
        spark.createDataFrame(tech).write.format("delta").mode("overwrite")
             .saveAsTable("cred_scored_reports_technical")
        spark.createDataFrame(exec).write.format("delta").mode("overwrite")
             .saveAsTable("cred_scored_reports_executive")
    """
    print(f"[CRED] Starting pipeline — {len(input_df)} reports")

    df = load_and_validate(input_df.copy())
    print(f"[CRED] Validation passed")

    df = build_fingerprints(df)
    print(f"[CRED] Fingerprints built")

    df = score_recency_usage(df)
    print(f"[CRED] Recency & usage scored")

    df = compute_pairwise_similarity(df)
    clusters = int(df["_cluster_id"].max()) + 1 if df["_cluster_id"].max() >= 0 else 0
    print(f"[CRED] Similarity computed — {clusters} consolidation clusters found")

    df = apply_cred_classification(df)
    print(f"[CRED] CRED classification complete")

    tech_df, exec_df, summary = prepare_outputs(df)

    print(f"\n[CRED] ── SUMMARY ────────────────────────────")
    print(f"  Total reports:     {summary['total_reports']}")
    print(f"  Consolidate:       {summary['consolidate']}")
    print(f"  Retain:            {summary['retain']}")
    print(f"  Eliminate:         {summary['eliminate']}")
    print(f"  Develop:           {summary['develop']}")
    print(f"  Migration saving:  ~{summary['estimated_migration_reduction_pct']}% reduction")
    print(f"──────────────────────────────────────────────\n")

    return tech_df, exec_df, summary


# ─────────────────────────────────────────────────────────────
# SECTION 9 — QUICK TEST WITH SAMPLE DATA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_data = {
        "report_id":          ["RPT001", "RPT002", "RPT003", "RPT004", "RPT005", "RPT006"],
        "report_name":        ["Monthly Sales", "Sales Summary", "Customer Churn",
                               "Inventory Levels", "Old HR Report", "Finance Dashboard"],
        "report_description": ["Monthly rev by region", "Sales by region monthly",
                               "Churn analysis", "Stock levels", "Legacy HR data", "Finance KPIs"],
        "source_platform":    ["Tableau", "Tableau", "Tableau", "Tableau", "Crystal", "Qlik"],
        "business_domain":    ["Sales", "Sales", "Customer", "Ops", "HR", "Finance"],
        "owner_email":        ["alice@co.com", "bob@co.com", "carol@co.com",
                               "dave@co.com", "eve@co.com", "frank@co.com"],
        "last_accessed_date": ["2025-12-01", "2025-11-28", "2026-02-15",
                               "2026-03-01", "2024-06-01", "2026-03-05"],
        "view_count_90d":     [45, 38, 12, 67, 1, 89],
        "source_tables":      [
            "sales_fact|region_dim|date_dim",
            "sales_fact|region_dim|date_dim|product_dim",
            "customer_fact|churn_model|date_dim",
            "inventory_fact|warehouse_dim|date_dim",
            "hr_legacy_table",
            "finance_fact|gl_dim|cost_centre_dim|date_dim"
        ],
        "metrics":            [
            "total_revenue|units_sold|avg_order",
            "total_revenue|units_sold|margin",
            "churn_rate|ltv|cohort_size",
            "stock_level|reorder_point|days_cover",
            "headcount|attrition",
            "revenue|ebitda|opex|capex"
        ],
        "dimensions":         [
            "region|month|product",
            "region|month|product|channel",
            "customer_segment|month",
            "warehouse|product|week",
            "department|month",
            "cost_centre|quarter|entity"
        ],
        "report_type":        ["scheduled", "scheduled", "ad-hoc", "operational", "scheduled", "executive"],
        "tags":               ["finance|sales", "finance|sales", "cx|analytics", "ops", "legacy|hr", "finance|exec"],
    }

    df_sample = pd.DataFrame(sample_data)
    tech, exc, summ = run_cred_pipeline(df_sample)

    print("\nTechnical Output (key columns):")
    print(tech[["report_id", "report_name", "recency_days", "view_count_90d",
                "_sim_score", "_cluster_id", "cred_label", "cred_confidence"]].to_string(index=False))
