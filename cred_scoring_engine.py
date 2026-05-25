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
    "report_id": [
        f"RPT{i:03d}" for i in range(1, 101)
    ],
    
    "report_name": [
        # 1-15: Active Core Reports (Retain / Core)
        "Monthly Sales Regional", "Finance Core Dashboard", "Customer Churn Analysis", "Inventory Tracking Live", "Marketing ROI Tracker",
        "Executive Strategy Deck", "Daily Revenue Flash", "OPEX Planning Tool", "Global Supply Chain KPI", "HR Employee Headcount",
        "Product Margin Deep-Dive", "Support Ticket Backlog", "Online Store Traffic", "Procurement Spend Audit", "Compliance Risk Matrix",
        # 16-35: Duplicates & Overlaps (Consolidation Candidates)
        "Sales Summary Regional", "Regional Sales Overview", "Territory Sales Breakdown", "Finance KPI Master", "Finance Key Metrics",
        "EBITDA Summary Report", "User Churn Insight", "Customer Retention Monitor", "Warehouse Stock Levels", "Stock Count Summary",
        "Logistics Route Efficiency", "Shipping Performance Live", "Ad Spend Performance", "Digital Marketing ROI", "Campaign Conversion V2",
        "Lead Gen Funnel Analysis", "Hourly Sales Monitor", "Intraday Revenue Tracker", "QBR Financial Slide", "Corporate Board Pack",
        # 36-65: Stale & Legacy Reports (Eliminate Candidates)
        "Old HR Report 2024", "Legacy Headcount Flat File", "System Outage Logs 2023", "Network Uptime Archive", "GDPR Purge Log v1",
        "2022 Marketing Archives", "Decommissioned Sales View", "Temporary Promo Check", "Test Report Alice", "Dump Table Validation",
        "Backup Inventory Crystal", "Legacy GL Reconciliation", "Ariba Spend Export 2024", "Former Customer Feedback", "Survey Results 2023",
        "Training Completion 2021", "Onboarding Progress Old", "IT Hardware Audit Legacy", "Server Rack Space 2022", "Mobile App Beta Feedback",
        "Web Traffic Archive 2023", "AdHoc Discount Audit", "Sales Compensation Draft", "Ex-Employee Account Audit", "Office Supply Inventory",
        "Travel Expense Dump 2024", "Catering Cost Tracker", "Print Queue Metrics", "Legacy Billing Exceptions", "Pre-Migration Mapping Data",
        # 66-100: Active Mid-to-Low Usage Operational / Specialized (Develop / Optimize / Retain)
        "Legal Contract Expirations", "Data Quality Null Checker", "User Role Access Matrix", "IAM Orphaned Accounts", "Tax Filing Prep Ledger",
        "Treasury Cash Flow Forecast", "Asset Depreciation Schedule", "R&D Project Milestones", "SaaS Subscription Optimizer", "Facilities Energy Usage",
        "Fleet Vehicle Maintenance", "Vendor SLA Breach Log", "Quality Assurance Defect Rate", "Manufacturing Line Yield", "Packaging Waste Metrics",
        "Customer NPS Feed", "Social Media Sentiment Score", "Affiliate Referrals Tracker", "SEO Keyword Ranking Performance", "Email Campaign Open Rates",
        "Wholesale Partner Revenue", "Returns & Refunds Breakdown", "Gift Card Liability Balance", "Fraud Prevention Alert Log", "Point of Sale Terminal Health",
        "DB Maintenance Job Status", "API Gateway Latency Logs", "Cloud Infrastructure Spend", "CI/CD Pipeline Build Times", "Data Warehouse Sync Status",
        "Internal Audit Log Weekly", "BCP Drills Progress Report", "Sustainability Carbon Footprint", "Executive Compensation View", "Mergers & Acquisitions Pipeline"
    ],
    
    "report_description": [
        # 1-15
        "Monthly rev by region", "Finance KPIs", "Churn analysis", "Stock levels", "Digital ad spend",
        "Board KPIs", "Daily sales volume", "EBITDA by cost center", "Vendor performance", "Current active employees",
        "Total revenue margin", "Open Jira tickets", "Web to lead conversion", "System audit logs", "Compliance training status",
        # 16-35
        "Sales by region monthly", "Sales by territory", "Territory sales revenue", "Finance metrics dashboard", "Corporate finance overview",
        "Revenue ebitda opex capex", "Customer churn tracking", "Customer retention rates", "Stock by warehouse", "Inventory items summary",
        "Delivery routes efficiency", "Transit times and delays", "Return on ad spend analytics", "Marketing channels ROI", "Web conversion metrics",
        "Marketing lead tracking", "Hourly sales volume track", "Intraday sales revenue", "Executive QBR metrics", "Board presentation metrics",
        # 36-65
        "Legacy HR data", "Old headcount tracking file", "Historic IT system outages", "2023 network availability", "GDPR cleanup data",
        "Historic ad performance data", "Old tableau sales view", "One time promotional check", "Testing data framework", "Data validation testing",
        "Old warehouse metrics", "Legacy general ledger data", "2024 vendor procurement data", "Customer satisfaction 2023", "Survey results archive",
        "Compliance course metrics 2021", "Old onboarding pipeline status", "2022 computer hardware audit", "Datacenter usage history", "Beta software review data",
        "Google analytics dump 2023", "Discount validations historic", "Commission mapping draft", "Terminated user security records", "Stationery inventory records",
        "Business travel spending 2024", "Event food metrics", "Printer log analytics", "Historic billing system errors", "System conversion crosswalk",
        # 66-100
        "Legal agreements renewal timeline", "Missing customer fields analysis", "Access rights matrix", "Terminated user access audits", "Tax ledger data prep",
        "Predictive corporate cash flow", "Fixed asset depreciation schedules", "Engineering phase tracking", "SaaS application licenses count", "Corporate building utility cost",
        "Company vehicles servicing records", "Vendor operational SLA metrics", "Software bug density report", "Factory assembly line output", "Material packaging sustainability",
        "Net promoter scores from survey", "Brand reputation scoring metrics", "External referral sales tracking", "Organic search tracking analysis", "Newsletter delivery rates",
        "B2B partner revenue splits", "Product returns itemization", "Unredeemed gift cards liability", "High risk financial transactions", "Retail store hardware checks",
        "Database indexing jobs health", "Microservice performance metrics", "AWS Azure cloud cost burn", "DevOps build pipeline speed", "ETL data pipeline execution times",
        "Weekly compliance access log", "Disaster recovery testing tracking", "Corporate greenhouse emissions", "Exec salary and stock tracking", "Corporate investment evaluations"
    ],
    
    "source_platform": [
        "Tableau", "Qlik", "Tableau", "Tableau", "Looker", "Power BI", "Power BI", "Looker", "Tableau", "Workday",
        "Tableau", "Power BI", "Tableau", "Custom App", "Looker", "Tableau", "Looker", "Power BI", "Qlik", "Power BI",
        "Power BI", "Tableau", "Looker", "Power BI", "Tableau", "Looker", "Tableau", "Tableau", "Looker", "Tableau",
        "Looker", "Power BI", "Power BI", "Power BI", "Power BI", "Crystal", "SSRS", "SSRS", "SSRS", "Custom App",
        "Tableau", "Tableau", "Looker", "Tableau", "Power BI", "Crystal", "Cognos", "SSRS", "Looker", "Qlik",
        "Cognos", "Tableau", "SSRS", "SSRS", "Custom App", "Looker", "Tableau", "Power BI", "Custom App", "Looker",
        "SSRS", "Tableau", "Power BI", "SSRS", "Looker", "Looker", "Custom App", "SSRS", "Custom App", "Looker",
        "Power BI", "Power BI", "Looker", "Looker", "Tableau", "Tableau", "Looker", "Tableau", "Looker", "Tableau",
        "Power BI", "Looker", "Tableau", "Custom App", "Power BI", "Custom App", "Custom App", "SSRS", "Power BI", "Looker",
        "Custom App", "Power BI", "Looker", "Power BI", "Tableau", "Custom App", "SSRS", "Looker", "Power BI", "Power BI"
    ],
    
    "business_domain": [
        "Sales", "Finance", "Customer", "Ops", "Marketing", "Executive", "Sales", "Finance", "Procurement", "HR",
        "Sales", "IT", "Marketing", "IT", "HR", "Sales", "Sales", "Sales", "Finance", "Finance",
        "Finance", "Customer", "Customer", "Ops", "Ops", "Ops", "Ops", "Marketing", "Marketing", "Marketing",
        "Marketing", "Sales", "Sales", "Executive", "Executive", "HR", "HR", "IT", "IT", "Legal",
        "Marketing", "Sales", "Marketing", "IT", "IT", "Ops", "Finance", "Procurement", "Customer", "Customer",
        "HR", "HR", "IT", "IT", "Product", "Marketing", "Sales", "Sales", "Security", "Ops",
        "Finance", "Ops", "IT", "Finance", "IT", "Legal", "Data", "Security", "Security", "Finance",
        "Finance", "Finance", "Product", "IT", "Ops", "Ops", "Procurement", "Product", "Ops", "Ops",
        "Customer", "Marketing", "Marketing", "Marketing", "Marketing", "Sales", "Product", "Finance", "Security", "Ops",
        "IT", "IT", "IT", "IT", "IT", "IT", "IT", "Legal", "Executive", "Finance"
    ],
    
    "owner_email": [
        "alice@co.com", "frank@co.com", "carol@co.com", "dave@co.com", "mike@co.com", "ceo@co.com", "alice@co.com", "frank@co.com", "lee@co.com", "eve@co.com",
        "bob@co.com", "sysadmin@co.com", "jen@co.com", "sec_ops@co.com", "eve@co.com", "bob@co.com", "tom@co.com", "alice@co.com", "frank@co.com", "sara@co.com",
        "sara@co.com", "carol@co.com", "carol@co.com", "dave@co.com", "dave@co.com", "lee@co.com", "lee@co.com", "mike@co.com", "mike@co.com", "jen@co.com",
        "jen@co.com", "alice@co.com", "alice@co.com", "ceo@co.com", "ceo@co.com", "eve@co.com", "eve@co.com", "inactive_user@co.com", "inactive_user@co.com", "legal@co.com",
        "inactive_user@co.com", "bob@co.com", "mike@co.com", "data_eng@co.com", "data_eng@co.com", "dave@co.com", "frank@co.com", "lee@co.com", "carol@co.com", "carol@co.com",
        "eve@co.com", "eve@co.com", "sysadmin@co.com", "sysadmin@co.com", "product_owner@co.com", "mike@co.com", "bob@co.com", "alice@co.com", "sec_ops@co.com", "dave@co.com",
        "frank@co.com", "dave@co.com", "sysadmin@co.com", "frank@co.com", "data_eng@co.com", "legal@co.com", "data_eng@co.com", "sec_ops@co.com", "sec_ops@co.com", "sara@co.com",
        "sara@co.com", "sara@co.com", "product_owner@co.com", "sysadmin@co.com", "dave@co.com", "lee@co.com", "lee@co.com", "product_owner@co.com", "dave@co.com", "dave@co.com",
        "carol@co.com", "mike@co.com", "mike@co.com", "mike@co.com", "jen@co.com", "bob@co.com", "product_owner@co.com", "sara@co.com", "sec_ops@co.com", "dave@co.com",
        "sysadmin@co.com", "sysadmin@co.com", "sysadmin@co.com", "sysadmin@co.com", "data_eng@co.com", "sec_ops@co.com", "sysadmin@co.com", "legal@co.com", "ceo@co.com", "frank@co.com"
    ],
    
    "last_accessed_date": [
        # 1-15: Recent
        "2026-05-20", "2026-05-19", "2026-05-10", "2026-05-21", "2026-05-15", "2026-05-01", "2026-05-21", "2026-05-18", "2025-11-20", "2026-05-10",
        "2026-05-14", "2026-05-21", "2026-04-30", "2026-05-20", "2026-03-05",
        # 16-35: Recent to Semi-Recent (Consolidation Targets)
        "2026-05-14", "2026-05-20", "2026-05-11", "2026-05-19", "2026-05-15", "2026-05-18", "2026-05-10", "2026-05-15", "2026-05-21", "2026-05-12",
        "2026-05-15", "2026-05-08", "2026-05-12", "2026-05-15", "2026-04-28", "2026-05-13", "2026-05-21", "2026-05-21", "2026-05-01", "2026-04-25",
        # 36-65: Stale / Ancient (Eliminate Targets)
        "2024-06-01", "2024-11-15", "2020-12-15", "2021-05-20", "2023-08-11", "2023-01-10", "2024-02-14", "2025-01-05", "2025-03-10", "2024-12-25",
        "2022-04-18", "2023-09-30", "2024-05-14", "2023-11-01", "2023-07-19", "2021-12-31", "2024-08-22", "2022-10-05", "2022-03-14", "2024-04-01",
        "2023-06-12", "2025-02-18", "2025-03-20", "2023-10-10", "2024-07-07", "2024-01-15", "2024-09-09", "2024-02-28", "2024-11-30", "2023-04-15",
        # 66-100: Active Mid/Low Usage
        "2026-05-18", "2026-05-20", "2026-05-10", "2026-05-10", "2026-05-05", "2026-05-19", "2026-05-16", "2026-05-12", "2026-05-15", "2026-05-11",
        "2026-05-14", "2026-05-02", "2026-05-20", "2026-05-21", "2026-05-19", "2026-05-15", "2026-05-14", "2026-05-17", "2026-05-20", "2026-05-19",
        "2026-05-11", "2026-05-15", "2026-05-09", "2026-05-20", "2026-05-21", "2026-05-20", "2026-05-21", "2026-05-21", "2026-05-21", "2026-05-21",
        "2026-05-20", "2026-05-05", "2026-05-13", "2026-05-01", "2026-05-18"
    ],
    
    "view_count_90d": [
        # 1-15
        145, 210, 12, 67, 340, 12, 450, 120, 5, 85, 38, 950, 95, 45, 89,
        # 16-35 (Duplicates with real use)
        138, 89, 44, 180, 92, 65, 112, 430, 1150, 310, 45, 28, 310, 105, 125, 66, 240, 195, 35, 18,
        # 36-65 (Stale / Zero-low usage)
        1, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 5, 2, 0, 0, 1, 0, 0, 0, 0,
        # 66-100 (Operational specialized)
        55, 25, 2, 18, 34, 76, 19, 42, 61, 23, 14, 52, 165, 290, 31, 430, 88, 114, 73, 152, 83, 91, 24, 118, 201,
        15, 84, 340, 112, 405, 27, 8, 16, 4, 11
    ],
    
    "source_tables": [
        # 1-15
        "sales_fact|region_dim|date_dim", "finance_fact|gl_dim|date_dim", "customer_fact|churn_model|date_dim", "inventory_fact|warehouse_dim|date_dim", "mktg_spend_fact|campaign_dim",
        "exec_kpi_summary_table", "sales_fact|date_dim", "finance_fact|cost_centre_dim", "po_fact|vendor_dim", "employee_dim|dept_dim",
        "sales_fact|region_dim|date_dim|product_dim", "jira_tickets_fact", "web_traffic_fact|lead_dim", "audit_log_fact", "lms_course_fact",
        # 16-35 (Intentionally overlapping schemas to trigger consolidation logic)
        "sales_fact|region_dim|date_dim", "sales_fact|territory_dim", "sales_fact|territory_dim", "finance_fact|gl_dim|date_dim", "finance_fact|gl_dim",
        "finance_fact|gl_dim|cost_centre_dim|date_dim", "customer_fact|churn_model|date_dim", "survey_results_fact|customer_dim", "inventory_fact|warehouse_dim", "inventory_fact|warehouse_dim",
        "shipping_fact|route_dim", "shipping_fact|route_dim", "mktg_spend_fact|platform_dim", "mktg_spend_fact|campaign_dim", "web_traffic_fact|lead_dim",
        "web_traffic_fact|lead_dim", "sales_fact|date_dim", "sales_fact|date_dim", "exec_kpi_summary_table", "exec_kpi_summary_table",
        # 36-65 (Legacy / Unused models)
        "hr_legacy_table", "hr_legacy_table", "net_logs_2019", "net_logs_2019", "gdpr_requests_fact",
        "mktg_legacy_flat_file", "sales_fact|region_dim", "promo_code_dim", "test_scratch_table", "temp_validation_run",
        "inventory_fact|warehouse_dim", "finance_fact|gl_dim", "po_fact|vendor_dim", "survey_results_fact", "survey_results_fact",
        "lms_course_fact", "employee_dim", "iam_users_dim", "iam_users_dim", "customer_fact",
        "web_traffic_fact", "sales_fact", "sales_fact", "iam_users_dim", "office_supplies_table",
        "finance_fact", "vendor_dim", "it_assets_dim", "finance_fact", "system_crosswalk_table",
        # 66-100 (Operational specific schemas)
        "legal_contracts_fact|vendor_dim", "customer_dim", "iam_users_dim", "iam_users_dim|hr_term_table", "tax_ledger_table",
        "finance_fact|cash_flow_model", "asset_depreciation_dim|gl_dim", "project_milestones_fact", "saas_licenses_dim", "facilities_utility_fact",
        "fleet_maintenance_fact|vehicle_dim", "po_fact|vendor_dim|sla_dim", "qa_defects_fact|product_dim", "manufacturing_yield_fact|plant_dim", "sustainability_waste_fact",
        "survey_results_fact|customer_dim", "social_sentiment_flat_file", "affiliate_sales_fact|partner_dim", "seo_rankings_fact|keyword_dim", "email_metrics_fact",
        "sales_fact|partner_dim|date_dim", "returns_fact|product_dim|date_dim", "gift_card_liability_table", "fraud_alerts_fact|customer_dim", "pos_hardware_dim|store_dim",
        "db_maintenance_logs", "api_gateway_logs", "cloud_cost_fact|infrastructure_dim", "devops_build_fact", "etl_pipeline_logs",
        "audit_log_fact|iam_users_dim", "dr_testing_fact", "carbon_emissions_fact", "employee_dim|payroll_fact", "ma_pipeline_dim"
    ],
    
    "metrics": [
        # 1-15
        "total_revenue|units_sold", "revenue|ebitda|opex", "churn_rate|ltv", "stock_level|days_cover", "roi|cpa|cac",
        "arr|nrr|ebitda|churn", "daily_sales|wow_growth", "ebitda|margin_pct", "sla_breach_count|avg_delay", "active_headcount|new_hires",
        "total_revenue|units_sold|margin", "open_tickets|avg_resolution_time", "conversion_rate|cost_per_lead", "login_count|failed_attempts", "completion_pct",
        # 16-35
        "total_revenue|units_sold|avg_order", "total_revenue|quota_attainment", "total_revenue|units_sold", "revenue|ebitda|opex", "revenue|opex",
        "revenue|ebitda|opex|capex", "churn_rate|ltv|cohort_size", "nps_score|response_rate", "stock_level|days_cover", "stock_level|reorder_point",
        "transit_time|on_time_pct", "transit_time|avg_delay", "cpa|impressions|clicks", "roi|cpa|cac", "conversion_rate|cost_per_lead",
        "cost_per_lead|clicks", "daily_sales|wow_growth", "daily_sales", "arr|nrr", "arr|nrr|ebitda",
        # 36-65
        "headcount|attrition", "headcount", "uptime_pct|downtime_hrs", "uptime_pct", "deletion_sla_breach",
        "spend|clicks", "total_revenue", "promo_count", "null_count", "row_count",
        "stock_level", "revenue", "spend", "nps_score", "response_rate",
        "completion_pct", "new_hires", "access_level_count", "access_level_count", "churn_rate",
        "clicks", "total_revenue", "total_revenue", "access_level_count", "supply_count",
        "revenue", "spend", "item_count", "revenue", "mapping_count",
        # 66-100
        "active_contracts|expired_contracts", "null_count|pct_missing", "access_level_count", "orphaned_account_count", "tax_liability|deductions",
        "forecasted_cash|net_variance", "depreciation_value|asset_lifespan", "completed_milestones|slippage_days", "license_count|total_spend|unused_licenses", "kwh_consumed|total_utility_cost",
        "maintenance_cost|downtime_days", "sla_breach_pct|avg_penalty_fee", "defect_count|rejection_rate", "units_produced|yield_pct", "waste_weight_kg|recycling_pct",
        "nps_score|promoter_count|detractor_count", "sentiment_score|mention_volume", "referral_revenue|payout_amount", "avg_keyword_position|search_volume", "open_rate|click_through_pct",
        "wholesale_revenue|partner_commission", "return_count|refund_amount_total", "outstanding_balance|expiration_volume", "fraud_alert_count|blocked_amount", "terminal_uptime_pct|incident_count",
        "backup_success_rate|index_fragmentation_pct", "p99_latency_ms|error_rate_pct", "total_cloud_spend|unblended_cost", "build_success_rate|avg_build_duration_mins", "rows_transferred|pipeline_duration_sec",
        "failed_login_attempts|privilege_escalation_count", "recovery_time_objective_ms|drill_success_rate", "co2_emissions_tons|offset_credits", "base_salary|bonus_pool|options_granted", "deal_value_millions|pipeline_stage_count"
    ],
    
    "dimensions": [
        # 1-15
        "region|month", "cost_centre|quarter", "customer_segment|month", "warehouse|product", "campaign|channel",
        "entity|quarter", "date", "cost_centre|month", "vendor|category", "department|role|location",
        "region|month|product|channel", "priority|assignee", "source|campaign", "system|user_role", "course|department",
        # 16-35
        "region|month|product", "territory|quarter", "territory|month", "cost_centre|quarter", "entity|year",
        "cost_centre|quarter|entity", "customer_segment|month", "segment|country", "warehouse|product", "warehouse|product|week",
        "route|carrier", "route|carrier|week", "platform|week", "campaign|channel", "source|campaign",
        "campaign|week", "date", "date|hour", "entity|quarter", "entity|quarter|year",
        # 36-65
        "department|month", "department", "node|month", "node", "country|request_type",
        "campaign|month", "region", "promo_code", "field_name", "table_name",
        "warehouse", "cost_centre", "vendor", "segment", "country",
        "course", "department", "system", "system", "customer_segment",
        "source", "region", "month", "system", "item_type",
        "cost_centre", "vendor", "asset_type", "cost_centre", "field_name",
        # 66-100
        "contract_type|expiry_month", "field_name|table_name", "system|department", "system|manager", "tax_year|country",
        "currency|forecast_month", "asset_category|depreciation_method", "project_id|team_lead", "vendor|department", "building_id|energy_source",
        "vehicle_type|location", "vendor_name|sla_tier", "product_line|defect_type", "factory_id|shift_leader", "facility|material_type",
        "customer_tier|quarter", "platform|sentiment_type", "partner_id|program_tier", "keyword|search_engine", "campaign_id|list_segment",
        "partner_name|product_category", "reason_code|store_location", "issuance_quarter|country", "risk_score_bucket|country", "store_id|device_type",
        "database_name|server_cluster", "api_endpoint|http_status", "cloud_provider|service_name|account_id", "repository_name|branch", "pipeline_name|target_table",
        "user_id|application_id", "business_unit|scenario_type", "facility_id|emission_scope", "grade_level|department", "target_company|industry_vertical"
    ],
    
    "report_type": [
        # 1-15
        "scheduled", "executive", "ad-hoc", "operational", "scheduled", "executive", "operational", "executive", "scheduled", "scheduled",
        "scheduled", "operational", "ad-hoc", "scheduled", "ad-hoc",
        # 16-35
        "scheduled", "ad-hoc", "scheduled", "scheduled", "executive", "executive", "ad-hoc", "scheduled", "operational", "operational",
        "ad-hoc", "operational", "ad-hoc", "scheduled", "ad-hoc", "ad-hoc", "operational", "operational", "executive", "executive",
        # 36-65
        "scheduled", "scheduled", "operational", "scheduled", "scheduled", "scheduled", "ad-hoc", "ad-hoc", "operational", "operational",
        "scheduled", "scheduled", "scheduled", "ad-hoc", "scheduled", "scheduled", "scheduled", "operational", "scheduled", "ad-hoc",
        "scheduled", "ad-hoc", "scheduled", "ad-hoc", "scheduled", "scheduled", "scheduled", "ad-hoc", "scheduled", "scheduled",
        # 66-100
        "scheduled", "operational", "scheduled", "ad-hoc", "scheduled", "executive", "scheduled", "operational", "scheduled", "operational",
        "scheduled", "operational", "operational", "operational", "scheduled", "scheduled", "ad-hoc", "scheduled", "ad-hoc", "scheduled",
        "scheduled", "operational", "executive", "operational", "operational", "operational", "operational", "executive", "operational", "operational",
        "scheduled", "ad-hoc", "scheduled", "executive", "executive"
    ],
    
    "tags": [
        # 1-15
        "finance|sales", "finance|exec", "cx|analytics", "ops", "growth|marketing", "exec|board", "sales|daily", "finance|exec", "procurement", "hr|core",
        "finance|sales", "it|helpdesk", "marketing|web", "it|security", "hr|compliance",
        # 16-35
        "finance|sales", "sales|geo", "sales|geo", "finance|exec", "finance", "finance|exec", "cx|analytics", "cx|survey", "ops|inventory", "ops|inventory",
        "ops|logistics", "ops|logistics", "marketing|digital", "growth|marketing", "marketing|web", "marketing|web", "sales|daily", "sales|daily", "exec|board", "exec|board",
        # 36-65
        "legacy|hr", "hr|legacy", "it|legacy", "it|legacy", "legal|privacy", "marketing|legacy", "sales|daily|deprecated", "marketing|legacy", "test", "test",
        "ops|legacy", "finance|legacy", "procurement|legacy", "cx|legacy", "cx|legacy", "hr|compliance", "hr|legacy", "security|legacy", "it|legacy", "cx|legacy",
        "marketing|legacy", "sales|legacy", "sales|legacy", "security|legacy", "ops|legacy", "finance|legacy", "procurement|legacy", "it|legacy", "finance|legacy", "it|legacy",
        # 66-100
        "legal|contracts", "data_eng|dq", "security|audit", "security|audit", "finance|tax", "finance|treasury", "finance|assets", "it|pm", "it|licensing", "ops|facilities",
        "ops|fleet", "procurement|sla", "product|qa", "ops|manufacturing", "ops|sustainability", "cx|survey", "marketing|sentiment", "marketing|affiliates", "marketing|seo", "marketing|email",
        "finance|sales", "ops|returns", "finance|liability", "security|fraud", "ops|retail", "it|database", "it|performance", "it|cloud", "it|devops", "data_eng|etl",
        "security|audit", "it|bcp", "ops|sustainability", "hr|exec", "exec|strategy"
    ]
}

    df_sample = pd.DataFrame(sample_data)
    tech, exc, summ = run_cred_pipeline(df_sample)

    print("\nTechnical Output (key columns):")
    print(tech[["report_id", "report_name", "recency_days", "view_count_90d",
                "_sim_score", "_cluster_id", "cred_label", "cred_confidence"]].to_string(index=False))
    print("\nExecutive Output:")
    print(exc.to_string(index=False))