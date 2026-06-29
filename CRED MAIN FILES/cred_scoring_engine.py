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

    # Develop — gap-based classifier (Adoption / Content / UX / Health)
    # Each gap is a 0–1 score; max score across the 4 must clear the threshold.
    "DEVELOP_GAP_THRESHOLD": 0.35,        # min dominant-gap score to label Develop
    "ADOPTION_RICHNESS_DIVISOR": 10.0,    # (metrics + 0.5*dims) / this = richness
    "CONTENT_THINNESS_DIVISOR": 8.0,      # 1 - (metrics + tables) / this = thinness
    "UX_HAS_CONTENT_DIVISOR": 5.0,        # (metrics + tables) / this = has_content

    # Develop — operational signals (Phase 2 enrichment).
    # All 5 source columns are OPTIONAL. Missing/NaN values default to neutral
    # so the engine still runs cleanly on tenants without refresh/activity logs.
    "HEALTH_DIVISOR":                0.30,   # refresh_fail_rate_30d / this → 0..1
    "UX_FAILURE_DIVISOR":            0.20,   # activity_fail_rate_30d / this → 0..1
    "MOBILE_SHARE_THRESHOLD":        0.40,   # ≥this share of mobile views triggers mismatch check
    "ADOPTION_EDIT_RATIO_THRESHOLD": 0.50,   # edits/views ≥ this boosts adoption_gap
    "ADOPTION_EDIT_RATIO_BOOST":     0.20,   # additive boost (capped at 1.0)

    # Boilerplate dimension redaction
    # Manual stoplist: known generic terms that inflate similarity without meaning
    "DIMENSION_STOPLIST": {
        "date", "month", "year", "region", "page 1", "page 2", "page 3",
        "overview", "details", "summary",
    },
    # Auto-detection: dimensions appearing in >70% of reports are treated as boilerplate
    "STOPLIST_PREVALENCE_THRESHOLD": 0.70,

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

def build_fingerprints(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Build structural fingerprint sets for each report."""
    df["_fp_tables"]     = df["source_tables"].apply(parse_pipe)
    df["_fp_metrics"]    = df["metrics"].apply(parse_pipe)
    df["_fp_dimensions"] = df["dimensions"].apply(parse_pipe)
    df["_fp_tags"]       = df["tags"].apply(parse_pipe)

    # ── Boilerplate dimension redaction ─────────────────────
    manual_stoplist = CONFIG.get("DIMENSION_STOPLIST", set())
    prevalence_threshold = CONFIG.get("STOPLIST_PREVALENCE_THRESHOLD", 0.70)
    n_reports = len(df)

    # Count how many reports use each dimension
    all_dims = pd.Series(
        [d for fp in df["_fp_dimensions"] for d in fp]
    ).value_counts()

    # Auto-detect: dimensions in >70% of reports
    auto_stoplist = set(
        all_dims[all_dims > (prevalence_threshold * n_reports)].index
    ) if n_reports > 0 else set()

    # Combined stoplist
    combined_stoplist = manual_stoplist | auto_stoplist

    if combined_stoplist:
        # Store what was redacted (for audit)
        df["_redacted_dims"] = df["_fp_dimensions"].apply(
            lambda fp: fp & combined_stoplist
        )
        # Remove boilerplate from fingerprint
        df["_fp_dimensions"] = df["_fp_dimensions"].apply(
            lambda fp: fp - combined_stoplist
        )
        if verbose:
            print(f"  [REDACT] Removed {len(combined_stoplist)} boilerplate dimensions: "
                  f"{', '.join(sorted(combined_stoplist))}")
    else:
        df["_redacted_dims"] = df["_fp_dimensions"].apply(lambda fp: set())
    # ───────────────────────────────────────────────────────

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

# ── Develop sub-classifier: 4 gap functions + orchestrator ───
# Each gap returns a 0–1 score where:
#   0.0 = no gap (the diagnosis does not apply)
#   1.0 = textbook case for this gap
# Adoption / Content / UX read the CLEANED fingerprint sets produced by
# build_fingerprints() — boilerplate dimensions have already been redacted.
# Health reads the optional refresh_fail_rate_30d column.

def _safe_get(row, col, default):
    """Return row[col] if present and not NaN, else default."""
    val = row.get(col, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


_DEVELOP_ACTIONS = {
    "Adoption": "promote and train users; do not rebuild — the asset is fine",
    "Content":  "expand the data model and add missing metrics — demand is proven",
    "UX":       "redesign visuals/layout or fix surface errors to revive engagement",
    "Health":   "fix the dataset refresh pipeline — content is fine but data is going stale",
}


def adoption_gap(row) -> float:
    """Adoption gap: rich content but weak usage. Action = promote, not rebuild.
    Sharpened by edits-vs-views ratio: a team actively editing a low-traffic
    report is the textbook adoption-gap case.
    """
    metrics = len(row["_fp_metrics"])    if isinstance(row.get("_fp_metrics"),    set) else 0
    dims    = len(row["_fp_dimensions"]) if isinstance(row.get("_fp_dimensions"), set) else 0
    richness = min(1.0, (metrics + 0.5 * dims) / CONFIG["ADOPTION_RICHNESS_DIVISOR"])
    usage_weakness = 1.0 - min(1.0, row["view_count_90d"] / CONFIG["HIGH_USAGE_THRESHOLD"])
    score = richness * usage_weakness

    # Optional boost: high edit-to-view ratio = active dev attention without consumption
    edits = float(_safe_get(row, "edit_count_90d", 0))
    views = max(1.0, float(row["view_count_90d"]))
    if (edits / views) >= CONFIG["ADOPTION_EDIT_RATIO_THRESHOLD"]:
        score = min(1.0, score + CONFIG["ADOPTION_EDIT_RATIO_BOOST"])

    return float(score)


def content_gap(row) -> float:
    """Content gap: thin model but strong usage. Action = expand the model."""
    metrics = len(row["_fp_metrics"]) if isinstance(row.get("_fp_metrics"), set) else 0
    tables  = len(row["_fp_tables"])  if isinstance(row.get("_fp_tables"),  set) else 0
    thinness = 1.0 - min(1.0, (metrics + tables) / CONFIG["CONTENT_THINNESS_DIVISOR"])
    usage_strength = min(1.0, row["view_count_90d"] / CONFIG["HIGH_USAGE_THRESHOLD"])
    return float(thinness * usage_strength)


def ux_gap(row) -> float:
    """UX gap: friction is preventing or eroding engagement. Action = redesign
    visuals/layout or fix surface errors. Friction is the WORST of three signals:
      (a) recency is slipping (the original fading signal),
      (b) users hit errors when opening the report (activity_fail_rate_30d),
      (c) audience is mostly mobile but the report isn't mobile-friendly.
    Signals (b) and (c) are optional — they contribute 0 if their input columns
    are absent, so the gap degrades gracefully to the original recency-only form.
    """
    metrics = len(row["_fp_metrics"]) if isinstance(row.get("_fp_metrics"), set) else 0
    tables  = len(row["_fp_tables"])  if isinstance(row.get("_fp_tables"),  set) else 0
    has_content = min(1.0, (metrics + tables) / CONFIG["UX_HAS_CONTENT_DIVISOR"])

    low_use, high_use = CONFIG["LOW_USAGE_THRESHOLD"], CONFIG["HIGH_USAGE_THRESHOLD"]
    moderate_usage = 1.0 if (low_use < row["view_count_90d"] < high_use) else 0.3

    fading_signal = min(1.0, row["recency_days"] / CONFIG["ELIMINATE_RECENCY_DAYS"])

    err_rate = float(_safe_get(row, "activity_fail_rate_30d", 0.0))
    error_signal = min(1.0, err_rate / CONFIG["UX_FAILURE_DIVISOR"])

    mobile_share = float(_safe_get(row, "mobile_view_share", 0.0))
    is_mobile_ready = bool(_safe_get(row, "is_mobile_ready", True))
    mobile_mismatch = (
        mobile_share if (not is_mobile_ready and mobile_share >= CONFIG["MOBILE_SHARE_THRESHOLD"])
        else 0.0
    )

    friction = max(fading_signal, error_signal, mobile_mismatch)
    return float(has_content * moderate_usage * friction)


def health_gap(row) -> float:
    """Health gap: the dataset refresh pipeline is breaking, so users see stale
    or missing data even if the report itself is well-designed.
    Action = fix the pipeline; do not rebuild the report.
    Optional signal — returns 0 if refresh_fail_rate_30d is absent.
    """
    fail_rate = float(_safe_get(row, "refresh_fail_rate_30d", 0.0))
    return float(min(1.0, fail_rate / CONFIG["HEALTH_DIVISOR"]))


def classify_develop_gap(row):
    """Pick the dominant gap; return (label, confidence, rationale) or None if no gap."""
    gaps = {
        "Adoption": adoption_gap(row),
        "Content":  content_gap(row),
        "UX":       ux_gap(row),
        "Health":   health_gap(row),
    }
    dominant_gap, dominant_score = max(gaps.items(), key=lambda kv: kv[1])

    if dominant_score < CONFIG["DEVELOP_GAP_THRESHOLD"]:
        return None  # no significant gap → caller falls through to default Retain

    confidence = (
        "HIGH"   if dominant_score >= 0.70 else
        "MEDIUM" if dominant_score >= 0.50 else
        "LOW"
    )
    score_summary = ", ".join(f"{name}={s:.2f}" for name, s in gaps.items())
    rationale = (
        f"Primary gap: {dominant_gap} ({dominant_score:.2f}). "
        f"Action: {_DEVELOP_ACTIONS[dominant_gap]}. "
        f"All gap scores: {score_summary}."
    )
    return (f"Develop — {dominant_gap}", confidence, rationale)


def classify_cred(row) -> tuple:
    """
    Apply CRED decision tree.
    Returns (cred_label, confidence, rationale)
    """
    elim_days  = CONFIG["ELIMINATE_RECENCY_DAYS"]
    low_use    = CONFIG["LOW_USAGE_THRESHOLD"]
    high_use   = CONFIG["HIGH_USAGE_THRESHOLD"]
    sim_high   = CONFIG["CONSOLIDATE_SIMILARITY_HIGH"]

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

    # ── DEVELOP — gap-based diagnosis ──────────────────────
    # Pick the dominant gap (Adoption / Content / UX). Each gap is a 0–1
    # diagnostic score combining content shape with behavioural signals.
    # See gap functions above for the exact math.
    develop_result = classify_develop_gap(row)
    if develop_result is not None:
        return develop_result

    # ── No-signal fallback: default to Retain ──────────────
    # If no Develop gap clears the threshold AND none of the Retain branches
    # fired earlier, the report is genuinely "quietly fine" — neither a
    # standout asset nor a clear problem. Default to a low-confidence Retain
    # rather than dumping it into Develop (the old fallback behaviour).
    return (
        "Retain",
        "LOW",
        f"No strong CRED signal: recency={recency}d, views_90d={usage}, "
        f"sim_score={sim_score:.0%}, "
        f"{'in cluster' if in_cluster else 'no cluster'}, "
        f"{'unique tables' if unique_src else 'shared tables'}. "
        f"Defaulting to Retain pending stakeholder review."
    )


def apply_cred_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Apply CRED classification to every row."""
    results = df.apply(classify_cred, axis=1)
    df["cred_label"]      = [r[0] for r in results]
    df["cred_confidence"] = [r[1] for r in results]
    df["cred_rationale"]  = [r[2] for r in results]

    # ── Persist Develop gap diagnostics for every row (audit trail) ─
    # Compute all four gap scores even for non-Develop rows so reviewers
    # can see why a report did NOT trip Develop.
    df["adoption_gap"] = df.apply(adoption_gap, axis=1).round(3)
    df["content_gap"]  = df.apply(content_gap,  axis=1).round(3)
    df["ux_gap"]       = df.apply(ux_gap,       axis=1).round(3)
    df["health_gap"]   = df.apply(health_gap,   axis=1).round(3)

    # ── Enrich rationale with cluster-level context ─────────
    df = _enrich_cluster_rationale(df)
    return df


def _enrich_cluster_rationale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process: enrich cred_rationale for Consolidate and Retain-anchor
    reports with detailed cluster documentation:
      - Which cluster (ID + size)
      - Why this anchor was chosen
      - Who the other members are
      - What unique capability each member loses on merge
    """
    cluster_ids = sorted(c for c in df["_cluster_id"].unique() if c >= 0)
    if not cluster_ids:
        return df

    for cid in cluster_ids:
        members = df[df["_cluster_id"] == cid]
        anchor_row = members[members["_is_cluster_anchor"] == True]
        if anchor_row.empty:
            continue
        anchor_idx = anchor_row.index[0]
        anchor_name = df.at[anchor_idx, "report_name"]
        anchor_views = df.at[anchor_idx, "view_count_90d"]
        anchor_tables = df.at[anchor_idx, "_fp_tables"] if isinstance(df.at[anchor_idx, "_fp_tables"], set) else set()
        anchor_metrics = df.at[anchor_idx, "_fp_metrics"] if isinstance(df.at[anchor_idx, "_fp_metrics"], set) else set()
        anchor_dims = df.at[anchor_idx, "_fp_dimensions"] if isinstance(df.at[anchor_idx, "_fp_dimensions"], set) else set()

        member_names = members["report_name"].tolist()
        non_anchor_names = [n for n in member_names if n != anchor_name]
        cluster_size = len(members)

        # Anchor reason: highest view_count_90d in cluster
        second_highest = members[members.index != anchor_idx]["view_count_90d"].max() if cluster_size > 1 else 0
        anchor_reason = (
            f"Chosen as anchor because it has the highest usage in the cluster "
            f"({anchor_views} views vs next-highest {int(second_highest)} views)."
        )

        # ── Enrich Retain-anchor rationale ──
        existing = df.at[anchor_idx, "cred_rationale"]
        enriched_anchor = (
            f"{existing} | CLUSTER {cid} ({cluster_size} members): "
            f"{anchor_reason} "
            f"Other members to merge in: {', '.join(non_anchor_names)}."
        )
        df.at[anchor_idx, "cred_rationale"] = enriched_anchor

        # ── Enrich each Consolidate member rationale ──
        for idx in members.index:
            if idx == anchor_idx:
                continue
            row = df.loc[idx]
            member_tables = row["_fp_tables"] if isinstance(row["_fp_tables"], set) else set()
            member_metrics = row["_fp_metrics"] if isinstance(row["_fp_metrics"], set) else set()
            member_dims = row["_fp_dimensions"] if isinstance(row["_fp_dimensions"], set) else set()

            # Unique capabilities this member has that the anchor does NOT
            unique_tables = member_tables - anchor_tables
            unique_metrics = member_metrics - anchor_metrics
            unique_dims = member_dims - anchor_dims

            capability_parts = []
            if unique_tables:
                capability_parts.append(f"tables: {', '.join(sorted(unique_tables))}")
            if unique_metrics:
                capability_parts.append(f"metrics: {', '.join(sorted(unique_metrics))}")
            if unique_dims:
                capability_parts.append(f"dimensions: {', '.join(sorted(unique_dims))}")

            if capability_parts:
                capability_str = "; ".join(capability_parts)
                capability_note = f"Unique capability at risk on merge: [{capability_str}]. Ensure anchor absorbs these before retiring this report."
            else:
                capability_note = "No unique capability beyond the anchor — safe to merge without data loss."

            existing = df.at[idx, "cred_rationale"]
            enriched_member = (
                f"{existing} | CLUSTER {cid} ({cluster_size} members): "
                f"Anchor = '{anchor_name}'. {anchor_reason} "
                f"Other cluster members: {', '.join(n for n in member_names if n != row['report_name'])}. "
                f"{capability_note}"
            )
            df.at[idx, "cred_rationale"] = enriched_member

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
    "cred_label", "cred_confidence", "cred_rationale",
    "adoption_gap", "content_gap", "ux_gap", "health_gap",
    "refresh_fail_rate_30d", "activity_fail_rate_30d",
    "edit_count_90d", "mobile_view_share", "is_mobile_ready",
    "cred_label_0.50", "cluster_id_0.50", "sim_partner_0.50",
    "cred_label_0.75", "cluster_id_0.75", "sim_partner_0.75",
    "cred_label_0.90", "cluster_id_0.90", "sim_partner_0.90",
    "tags"
]

EXEC_COLUMNS = [
    "report_id", "report_name", "business_domain", "owner_email",
    "last_accessed_date", "view_count_90d",
    "cred_label", "cred_confidence", "cred_rationale",
    "cred_label_0.50", "cred_label_0.75", "cred_label_0.90",
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
# SECTION 8a — SENSITIVITY ANALYSIS (per-report, inline columns)
# ─────────────────────────────────────────────────────────────

SENSITIVITY_THRESHOLDS = [0.50, 0.75, 0.90]

def run_sensitivity_analysis(input_df: pd.DataFrame,
                             thresholds=None) -> pd.DataFrame:
    """
    Run the CRED pipeline at multiple similarity thresholds and return
    per-report columns showing label/cluster/partner at each threshold.

    Args:
        input_df: Raw input DataFrame (same as run_cred_pipeline expects)
        thresholds: List of similarity thresholds to test

    Returns:
        DataFrame indexed by report_id with columns:
          cred_label_0.50, cluster_id_0.50, sim_partner_0.50,
          cred_label_0.75, cluster_id_0.75, sim_partner_0.75,
          cred_label_0.90, cluster_id_0.90, sim_partner_0.90
    """
    if thresholds is None:
        thresholds = SENSITIVITY_THRESHOLDS

    original_threshold = CONFIG["CONSOLIDATE_SIMILARITY_HIGH"]
    all_results = {}

    for thresh in thresholds:
        CONFIG["CONSOLIDATE_SIMILARITY_HIGH"] = thresh

        df = load_and_validate(input_df.copy())
        df = build_fingerprints(df, verbose=False)
        df = score_recency_usage(df)
        df = compute_pairwise_similarity(df)
        df = apply_cred_classification(df)

        suffix = f"_{thresh:.2f}"
        all_results[f"cred_label{suffix}"] = df.set_index("report_id")["cred_label"]
        all_results[f"cluster_id{suffix}"] = df.set_index("report_id")["_cluster_id"]
        all_results[f"sim_partner{suffix}"] = df.set_index("report_id")["_sim_partner_name"]

    CONFIG["CONSOLIDATE_SIMILARITY_HIGH"] = original_threshold

    return pd.DataFrame(all_results)


# ─────────────────────────────────────────────────────────────
# SECTION 8b — MAIN PIPELINE
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

    # Sensitivity: run at multiple thresholds, merge per-report columns
    print(f"[CRED] Running sensitivity analysis (0.50 / 0.75 / 0.90)...")
    sensitivity_cols = run_sensitivity_analysis(input_df)
    df = df.merge(sensitivity_cols, left_on="report_id", right_index=True, how="left")
    print(f"[CRED] Sensitivity columns merged")

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
            "RPT-001",
            "RPT-002",
            "RPT-003",
            "RPT-004",
            "RPT-005",
            "RPT-006",
            "RPT-007",
            "RPT-008",
            "RPT-009",
            "RPT-010",
            "RPT-011",
            "RPT-012",
            "RPT-013",
            "RPT-014",
            "RPT-015",
            "RPT-016",
            "RPT-017",
            "RPT-018",
            "RPT-019",
            "RPT-020",
            "RPT-021",
            "RPT-022",
            "RPT-023",
            "RPT-024",
            "RPT-025",
            "RPT-026",
            "RPT-027",
            "RPT-028",
            "RPT-029",
            "RPT-030",
        ],

        "report_name": [
            "far pivot report",
            "8:15 kpi dashboard",
            "maintenance work orders",
            "machine downtime analysis",
            "executive kpi summary",
            "planned maintenance schedule",
            "oee trend report",
            "cost centre p&l",
            "headcount by plant",
            "predictive failure scores",
            "supplier quality report",
            "production line throughput",
            "far monthly variance",
            "mttr mtbf analysis",
            "energy consumption kpis",
            "scrap & rework report",
            "asset register",
            "shift handover summary",
            "safety incidents report",
            "inventory & parts tracker",
            "production efficiency",
            "finance dashboard",
            "supply chain kpis",
            "shift performance",
            "customer quality scorecard",
            "hr attrition analysis",
            "dev test report 1",
            "dev test report 2",
            "uat far validation",
            "uat 8:15 validation",
        ],

        "report_description": [
            "far pivot report | dataset: far semantic model",
            "8:15 kpi dashboard | dataset: 8:15 kpi semantic model",
            "maintenance work orders | dataset: maintenance semantic model",
            "machine downtime analysis | dataset: 8:15 kpi semantic model",
            "executive kpi summary | dataset: executive summary dataset",
            "planned maintenance schedule | dataset: maintenance semantic model",
            "oee trend report | dataset: 8:15 kpi semantic model",
            "cost centre p&l | dataset: far semantic model",
            "headcount by plant | dataset: hr semantic model",
            "predictive failure scores | dataset: predictive model dataset",
            "supplier quality report | dataset: supply chain dataset",
            "production line throughput | dataset: 8:15 kpi semantic model",
            "far monthly variance | dataset: far semantic model",
            "mttr mtbf analysis | dataset: maintenance semantic model",
            "energy consumption kpis | dataset: energy dataset",
            "scrap & rework report | dataset: quality dataset",
            "asset register | dataset: maintenance semantic model",
            "shift handover summary | dataset: 8:15 kpi semantic model",
            "safety incidents report | dataset: safety dataset",
            "inventory & parts tracker | dataset: inventory & parts",
            "production efficiency | dataset: production planning",
            "finance dashboard | dataset: finance consolidated",
            "supply chain kpis | dataset: supply chain dataset",
            "shift performance | dataset: shift scheduling",
            "customer quality scorecard | dataset: customer quality",
            "hr attrition analysis | dataset: hr semantic model",
            "dev test report 1 | dataset: dev - far test",
            "dev test report 2 | dataset: dev - 8:15 test",
            "uat far validation | dataset: uat - far",
            "uat 8:15 validation | dataset: uat - 8:15",
        ],

        "source_platform": [
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
            "power bi",
        ],

        "business_domain": [
            "finance",
            "operations",
            "maintenance",
            "production",
            "executive",
            "maintenance",
            "operations",
            "finance",
            "finance",
            "data science",
            "supply chain",
            "production",
            "finance",
            "maintenance",
            "production",
            "production",
            "maintenance",
            "operations",
            "production",
            "maintenance",
            "production",
            "finance",
            "supply chain",
            "production",
            "supply chain",
            "hr",
            "dev/test",
            "dev/test",
            "dev/test",
            "dev/test",
        ],

        "owner_email": [
            "analyst1@michiganauto.com",
            "dev.team@michiganauto.com",
            "smoore@michiganauto.com",
            "analyst1@michiganauto.com",
            "bjackson@michiganauto.com",
            "kwhite@michiganauto.com",
            "danderson@michiganauto.com",
            "ctaylor@michiganauto.com",
            "ds.lead@michiganauto.com",
            "pmartinez@michiganauto.com",
            "mthompson@michiganauto.com",
            "drobinson@michiganauto.com",
            "mthompson@michiganauto.com",
            "finance@michiganauto.com",
            "jbrown@michiganauto.com",
            "maint.mgr@michiganauto.com",
            "analyst1@michiganauto.com",
            "bjackson@michiganauto.com",
            "john.smith@michiganauto.com",
            "pmartinez@michiganauto.com",
            "pmartinez@michiganauto.com",
            "gthomas@michiganauto.com",
            "plee@michiganauto.com",
            "gthomas@michiganauto.com",
            "exec.admin@michiganauto.com",
            "pmartinez@michiganauto.com",
            "lgarcia@michiganauto.com",
            "nadams@michiganauto.com",
            "danderson@michiganauto.com",
            "kwhite@michiganauto.com",
        ],

        "last_accessed_date": [
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-24",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
            "2026-05-25",
        ],

        "view_count_90d": [
            236,
            255,
            272,
            245,
            266,
            254,
            239,
            237,
            267,
            237,
            252,
            259,
            227,
            230,
            255,
            251,
            275,
            250,
            256,
            236,
            250,
            277,
            265,
            274,
            271,
            239,
            249,
            239,
            275,
            240,
        ],

        "source_tables": [
            "vw_far_semantic_model",
            "vw_8:15_kpi_semantic_model",
            "vw_maintenance_semantic_model",
            "vw_8:15_kpi_semantic_model",
            "vw_executive_summary_dataset",
            "vw_maintenance_semantic_model",
            "vw_8:15_kpi_semantic_model",
            "vw_far_semantic_model",
            "vw_hr_semantic_model",
            "vw_predictive_model_dataset",
            "vw_supply_chain_dataset",
            "vw_8:15_kpi_semantic_model",
            "vw_far_semantic_model",
            "vw_maintenance_semantic_model",
            "vw_energy_dataset",
            "vw_quality_dataset",
            "vw_maintenance_semantic_model",
            "vw_8:15_kpi_semantic_model",
            "vw_safety_dataset",
            "vw_inventory_&_parts",
            "vw_production_planning",
            "vw_finance_consolidated",
            "vw_supply_chain_dataset",
            "vw_shift_scheduling",
            "vw_customer_quality",
            "vw_hr_semantic_model",
            "vw_dev_-_far_test",
            "vw_dev_-_8:15_test",
            "vw_uat_-_far",
            "vw_uat_-_8:15",
        ],

        "metrics": [
            "actual spend|budget spend|ebitda|ebitda margin %|far cost centre rank|far variance %|mtd far spend|revenue|rolling 12m far spend|total far spend|ytd far spend",
            "actual output|availability %|downtime hours|good parts|oee %|oee target gap|performance %|planned production time|quality rate %|rolling 13m downtime|run time|scrap count|scrap rate %|total parts",
            "actual cost|cost variance %|mtbf (days)|mttr (hours)|planned cost|preventive wo %|work orders closed|work orders open|work orders overdue",
            "actual output|availability %|downtime hours|good parts|oee %|oee target gap|performance %|planned production time|quality rate %|rolling 13m downtime|run time|scrap count|scrap rate %|total parts",
            "",
            "actual cost|cost variance %|mtbf (days)|mttr (hours)|planned cost|preventive wo %|work orders closed|work orders open|work orders overdue",
            "actual output|availability %|downtime hours|good parts|oee %|oee target gap|performance %|planned production time|quality rate %|rolling 13m downtime|run time|scrap count|scrap rate %|total parts",
            "actual spend|budget spend|ebitda|ebitda margin %|far cost centre rank|far variance %|mtd far spend|revenue|rolling 12m far spend|total far spend|ytd far spend",
            "headcount fte|turnover rate %",
            "high risk asset count|predictive failure score",
            "on time delivery %|supplier dppm",
            "actual output|availability %|downtime hours|good parts|oee %|oee target gap|performance %|planned production time|quality rate %|rolling 13m downtime|run time|scrap count|scrap rate %|total parts",
            "actual spend|budget spend|ebitda|ebitda margin %|far cost centre rank|far variance %|mtd far spend|revenue|rolling 12m far spend|total far spend|ytd far spend",
            "actual cost|cost variance %|mtbf (days)|mttr (hours)|planned cost|preventive wo %|work orders closed|work orders open|work orders overdue",
            "energy cost|energy per unit",
            "",
            "actual cost|cost variance %|mtbf (days)|mttr (hours)|planned cost|preventive wo %|work orders closed|work orders open|work orders overdue",
            "actual output|availability %|downtime hours|good parts|oee %|oee target gap|performance %|planned production time|quality rate %|rolling 13m downtime|run time|scrap count|scrap rate %|total parts",
            "near miss count|safety incident rate",
            "",
            "",
            "",
            "on time delivery %|supplier dppm",
            "",
            "",
            "headcount fte|turnover rate %",
            "",
            "",
            "",
            "",
        ],

        "dimensions": [
            "gl drill-through|overview|prior year comp|variance analysis|ytd summary",
            "downtime analysis|kpi overview|machine status|shift summary",
            "completed orders|cost analysis|open orders|schedule view|work order summary",
            "downtime overview|machine detail|pareto analysis",
            "financial summary",
            "compliance status|plant view|schedule overview",
            "machine breakdown",
            "budget vs actual|cost centre view|gl detail|variance",
            "page 1|page 2|page 3",
            "page 2|page 3|page 4",
            "page 2",
            "page 2|page 3",
            "page 2|page 4",
            "page 1",
            "",
            "page 2|page 4",
            "page 1|page 2",
            "page 1|page 2",
            "page 1|page 2|page 3",
            "page 1|page 2|page 3",
            "page 2|page 4",
            "page 1|page 2|page 3|page 4",
            "page 1|page 3|page 4",
            "page 1|page 2|page 3",
            "page 1",
            "page 2|page 3|page 4",
            "page 1|page 2",
            "page 1",
            "page 1|page 3",
            "page 1|page 2|page 3",
        ],

        "report_type": [
            "operational",
            "operational",
            "operational",
            "scheduled",
            "operational",
            "operational",
            "operational",
            "operational",
            "scheduled",
            "operational",
            "operational",
            "operational",
            "scheduled",
            "executive",
            "operational",
            "operational",
            "operational",
            "operational",
            "operational",
            "operational",
            "scheduled",
            "operational",
            "operational",
            "scheduled",
            "executive",
            "operational",
            "ad-hoc",
            "ad-hoc",
            "ad-hoc",
            "ad-hoc",
        ],

        "tags": [
            "none|michigan auto - far reporting",
            "none|michigan auto - 8:15 kpis",
            "none|michigan auto - maintenance",
            "promoted|michigan auto - production",
            "none|michigan auto - executive",
            "none|michigan auto - maintenance",
            "none|michigan auto - 8:15 kpis",
            "none|michigan auto - finance",
            "promoted|michigan auto - finance",
            "none|michigan auto - data science",
            "none|michigan auto - supply chain",
            "none|michigan auto - production",
            "promoted|michigan auto - far reporting",
            "certified|michigan auto - maintenance",
            "none|michigan auto - production",
            "none|michigan auto - production",
            "none|michigan auto - maintenance",
            "none|michigan auto - 8:15 kpis",
            "none|michigan auto - production",
            "none|michigan auto - maintenance",
            "promoted|michigan auto - production",
            "none|michigan auto - finance",
            "none|michigan auto - supply chain",
            "promoted|michigan auto - production",
            "certified|michigan auto - supply chain",
            "none|michigan auto - hr analytics",
            "none|michigan auto - dev",
            "none|michigan auto - dev",
            "none|michigan auto - uat",
            "promoted|michigan auto - uat",
        ],

        "refresh_fail_rate_30d": [
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        "activity_fail_rate_30d": [
            0.0,
            0.214,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.053,
            0.0,
            0.0,
            0.0,
            0.059,
            0.0,
            0.083,
            0.0,
            0.029,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.125,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        "edit_count_90d": [
            22,
            22,
            18,
            20,
            22,
            24,
            21,
            29,
            22,
            21,
            26,
            18,
            11,
            20,
            21,
            18,
            20,
            17,
            18,
            15,
            27,
            17,
            15,
            24,
            21,
            24,
            24,
            25,
            23,
            18,
        ],

        "mobile_view_share": [
            0.267,
            0.275,
            0.165,
            0.184,
            0.177,
            0.185,
            0.234,
            0.198,
            0.165,
            0.194,
            0.198,
            0.147,
            0.185,
            0.178,
            0.18,
            0.175,
            0.2,
            0.228,
            0.195,
            0.195,
            0.164,
            0.177,
            0.211,
            0.19,
            0.185,
            0.172,
            0.201,
            0.255,
            0.196,
            0.188,
        ],

        "is_mobile_ready": [
            False,
            False,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
        ],

    }

    df_sample = pd.DataFrame(sample_data)
    tech, exc, summ = run_cred_pipeline(df_sample)

    # ─────────────────────────────────────────────────────────────
    # SECTION 10 — EXCEL OUTPUT (Single File, Multiple Sheets)
    # ─────────────────────────────────────────────────────────────

    import os
    # Code lives in CRED MAIN FILES/, outputs/ is at project root (one level up)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ── Build Consolidation Clusters sheet ──
    clustered = tech[tech["_cluster_id"] >= 0].copy()
    if not clustered.empty:
        clustered = clustered.sort_values(["_cluster_id", "_is_cluster_anchor"], ascending=[True, False])
        cluster_enrichment = []
        for cid in sorted(clustered["_cluster_id"].unique()):
            members = clustered[clustered["_cluster_id"] == cid]
            anchor_row = members[members["_is_cluster_anchor"] == True]
            if anchor_row.empty:
                continue
            anchor_name = anchor_row.iloc[0]["report_name"]
            anchor_views = anchor_row.iloc[0]["view_count_90d"]
            anchor_tables = {t.strip() for t in str(anchor_row.iloc[0].get("source_tables", "")).split("|") if t.strip()}
            anchor_metrics = {m.strip() for m in str(anchor_row.iloc[0].get("metrics", "")).split("|") if m.strip()}
            anchor_dims = {d.strip() for d in str(anchor_row.iloc[0].get("dimensions", "")).split("|") if d.strip()}
            all_member_names = members["report_name"].tolist()
            cluster_size = len(members)
            second_highest = members[members["_is_cluster_anchor"] != True]["view_count_90d"].max() if cluster_size > 1 else 0

            for idx, row in members.iterrows():
                is_anchor = row["_is_cluster_anchor"]
                member_tables = {t.strip() for t in str(row.get("source_tables", "")).split("|") if t.strip()}
                member_metrics = {m.strip() for m in str(row.get("metrics", "")).split("|") if m.strip()}
                member_dims = {d.strip() for d in str(row.get("dimensions", "")).split("|") if d.strip()}

                if is_anchor:
                    unique_cap = "N/A (this IS the anchor)"
                    anchor_reason = f"Highest usage in cluster ({anchor_views} views vs next-highest {int(second_highest)})"
                else:
                    unique_t = member_tables - anchor_tables
                    unique_m = member_metrics - anchor_metrics
                    unique_d = member_dims - anchor_dims
                    parts = []
                    if unique_t:
                        parts.append(f"tables: {', '.join(sorted(unique_t))}")
                    if unique_m:
                        parts.append(f"metrics: {', '.join(sorted(unique_m))}")
                    if unique_d:
                        parts.append(f"dims: {', '.join(sorted(unique_d))}")
                    unique_cap = "; ".join(parts) if parts else "None — safe to merge without data loss"
                    anchor_reason = f"Anchor '{anchor_name}' chosen for highest usage ({anchor_views} views)"

                cluster_enrichment.append({
                    "idx": idx,
                    "anchor_report_name": anchor_name,
                    "cluster_size": cluster_size,
                    "all_cluster_members": " | ".join(all_member_names),
                    "anchor_selection_reason": anchor_reason,
                    "unique_capability_at_risk": unique_cap,
                })

        enrichment_df = pd.DataFrame(cluster_enrichment).set_index("idx")
        clustered = clustered.join(enrichment_df)
        clusters_sheet = (
            clustered[["_cluster_id", "_is_cluster_anchor", "report_id", "report_name",
              "business_domain", "owner_email", "view_count_90d", "_sim_score",
              "_sim_partner_name", "cred_label",
              "anchor_report_name", "cluster_size", "all_cluster_members",
              "anchor_selection_reason", "unique_capability_at_risk"]]
            .rename(columns={
                "_cluster_id": "cluster_id",
                "_is_cluster_anchor": "is_anchor",
                "_sim_score": "similarity_score",
                "_sim_partner_name": "most_similar_to",
            })
        )
    else:
        clusters_sheet = pd.DataFrame()

    output_file = os.path.join(output_dir, "CRED_Output.xlsx")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        # ── Sheet 1: Technical View ──
        tech.to_excel(writer, index=False, sheet_name="Technical View")

        # ── Sheet 2: Executive View ──
        exc.to_excel(writer, index=False, sheet_name="Executive View")

        # ── Sheet 3: Consolidation Clusters ──
        if not clusters_sheet.empty:
            clusters_sheet.to_excel(writer, index=False, sheet_name="Consolidation Clusters")

        # ── Sheet 4: Pipeline Summary ──
        pd.DataFrame([summ]).to_excel(writer, index=False, sheet_name="Pipeline Summary")

    sheet_count = 3 + (1 if not clusters_sheet.empty else 0)
    print(f"\n[EXPORT] All outputs saved to: {output_file}")
    print(f"[DONE] Single Excel file with {sheet_count} sheets exported successfully.")