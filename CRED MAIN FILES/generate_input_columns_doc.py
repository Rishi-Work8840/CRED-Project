"""
generate_input_columns_doc.py
─────────────────────────────
Generates an Excel reference sheet that documents every required input column
for the CRED scoring engine:

  - Column name (matches cred_scoring_engine.REQUIRED_COLUMNS exactly)
  - Data type and format
  - Example value
  - Whether it's required or optional
  - Plain-English description / justification
  - Which CRED category (Eliminate / Consolidate / Retain / Develop) it feeds
  - Where in the engine it's consumed

Run with:  python generate_input_columns_doc.py
Output:    outputs/CRED_Input_Columns_Reference.xlsx
"""

import os
import pandas as pd

# ── Locate output folder ─────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "CRED_Input_Columns_Reference.xlsx")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Column documentation table ───────────────────────────────────────────
# Each entry mirrors a column in cred_scoring_engine.REQUIRED_COLUMNS.
# The "consumed_by" column lists every CRED category that reads this signal,
# directly or indirectly (e.g. via fingerprints or clustering).

COLUMN_DOCS = [
    # ── Identity ────────────────────────────────────────────────────────
    {
        "column_name"        : "report_id",
        "data_type"          : "string",
        "format_or_example"  : "RPT-001",
        "required_optional"  : "Required",
        "justification"      : "Unique key to identify each report across all pipelines.",
        "consumed_by"        : "All categories",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "report_name",
        "data_type"          : "string",
        "format_or_example"  : "Sales Performance Dashboard",
        "required_optional"  : "Required",
        "justification"      : "Human-readable label shown in every output sheet and rationale.",
        "consumed_by"        : "All categories",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "report_description",
        "data_type"          : "string (free text)",
        "format_or_example"  : "Daily sales by region; refreshes 6am ET",
        "required_optional"  : "Optional",
        "justification"      : "",
        "consumed_by"        : "",
        "engine_usage"       : "",
    },

    # ── Provenance / context ────────────────────────────────────────────
    {
        "column_name"        : "source_platform",
        "data_type"          : "string",
        "format_or_example"  : "powerbi | tableau | excel",
        "required_optional"  : "Required",
        "justification"      : "Originating BI tool, used for per-platform breakdown reporting.",
        "consumed_by"        : "Breakdown reporting",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "business_domain",
        "data_type"          : "string",
        "format_or_example"  : "finance | sales | hr | operations",
        "required_optional"  : "Required",
        "justification"      : "Functional area the report serves, used for domain-level breakdown reporting.",
        "consumed_by"        : "Breakdown reporting",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "owner_email",
        "data_type"          : "string (email)",
        "format_or_example"  : "jane.smith@company.com",
        "required_optional"  : "Required",
        "justification"      : "Stakeholder contact for the report, shown in rationales.",
        "consumed_by"        : "All categories",
        "engine_usage"       : "",
    },

    # ── Behavioural signals (the heart of the decision tree) ────────────
    {
        "column_name"        : "last_accessed_date",
        "data_type"          : "date (YYYY-MM-DD)",
        "format_or_example"  : "2026-05-14",
        "required_optional"  : "Required",
        "justification"      : "Primary freshness signal; converted to recency_days to detect stale or fading reports.",
        "consumed_by"        : "Eliminate, Retain, Develop",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "view_count_90d",
        "data_type"          : "integer",
        "format_or_example"  : "247",
        "required_optional"  : "Required",
        "justification"      : "Primary engagement signal; drives every classification branch.",
        "consumed_by"        : "Eliminate, Consolidate, Retain, Develop",
        "engine_usage"       : "",
    },

    # ── Structural fingerprint (similarity + gaps) ──────────────────────
    {
        "column_name"        : "source_tables",
        "data_type"          : "pipe-delimited string",
        "format_or_example"  : "sales_fact|customer_dim|date_dim",
        "required_optional"  : "Required",
        "justification"      : "Tables the report consumes; heaviest weight in similarity scoring and used to detect unique data assets.",
        "consumed_by"        : "Consolidate, Retain, Develop",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "metrics",
        "data_type"          : "pipe-delimited string",
        "format_or_example"  : "total_revenue|avg_order_value|conversion_rate",
        "required_optional"  : "Required",
        "justification"      : "Measures the report exposes; contributes to similarity scoring and all Develop gap calculations.",
        "consumed_by"        : "Consolidate, Develop",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "dimensions",
        "data_type"          : "pipe-delimited string",
        "format_or_example"  : "region|product_category|month",
        "required_optional"  : "Required",
        "justification"      : "Slicers/filters the report exposes; contributes to similarity scoring after boilerplate redaction.",
        "consumed_by"        : "Consolidate, Develop",
        "engine_usage"       : "",
    },

    # ── Lightweight metadata ────────────────────────────────────────────
    {
        "column_name"        : "report_type",
        "data_type"          : "string",
        "format_or_example"  : "operational | executive | ad-hoc | scheduled",
        "required_optional"  : "Required",
        "justification"      : "Functional classification of the report, used for output grouping.",
        "consumed_by"        : "Breakdown reporting",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "tags",
        "data_type"          : "pipe-delimited string",
        "format_or_example"  : "finance|monthly|legacy",
        "required_optional"  : "Optional",
        "justification"      : "",
        "consumed_by"        : "",
        "engine_usage"       : "",
    },

    # ── Operational signals (Phase 2 enrichment) ──────────────────────
    # All Optional. Engine treats missing/NaN as neutral so it still runs
    # cleanly on tenants without refresh/activity logs.
    {
        "column_name"        : "refresh_fail_rate_30d",
        "data_type"          : "float (0.0-1.0)",
        "format_or_example"  : "0.25",
        "required_optional"  : "Optional",
        "justification"      : "Share of failed dataset refreshes in the last 30 days; flags broken pipelines that make reports show stale data.",
        "consumed_by"        : "Develop (Health gap)",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "activity_fail_rate_30d",
        "data_type"          : "float (0.0-1.0)",
        "format_or_example"  : "0.10",
        "required_optional"  : "Optional",
        "justification"      : "Share of failed user sessions in the last 30 days; direct evidence of UX friction or rendering errors.",
        "consumed_by"        : "Develop (UX gap)",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "edit_count_90d",
        "data_type"          : "integer",
        "format_or_example"  : "12",
        "required_optional"  : "Optional",
        "justification"      : "Number of EditReport events in last 90 days; high edits vs low views points to active development without audience uptake.",
        "consumed_by"        : "Develop (Adoption gap)",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "mobile_view_share",
        "data_type"          : "float (0.0-1.0)",
        "format_or_example"  : "0.45",
        "required_optional"  : "Optional",
        "justification"      : "Share of views from Mobile/Embedded clients in last 90 days; used with is_mobile_ready to detect compatibility mismatch.",
        "consumed_by"        : "Develop (UX gap)",
        "engine_usage"       : "",
    },
    {
        "column_name"        : "is_mobile_ready",
        "data_type"          : "boolean",
        "format_or_example"  : "True",
        "required_optional"  : "Optional",
        "justification"      : "Indicates whether the report renders on mobile/embedded surfaces; combined with mobile_view_share to flag audience-platform mismatch.",
        "consumed_by"        : "Develop (UX gap)",
        "engine_usage"       : "",
    },
]


# ── Build the workbook ───────────────────────────────────────────────────
def build_workbook() -> None:
    # Keep only the 5 requested columns
    keep_cols = ["column_name", "data_type", "required_optional", "justification", "consumed_by"]
    columns_df = pd.DataFrame(COLUMN_DOCS)[keep_cols]

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        columns_df.to_excel(writer, index=False, sheet_name="Column Reference")

        # Auto-fit column widths for readability
        ws = writer.sheets["Column Reference"]
        for col_cells in ws.columns:
            max_len = 0
            for cell in col_cells:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            # Cap width so very long text wraps instead of stretching to absurd widths
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 60)

    print(f"[DONE] Reference workbook written to: {OUTPUT_FILE}")
    print(f"       Documented {len(COLUMN_DOCS)} input columns in 1 sheet.")


if __name__ == "__main__":
    build_workbook()
