"""
Helper: generate michigan_sample_data.py from the raw Power BI metadata.

This produces a `sample_data` dict literal in the exact format used at
the bottom of cred_scoring_engine.py — so you can copy-paste the
dict in and run the engine directly on Michigan Auto data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cred_michigan_auto import load_raw_files, transform_to_cred_input

raw = load_raw_files()
cred_df, _ = transform_to_cred_input(raw)

# Stable order + clean date format
cred_df = cred_df.sort_values("report_id").reset_index(drop=True)
cred_df["last_accessed_date"] = (
    cred_df["last_accessed_date"].dt.strftime("%Y-%m-%d").fillna("")
)

COL_ORDER = [
    "report_id", "report_name", "report_description",
    "source_platform", "business_domain", "owner_email",
    "last_accessed_date", "view_count_90d",
    "source_tables", "metrics", "dimensions",
    "report_type", "tags",
    # Optional operational signals (Phase 2 enrichment)
    "refresh_fail_rate_30d", "activity_fail_rate_30d",
    "edit_count_90d", "mobile_view_share", "is_mobile_ready",
]

HEADER = (
    '"""\n'
    'Michigan Auto sample_data - drop-in replacement for the sample_data\n'
    'block at the bottom of cred_scoring_engine.py.\n'
    '\n'
    'Usage\n'
    '-----\n'
    '1. Open  cred_scoring_engine.py\n'
    '2. Find the block at the very bottom that starts with:\n'
    '       if __name__ == "__main__":\n'
    '           sample_data = {\n'
    '3. Replace that dict with the sample_data dict from this file.\n'
    '4. Run:  python cred_scoring_engine.py\n'
    '\n'
    'Built from the 10 raw Power BI metadata CSVs in /Raw Metadata/.\n'
    f'Total reports: {len(cred_df)}\n'
    '"""\n\n'
)

out_path = "michigan_sample_data.py"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(HEADER)
    f.write("sample_data = {\n")
    for col in COL_ORDER:
        vals = cred_df[col].tolist()
        f.write(f'    "{col}": [\n')
        for v in vals:
            if isinstance(v, str):
                safe = v.replace('"', '\\"')
                f.write(f'        "{safe}",\n')
            else:
                f.write(f"        {v},\n")
        f.write("    ],\n\n")
    f.write("}\n")

print(f"[DONE] Wrote {out_path}  ({len(cred_df)} reports, {len(COL_ORDER)} columns)")
