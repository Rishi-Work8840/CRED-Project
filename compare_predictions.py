"""Compare CRED scoring engine output vs LLM predictions."""
import pandas as pd
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from cred_scoring_engine_copy import *

# ── LLM Ground Truth ──
llm_labels = {}
for i in range(1, 16):
    llm_labels[f"RPT{i:03d}"] = "Retain"
for i in range(16, 36):
    llm_labels[f"RPT{i:03d}"] = "Consolidate"
for i in range(36, 66):
    llm_labels[f"RPT{i:03d}"] = "Eliminate"
for i in range(66, 101):
    llm_labels[f"RPT{i:03d}"] = "Develop"

# ── Run the pipeline (suppress print) ──
import contextlib

# Just run the file as __main__ and capture tech
import subprocess, json
with contextlib.redirect_stdout(io.StringIO()):
    # Import the module functions, then build sample and run
    pass

# Simpler approach: run the pipeline directly with the sample data from the file
import importlib.util
spec = importlib.util.spec_from_file_location("cred_copy", "cred_scoring_engine_copy.py")
mod = importlib.util.module_from_spec(spec)

# We already imported everything from cred_scoring_engine_copy
# Just build the sample_data here by reading the file and extracting it
with open('cred_scoring_engine_copy.py', encoding='utf-8') as f:
    lines = f.readlines()

# Find the sample_data start and the df_sample line
in_main = False
sample_lines = []
for line in lines:
    if 'if __name__' in line:
        in_main = True
        continue
    if in_main:
        sample_lines.append(line)

# Execute sample_data construction with no indentation issues
# Strip exactly 4 spaces from each line (the if __main__ indent)
dedented = []
for line in sample_lines:
    if line.startswith('    '):
        dedented.append(line[4:])
    else:
        dedented.append(line)

code_to_run = ''.join(dedented)
local_ns = {**globals()}
with contextlib.redirect_stdout(io.StringIO()):
    exec(code_to_run, local_ns)

tech = local_ns['tech']
df_results = tech[["report_id", "cred_label"]].copy()
df_results["llm_label"] = df_results["report_id"].map(llm_labels)
df_results["match"] = df_results["cred_label"] == df_results["llm_label"]

# ── METRICS ──
print("=" * 70)
print("   CRED SCORING ENGINE vs LLM PREDICTIONS - COMPARISON REPORT")
print("=" * 70)

total = len(df_results)
correct = df_results["match"].sum()
accuracy = correct / total * 100

print(f"\n1. OVERALL ACCURACY")
print(f"   Correct: {correct} / {total}")
print(f"   Accuracy: {accuracy:.1f}%")
print(f"   Mismatches: {total - correct}")

# ── Per-Category Metrics ──
categories = ["Retain", "Consolidate", "Eliminate", "Develop"]
print(f"\n2. PER-CATEGORY BREAKDOWN")
print(f"   {'Category':<14} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Support (LLM)'}")
print(f"   {'-'*14} {'-'*12} {'-'*12} {'-'*12} {'-'*14}")

for cat in categories:
    # True Positives: code says cat AND llm says cat
    tp = ((df_results["cred_label"] == cat) & (df_results["llm_label"] == cat)).sum()
    # False Positives: code says cat BUT llm says something else
    fp = ((df_results["cred_label"] == cat) & (df_results["llm_label"] != cat)).sum()
    # False Negatives: llm says cat BUT code says something else
    fn = ((df_results["cred_label"] != cat) & (df_results["llm_label"] == cat)).sum()
    
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    support = (df_results["llm_label"] == cat).sum()
    
    print(f"   {cat:<14} {precision:>8.1f}%   {recall:>8.1f}%   {f1:>8.1f}%   {support:>5}")

# ── Confusion Matrix ──
print(f"\n3. CONFUSION MATRIX")
print(f"   (Rows = Code prediction, Columns = LLM expected)")
print(f"\n   {'Code \\ LLM':<14}", end="")
for cat in categories:
    print(f" {cat[:5]:>7}", end="")
print(f"  {'Total':>6}")
print(f"   {'-'*60}")

for code_cat in categories:
    code_mask = df_results["cred_label"] == code_cat
    print(f"   {code_cat:<14}", end="")
    row_total = 0
    for llm_cat in categories:
        llm_mask = df_results["llm_label"] == llm_cat
        count = (code_mask & llm_mask).sum()
        row_total += count
        print(f" {count:>7}", end="")
    print(f"  {row_total:>6}")

print(f"   {'-'*60}")
print(f"   {'Total':<14}", end="")
for llm_cat in categories:
    print(f" {(df_results['llm_label'] == llm_cat).sum():>7}", end="")
print(f"  {total:>6}")

# ── Misclassified Reports Detail ──
mismatches = df_results[~df_results["match"]].copy()
print(f"\n4. MISCLASSIFIED REPORTS ({len(mismatches)} total)")
print(f"   {'Report':<8} {'Code Says':<14} {'LLM Says':<14} {'Direction'}")
print(f"   {'-'*8} {'-'*14} {'-'*14} {'-'*30}")
for _, row in mismatches.iterrows():
    direction = f"{row['cred_label']} -> should be {row['llm_label']}"
    print(f"   {row['report_id']:<8} {row['cred_label']:<14} {row['llm_label']:<14} {direction}")

# ── Migration Impact Comparison ──
print(f"\n5. MIGRATION IMPACT COMPARISON")
code_elim = (df_results["cred_label"] == "Eliminate").sum()
code_cons = (df_results["cred_label"] == "Consolidate").sum()
llm_elim = (df_results["llm_label"] == "Eliminate").sum()
llm_cons = (df_results["llm_label"] == "Consolidate").sum()
code_saving = (code_elim + code_cons) / total * 100
llm_saving = (llm_elim + llm_cons) / total * 100

print(f"   Code migration reduction:  {code_saving:.1f}% ({code_elim} eliminate + {code_cons} consolidate)")
print(f"   LLM migration reduction:   {llm_saving:.1f}% ({llm_elim} eliminate + {llm_cons} consolidate)")
print(f"   Difference:                 {abs(code_saving - llm_saving):.1f} percentage points")

# ── Agreement by report group ──
print(f"\n6. AGREEMENT BY REPORT GROUP")
groups = [("RPT001-015 (Core Active)", 1, 16), 
          ("RPT016-035 (Duplicates)", 16, 36),
          ("RPT036-065 (Stale/Legacy)", 36, 66),
          ("RPT066-100 (Operational)", 66, 101)]
for name, start, end in groups:
    group_ids = [f"RPT{i:03d}" for i in range(start, end)]
    group_df = df_results[df_results["report_id"].isin(group_ids)]
    group_acc = group_df["match"].sum() / len(group_df) * 100
    print(f"   {name:<35} {group_acc:>5.1f}% agreement ({group_df['match'].sum()}/{len(group_df)})")

print("\n" + "=" * 70)
