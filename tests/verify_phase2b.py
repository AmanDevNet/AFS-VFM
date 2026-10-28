"""Quick verification of Phase 2B signal data."""
import os
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(ROOT_DIR, "results", "phase2b", "signals", "frame_signals.csv")
df = pd.read_csv(csv_path)

print("=" * 60)
print("PHASE 2B - SIGNAL DATA VERIFICATION")
print("=" * 60)

img_col = "image"
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Unique images: {df[img_col].nunique()}")
print(f"Degradation types: {sorted(df['degradation'].unique())}")
print(f"Frame range: {df['frame'].min()} - {df['frame'].max()}")
print(f"Severity range: {df['severity_pct'].min()}% - {df['severity_pct'].max()}%")

expected_rows = df[img_col].nunique() * 5 * 20
print(f"\nExpected rows: {expected_rows:,}")
print(f"Actual rows:   {len(df):,}")
print(f"Match: {'YES' if len(df) == expected_rows else 'NO - MISMATCH'}")

print(f"\n--- MISSING VALUES ---")
has_missing = False
for col in df.columns:
    n = df[col].isna().sum()
    if n > 0:
        has_missing = True
        print(f"  {col}: {n} ({n/len(df)*100:.1f}%)")
if not has_missing:
    print("  None! All columns complete.")

print(f"\n--- DINOV2 SIGNALS ---")
d_conf = df["dinov2_top1_confidence"]
d_margin = df["dinov2_logit_margin"]
d_entropy = df["dinov2_attention_entropy"]
d_cos = df["dinov2_cosine_sim"]
d_l2 = df["dinov2_l2_distance"]
d_top5 = df["dinov2_top5_overlap"]
print(f"  top1_confidence: mean={d_conf.mean():.4f}, min={d_conf.min():.4f}, max={d_conf.max():.4f}")
print(f"  logit_margin:    mean={d_margin.mean():.4f}, min={d_margin.min():.4f}, max={d_margin.max():.4f}")
print(f"  attn_entropy:    mean={d_entropy.mean():.4f}, min={d_entropy.min():.4f}, max={d_entropy.max():.4f}")
print(f"  cosine_sim:      mean={d_cos.mean():.4f}, min={d_cos.min():.4f}, max={d_cos.max():.4f}")
print(f"  l2_distance:     mean={d_l2.mean():.4f}, min={d_l2.min():.4f}, max={d_l2.max():.4f}")
print(f"  top5_overlap:    mean={d_top5.mean():.4f}")

print(f"\n--- CLIP SIGNALS ---")
c_conf = df["clip_top1_confidence"]
c_margin = df["clip_logit_margin"]
c_cos = df["clip_cosine_sim"]
print(f"  top1_confidence: mean={c_conf.mean():.4f}, min={c_conf.min():.4f}, max={c_conf.max():.4f}")
print(f"  logit_margin:    mean={c_margin.mean():.4f}, min={c_margin.min():.4f}, max={c_margin.max():.4f}")
print(f"  cosine_sim:      mean={c_cos.mean():.4f}, min={c_cos.min():.4f}")

print(f"\n--- DETR SIGNALS ---")
t_conf = df["detr_top_detection_confidence"]
t_obj = df["detr_objectness"]
t_num = df["detr_num_detections"]
print(f"  detection_conf:  mean={t_conf.mean():.4f}, min={t_conf.min():.4f}, max={t_conf.max():.4f}")
print(f"  objectness:      mean={t_obj.mean():.4f}")
print(f"  num_detections:  mean={t_num.mean():.1f}, min={t_num.min()}, max={t_num.max()}")

# Key check: Do signals degrade with increasing severity?
print(f"\n--- SIGNAL vs SEVERITY (do signals degrade as expected?) ---")
for sev in [0, 26, 52, 78, 100]:
    s = df[df["severity_pct"] == sev]
    if len(s) > 0:
        print(f"  Severity {sev:3d}%: "
              f"dinov2_cos={s['dinov2_cosine_sim'].mean():.3f}  "
              f"clip_cos={s['clip_cosine_sim'].mean():.3f}  "
              f"dinov2_conf={s['dinov2_top1_confidence'].mean():.3f}  "
              f"detr_det={s['detr_num_detections'].mean():.1f}")

# Check frame 1 baselines are correct
frame1 = df[df["frame"] == 1]
print(f"\n--- FRAME 1 BASELINE CHECKS ---")
print(f"  All dinov2_cosine_sim == 1.0: {(frame1['dinov2_cosine_sim'] == 1.0).all()}")
print(f"  All clip_cosine_sim == 1.0:   {(frame1['clip_cosine_sim'] == 1.0).all()}")
print(f"  All dinov2_l2_distance == 0:  {(frame1['dinov2_l2_distance'] == 0.0).all()}")

print(f"\n--- VERDICT ---")
checks = [
    len(df) == expected_rows,
    d_cos.min() >= -1.0 and d_cos.max() <= 1.01,
    c_cos.min() >= -1.0 and c_cos.max() <= 1.01,
    d_conf.min() >= 0 and d_conf.max() <= 1.01,
    (frame1["dinov2_cosine_sim"] == 1.0).all(),
]
if all(checks):
    print("  ALL CHECKS PASSED - Data looks correct!")
else:
    print("  SOME CHECKS FAILED - Review the output above.")
