import pandas as pd

df = pd.read_csv(r"d:\Research Project\results\full_benchmark.csv")
print("=" * 60)
print("PHASE 1 BENCHMARK - DATA QUALITY REPORT")
print("=" * 60)
print(f"Total rows:           {len(df):,}")
print(f"Unique images:        {df['image'].nunique():,}")
print(f"Datasets:             {dict(df['dataset'].value_counts())}")
print(f"Degradation types:    {sorted(df['degradation'].unique())}")
print(f"Frames per sequence:  {df['frame'].nunique()}")
print(f"Severity range:       {df['severity_pct'].min()}% - {df['severity_pct'].max()}%")
print()
print("--- FAILURE RATES BY MODEL ---")
for m in ["dinov2", "clip", "detr"]:
    col = f"{m}_failure"
    rate = df[col].sum() / len(df) * 100
    print(f"  {m.upper():8s}: {rate:.1f}% overall failure rate")
print()
print("--- FAILURE RATE BY DEGRADATION TYPE ---")
for deg in sorted(df["degradation"].unique()):
    sub = df[df["degradation"] == deg]
    d = sub["dinov2_failure"].sum() / len(sub) * 100
    c = sub["clip_failure"].sum() / len(sub) * 100
    t = sub["detr_failure"].sum() / len(sub) * 100
    print(f"  {deg:12s}  DINOv2={d:5.1f}%  CLIP={c:5.1f}%  DETR={t:5.1f}%")
print()
print("--- FAILURE RATE BY SEVERITY BRACKET ---")
for lo, hi, label in [
    (0, 20, "Low (0-20%)"),
    (21, 50, "Med (21-50%)"),
    (51, 80, "High (51-80%)"),
    (81, 100, "Extreme (81-100%)"),
]:
    sub = df[(df["severity_pct"] >= lo) & (df["severity_pct"] <= hi)]
    d = sub["dinov2_failure"].sum() / len(sub) * 100
    c = sub["clip_failure"].sum() / len(sub) * 100
    t = sub["detr_failure"].sum() / len(sub) * 100
    print(f"  {label:18s}  DINOv2={d:5.1f}%  CLIP={c:5.1f}%  DETR={t:5.1f}%")
print()
print("--- HALLUCINATION EXAMPLES (Scale @ 100%) ---")
scale100 = df[(df["degradation"] == "scale") & (df["severity_pct"] == 100)]
print(
    f"  DINOv2 top hallucinations: {dict(scale100[scale100['dinov2_failure'] == True]['dinov2_prediction'].value_counts().head(5))}"
)
print(
    f"  DETR top hallucinations:   {dict(scale100[scale100['detr_failure'] == True]['detr_prediction'].value_counts().head(5))}"
)
