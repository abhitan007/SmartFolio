import joblib
import os

# Use absolute path - this is more reliable
artifact_dir = os.path.abspath('risk_artifacts')

print(f"Loading from: {artifact_dir}\n")

# Verify directory exists
if not os.path.isdir(artifact_dir):
    print(f"ERROR: Directory does not exist: {artifact_dir}")
    print(f"Current directory: {os.getcwd()}")
    print(f"\nTrying to list files in current directory:")
    print(os.listdir('.'))
    exit(1)

print(f"✓ Directory exists\n")

# ===== LOAD FILES =====
preprocessors_path = os.path.join(artifact_dir, "risk_artifacts/preprocessors.joblib")
cluster_context_path = os.path.join(artifact_dir, "risk_artifacts/cluster_context.joblib")

print(f"Looking for:")
print(f"  1. {preprocessors_path}")
print(f"  2. {cluster_context_path}\n")

# Verify files exist before loading
if not os.path.isfile(preprocessors_path):
    print(f"✗ ERROR: File not found: {preprocessors_path}")
    print(f"\nFiles in {artifact_dir}:")
    try:
        files = os.listdir(artifact_dir)
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"  Could not list files: {e}")
    exit(1)

if not os.path.isfile(cluster_context_path):
    print(f"✗ ERROR: File not found: {cluster_context_path}")
    exit(1)

# Load the files
try:
    bundle = joblib.load(preprocessors_path)
    print("✓ Successfully loaded preprocessors.joblib")
except Exception as e:
    print(f"✗ ERROR loading preprocessors.joblib: {e}")
    exit(1)

try:
    cluster_context = joblib.load(cluster_context_path)
    print("✓ Successfully loaded cluster_context.joblib")
except Exception as e:
    print(f"✗ ERROR loading cluster_context.joblib: {e}")
    exit(1)

# Print the contents
print("\n" + "="*70)
print("RISK RANGES (from risk_cli artifacts):")
print("="*70)
risk_ranges = bundle.get("risk_ranges", {})
if risk_ranges:
    for label, ranges in risk_ranges.items():
        print(f"  {label}: min={ranges['min']}, max={ranges['max']}")
else:
    print("  (empty or not found)")
print()

print("="*70)
print("CLUSTER TO LABEL MAPPING:")
print("="*70)
cluster_to_label = cluster_context.get("cluster_to_label", {})
if cluster_to_label:
    for cluster_id, label in cluster_to_label.items():
        print(f"  Cluster {cluster_id} → {label}")
else:
    print("  (empty or not found)")
print()

print("="*70)
print("CLUSTER DISTANCE STATS:")
print("="*70)
distance_stats = cluster_context.get("cluster_distance_stats", {})
if distance_stats:
    for cluster_id, stats in distance_stats.items():
        print(f"  Cluster {cluster_id}: {stats}")
else:
    print("  (empty or not found)")
print()

print("="*70)
print("SUMMARY")
print("="*70)
print(f"Risk Ranges: {list(risk_ranges.keys())}")
print(f"Clusters: {list(cluster_to_label.keys())}")
print(f"All Keys in Bundle: {list(bundle.keys())}")
print(f"All Keys in Context: {list(cluster_context.keys())}")