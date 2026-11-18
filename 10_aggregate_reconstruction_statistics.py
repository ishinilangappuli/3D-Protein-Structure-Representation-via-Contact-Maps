
import pandas as pd

from pathlib import Path

PROJECT_DIR = Path(
    r"C:\Users\ishin\OneDrive\Desktop\Complex_Networks"
    r"\3D-Protein-Structure-Representation-via-Contact-Maps"
)

SUMMARY_FILE = (
    PROJECT_DIR
    / "results_ecoli_8A"
    / "ecoli_shrec3d_8A_summary.csv"
)

df = pd.read_csv(SUMMARY_FILE)

# Keep successfully reconstructed proteins
successful = df[df["Status"] == "OK"].copy()

metrics = [
    "Pearson_correlation",
    "Spearman_correlation",
    "Scale_factor",
    "Coordinate_scale_factor",
    "Mean_absolute_error_A",
    "Maximum_absolute_error_A",
    "Distance_RMSE_A",
    "Aligned_RMSD_A"
]

summary = successful[metrics].agg(
    ["mean", "median", "std", "min", "max"]
).T

print("NUMBER OF PROTEINS")
print("Total selected:", len(df))
print("Successful:", len(successful))
print("Disconnected:", (df["Status"] == "DISCONNECTED").sum())

print("\nAGGREGATE RECONSTRUCTION STATISTICS")
print(summary.round(4))

summary.to_csv(
    PROJECT_DIR
    / "results_ecoli_8A"
    / "aggregate_reconstruction_statistics.csv"
)

print("\nAggregate statistics saved.")
