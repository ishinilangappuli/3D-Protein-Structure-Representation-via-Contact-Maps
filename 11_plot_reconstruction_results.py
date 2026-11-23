
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(
    r"C:\Users\ishin\OneDrive\Desktop\Complex_Networks"
    r"\3D-Protein-Structure-Representation-via-Contact-Maps"
)

RESULTS_DIR = PROJECT_DIR / "results_ecoli_8A"

SUMMARY_FILE = RESULTS_DIR / "ecoli_shrec3d_8A_summary.csv"

PLOT_DIR = RESULTS_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. LOAD SUCCESSFUL RECONSTRUCTIONS
# ============================================================

df = pd.read_csv(SUMMARY_FILE)

successful = df[df["Status"] == "OK"].copy()

print("TOTAL STRUCTURES:", len(df))
print("SUCCESSFUL RECONSTRUCTIONS:", len(successful))
print("DISCONNECTED STRUCTURES:",
      (df["Status"] == "DISCONNECTED").sum())

# ============================================================
# 3. PEARSON CORRELATION HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    successful["Pearson_correlation"],
    bins=12,
    edgecolor="black"
)

plt.axvline(
    successful["Pearson_correlation"].mean(),
    linestyle="--",
    label=f'Mean = {successful["Pearson_correlation"].mean():.3f}'
)

plt.xlabel("Pearson correlation")
plt.ylabel("Number of proteins")
plt.title("Distribution of Pearson Correlation")
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_DIR / "01_pearson_distribution.png", dpi=300)
plt.show()

# ============================================================
# 4. ALIGNED RMSD HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    successful["Aligned_RMSD_A"],
    bins=12,
    edgecolor="black"
)

plt.axvline(
    successful["Aligned_RMSD_A"].mean(),
    linestyle="--",
    label=f'Mean = {successful["Aligned_RMSD_A"].mean():.2f} Å'
)

plt.axvline(
    successful["Aligned_RMSD_A"].median(),
    linestyle=":",
    label=f'Median = {successful["Aligned_RMSD_A"].median():.2f} Å'
)

plt.xlabel("Aligned RMSD (Å)")
plt.ylabel("Number of proteins")
plt.title("Distribution of Aligned RMSD")
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_DIR / "02_aligned_RMSD_distribution.png", dpi=300)
plt.show()

# ============================================================
# 5. PEARSON CORRELATION VS ALIGNED RMSD
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    successful["Pearson_correlation"],
    successful["Aligned_RMSD_A"]
)

plt.xlabel("Pearson correlation")
plt.ylabel("Aligned RMSD (Å)")
plt.title("Correlation vs Reconstruction Error")
plt.tight_layout()

plt.savefig(PLOT_DIR / "03_correlation_vs_RMSD.png", dpi=300)
plt.show()

# ============================================================
# 6. ALIGNED RMSD FOR EACH PROTEIN
# ============================================================

rmsd_data = successful.sort_values(
    "Aligned_RMSD_A",
    ascending=True
)

plt.figure(figsize=(12, 9))

plt.barh(
    rmsd_data["Protein"],
    rmsd_data["Aligned_RMSD_A"]
)

plt.axvline(
    successful["Aligned_RMSD_A"].mean(),
    linestyle="--",
    label=f'Mean = {successful["Aligned_RMSD_A"].mean():.2f} Å'
)

plt.xlabel("Aligned RMSD (Å)")
plt.ylabel("Protein")
plt.title("Reconstruction Error Across Proteins")
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_DIR / "04_RMSD_per_protein.png", dpi=300)
plt.show()

print("\nALL FOUR PLOTS SAVED SUCCESSFULLY.")
print("Output folder:", PLOT_DIR)
