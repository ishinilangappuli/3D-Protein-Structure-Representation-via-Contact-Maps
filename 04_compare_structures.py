
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr


# ==================================================
# 1. FILE PATHS
# ==================================================

BASE_DIR = Path(
    r"C:\Users\ishin\OneDrive\Desktop\Complex_Networks"
)

PDB_FILE = BASE_DIR / "1UBQ.pdb"

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECONSTRUCTED_FILE = (
    OUTPUT_DIR / "1UBQ_reconstructed_coordinates.csv"
)


# ==================================================
# 2. LOAD ORIGINAL PROTEIN STRUCTURE
# ==================================================

parser = PDBParser(QUIET=True)

structure = parser.get_structure(
    "1UBQ",
    str(PDB_FILE)
)

model = next(structure.get_models())
chain = next(model.get_chains())

original_coordinates = []
residue_labels = []

for residue in chain.get_residues():

    if residue.id[0] != " ":
        continue

    if "CA" not in residue:
        continue

    original_coordinates.append(
        residue["CA"].coord
    )

    residue_labels.append(
        f"{residue.id[1]}_{residue.resname}"
    )

original_coordinates = np.asarray(
    original_coordinates,
    dtype=float
)


# ==================================================
# 3. LOAD SHREC3D RECONSTRUCTION
# ==================================================

reconstructed_df = pd.read_csv(
    RECONSTRUCTED_FILE
)

reconstructed_coordinates = (
    reconstructed_df[["X", "Y", "Z"]].to_numpy()
)

# Check residue correspondence
if "Residue" in reconstructed_df.columns:
    if reconstructed_df["Residue"].tolist() != residue_labels:
        raise ValueError(
            "Residue labels do not match between "
            "original and reconstructed structures."
        )

if original_coordinates.shape != reconstructed_coordinates.shape:
    raise ValueError(
        "Original and reconstructed coordinate "
        "dimensions do not match."
    )

if not np.all(np.isfinite(reconstructed_coordinates)):
    raise ValueError("Reconstructed coordinates contain invalid values.")

print("\n----- Input Information -----")
print("Protein: 1UBQ")
print("Chain:", chain.id)
print("Original structure:", original_coordinates.shape)
print("Reconstructed structure:", reconstructed_coordinates.shape)


# ==================================================
# 4. SCALE-ADJUSTED STRUCTURAL ALIGNMENT
# ==================================================

def align_structures(reference, moving):

    # Center both structures
    reference_center = reference.mean(axis=0)
    moving_center = moving.mean(axis=0)

    X = reference - reference_center
    Y = moving - moving_center

    # Find optimal rotation
    U, singular_values, Vt = np.linalg.svd(
        Y.T @ X
    )

    rotation = U @ Vt

    # Correct possible reflection
    correction = np.eye(3)

    if np.linalg.det(rotation) < 0:
        correction[-1, -1] = -1

    rotation = U @ correction @ Vt

    # Optimal scale factor
    scale = (
        np.sum(singular_values * np.diag(correction))
        / np.sum(Y ** 2)
    )

    # Align reconstructed coordinates
    aligned = (
        scale * Y @ rotation
        + reference_center
    )

    return aligned, scale


aligned_coordinates, scale_factor = align_structures(
    original_coordinates,
    reconstructed_coordinates
)


# ==================================================
# 5. CALCULATE RMSD
# ==================================================

coordinate_errors = (
    original_coordinates - aligned_coordinates
)

squared_errors = np.sum(
    coordinate_errors ** 2,
    axis=1
)

rmsd = np.sqrt(
    np.mean(squared_errors)
)


# ==================================================
# 6. CALCULATE PAIRWISE DISTANCE MATRICES
# ==================================================

original_distances = cdist(
    original_coordinates,
    original_coordinates,
    metric="euclidean"
)

reconstructed_distances = cdist(
    reconstructed_coordinates,
    reconstructed_coordinates,
    metric="euclidean"
)

aligned_distances = cdist(
    aligned_coordinates,
    aligned_coordinates,
    metric="euclidean"
)


# Use unique residue pairs only
upper_triangle = np.triu_indices(
    len(original_coordinates),
    k=1
)

true_values = original_distances[upper_triangle]

# Distances after scale-adjusted alignment
predicted_values = aligned_distances[upper_triangle]

# Calculate errors
absolute_errors = np.abs(
    true_values - predicted_values
)


# ==================================================
# 7. CORRELATION ANALYSIS
# ==================================================

pearson_correlation, pearson_p = pearsonr(
    true_values,
    predicted_values
)

spearman_correlation, spearman_p = spearmanr(
    true_values,
    predicted_values
)

mae = np.mean(absolute_errors)

maximum_error = np.max(absolute_errors)

distance_rmse = np.sqrt(
    np.mean(absolute_errors ** 2)
)


# ==================================================
# 8. PRINT COMPARISON RESULTS
# ==================================================

print("\n===== STRUCTURAL COMPARISON RESULTS =====")

print(f"Number of residues: {len(original_coordinates)}")

print(f"Optimal scale factor: {scale_factor:.4f}")

print(f"Aligned RMSD: {rmsd:.4f} Å")

print("\n----- Distance Comparison -----")

print(f"Pearson correlation: {pearson_correlation:.4f}")
print(f"Pearson p-value: {pearson_p:.6g}")

print(f"Spearman correlation: {spearman_correlation:.4f}")
print(f"Spearman p-value: {spearman_p:.6g}")

print(f"Mean absolute error: {mae:.4f} Å")

print(f"Maximum absolute error: {maximum_error:.4f} Å")

print(f"Distance RMSE: {distance_rmse:.4f} Å")


# ==================================================
# 9. SAVE ALIGNED COORDINATES
# ==================================================

aligned_df = pd.DataFrame(
    aligned_coordinates,
    columns=["X", "Y", "Z"]
)

aligned_df.insert(
    0,
    "Residue",
    residue_labels
)

aligned_df.to_csv(
    OUTPUT_DIR / "1UBQ_aligned_coordinates.csv",
    index=False
)


# ==================================================
# 10. SAVE COMPARISON STATISTICS
# ==================================================

statistics = pd.DataFrame({
    "Protein": ["1UBQ"],
    "Number_of_residues": [len(original_coordinates)],
    "Scale_factor": [scale_factor],
    "Aligned_RMSD_A": [rmsd],
    "Pearson_correlation": [pearson_correlation],
    "Pearson_p_value": [pearson_p],
    "Spearman_correlation": [spearman_correlation],
    "Spearman_p_value": [spearman_p],
    "Mean_absolute_error_A": [mae],
    "Maximum_absolute_error_A": [maximum_error],
    "Distance_RMSE_A": [distance_rmse]
})

statistics.to_csv(
    OUTPUT_DIR / "1UBQ_comparison_statistics.csv",
    index=False
)


# ==================================================
# 11. VISUALIZE ORIGINAL AND RECONSTRUCTED STRUCTURES
# ==================================================

fig = plt.figure(figsize=(14, 6))

# Original structure
ax1 = fig.add_subplot(121, projection="3d")

ax1.plot(
    original_coordinates[:, 0],
    original_coordinates[:, 1],
    original_coordinates[:, 2],
    marker="o",
    markersize=3
)

ax1.set_title("Original Ubiquitin (1UBQ)")
ax1.set_xlabel("X (Å)")
ax1.set_ylabel("Y (Å)")
ax1.set_zlabel("Z (Å)")


# Aligned reconstructed structure
ax2 = fig.add_subplot(122, projection="3d")

ax2.plot(
    aligned_coordinates[:, 0],
    aligned_coordinates[:, 1],
    aligned_coordinates[:, 2],
    marker="o",
    markersize=3
)

ax2.set_title(
    f"Scale-adjusted ShRec3D Reconstruction\n"
    f"RMSD = {rmsd:.3f} Å"
)

ax2.set_xlabel("X (Å)")
ax2.set_ylabel("Y (Å)")
ax2.set_zlabel("Z (Å)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "1UBQ_final_structure_comparison.png",
    dpi=300
)

plt.show()


# ==================================================
# 12. DISTANCE SCATTER PLOT
# ==================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    true_values,
    predicted_values,
    alpha=0.5,
    s=15
)

# Ideal agreement line
maximum_distance = max(
    np.max(true_values),
    np.max(predicted_values)
)

plt.plot(
    [0, maximum_distance],
    [0, maximum_distance],
    linestyle="--",
    label="Ideal agreement"
)

plt.xlabel("Original distances (Å)")
plt.ylabel("Reconstructed distances (Å)")

plt.title(
    f"Original vs Reconstructed Distances\n"
    f"Pearson correlation = {pearson_correlation:.3f}"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "1UBQ_distance_scatter_comparison.png",
    dpi=300
)

plt.show()


print("\nComparison completed successfully!")
print("Results saved in:", OUTPUT_DIR)
