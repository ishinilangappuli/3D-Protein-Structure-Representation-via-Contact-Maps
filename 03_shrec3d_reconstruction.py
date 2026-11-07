
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, spearmanr


# --------------------------------------------------
# 1. Paths and parameters
# --------------------------------------------------
PDB_FILE = Path(
    r"C:\Users\ishin\OneDrive\Desktop\Complex_Networks\1UBQ.pdb"
)

OUTPUT_DIR = Path(
    r"C:\Users\ishin\OneDrive\Desktop\Complex_Networks\results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 8.0
N_DIMENSIONS = 3


# --------------------------------------------------
# 2. Extract C-alpha coordinates
# --------------------------------------------------
parser = PDBParser(QUIET=True)
structure = parser.get_structure("1UBQ", str(PDB_FILE))

model = next(structure.get_models())
chain = next(model.get_chains())

coordinates = []
residue_labels = []

for residue in chain.get_residues():

    if residue.id[0] != " ":
        continue

    if "CA" not in residue:
        continue

    coordinates.append(residue["CA"].coord)

    residue_labels.append(
        f"{residue.id[1]}_{residue.resname}"
    )

coordinates = np.asarray(coordinates, dtype=float)


# --------------------------------------------------
# 3. Calculate original pairwise distances
# --------------------------------------------------
original_distances = cdist(
    coordinates,
    coordinates,
    metric="euclidean"
)


# --------------------------------------------------
# 4. Generate contact map
# --------------------------------------------------
contact_map = (
    original_distances < THRESHOLD
).astype(int)

np.fill_diagonal(contact_map, 0)


# --------------------------------------------------
# 5. Calculate shortest-path distances
# --------------------------------------------------
shortest_path_matrix = shortest_path(
    contact_map,
    directed=False,
    unweighted=True
)

# Check that every residue is connected
if not np.all(np.isfinite(shortest_path_matrix)):
    raise ValueError(
        "The contact graph is disconnected. "
        "Try a different threshold or inspect the protein."
    )


# --------------------------------------------------
# 6. Classical MDS reconstruction
# --------------------------------------------------
def classical_mds(distance_matrix, n_components=3):

    n = distance_matrix.shape[0]

    # Square distances
    squared_distances = distance_matrix ** 2

    # Centering matrix
    centering_matrix = (
        np.eye(n) - np.ones((n, n)) / n
    )

    # Gram matrix
    gram_matrix = (
        -0.5
        * centering_matrix
        @ squared_distances
        @ centering_matrix
    )

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

    # Sort eigenvalues from largest to smallest
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Keep only positive eigenvalues
    positive = eigenvalues > 0

    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]

    # Select the first three dimensions
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    reconstructed_coordinates = (
        eigenvectors * np.sqrt(eigenvalues)
    )

    return reconstructed_coordinates


reconstructed_coordinates = classical_mds(
    shortest_path_matrix,
    n_components=N_DIMENSIONS
)


# --------------------------------------------------
# 7. Calculate reconstructed pairwise distances
# --------------------------------------------------
reconstructed_distances = cdist(
    reconstructed_coordinates,
    reconstructed_coordinates,
    metric="euclidean"
)


# --------------------------------------------------
# 8. Compare original and reconstructed distances
# --------------------------------------------------
upper_triangle = np.triu_indices(
    len(coordinates),
    k=1
)

true_values = original_distances[upper_triangle]
reconstructed_values = reconstructed_distances[upper_triangle]

absolute_errors = np.abs(
    true_values - reconstructed_values
)

pearson_correlation = pearsonr(
    true_values,
    reconstructed_values
).statistic

spearman_correlation = spearmanr(
    true_values,
    reconstructed_values
).statistic

mean_absolute_error = np.mean(absolute_errors)
maximum_absolute_error = np.max(absolute_errors)


# --------------------------------------------------
# 9. Print results
# --------------------------------------------------
print("\n----- ShRec3D Results -----")
print(f"Protein: 1UBQ")
print(f"Chain: {chain.id}")
print(f"Number of residues: {len(coordinates)}")
print(f"Contact threshold: {THRESHOLD} Å")

print(
    f"Reconstructed coordinate shape: "
    f"{reconstructed_coordinates.shape}"
)

print(
    f"Pearson correlation: "
    f"{pearson_correlation:.4f}"
)

print(
    f"Spearman correlation: "
    f"{spearman_correlation:.4f}"
)

print(
    f"Mean absolute distance error: "
    f"{mean_absolute_error:.4f} Å"
)

print(
    f"Maximum absolute distance error: "
    f"{maximum_absolute_error:.4f} Å"
)


# --------------------------------------------------
# 10. Save reconstructed coordinates
# --------------------------------------------------
coordinates_df = pd.DataFrame(
    reconstructed_coordinates,
    columns=["X", "Y", "Z"]
)

coordinates_df.insert(
    0,
    "Residue",
    residue_labels
)

coordinates_df.to_csv(
    OUTPUT_DIR / "1UBQ_reconstructed_coordinates.csv",
    index=False
)


# --------------------------------------------------
# 11. Save evaluation results
# --------------------------------------------------
results_df = pd.DataFrame({
    "Protein": ["1UBQ"],
    "Threshold_A": [THRESHOLD],
    "Number_of_residues": [len(coordinates)],
    "Pearson_correlation": [pearson_correlation],
    "Spearman_correlation": [spearman_correlation],
    "Mean_absolute_error_A": [mean_absolute_error],
    "Maximum_absolute_error_A": [maximum_absolute_error]
})

results_df.to_csv(
    OUTPUT_DIR / "1UBQ_reconstruction_statistics.csv",
    index=False
)


# --------------------------------------------------
# 12. Plot true vs reconstructed distances
# --------------------------------------------------
plt.figure(figsize=(7, 6))

plt.scatter(
    true_values,
    reconstructed_values,
    alpha=0.5,
    s=15
)

plt.xlabel("Original distances (Å)")
plt.ylabel("Reconstructed distances (MDS units)")

plt.title(
    f"1UBQ ShRec3D Reconstruction\n"
    f"Pearson correlation = {pearson_correlation:.3f}"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "1UBQ_distance_comparison.png",
    dpi=300
)

plt.show()


from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# 13. Visualize original and reconstructed structures
# --------------------------------------------------

fig = plt.figure(figsize=(12, 5))

# Original protein
ax1 = fig.add_subplot(121, projection="3d")

ax1.plot(
    coordinates[:, 0],
    coordinates[:, 1],
    coordinates[:, 2],
    marker="o",
    markersize=3,
    linewidth=1
)

ax1.set_title("Original Ubiquitin (1UBQ)")
ax1.set_xlabel("X (Å)")
ax1.set_ylabel("Y (Å)")
ax1.set_zlabel("Z (Å)")


# Reconstructed protein
ax2 = fig.add_subplot(122, projection="3d")

ax2.plot(
    reconstructed_coordinates[:, 0],
    reconstructed_coordinates[:, 1],
    reconstructed_coordinates[:, 2],
    marker="o",
    markersize=3,
    linewidth=1
)

ax2.set_title("ShRec3D Reconstruction")
ax2.set_xlabel("X (MDS units)")
ax2.set_ylabel("Y (MDS units)")
ax2.set_zlabel("Z (MDS units)")


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "1UBQ_3D_comparison.png",
    dpi=300
)

plt.show()

