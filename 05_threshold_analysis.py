
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, spearmanr


# ==================================================
# 1. PATHS AND PARAMETERS
# ==================================================

BASE_DIR = Path(
    r"C:\Users\ishin\OneDrive\Desktop\Complex_Networks"
)

PDB_FILE = BASE_DIR / "1UBQ.pdb"

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [8.0, 10.0, 12.0]


# ==================================================
# 2. EXTRACT ORIGINAL PROTEIN COORDINATES
# ==================================================

parser = PDBParser(QUIET=True)
structure = parser.get_structure("1UBQ", str(PDB_FILE))

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

n_residues = len(original_coordinates)

print("\nProtein: 1UBQ")
print("Chain:", chain.id)
print("Number of residues:", n_residues)


# ==================================================
# 3. ORIGINAL DISTANCE MATRIX
# ==================================================

original_distances = cdist(
    original_coordinates,
    original_coordinates,
    metric="euclidean"
)

upper_triangle = np.triu_indices(
    n_residues,
    k=1
)

true_values = original_distances[upper_triangle]


# ==================================================
# 4. CLASSICAL MDS FUNCTION
# ==================================================

def classical_mds(distance_matrix, dimensions=3):

    n = distance_matrix.shape[0]

    centering = (
        np.eye(n) - np.ones((n, n)) / n
    )

    squared_distances = distance_matrix ** 2

    gram_matrix = (
        -0.5
        * centering
        @ squared_distances
        @ centering
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        gram_matrix
    )

    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Keep the three largest positive eigenvalues
    positive = np.where(eigenvalues > 1e-10)[0]

    if len(positive) < dimensions:
        raise ValueError(
            "Insufficient positive eigenvalues for 3D reconstruction."
        )

    selected = positive[:dimensions]

    eigenvalues = eigenvalues[selected]
    eigenvectors = eigenvectors[:, selected]

    reconstructed = (
        eigenvectors * np.sqrt(eigenvalues)
    )

    return reconstructed


# ==================================================
# 5. SCALE-ADJUSTED ALIGNMENT
# ==================================================

def align_structures(reference, moving):

    X = reference - reference.mean(axis=0)
    Y = moving - moving.mean(axis=0)

    U, singular_values, Vt = np.linalg.svd(
        Y.T @ X
    )

    correction = np.eye(3)

    rotation = U @ Vt

    if np.linalg.det(rotation) < 0:
        correction[-1, -1] = -1

    rotation = U @ correction @ Vt

    scale = (
        np.sum(singular_values * np.diag(correction))
        / np.sum(Y ** 2)
    )

    aligned = (
        scale * Y @ rotation
        + reference.mean(axis=0)
    )

    return aligned, scale


# ==================================================
# 6. ANALYSE EACH CONTACT THRESHOLD
# ==================================================

all_results = []
all_coordinates = {}

for threshold in THRESHOLDS:

    print(f"\nProcessing threshold: {threshold} Å")

    # Generate contact map
    contact_map = (
        original_distances < threshold
    ).astype(int)

    np.fill_diagonal(contact_map, 0)

    # Number of unique contacts
    number_of_contacts = int(
        np.sum(contact_map) // 2
    )

    possible_pairs = n_residues * (n_residues - 1) // 2

    contact_density = (
        number_of_contacts / possible_pairs
    )

    # Shortest paths
    shortest_distances = shortest_path(
        contact_map,
        directed=False,
        unweighted=True
    )

    # Check graph connectivity
    if not np.all(np.isfinite(shortest_distances)):

        print(
            f"WARNING: Graph disconnected at {threshold} Å"
        )

        all_results.append({
            "Protein": "1UBQ",
            "Threshold_A": threshold,
            "Number_of_residues": n_residues,
            "Number_of_contacts": number_of_contacts,
            "Contact_density": contact_density,
            "Scale_factor": np.nan,
            "Aligned_RMSD_A": np.nan,
            "Pearson_correlation": np.nan,
            "Pearson_p_value": np.nan,
            "Spearman_correlation": np.nan,
            "Spearman_p_value": np.nan,
            "Mean_absolute_error_A": np.nan,
            "Maximum_absolute_error_A": np.nan,
            "Distance_RMSE_A": np.nan,
            "Status": "Disconnected graph"
        })

        continue

    # Reconstruct coordinates
    reconstructed = classical_mds(
        shortest_distances,
        dimensions=3
    )

    # Scale-adjusted alignment
    aligned, scale = align_structures(
        original_coordinates,
        reconstructed
    )

    all_coordinates[threshold] = aligned

    # RMSD
    rmsd = np.sqrt(
        np.mean(
            np.sum(
                (original_coordinates - aligned) ** 2,
                axis=1
            )
        )
    )

    # Pairwise reconstructed distances
    reconstructed_distances = cdist(
        aligned,
        aligned,
        metric="euclidean"
    )

    predicted_values = reconstructed_distances[
        upper_triangle
    ]

    # Distance errors
    absolute_errors = np.abs(
        true_values - predicted_values
    )

    mae = np.mean(absolute_errors)

    maximum_error = np.max(absolute_errors)

    distance_rmse = np.sqrt(
        np.mean(absolute_errors ** 2)
    )

    # Correlations
    pearson_correlation, pearson_p = pearsonr(
        true_values,
        predicted_values
    )

    spearman_correlation, spearman_p = spearmanr(
        true_values,
        predicted_values
    )

    # Save results
    all_results.append({
        "Protein": "1UBQ",
        "Threshold_A": threshold,
        "Number_of_residues": n_residues,
        "Number_of_contacts": number_of_contacts,
        "Contact_density": contact_density,
        "Scale_factor": scale,
        "Aligned_RMSD_A": rmsd,
        "Pearson_correlation": pearson_correlation,
        "Pearson_p_value": pearson_p,
        "Spearman_correlation": spearman_correlation,
        "Spearman_p_value": spearman_p,
        "Mean_absolute_error_A": mae,
        "Maximum_absolute_error_A": maximum_error,
        "Distance_RMSE_A": distance_rmse,
        "Status": "Success"
    })

    print(f"Contacts: {number_of_contacts}")
    print(f"Scale factor: {scale:.4f}")
    print(f"RMSD: {rmsd:.4f} Å")
    print(f"Pearson: {pearson_correlation:.4f}")
    print(f"Spearman: {spearman_correlation:.4f}")
    print(f"MAE: {mae:.4f} Å")
    print(f"Maximum error: {maximum_error:.4f} Å")


# ==================================================
# 7. SAVE COMPARISON RESULTS
# ==================================================

results_df = pd.DataFrame(all_results)

results_file = (
    OUTPUT_DIR / "1UBQ_threshold_analysis.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print("\n===== THRESHOLD ANALYSIS SUMMARY =====")

print(
    results_df.to_string(index=False)
)

print("\nResults saved to:", results_file)


# ==================================================
# 8. PLOT CORRELATION VS THRESHOLD
# ==================================================

successful = results_df[
    results_df["Status"] == "Success"
]

if not successful.empty:

    plt.figure(figsize=(8, 5))

    plt.plot(
        successful["Threshold_A"],
        successful["Pearson_correlation"],
        marker="o",
        label="Pearson"
    )

    plt.plot(
        successful["Threshold_A"],
        successful["Spearman_correlation"],
        marker="s",
        label="Spearman"
    )

    plt.xlabel("Contact threshold (Å)")
    plt.ylabel("Correlation coefficient")

    plt.title(
        "1UBQ: Reconstruction Correlation vs Threshold"
    )

    plt.xticks(THRESHOLDS)
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "1UBQ_threshold_correlations.png",
        dpi=300
    )

    plt.show()


# ==================================================
# 9. PLOT RMSD AND MAE VS THRESHOLD
# ==================================================

if not successful.empty:

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(
        successful["Threshold_A"],
        successful["Aligned_RMSD_A"],
        marker="o"
    )

    axes[0].set_title("Aligned RMSD")
    axes[0].set_xlabel("Threshold (Å)")
    axes[0].set_ylabel("RMSD (Å)")
    axes[0].set_xticks(THRESHOLDS)

    axes[1].plot(
        successful["Threshold_A"],
        successful["Mean_absolute_error_A"],
        marker="s"
    )

    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Threshold (Å)")
    axes[1].set_ylabel("MAE (Å)")
    axes[1].set_xticks(THRESHOLDS)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "1UBQ_threshold_errors.png",
        dpi=300
    )

    plt.show()


print("\nThreshold analysis completed!")
