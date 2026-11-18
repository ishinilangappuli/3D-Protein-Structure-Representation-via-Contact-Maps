
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.distance import cdist, squareform
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.stats import pearsonr, spearmanr


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\ishin\OneDrive\Desktop\Complex_Networks"
    r"\3D-Protein-Structure-Representation-via-Contact-Maps"
)

DATASET_DIR = PROJECT_DIR / "ecoli_dataset"

COORDINATE_DIR = DATASET_DIR / "coordinates"
METADATA_DIR = DATASET_DIR / "metadata"

UNIQUE_PROTEINS_FILE = (
    METADATA_DIR / "ecoli_unique_proteins.csv"
)

VALIDATION_FILE = (
    METADATA_DIR / "ecoli_coordinate_validation_summary.csv"
)

OUTPUT_DIR = PROJECT_DIR / "results_ecoli_8A"

CONTACT_MAP_DIR = OUTPUT_DIR / "contact_maps"
RECONSTRUCTION_DIR = OUTPUT_DIR / "reconstructed_coordinates"

OUTPUT_DIR.mkdir(exist_ok=True)
CONTACT_MAP_DIR.mkdir(exist_ok=True)
RECONSTRUCTION_DIR.mkdir(exist_ok=True)

THRESHOLD = 8.0
DIMENSIONS = 3


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):
    """
    Find a column without depending on capitalization.
    """
    lookup = {str(c).strip().upper(): c for c in df.columns}

    for name in possible_names:
        if name.upper() in lookup:
            return lookup[name.upper()]

    return None


def get_coordinate_file(entity_key, validation_df):
    """
    Find the coordinate CSV corresponding to an entity.
    First use the validation summary, then search the
    coordinate folder using the entity key.
    """

    entity_col = find_column(
        validation_df,
        ["ENTITY_KEY"]
    )

    file_col = find_column(
        validation_df,
        ["COORDINATE_FILE", "COORDINATE_FILENAME", "FILE_NAME"]
    )

    if entity_col is not None and file_col is not None:

        matches = validation_df[
            validation_df[entity_col].astype(str).str.strip()
            == str(entity_key).strip()
        ]

        if not matches.empty:

            filename = str(matches.iloc[0][file_col]).strip()

            if filename and filename.lower() != "nan":

                candidate = Path(filename)

                if not candidate.is_absolute():
                    candidate = COORDINATE_DIR / candidate

                if candidate.exists():
                    return candidate

    # Fallback: search by entity key
    matches = list(
        COORDINATE_DIR.glob(f"*{entity_key}*.csv")
    )

    if matches:
        return matches[0]

    return None



def load_coordinates(csv_file):

    df = pd.read_csv(csv_file)

    # Normalize column names
    df.columns = df.columns.str.strip()

    required = [
        "X_Angstrom",
        "Y_Angstrom",
        "Z_Angstrom"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Extract coordinates
    df = df.dropna(subset=required).reset_index(drop=True)

    coordinates = df[
        ["X_Angstrom", "Y_Angstrom", "Z_Angstrom"]
    ].to_numpy(dtype=float)

    if len(coordinates) < 4:
        raise ValueError("Not enough coordinates.")

    if not np.isfinite(coordinates).all():
        raise ValueError("Invalid coordinate values.")

    # Residue labels
    residue_labels = (
        df["Residue_Number"].astype(str)
        + "_"
        + df["Residue_Name"].astype(str)
    ).tolist()

    return coordinates, residue_labels



# ============================================================
# 3. CONTACT MAP GENERATION
# ============================================================

def generate_contact_map(coordinates):

    distance_matrix = cdist(
        coordinates,
        coordinates,
        metric="euclidean"
    )

    contact_map = (
        distance_matrix < THRESHOLD
    ).astype(int)

    np.fill_diagonal(contact_map, 0)

    return distance_matrix, contact_map


# ============================================================
# 4. SHREC3D-STYLE RECONSTRUCTION
# ============================================================

def reconstruct_structure(contact_map):

    number_of_residues = contact_map.shape[0]

    # Check whether the contact network is connected
    number_of_components, labels = connected_components(
        contact_map,
        directed=False,
        return_labels=True
    )

    if number_of_components > 1:
        raise ValueError(
            f"DISCONNECTED: {number_of_components} components"
        )

    # Shortest-path distance matrix
    graph_distances = shortest_path(
        contact_map,
        directed=False,
        unweighted=True
    )

    if not np.isfinite(graph_distances).all():
        raise ValueError("DISCONNECTED: Infinite shortest paths.")

    # Classical multidimensional scaling (MDS)
    squared_distances = graph_distances ** 2

    identity = np.eye(number_of_residues)

    centering_matrix = (
        identity -
        np.ones((number_of_residues, number_of_residues))
        / number_of_residues
    )

    gram_matrix = (
        -0.5
        * centering_matrix
        @ squared_distances
        @ centering_matrix
    )

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

    # Sort eigenvalues in descending order
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Keep the three largest positive eigenvalues
    positive_indices = np.where(eigenvalues > 1e-10)[0]

    if len(positive_indices) < DIMENSIONS:
        raise ValueError(
            "Not enough positive eigenvalues for 3D reconstruction."
        )

    selected_indices = positive_indices[:DIMENSIONS]

    selected_values = eigenvalues[selected_indices]
    selected_vectors = eigenvectors[:, selected_indices]

    reconstructed_coordinates = (
        selected_vectors @ np.diag(np.sqrt(selected_values))
    )

    return reconstructed_coordinates


# ============================================================
# 5. STRUCTURE COMPARISON
# ============================================================

def compare_structures(original, reconstructed):

    original_distances = cdist(
        original,
        original,
        metric="euclidean"
    )

    reconstructed_distances = cdist(
        reconstructed,
        reconstructed,
        metric="euclidean"
    )

    # Use only the upper triangle (unique residue pairs)
    upper_indices = np.triu_indices(
        len(original),
        k=1
    )

    true_values = original_distances[upper_indices]
    reconstructed_values = reconstructed_distances[upper_indices]

    # Correlations
    pearson_r, pearson_p = pearsonr(
        true_values,
        reconstructed_values
    )

    spearman_r, spearman_p = spearmanr(
        true_values,
        reconstructed_values
    )

    # Least-squares scale adjustment for pairwise distances
    denominator = np.sum(reconstructed_values ** 2)

    if denominator == 0:
        raise ValueError("Cannot calculate scale factor.")

    scale_factor = (
        np.sum(true_values * reconstructed_values)
        / denominator
    )

    scaled_values = reconstructed_values * scale_factor

    absolute_errors = np.abs(
        true_values - scaled_values
    )

    mae = np.mean(absolute_errors)
    maximum_error = np.max(absolute_errors)

    rmse = np.sqrt(
        np.mean((true_values - scaled_values) ** 2)
    )

    # Scale + rotation + translation alignment for coordinate RMSD
    original_centered = original - np.mean(original, axis=0)

    reconstructed_centered = (
        reconstructed - np.mean(reconstructed, axis=0)
    )

    covariance = (
        reconstructed_centered.T @ original_centered
    )

    U, singular_values, Vt = np.linalg.svd(covariance)

    correction = np.eye(3)

    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1

    rotation = U @ correction @ Vt

    coordinate_scale = (
        np.sum(singular_values * np.diag(correction))
        / np.sum(reconstructed_centered ** 2)
    )

    aligned = (
        coordinate_scale
        * reconstructed_centered
        @ rotation
    )

    aligned = aligned + np.mean(original, axis=0)

    aligned_rmsd = np.sqrt(
        np.mean(
            np.sum((original - aligned) ** 2, axis=1)
        )
    )

    return {
        "Pearson_correlation": pearson_r,
        "Pearson_p_value": pearson_p,
        "Spearman_correlation": spearman_r,
        "Spearman_p_value": spearman_p,
        "Scale_factor": scale_factor,
        "Mean_absolute_error_A": mae,
        "Maximum_absolute_error_A": maximum_error,
        "Distance_RMSE_A": rmse,
        "Aligned_RMSD_A": aligned_rmsd,
        "Aligned_coordinates": aligned
    }


# ============================================================
# 6. LOAD DATASET
# ============================================================

print("\nLoading E. coli protein dataset...")

unique_df = pd.read_csv(UNIQUE_PROTEINS_FILE)

validation_df = pd.read_csv(VALIDATION_FILE)

unique_col = find_column(
    unique_df,
    ["IS_UNIQUE"]
)

entity_key_col = find_column(
    unique_df,
    ["ENTITY_KEY"]
)

representative_col = find_column(
    unique_df,
    ["REPRESENTATIVE_ENTITY"]
)

if unique_col is None:
    raise ValueError("IS_UNIQUE column not found.")

if entity_key_col is None:
    raise ValueError("ENTITY_KEY column not found.")

# Select exact-sequence-unique proteins
unique_df[unique_col] = (
    unique_df[unique_col]
    .astype(str)
    .str.strip()
    .str.lower()
)

unique_df = unique_df[
    unique_df[unique_col].isin(["true", "1", "yes"])
].copy()

# Use the representative entity key when available
if representative_col is not None:

    unique_df["_PROCESS_KEY"] = (
        unique_df[representative_col]
        .fillna(unique_df[entity_key_col])
        .astype(str)
        .str.strip()
    )

else:

    unique_df["_PROCESS_KEY"] = (
        unique_df[entity_key_col]
        .astype(str)
        .str.strip()
    )

# Remove duplicate keys
unique_df = unique_df.drop_duplicates(
    subset=["_PROCESS_KEY"]
).reset_index(drop=True)

print(f"Unique proteins selected: {len(unique_df)}")
print(f"Contact threshold: {THRESHOLD} Å")


# ============================================================
# 7. BATCH PROCESSING
# ============================================================

results = []

for index, row in unique_df.iterrows():

    entity_key = row["_PROCESS_KEY"]

    print("\n" + "=" * 60)
    print(f"Processing {index + 1}/{len(unique_df)}: {entity_key}")
    print("=" * 60)

    result = {
        "Protein": entity_key,
        "Threshold_A": THRESHOLD,
        "Status": "FAILED"
    }

    try:

        # ----------------------------------------------------
        # Find and load coordinate CSV
        # ----------------------------------------------------

        coordinate_file = get_coordinate_file(
            entity_key,
            validation_df
        )

        if coordinate_file is None:
            raise FileNotFoundError(
                f"Coordinate CSV not found for {entity_key}"
            )

        original_coordinates, residue_labels = (
            load_coordinates(coordinate_file)
        )

        number_of_residues = len(original_coordinates)

        # ----------------------------------------------------
        # Generate contact map
        # ----------------------------------------------------

        distance_matrix, contact_map = generate_contact_map(
            original_coordinates
        )

        number_of_contacts = int(
            np.sum(contact_map) // 2
        )

        possible_contacts = (
            number_of_residues
            * (number_of_residues - 1)
            // 2
        )

        contact_density = (
            number_of_contacts / possible_contacts
            if possible_contacts > 0 else 0
        )

        result.update({
            "Number_of_residues": number_of_residues,
            "Number_of_contacts": number_of_contacts,
            "Contact_density": contact_density
        })

        # Save contact map
        contact_df = pd.DataFrame(
            contact_map,
            index=residue_labels,
            columns=residue_labels
        )

        contact_df.index.name = "Residue"

        contact_file = (
            CONTACT_MAP_DIR
            / f"{entity_key}_contact_map.csv"
        )

        contact_df.to_csv(contact_file)

        # ----------------------------------------------------
        # Reconstruct 3D structure
        # ----------------------------------------------------

        reconstructed_coordinates = reconstruct_structure(
            contact_map
        )

        reconstruction_df = pd.DataFrame(
            reconstructed_coordinates,
            columns=["X", "Y", "Z"]
        )

        reconstruction_df.insert(
            0,
            "Residue",
            residue_labels
        )

        reconstruction_file = (
            RECONSTRUCTION_DIR
            / f"{entity_key}_reconstructed_coordinates.csv"
        )

        reconstruction_df.to_csv(
            reconstruction_file,
            index=False
        )

        # ----------------------------------------------------
        # Compare original and reconstructed structures
        # ----------------------------------------------------

        comparison = compare_structures(
            original_coordinates,
            reconstructed_coordinates
        )

        aligned_df = pd.DataFrame(
            comparison.pop("Aligned_coordinates"),
            columns=["X", "Y", "Z"]
        )

        aligned_df.insert(
            0,
            "Residue",
            residue_labels
        )

        aligned_file = (
            RECONSTRUCTION_DIR
            / f"{entity_key}_aligned_coordinates.csv"
        )

        aligned_df.to_csv(
            aligned_file,
            index=False
        )

        result.update(comparison)

        result["Status"] = "OK"

        print(f"Residues: {number_of_residues}")
        print(f"Contacts: {number_of_contacts}")
        print(
            "Pearson correlation: "
            f"{result['Pearson_correlation']:.4f}"
        )
        print(
            "Aligned RMSD: "
            f"{result['Aligned_RMSD_A']:.4f} Å"
        )

    except Exception as error:

        error_message = str(error)

        if "DISCONNECTED" in error_message.upper():
            result["Status"] = "DISCONNECTED"
        else:
            result["Status"] = "FAILED"

        result["Error"] = error_message

        print(f"Status: {result['Status']}")
        print(f"Reason: {error_message}")

    results.append(result)


# ============================================================
# 8. SAVE FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(results)

summary_file = (
    OUTPUT_DIR / "ecoli_shrec3d_8A_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 9. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("BATCH PROCESSING COMPLETED")
print("=" * 60)

print(f"Total unique proteins: {len(summary_df)}")

if "Status" in summary_df.columns:

    print("\nProcessing status:")

    print(
        summary_df["Status"]
        .value_counts(dropna=False)
    )

print(f"\nSummary saved to:\n{summary_file}")

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nAll processing finished.")
