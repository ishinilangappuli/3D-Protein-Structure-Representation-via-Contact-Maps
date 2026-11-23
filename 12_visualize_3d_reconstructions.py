
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


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

VALIDATION_FILE = (
    METADATA_DIR / "ecoli_coordinate_validation_summary.csv"
)

OUTPUT_DIR = PROJECT_DIR / "results_ecoli_8A"

SUMMARY_FILE = (
    OUTPUT_DIR / "ecoli_shrec3d_8A_summary.csv"
)

RECONSTRUCTION_DIR = (
    OUTPUT_DIR / "reconstructed_coordinates"
)

VISUALIZATION_DIR = (
    OUTPUT_DIR / "3D_visualizations"
)

VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):

    lookup = {
        str(column).strip().upper(): column
        for column in df.columns
    }

    for name in possible_names:

        if name.upper() in lookup:
            return lookup[name.upper()]

    return None


def get_coordinate_file(entity_key, validation_df):

    entity_col = find_column(
        validation_df,
        ["ENTITY_KEY"]
    )

    file_col = find_column(
        validation_df,
        [
            "COORDINATE_FILE",
            "COORDINATE_FILENAME",
            "FILE_NAME"
        ]
    )

    if entity_col is not None and file_col is not None:

        matches = validation_df[
            validation_df[entity_col].astype(str).str.strip()
            == str(entity_key).strip()
        ]

        if not matches.empty:

            filename = str(
                matches.iloc[0][file_col]
            ).strip()

            if filename and filename.lower() != "nan":

                candidate = Path(filename)

                if not candidate.is_absolute():
                    candidate = COORDINATE_DIR / candidate

                if candidate.exists():
                    return candidate

    # Fallback search
    matches = list(
        COORDINATE_DIR.glob(f"*{entity_key}*.csv")
    )

    if matches:
        return matches[0]

    return None


def load_original_coordinates(csv_file):

    df = pd.read_csv(csv_file)

    df.columns = df.columns.str.strip()

    required = [
        "X_Angstrom",
        "Y_Angstrom",
        "Z_Angstrom"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing original coordinate columns: {missing}"
        )

    df = df.dropna(subset=required).reset_index(drop=True)

    coordinates = df[
        required
    ].to_numpy(dtype=float)

    if len(coordinates) < 4:
        raise ValueError("Not enough original coordinates.")

    if not np.isfinite(coordinates).all():
        raise ValueError("Invalid original coordinate values.")

    return coordinates, df


def load_aligned_coordinates(csv_file):

    df = pd.read_csv(csv_file)

    df.columns = df.columns.str.strip()

    required = ["X", "Y", "Z"]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing reconstructed coordinate columns: {missing}"
        )

    coordinates = df[
        required
    ].to_numpy(dtype=float)

    if not np.isfinite(coordinates).all():
        raise ValueError("Invalid reconstructed coordinate values.")

    return coordinates, df


# ============================================================
# 3. 3D PLOTTING FUNCTION
# ============================================================

def set_equal_axes(ax, all_coordinates):

    minimum = np.min(all_coordinates, axis=0)
    maximum = np.max(all_coordinates, axis=0)

    center = (minimum + maximum) / 2

    maximum_range = np.max(maximum - minimum)

    if maximum_range == 0:
        maximum_range = 1.0

    half_range = maximum_range / 2 * 1.05

    ax.set_xlim(
        center[0] - half_range,
        center[0] + half_range
    )

    ax.set_ylim(
        center[1] - half_range,
        center[1] + half_range
    )

    ax.set_zlim(
        center[2] - half_range,
        center[2] + half_range
    )

    ax.set_box_aspect((1, 1, 1))


def plot_structure(ax, coordinates, title, norm, cmap):

    number_of_residues = len(coordinates)

    residue_index = np.arange(number_of_residues)

    # Backbone connection
    ax.plot(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        color="grey",
        linewidth=0.7,
        alpha=0.65
    )

    # C-alpha atoms coloured by residue order
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=residue_index,
        cmap=cmap,
        norm=norm,
        s=13,
        depthshade=True
    )

    ax.set_title(title, fontsize=11, fontweight="bold")

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    ax.view_init(elev=22, azim=40)

    return scatter


def create_comparison_plot(
    protein,
    original,
    reconstructed,
    pearson,
    rmsd,
    output_file
):

    number_of_residues = len(original)

    if len(reconstructed) != number_of_residues:
        raise ValueError(
            "Original and reconstructed structures have "
            "different numbers of residues."
        )

    all_coordinates = np.vstack([
        original,
        reconstructed
    ])

    norm = Normalize(
        vmin=0,
        vmax=max(number_of_residues - 1, 1)
    )

    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(14, 7))

    fig.suptitle(
        f"{protein}: Original vs Reconstructed 3D Structure",
        fontsize=15,
        fontweight="bold"
    )

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    plot_structure(
        ax1,
        original,
        "Original Cα Backbone",
        norm,
        cmap
    )

    scatter = plot_structure(
        ax2,
        reconstructed,
        "Reconstructed Cα Backbone (Aligned)",
        norm,
        cmap
    )

    set_equal_axes(ax1, all_coordinates)
    set_equal_axes(ax2, all_coordinates)

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=[ax1, ax2],
        shrink=0.65,
        pad=0.08
    )

    colorbar.set_label("Residue sequence position")

    fig.text(
        0.5,
        0.035,
        f"Residues: {number_of_residues}   |   "
        f"Pearson correlation: {pearson:.4f}   |   "
        f"Aligned RMSD: {rmsd:.3f} Å",
        ha="center",
        fontsize=11
    )

    plt.subplots_adjust(
        top=0.87,
        bottom=0.13,
        wspace=0.12
    )

    plt.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 4. LOAD SUMMARY AND VALIDATION DATA
# ============================================================

print("\nLoading reconstruction results...")

if not SUMMARY_FILE.exists():
    raise FileNotFoundError(
        f"Summary file not found:\n{SUMMARY_FILE}"
    )

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )

summary_df = pd.read_csv(SUMMARY_FILE)

validation_df = pd.read_csv(VALIDATION_FILE)

successful_df = summary_df[
    summary_df["Status"].astype(str).str.upper() == "OK"
].copy()

print("Total proteins:", len(summary_df))
print("Successful reconstructions:", len(successful_df))

print("\nStarting 3D visualisation...")


# ============================================================
# 5. GENERATE VISUALISATIONS
# ============================================================

visualization_results = []

for index, row in successful_df.iterrows():

    protein = str(row["Protein"]).strip()

    print("\n" + "=" * 55)
    print(f"Protein: {protein}")
    print("=" * 55)

    result = {
        "Protein": protein,
        "Status": "FAILED"
    }

    try:

        # ----------------------------------------------------
        # Find original coordinate file
        # ----------------------------------------------------

        original_file = get_coordinate_file(
            protein,
            validation_df
        )

        if original_file is None:
            raise FileNotFoundError(
                f"Original coordinate file not found: {protein}"
            )

        # ----------------------------------------------------
        # Find aligned reconstructed coordinates
        # ----------------------------------------------------

        aligned_file = (
            RECONSTRUCTION_DIR
            / f"{protein}_aligned_coordinates.csv"
        )

        if not aligned_file.exists():
            raise FileNotFoundError(
                f"Aligned coordinate file not found:\n{aligned_file}"
            )

        # ----------------------------------------------------
        # Load both structures
        # ----------------------------------------------------

        original, original_df = load_original_coordinates(
            original_file
        )

        reconstructed, reconstructed_df = (
            load_aligned_coordinates(aligned_file)
        )

        # ----------------------------------------------------
        # Verify residue counts
        # ----------------------------------------------------

        if len(original) != len(reconstructed):

            raise ValueError(
                f"Residue count mismatch: "
                f"original={len(original)}, "
                f"reconstructed={len(reconstructed)}"
            )

        # ----------------------------------------------------
        # Verify residue ordering where labels are available
        # ----------------------------------------------------

        if (
            "Residue" in original_df.columns
            and "Residue" in reconstructed_df.columns
        ):

            original_labels = (
                original_df["Residue"].astype(str).tolist()
            )

            reconstructed_labels = (
                reconstructed_df["Residue"].astype(str).tolist()
            )

            if original_labels != reconstructed_labels:

                raise ValueError(
                    "Residue ordering does not match between "
                    "original and reconstructed structures."
                )

        # ----------------------------------------------------
        # Create output figure
        # ----------------------------------------------------

        output_file = (
            VISUALIZATION_DIR
            / f"{protein}_3D_comparison.png"
        )

        create_comparison_plot(
            protein=protein,
            original=original,
            reconstructed=reconstructed,
            pearson=float(row["Pearson_correlation"]),
            rmsd=float(row["Aligned_RMSD_A"]),
            output_file=output_file
        )

        result.update({
            "Status": "OK",
            "Number_of_residues": len(original),
            "Pearson_correlation": row["Pearson_correlation"],
            "Aligned_RMSD_A": row["Aligned_RMSD_A"],
            "Figure": str(output_file)
        })

        print("Original coordinates loaded.")
        print("Aligned reconstructed coordinates loaded.")
        print(f"Residues: {len(original)}")
        print(f"Figure saved: {output_file.name}")

    except Exception as error:

        result["Error"] = str(error)

        print("Visualisation failed:", error)

    visualization_results.append(result)


# ============================================================
# 6. SAVE VISUALISATION REPORT
# ============================================================

visualization_df = pd.DataFrame(
    visualization_results
)

report_file = (
    VISUALIZATION_DIR
    / "3D_visualization_status.csv"
)

visualization_df.to_csv(
    report_file,
    index=False
)


# ============================================================
# 7. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("3D VISUALISATION COMPLETED")
print("=" * 60)

print(
    visualization_df["Status"].value_counts(dropna=False)
)

print(
    "\nSuccessfully generated:",
    (visualization_df["Status"] == "OK").sum()
)

print(
    "Failed:",
    (visualization_df["Status"] != "OK").sum()
)

print("\nVisualisation folder:")
print(VISUALIZATION_DIR)

print("\nStatus report:")
print(report_file)

print("\nAll visualisation tasks finished.")
