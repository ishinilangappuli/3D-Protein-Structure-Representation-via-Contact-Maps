
import os
import re
import warnings
import numpy as np
import pandas as pd

from Bio.PDB import MMCIFParser, MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# =====================================================
# 1. PATHS AND SETTINGS
# =====================================================

BASE_DIR = "ecoli_dataset"

CIF_DIR = os.path.join(BASE_DIR, "cif_files")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
COORDINATE_DIR = os.path.join(BASE_DIR, "coordinates")

REPORT_PATH = os.path.join(
    METADATA_DIR,
    "ecoli_protein_download_report.csv"
)

SUMMARY_PATH = os.path.join(
    METADATA_DIR,
    "ecoli_coordinate_validation_summary.csv"
)

os.makedirs(COORDINATE_DIR, exist_ok=True)

warnings.simplefilter("ignore", PDBConstructionWarning)

# Use label_asym_id as the chain identifier.
parser = MMCIFParser(
    QUIET=True,
    auth_chains=False,
    auth_residues=True
)


# =====================================================
# 2. HELPER FUNCTION
# =====================================================

def to_list(value):
    """Convert an MMCIF dictionary value into a list."""

    if isinstance(value, list):
        return value

    return [value]


def safe_filename(value):
    """Make a value safe for use in a filename."""

    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


# =====================================================
# 3. READ DOWNLOADED ENTITY REPORT
# =====================================================

if not os.path.exists(REPORT_PATH):
    raise FileNotFoundError(
        f"Download report not found: {REPORT_PATH}"
    )

entities_df = pd.read_csv(REPORT_PATH)

required_columns = [
    "PDB_ID",
    "ENTITY_ID",
    "ENTITY_KEY",
    "STATUS"
]

for column in required_columns:
    if column not in entities_df.columns:
        raise ValueError(
            f"Missing column in download report: {column}"
        )

print("======================================")
print("E. COLI COORDINATE VALIDATION")
print("======================================")

print("Protein entities in report:", len(entities_df))


# =====================================================
# 4. PROCESS EACH PROTEIN ENTITY
# =====================================================

results = []

for index, row in entities_df.iterrows():

    pdb_id = str(row["PDB_ID"]).strip().upper()
    entity_id = str(row["ENTITY_ID"]).strip()
    entity_key = str(row["ENTITY_KEY"]).strip()

    cif_path = os.path.join(
        CIF_DIR,
        f"{pdb_id}.cif"
    )

    print(
        f"\nProcessing {index + 1}/{len(entities_df)}: "
        f"{entity_key}"
    )

    result = {
        "PDB_ID": pdb_id,
        "ENTITY_ID": entity_id,
        "ENTITY_KEY": entity_key,
        "CHAIN_ID": "",
        "NUMBER_OF_CA_ATOMS": 0,
        "NUMBER_OF_RESIDUES": 0,
        "COORDINATE_FILE": "",
        "STATUS": ""
    }

    try:

        # ---------------------------------------------
        # Check CIF file
        # ---------------------------------------------

        if not os.path.exists(cif_path):
            result["STATUS"] = "FILE_NOT_FOUND"
            results.append(result)
            print("File not found.")
            continue

        # ---------------------------------------------
        # Read CIF metadata
        # ---------------------------------------------

        cif_dict = MMCIF2Dict.MMCIF2Dict(cif_path)

        asym_ids = to_list(
            cif_dict["_struct_asym.id"]
        )

        asym_entity_ids = to_list(
            cif_dict["_struct_asym.entity_id"]
        )

        matching_chains = [
            asym_id
            for asym_id, asym_entity_id
            in zip(asym_ids, asym_entity_ids)
            if str(asym_entity_id) == entity_id
        ]

        if not matching_chains:
            result["STATUS"] = "ENTITY_CHAIN_NOT_FOUND"
            results.append(result)
            print("No matching chain found.")
            continue

        # ---------------------------------------------
        # Parse 3D structure
        # ---------------------------------------------

        structure = parser.get_structure(
            pdb_id,
            cif_path
        )

        model = structure[0]

        # ---------------------------------------------
        # Find C-alpha coordinates for matching chains
        # ---------------------------------------------

        chain_candidates = []

        for chain_id in matching_chains:

            if chain_id not in model:
                continue

            chain = model[chain_id]

            coordinates = []
            residue_numbers = []
            insertion_codes = []
            residue_names = []

            for residue in chain:

                # Exclude water and other non-standard residues.
                if residue.id[0] != " ":
                    continue

                if "CA" not in residue:
                    continue

                ca_atom = residue["CA"]

                coordinates.append(ca_atom.coord)

                residue_numbers.append(residue.id[1])
                insertion_codes.append(residue.id[2])
                residue_names.append(residue.resname)

            if coordinates:

                chain_candidates.append({
                    "chain_id": chain_id,
                    "coordinates": np.asarray(
                        coordinates,
                        dtype=float
                    ),
                    "residue_numbers": residue_numbers,
                    "insertion_codes": insertion_codes,
                    "residue_names": residue_names
                })

        if not chain_candidates:
            result["STATUS"] = "NO_CA_COORDINATES"
            results.append(result)
            print("No C-alpha coordinates found.")
            continue

        # ---------------------------------------------
        # Select the chain with the most C-alpha atoms
        # ---------------------------------------------

        selected_chain = max(
            chain_candidates,
            key=lambda item: len(item["coordinates"])
        )

        coords = selected_chain["coordinates"]

        chain_id = selected_chain["chain_id"]

        number_of_ca = len(coords)

        # ---------------------------------------------
        # Validate coordinate matrix
        # ---------------------------------------------

        if coords.shape != (number_of_ca, 3):
            result["STATUS"] = "INVALID_COORDINATE_SHAPE"
            results.append(result)
            print("Invalid coordinate matrix.")
            continue

        if not np.all(np.isfinite(coords)):
            result["STATUS"] = "INVALID_COORDINATE_VALUES"
            results.append(result)
            print("Non-finite coordinate values.")
            continue

        # ---------------------------------------------
        # Save coordinates
        # ---------------------------------------------

        coordinate_data = pd.DataFrame({
            "Residue_Number": selected_chain["residue_numbers"],
            "Insertion_Code": selected_chain["insertion_codes"],
            "Residue_Name": selected_chain["residue_names"],
            "X_Angstrom": coords[:, 0],
            "Y_Angstrom": coords[:, 1],
            "Z_Angstrom": coords[:, 2]
        })

        filename = (
            f"{safe_filename(entity_key)}_"
            f"chain_{safe_filename(chain_id)}_coordinates.csv"
        )

        coordinate_path = os.path.join(
            COORDINATE_DIR,
            filename
        )

        coordinate_data.to_csv(
            coordinate_path,
            index=False
        )

        # ---------------------------------------------
        # Record results
        # ---------------------------------------------

        result["CHAIN_ID"] = chain_id
        result["NUMBER_OF_CA_ATOMS"] = number_of_ca
        result["NUMBER_OF_RESIDUES"] = number_of_ca
        result["COORDINATE_FILE"] = filename

        if number_of_ca < 10:
            result["STATUS"] = "LOW_CA_COUNT"
        else:
            result["STATUS"] = "OK"

        results.append(result)

        print("Selected chain:", chain_id)
        print("C-alpha atoms:", number_of_ca)
        print("Coordinate matrix:", coords.shape)
        print("Status:", result["STATUS"])

    except Exception as error:

        result["STATUS"] = "ERROR"
        result["ERROR_MESSAGE"] = str(error)

        results.append(result)

        print("Processing error:", error)


# =====================================================
# 5. SAVE VALIDATION SUMMARY
# =====================================================

summary_df = pd.DataFrame(results)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)


# =====================================================
# 6. FINAL STATISTICS
# =====================================================

successful = (summary_df["STATUS"] == "OK").sum()

low_count = (
    summary_df["STATUS"] == "LOW_CA_COUNT"
).sum()

failed = len(summary_df) - successful - low_count

print("\n======================================")
print("VALIDATION COMPLETED")
print("======================================")

print("Total entities:", len(summary_df))
print("Successfully extracted:", successful)
print("Low C-alpha count:", low_count)
print("Failed entities:", failed)

print("\nCoordinate files saved in:")
print(COORDINATE_DIR)

print("\nValidation report saved in:")
print(SUMMARY_PATH)

print("\n======================================")
print("STATUS COUNTS")
print("======================================")

print(summary_df["STATUS"].value_counts())

print("\nProcess completed.")
