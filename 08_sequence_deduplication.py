
import os
import re
import pandas as pd

from Bio.PDB import MMCIF2Dict

# =====================================================
# 1. PATHS
# =====================================================

BASE_DIR = "ecoli_dataset"

CIF_DIR = os.path.join(BASE_DIR, "cif_files")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
SEQUENCE_DIR = os.path.join(BASE_DIR, "sequences")

DOWNLOAD_REPORT = os.path.join(
    METADATA_DIR,
    "ecoli_protein_download_report.csv"
)

VALIDATION_REPORT = os.path.join(
    METADATA_DIR,
    "ecoli_coordinate_validation_summary.csv"
)

OUTPUT_REPORT = os.path.join(
    METADATA_DIR,
    "ecoli_sequence_deduplication_report.csv"
)

UNIQUE_REPORT = os.path.join(
    METADATA_DIR,
    "ecoli_unique_proteins.csv"
)

FASTA_PATH = os.path.join(
    SEQUENCE_DIR,
    "ecoli_protein_sequences.fasta"
)

os.makedirs(SEQUENCE_DIR, exist_ok=True)

# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def get_list(value):
    if isinstance(value, list):
        return value
    return [value]

def clean_sequence(sequence):
    """
    Remove whitespace and convert sequence to uppercase.
    Keep only standard one-letter amino acid symbols
    and X for unknown residues.
    """

    sequence = re.sub(r"\s+", "", str(sequence))
    sequence = sequence.upper()

    sequence = re.sub(
        r"[^ACDEFGHIKLMNPQRSTVWYX]",
        "",
        sequence
    )

    return sequence

# =====================================================
# 3. LOAD REPORTS
# =====================================================

if not os.path.exists(DOWNLOAD_REPORT):
    raise FileNotFoundError(DOWNLOAD_REPORT)

if not os.path.exists(VALIDATION_REPORT):
    raise FileNotFoundError(VALIDATION_REPORT)

download_df = pd.read_csv(DOWNLOAD_REPORT)
validation_df = pd.read_csv(VALIDATION_REPORT)

# Only use successfully validated entities.
valid_entities = validation_df[
    validation_df["STATUS"] == "OK"
].copy()

print("======================================")
print("PROTEIN SEQUENCE DEDUPLICATION")
print("======================================")

print("Downloaded entities:", len(download_df))
print("Validated entities:", len(valid_entities))

# =====================================================
# 4. EXTRACT SEQUENCES
# =====================================================

sequence_records = []

for index, row in valid_entities.iterrows():

    pdb_id = str(row["PDB_ID"]).strip().upper()
    entity_id = str(row["ENTITY_ID"]).strip()
    entity_key = str(row["ENTITY_KEY"]).strip()

    cif_path = os.path.join(
        CIF_DIR,
        f"{pdb_id}.cif"
    )

    print(
        f"\nExtracting {index + 1}/{len(valid_entities)}: "
        f"{entity_key}"
    )

    try:

        cif_dict = MMCIF2Dict.MMCIF2Dict(cif_path)

        entity_ids = get_list(
            cif_dict["_entity_poly.entity_id"]
        )

        entity_sequences = get_list(
            cif_dict[
                "_entity_poly.pdbx_seq_one_letter_code_can"
            ]
        )

        # Find the sequence belonging to this entity.
        sequence = None

        for current_id, current_sequence in zip(
            entity_ids,
            entity_sequences
        ):

            if str(current_id) == entity_id:
                sequence = clean_sequence(current_sequence)
                break

        if not sequence:
            print("Sequence not found:", entity_key)

            sequence_records.append({
                "PDB_ID": pdb_id,
                "ENTITY_ID": entity_id,
                "ENTITY_KEY": entity_key,
                "SEQUENCE": "",
                "SEQUENCE_LENGTH": 0,
                "STATUS": "SEQUENCE_NOT_FOUND"
            })

            continue

        sequence_records.append({
            "PDB_ID": pdb_id,
            "ENTITY_ID": entity_id,
            "ENTITY_KEY": entity_key,
            "SEQUENCE": sequence,
            "SEQUENCE_LENGTH": len(sequence),
            "STATUS": "OK"
        })

        print("Sequence length:", len(sequence))

    except Exception as error:

        print("Error:", error)

        sequence_records.append({
            "PDB_ID": pdb_id,
            "ENTITY_ID": entity_id,
            "ENTITY_KEY": entity_key,
            "SEQUENCE": "",
            "SEQUENCE_LENGTH": 0,
            "STATUS": "ERROR"
        })

# =====================================================
# 5. CREATE SEQUENCE DATAFRAME
# =====================================================

sequence_df = pd.DataFrame(sequence_records)

successful_sequences = sequence_df[
    sequence_df["STATUS"] == "OK"
].copy()

if successful_sequences.empty:
    raise RuntimeError(
        "No protein sequences were successfully extracted."
    )

# =====================================================
# 6. IDENTIFY EXACT DUPLICATES
# =====================================================

successful_sequences["SEQUENCE_GROUP"] = (
    successful_sequences
    .groupby("SEQUENCE")
    .ngroup()
    .add(1)
)

successful_sequences["IS_UNIQUE"] = (
    ~successful_sequences.duplicated(
        subset=["SEQUENCE"],
        keep="first"
    )
)

successful_sequences["REPRESENTATIVE_ENTITY"] = (
    successful_sequences
    .groupby("SEQUENCE")["ENTITY_KEY"]
    .transform("first")
)

successful_sequences["DUPLICATE_COUNT"] = (
    successful_sequences
    .groupby("SEQUENCE")["SEQUENCE"]
    .transform("size")
)

successful_sequences["DUPLICATE_STATUS"] = (
    successful_sequences["IS_UNIQUE"]
    .map({
        True: "UNIQUE",
        False: "EXACT_DUPLICATE"
    })
)

# =====================================================
# 7. COMBINE SUCCESSFUL AND FAILED RECORDS
# =====================================================

failed_sequences = sequence_df[
    sequence_df["STATUS"] != "OK"
].copy()

failed_sequences["SEQUENCE_GROUP"] = ""
failed_sequences["IS_UNIQUE"] = False
failed_sequences["REPRESENTATIVE_ENTITY"] = ""
failed_sequences["DUPLICATE_COUNT"] = 0
failed_sequences["DUPLICATE_STATUS"] = "NOT_CHECKED"

final_df = pd.concat(
    [successful_sequences, failed_sequences],
    ignore_index=True
)

final_df.to_csv(
    OUTPUT_REPORT,
    index=False
)

# =====================================================
# 8. SAVE UNIQUE PROTEIN LIST
# =====================================================

unique_df = successful_sequences[
    successful_sequences["IS_UNIQUE"]
].copy()

unique_df.to_csv(
    UNIQUE_REPORT,
    index=False
)

# =====================================================
# 9. SAVE FASTA FILE
# =====================================================

with open(FASTA_PATH, "w") as fasta_file:

    for _, row in successful_sequences.iterrows():

        fasta_file.write(
            f">{row['ENTITY_KEY']}\n"
        )

        sequence = row["SEQUENCE"]

        for start in range(0, len(sequence), 60):
            fasta_file.write(
                sequence[start:start + 60] + "\n"
            )

# =====================================================
# 10. FINAL SUMMARY
# =====================================================

total_entities = len(sequence_df)
successful_count = len(successful_sequences)
failed_count = len(failed_sequences)

unique_count = len(unique_df)

duplicate_count = (
    successful_count - unique_count
)

print("\n======================================")
print("SEQUENCE DEDUPLICATION COMPLETED")
print("======================================")

print("Total validated entities:", total_entities)
print("Sequences extracted:", successful_count)
print("Sequence extraction failures:", failed_count)

print("\nUnique sequences:", unique_count)
print("Exact duplicate entries:", duplicate_count)

print("\nSequence report saved:")
print(OUTPUT_REPORT)

print("\nUnique protein report saved:")
print(UNIQUE_REPORT)

print("\nFASTA file saved:")
print(FASTA_PATH)

print("\n======================================")