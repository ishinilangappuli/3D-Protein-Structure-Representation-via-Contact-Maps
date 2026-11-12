
import os
import time
import random
import requests
import pandas as pd

# ==================================================
# SETTINGS
# ==================================================

TAXONOMY_ID = 562
NUMBER_OF_PROTEINS = 60
RANDOM_SEED = 42

BASE_DIR = "ecoli_dataset"
CIF_DIR = os.path.join(BASE_DIR, "cif_files")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

os.makedirs(CIF_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "EcoliProteinResearch/1.0"
})

# ==================================================
# STEP 1: SEARCH RCSB
# ==================================================

search_url = (
    "https://search.rcsb.org/rcsbsearch/v2/query"
)


query = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entity_source_organism.ncbi_scientific_name",
                    "operator": "exact_match",
                    "value": "Escherichia coli"
                }
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match",
                    "value": "Protein"
                }
            }
        ]
    },
    "return_type": "polymer_entity",
    "request_options": {
        "paginate": {
            "start": 0,
            "rows": 10000
        },
        "results_content_type": ["experimental"]
    }
}

print("Searching RCSB for E. coli proteins...")

response = session.post(
    search_url,
    json=query,
    timeout=90
)

if response.status_code != 200:
    print("HTTP Status:", response.status_code)
    print("RCSB Error Message:")
    print(response.text)
    raise SystemExit("RCSB search request failed.")

search_data = response.json()

entity_ids = sorted(set(
    item["identifier"]
    for item in search_data.get("result_set", [])
))

print("Protein entities found:", len(entity_ids))

if len(entity_ids) < NUMBER_OF_PROTEINS:
    raise RuntimeError(
        "Not enough protein entities were found. "
        "Check the RCSB search results."
    )

# ==================================================
# STEP 2: RANDOM SELECTION
# ==================================================

rng = random.Random(RANDOM_SEED)

selected_entities = rng.sample(
    entity_ids,
    NUMBER_OF_PROTEINS
)

print("\nSelected protein entities:", len(selected_entities))

# Each entity ID looks like PDB_ID_ENTITY_ID
selected_records = []

for entity_key in selected_entities:

    pdb_id, entity_id = entity_key.split("_", 1)

    selected_records.append({
        "PDB_ID": pdb_id,
        "ENTITY_ID": entity_id,
        "ENTITY_KEY": entity_key
    })

selected_df = pd.DataFrame(selected_records)

# ==================================================
# STEP 3: DOWNLOAD UNIQUE CIF FILES
# ==================================================

unique_pdb_ids = sorted(
    selected_df["PDB_ID"].unique()
)

print("\nUnique PDB structures to download:",
      len(unique_pdb_ids))

download_status = {}

for index, pdb_id in enumerate(unique_pdb_ids, start=1):

    file_path = os.path.join(
        CIF_DIR,
        f"{pdb_id}.cif"
    )

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print(f"{index}: {pdb_id} already downloaded")
        download_status[pdb_id] = "Already downloaded"
        continue

    download_url = (
        f"https://files.rcsb.org/download/{pdb_id}.cif"
    )

    print(
        f"\nDownloading {index}/{len(unique_pdb_ids)}: "
        f"{pdb_id}"
    )

    success = False

    for attempt in range(3):

        try:
            file_response = session.get(
                download_url,
                timeout=90
            )

            file_response.raise_for_status()

            content = file_response.content

            if b"_atom_site." not in content:
                raise ValueError(
                    "Downloaded file does not contain "
                    "expected atom coordinate data."
                )

            with open(file_path, "wb") as file:
                file.write(content)

            success = True
            break

        except Exception as error:

            print(
                f"Attempt {attempt + 1} failed: {error}"
            )

            time.sleep(2 * (attempt + 1))

    if success:
        print("Downloaded:", pdb_id)
        download_status[pdb_id] = "Downloaded"
    else:
        print("Failed:", pdb_id)
        download_status[pdb_id] = "Failed"

    time.sleep(0.3)

# ==================================================
# STEP 4: CREATE METADATA REPORT
# ==================================================

selected_df["FILE"] = selected_df["PDB_ID"].apply(
    lambda x: f"{x}.cif"
)

selected_df["STATUS"] = selected_df["PDB_ID"].map(
    download_status
)

report_path = os.path.join(
    METADATA_DIR,
    "ecoli_protein_download_report.csv"
)

selected_df.to_csv(
    report_path,
    index=False
)

# ==================================================
# FINAL SUMMARY
# ==================================================

successful = selected_df["STATUS"].isin(
    ["Downloaded", "Already downloaded"]
).sum()

failed = NUMBER_OF_PROTEINS - successful

print("\n====================================")
print("E. COLI DATASET DOWNLOAD COMPLETED")
print("====================================")

print("Selected protein entities:", NUMBER_OF_PROTEINS)
print("Unique structure files:", len(unique_pdb_ids))
print("Successful entities:", successful)
print("Failed entities:", failed)

print("\nCIF files saved in:")
print(CIF_DIR)

print("\nMetadata report saved in:")
print(report_path)
