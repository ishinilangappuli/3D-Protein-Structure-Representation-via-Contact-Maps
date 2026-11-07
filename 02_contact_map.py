import pandas as pd

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist

# --------------------------------------------------
# 1. Load the protein structure
# --------------------------------------------------
PDB_FILE = Path(
    r"C:\Users\ishin\OneDrive\Desktop\Complex_Networks\1UBQ.pdb"
)

OUTPUT_DIR = Path(
    r"C:\Users\ishin\OneDrive\Desktop\Complex_Networks\results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

parser = PDBParser(QUIET=True)
structure = parser.get_structure("1UBQ", str(PDB_FILE))

model = next(structure.get_models())
chain = next(model.get_chains())

coordinates = []
residue_ids = []
residue_labels = []

for residue in chain.get_residues():
    if residue.id[0] != " ":
        continue

    if "CA" not in residue:
        continue

    coordinates.append(residue["CA"].coord)
    residue_ids.append(residue.id[1])
    residue_labels.append(
    f"{residue.id[1]}_{residue.resname}"
)

coordinates = np.asarray(coordinates, dtype=float)

# --------------------------------------------------
# 2. Calculate pairwise Euclidean distances
# --------------------------------------------------
distance_matrix = cdist(coordinates, coordinates, metric="euclidean")

# --------------------------------------------------
# 3. Generate the contact map using an 8 Å threshold
# --------------------------------------------------
THRESHOLD = 8.0

contact_map = (distance_matrix < THRESHOLD).astype(int)

# --------------------------------------------------
# 4. Exclude self-contacts from the contact map
# --------------------------------------------------
np.fill_diagonal(contact_map, 0)

# --------------------------------------------------
# 5. Save numerical results
# --------------------------------------------------
# Create labelled DataFrames
distance_df = pd.DataFrame(
    distance_matrix,
    index=residue_labels,
    columns=residue_labels
)

contact_df = pd.DataFrame(
    contact_map,
    index=residue_labels,
    columns=residue_labels
)

# Save labelled CSV files
distance_df.to_csv(
    OUTPUT_DIR / "1UBQ_distance_matrix_labelled.csv",
    index_label="Residue"
)

contact_df.to_csv(
    OUTPUT_DIR / "1UBQ_contact_map_8A_labelled.csv",
    index_label="Residue"
)

print("Labelled matrices saved successfully!")

# --------------------------------------------------
# 6. Display basic statistics
# --------------------------------------------------
n_residues = len(coordinates)
n_contacts = int(np.sum(contact_map) // 2)
possible_pairs = n_residues * (n_residues - 1) // 2

print(f"Protein: 1UBQ")
print(f"Chain: {chain.id}")
print(f"Number of residues: {n_residues}")
print(f"Distance matrix shape: {distance_matrix.shape}")
print(f"Contact map shape: {contact_map.shape}")
print(f"Threshold: {THRESHOLD} Å")
print(f"Number of unique contacts: {n_contacts}")
print(f"Possible residue pairs: {possible_pairs}")
print(f"Contact density: {n_contacts / possible_pairs:.4f}")

# --------------------------------------------------
# 7. Plot the contact map
# --------------------------------------------------
plt.figure(figsize=(7, 6))

plt.imshow(
    contact_map,
    cmap="Greys",
    origin="lower",
    interpolation="nearest"
)

plt.xlabel("Residue index")
plt.ylabel("Residue index")
plt.title("Ubiquitin (1UBQ) Contact Map — 8 Å")
plt.colorbar(label="Contact (1 = yes, 0 = no)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "1UBQ_contact_map_8A.png",
    dpi=300
)

plt.show()
