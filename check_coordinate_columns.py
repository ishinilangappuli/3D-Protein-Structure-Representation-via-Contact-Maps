
import pandas as pd
from pathlib import Path

COORDINATE_DIR = Path(
    r"C:\Users\ishin\OneDrive\Desktop\Complex_Networks"
    r"\3D-Protein-Structure-Representation-via-Contact-Maps"
    r"\ecoli_dataset\coordinates"
)

files = list(COORDINATE_DIR.glob("*.csv"))

print("Number of coordinate CSV files:", len(files))

if not files:
    print("ERROR: No CSV files found!")
else:
    file = files[0]

    print("\nTesting file:", file.name)

    df = pd.read_csv(file)

    print("\nACTUAL COLUMN NAMES:")
    for column in df.columns:
        print(repr(column))

    print("\nFIRST 5 ROWS:")
    print(df.head().to_string())

    print("\nDATA TYPES:")
    print(df.dtypes)
