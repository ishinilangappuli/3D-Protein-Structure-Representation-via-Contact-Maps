{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7c050677",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Protein file: C:\\Users\\ishin\\OneDrive\\Desktop\\Complex_Networks\\1UBQ.pdb\n",
      "Chain selected: A\n",
      "Number of C-alpha atoms: 76\n",
      "Coordinate matrix shape: (76, 3)\n",
      "\n",
      "First five coordinates (Å):\n",
      "[[26.26600075 25.41300011  2.84200001]\n",
      " [26.85000038 29.02099991  3.898     ]\n",
      " [26.23500061 30.05800056  7.49700022]\n",
      " [26.77199936 33.43600082  9.19699955]\n",
      " [28.60499954 33.96500015 12.50300026]]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "from pathlib import Path\n",
    "\n",
    "import numpy as np\n",
    "from Bio.PDB import PDBParser\n",
    "\n",
    "# Project paths\n",
    "PDB_FILE = Path(r\"C:\\Users\\ishin\\OneDrive\\Desktop\\Complex_Networks\\1UBQ.pdb\")\n",
    "\n",
    "# Read the PDB structure\n",
    "parser = PDBParser(QUIET=True)\n",
    "structure = parser.get_structure(\"protein\", str(PDB_FILE))\n",
    "\n",
    "# Use the first model and first chain for this initial test\n",
    "model = next(structure.get_models())\n",
    "chain = next(model.get_chains())\n",
    "\n",
    "coordinates = []\n",
    "residue_ids = []\n",
    "\n",
    "for residue in chain.get_residues():\n",
    "    # Keep standard amino-acid residues only\n",
    "    if residue.id[0] != \" \":\n",
    "        continue\n",
    "\n",
    "    # Each included residue must have a C-alpha atom\n",
    "    if \"CA\" not in residue:\n",
    "        continue\n",
    "\n",
    "    coordinates.append(residue[\"CA\"].coord)\n",
    "    residue_ids.append(residue.id[1])\n",
    "\n",
    "coordinates = np.asarray(coordinates, dtype=float)\n",
    "\n",
    "print(f\"Protein file: {PDB_FILE}\")\n",
    "print(f\"Chain selected: {chain.id}\")\n",
    "print(f\"Number of C-alpha atoms: {len(coordinates)}\")\n",
    "print(f\"Coordinate matrix shape: {coordinates.shape}\")\n",
    "print(\"\\nFirst five coordinates (Å):\")\n",
    "print(coordinates[:5])\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
