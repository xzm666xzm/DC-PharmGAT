# Docking Scripts

This directory keeps representative, target-agnostic docking utilities instead
of one script per PDB ID or receptor.

## Scripts

- `prepare_receptor.py`: extract the co-crystal ligand, clean the receptor,
  generate `receptor.pdbqt`, and write `docking_config.txt`.
- `extract_ligand.py`: extract only the co-crystal ligand from a PDB file.
- `dock_zinc.py`: dock a SMILES library against a prepared receptor with Smina.

## Examples

Prepare a receptor:

```bash
python prepare_receptor.py --pdb 3ERT.pdb --ligand-resname OHT --obabel /path/to/obabel --output-dir 3ert_prepared
```

Dock a SMILES library:

```bash
python dock_zinc.py \
  --smina /path/to/smina \
  --receptor 3ert_prepared/receptor.pdbqt \
  --smiles zinc_60k_smiles.smi \
  --center 30.0 10.0 20.0 \
  --size 18.0 20.0 22.0 \
  --output docking_results.csv
```

Use the center and size values from `docking_config.txt`, or provide a manually
curated docking box.
