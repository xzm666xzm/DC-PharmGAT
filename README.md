# DC-PharmGAT

Source code accompanying the manuscript:

**DC-PharmGAT: A Target-Conditioned, Pharmacophore-Guided Dual-Channel Graph Neural Network for Interpretable Virtual Screening**

This repository contains the model implementation, configurable training workflows, molecular-feature and pharmacophore extraction, data-preprocessing utilities, and molecular-docking scripts.

## Repository layout

```text
src/
  Core model, molecular features, graph featurization, and pharmacophore extraction.
experiments/
  Training and data-inspection code.
experiments/target_training/
  Configurable complete/random-split and scaffold-split training entry points.
scripts/data_preprocessing/
  Dataset preparation, curation, and split-verification code.
scripts/docking/
  Receptor preparation, ligand extraction, and Smina docking code.
requirements.txt
  Python dependencies.
```

## Installation

Install the Python dependencies in an environment compatible with the required RDKit, PyTorch, and PyTorch Geometric packages:

```bash
python -m pip install -r requirements.txt
```

Docking utilities additionally require the relevant Open Babel and Smina executables. See [scripts/docking/README.md](scripts/docking/README.md) for their command-line arguments.

## Data inputs and local outputs

Prepare input datasets separately using the original DUD-E, LIT-PCBA, ZINC, ChEMBL, and Protein Data Bank sources, together with the study's supplementary materials where applicable. Supply the local input directory with `--data-root`.

Training produces model checkpoints, selection metadata, and evaluation outputs locally. The repository tracks source code, dependency information, and usage documentation. Generated data, model binaries, caches, logs, and result files are excluded by `.gitignore`.

## Training

Run commands from the repository root and expose the core modules through `PYTHONPATH`.

PowerShell:

```powershell
$env:PYTHONPATH = "src"
```

Linux/macOS:

```bash
export PYTHONPATH=src
```

With the target dataset prepared, preview a complete/random-split training run with count-aware TPP conditioning:

```bash
python experiments/target_training/train_complete.py --target TARGET --data-root DUD-E --enable-tpp-conditioning --tpp-input-mode binary_counts --dry-run
```

Replace `TARGET` and the dataset directory with the desired local inputs. Omit `--dry-run` to launch training. For complete/random and scaffold workflows, data formats, and model-selection options, see [experiments/target_training/README.md](experiments/target_training/README.md).

## Citation

Citation information will be added after publication.
