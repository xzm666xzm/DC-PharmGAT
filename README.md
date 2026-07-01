# DC-PharmGAT

Clean code release for the manuscript:

**DC-PharmGAT: A Target-Conditioned Pharmacophore-Guided Dual-Channel Graph Neural Network for Interpretable Virtual Screening**

This repository provides the source code for model training, benchmark comparison, ablation studies, ZINC screening, ChEMBL external validation, docking-score label generation, and interpretability analysis.

## Repository layout

```text
src/
  Core model, molecular feature, utility, and pharmacophore extraction modules.
experiments/
  Training scripts for benchmark targets, scaffold splits, baselines, ablation, ZINC, and ChEMBL validation.
scripts/
  Data preprocessing, receptor/docking preparation, screening, and interpretability scripts.
data/
  Recommended location for raw datasets and processed splits.
checkpoints/
  Recommended location for trained model weights.
results/
  Recommended location for generated predictions, metrics, tables, and figures.
```

## Data and checkpoints

This repository keeps the source code under Git version control. The complete data and model reproducibility archive is publicly available on Zenodo:

- Version-specific DOI: https://doi.org/10.5281/zenodo.21109686
        
        
        
        
- All-versions DOI: https://doi.org/10.5281/zenodo.20669417
        
        
        
        
- Archive file: `DCGAT_v102_public_clean.zip`

The Zenodo archive contains the processed datasets, fixed train/validation/test splits, final trained model checkpoints, target pharmacophore profile files, model-selection metadata, result source files, manifest file, and reproducibility README required to reproduce the manuscript results and access the final models.

Raw datasets should be obtained from their original sources:

- DUD-E
- LIT-PCBA
- ZINC
- ChEMBL
- Protein Data Bank

After downloading `DCGAT_v102_public_clean.zip`, extract `data/` and `models/` from the archive and place them under the corresponding repository directories, or keep the extracted archive as a sibling reproducibility directory and update script paths accordingly.

## Typical usage

Run from the repository root and expose the core modules through `PYTHONPATH`:

```bash
set PYTHONPATH=src
python experiments/train.py
```

For Linux/macOS:

```bash
export PYTHONPATH=src
python experiments/train.py
```

Representative target-training workflows are kept under `experiments/target_training/`. They are parameterized by command-line arguments instead of being duplicated for each target. Data-processing and docking scripts are under `scripts/`.

## Citation

Citation information will be added after publication.
