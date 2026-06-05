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

Following common open-source practice, this repository keeps code under Git version control and uses external archives for large research artifacts.

Raw datasets should be obtained from their original sources:

- DUD-E
- LIT-PCBA
- ZINC
- ChEMBL
- Protein Data Bank

For publication and long-term reproducibility, processed data splits, trained checkpoints, and large generated results can be distributed through a versioned archive such as Zenodo, Figshare, OSF, institutional storage, Google Drive, or Hugging Face Hub. Once available, add the download link or DOI here and place the downloaded files under the corresponding directories above.

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
