# Checkpoints

Use this directory for trained model weights and saved training states.

Recommended organization:

```text
checkpoints/
  dude/
  litpcba/
  zinc/
  chembl/
```

Final trained model checkpoints for the manuscript are publicly archived on Zenodo:

- Version-specific DOI: https://doi.org/10.5281/zenodo.21109686
        
        
- All-versions DOI: https://doi.org/10.5281/zenodo.20669417
        
        
- Archive file: `DCGAT_v102_public_clean.zip`

The archive includes target-level `best_stage1.pth` files, target pharmacophore profile files, selected-checkpoint metadata, and model-selection/result source files. Download and extract the archive, then copy or symlink the relevant `models/` target folders into this directory if local inference or reproduction scripts expect checkpoint files here.
