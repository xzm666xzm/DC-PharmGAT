# Target Training

This directory keeps representative, target-agnostic training entry points.
The previous `train_<target>_complete.py` and `train_<target>_scaffold.py`
copies were consolidated so GitHub users can see one reusable workflow instead
of many target-specific duplicates.

## Scripts

- `train_target.py`: full configurable workflow.
- `train_complete.py`: thin wrapper for complete/random-split training.
- `train_scaffold.py`: thin wrapper for precomputed scaffold-split training.

## Examples

Complete DUD-E-style training:

```bash
python train_complete.py --target <target_name> --data-root DUD-E
```

Complete LIT-PCBA-style training:

```bash
python train_complete.py --target <target_name> --dataset lit-pcba --data-root LIT-PCBA
```

Scaffold-split training:

```bash
python train_scaffold.py --target <target_name> --data-root LIT-PCBA --target-folder <dataset_folder>
```

Use `--models 1-3` for a short smoke run, or `--dry-run` to inspect commands
without launching training.

During the hyperparameter search, `train_target.py` calls `train.py` with
`--skip_test_eval`, selects the checkpoint by validation AUC, and then runs a
single held-out test evaluation of the selected checkpoint with
`--final_test_only`.
