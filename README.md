# REASSEMBLE Minimal Observer Experiment

This repository hosts a minimal, reproducible Python scaffold for REASSEMBLE-based contact-state experiments. The current implementation covers:

- Milestone 1: project scaffold, configuration system, developer agreements, and runnable CLI entry points
- Milestone 2: raw file scanning, manifest generation, and trial-level indexing with segment metadata extraction
- Milestone 3: insert-segment extraction with unique insert IDs and low-level skill sequences
- Milestone 4: timestamp alignment onto a shared sample axis for insert segments
- Milestone 5: phase remapping and weak contact labels for aligned insert samples

The design intentionally prioritizes F/T and pose metadata first, keeps dataset field names configurable, and avoids assuming fixed REASSEMBLE internals when the schema is still uncertain.

## Project Layout

- `configs/`: YAML configuration for dataset paths and training placeholders
- `scripts/`: command-line entry points; each script supports `--help`
- `src/reassemble_minexp/io/`: raw HDF5 / JSON readers and segment parsing
- `src/reassemble_minexp/dataset/`: manifest and trial-index building logic
- `src/reassemble_minexp/utils/`: shared configuration and path utilities
- `tests/`: minimal regression coverage for configuration loading and Milestone 2 indexing
- `data/processed/`: generated CSV outputs from scanning and indexing
- `docs/`: lightweight notes when milestone changes need extra explanation

## Quick Start

1. Create a virtual environment and install dependencies.
2. Update `configs/dataset.yaml` so `dataset.raw_root` points at the extracted `data/` directory and `dataset.pose_root` points at the extracted `poses/` directory.
3. Run the scan step:

```bash
python3 scripts/00_scan_files.py --config configs/dataset.yaml
```

4. Build the trial index:

```bash
python3 scripts/01_build_index.py --config configs/dataset.yaml
```

5. Extract insert segments:

```bash
python3 scripts/02_extract_insert_segments.py --config configs/dataset.yaml
```

6. Align modalities onto a shared timeline:

```bash
python3 scripts/03_align_modalities.py --config configs/dataset.yaml
```

7. Generate phase labels:

```bash
python3 scripts/04_make_phase_labels.py --config configs/dataset.yaml
```

8. Generate weak contact labels:

```bash
python3 scripts/05_make_contact_labels.py --config configs/dataset.yaml
```

If you want data-driven threshold suggestions from the generated `ft_value_norm` distribution, run:

```bash
python3 scripts/05_make_contact_labels.py --config configs/dataset.yaml --suggest-thresholds
```

Generated files default to `data/processed/file_manifest.csv`, `data/processed/trial_index.csv`, `data/processed/segment_index.csv`, `data/processed/insert_index.csv`, `data/processed/aligned_samples.csv`, `data/processed/phase_labels.csv`, and `data/processed/contact_labels.csv`.

For the official REASSEMBLE layout, HDF5 trials live in a `data/` directory while the matching `*_poses.json` files live in a separate `poses/` directory. Files are matched by their shared timestamp filename stem.

In `trial_index.csv`, `trial_success_all_actions` is interpreted as demonstration-level success inferred from the high-level action segments: all non-`No action.` segments must be marked successful. `trial_success_last_action` keeps the success flag of the last non-`No action.` high-level segment. Because REASSEMBLE demonstrations can span multiple objects, `object_names` is filled from the ordered unique objects mentioned in high-level action text when no canonical trial-level object label is present.

`aligned_samples.csv` stores time-aligned sample indices rather than copied sensor data. Pose and F/T alignment use interpolation index pairs plus weights, while RGB alignment uses nearest-frame indices for each configured camera stream.

`phase_labels.csv` maps each aligned sample to the active official low-level skill and a configurable phase label. `contact_labels.csv` uses interpolated F/T magnitude thresholds and low-level skill groups to assign weak labels from the set `free`, `touch`, `search_contact`, `insertion_contact`, and `jam_or_abnormal`.

## Milestone Order

1. Milestone 1: scaffold and config system
2. Milestone 2: file scanning and trial index
3. Milestone 3: insert segment extraction
4. Milestone 4: time alignment
5. Milestone 5: labeling
6. Milestone 6+: windows, training, and evaluation

Unless blocked, development should follow this order.

## Validation

- Script interface check: `python3 scripts/00_scan_files.py --help`
- Script interface check: `python3 scripts/01_build_index.py --help`
- Script interface check: `python3 scripts/02_extract_insert_segments.py --help`
- Script interface check: `python3 scripts/03_align_modalities.py --help`
- Script interface check: `python3 scripts/04_make_phase_labels.py --help`
- Script interface check: `python3 scripts/05_make_contact_labels.py --help`
- Regression tests: `python3 -m pytest tests/test_config.py tests/test_trial_index.py tests/test_insert_index.py tests/test_timestamp_aligner.py tests/test_phase_mapper.py tests/test_contact_rule_labeler.py`

## TODO

- Confirm canonical HDF5 dataset keys for F/T timestamps and modality availability flags.
- Replace the current segment-text fallback for `object_names` with a direct field if a canonical trial-level object label exists in released HDF5 files.
- Decide whether future milestones should export one segment CSV or keep segment rows nested in the trial index.
- Add richer config validation once the full training pipeline is present.
