from pathlib import Path

from reassemble_minexp.labels.phase_mapper import map_phase_labels
from reassemble_minexp.utils.config import load_config


def test_map_phase_labels_uses_active_low_level_skill_remap(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        """
dataset:
  raw_root: data/raw
  pose_root: data/poses
  processed_root: data/processed
  h5_glob: "*.h5"
  pose_suffix: "_poses.json"
  manifest_name: file_manifest.csv
  trial_index_name: trial_index.csv
  segment_index_name: segment_index.csv
  insert_index_name: insert_index.csv
alignment:
  target_frequency_hz: 20.0
  pose_mode: linear
  ft_mode: linear
  rgb_mode: nearest
  pose_source_keys: [pose]
  ft_source_keys: [measured_force]
  rgb_source_keys: [hama1]
  aligned_index_name: aligned_samples.csv
labels:
  phase_index_name: phase_labels.csv
  contact_index_name: contact_labels.csv
  phase_map:
    Approach: search
    Align: search
    Push: insertion
  contact:
    touch_ft_norm_threshold: 1.0
    jam_ft_norm_threshold: 5.0
    search_skills: [Approach, Align, Nudge]
    insertion_skills: [Push, Twist]
schema:
  segments_key: segments_info
  result_keys: [success]
  object_keys: [object_name]
  timestamp_keys:
    start: [start]
    end: [end]
  modality_keys:
    ft: [measured_force]
    pose: [pose]
    rgb: [hama1]
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    aligned_rows = [
        {
            "sample_id": "trial_001::insert::1::sample::00000",
            "insert_id": "trial_001::insert::1",
            "trial_id": "trial_001",
            "aligned_time": "0.25",
        },
        {
            "sample_id": "trial_001::insert::1::sample::00001",
            "insert_id": "trial_001::insert::1",
            "trial_id": "trial_001",
            "aligned_time": "0.75",
        },
    ]
    segment_rows = [
        {
            "trial_id": "trial_001",
            "segment_level": "low",
            "segment_id": "1::0",
            "parent_segment_id": "1",
            "segment_name": "Approach",
            "start_time": "0.0",
            "end_time": "0.5",
        },
        {
            "trial_id": "trial_001",
            "segment_level": "low",
            "segment_id": "1::1",
            "parent_segment_id": "1",
            "segment_name": "Push",
            "start_time": "0.5",
            "end_time": "1.0",
        },
    ]

    rows = map_phase_labels(aligned_rows, segment_rows, config)

    assert rows[0]["active_low_level_skill"] == "Approach"
    assert rows[0]["y_phase"] == "search"
    assert rows[1]["active_low_level_skill"] == "Push"
    assert rows[1]["y_phase"] == "insertion"
