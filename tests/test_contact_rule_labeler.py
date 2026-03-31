import csv
from pathlib import Path

import h5py

from reassemble_minexp.labels.contact_rule_labeler import classify_contact_label, run_contact_label_pipeline
from reassemble_minexp.utils.config import load_config


def test_classify_contact_label_covers_required_classes(tmp_path: Path) -> None:
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

    assert classify_contact_label("Approach", 0.5, config)[0] == "free"
    assert classify_contact_label("Approach", 2.0, config)[0] == "search_contact"
    assert classify_contact_label("Push", 2.0, config)[0] == "insertion_contact"
    assert classify_contact_label("Lift", 2.0, config)[0] == "touch"
    assert classify_contact_label("Push", 6.0, config)[0] == "jam_or_abnormal"


def test_run_contact_label_pipeline_writes_contact_labels(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    h5_path = raw_root / "trial_001.h5"
    with h5py.File(h5_path, "w") as handle:
        robot_state = handle.create_group("robot_state")
        robot_state.create_dataset("measured_force", data=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 6.0]])

    aligned_path = processed_root / "aligned_samples.csv"
    aligned_path.write_text(
        "\n".join(
            [
                "sample_id,insert_id,trial_id,sample_index,aligned_time,pose_source_key,pose_index_0,pose_index_1,pose_weight_0,pose_weight_1,ft_source_key,ft_index_0,ft_index_1,ft_weight_0,ft_weight_1,rgb_hama1_index,h5_path,pose_path",
                f"trial_001::insert::1::sample::00000,trial_001::insert::1,trial_001,0,0.0,pose,0,0,1.0,0.0,measured_force,0,0,1.0,0.0,0,{h5_path.as_posix()},/tmp/trial_001_poses.json",
                f"trial_001::insert::1::sample::00001,trial_001::insert::1,trial_001,1,0.5,pose,0,0,1.0,0.0,measured_force,1,1,1.0,0.0,1,{h5_path.as_posix()},/tmp/trial_001_poses.json",
                f"trial_001::insert::1::sample::00002,trial_001::insert::1,trial_001,2,1.0,pose,0,0,1.0,0.0,measured_force,2,2,1.0,0.0,2,{h5_path.as_posix()},/tmp/trial_001_poses.json",
            ]
        ),
        encoding="utf-8",
    )

    phase_path = processed_root / "phase_labels.csv"
    phase_path.write_text(
        "\n".join(
            [
                "sample_id,insert_id,trial_id,aligned_time,active_low_level_segment_id,active_low_level_skill,y_phase",
                "trial_001::insert::1::sample::00000,trial_001::insert::1,trial_001,0.0,1::0,Approach,search",
                "trial_001::insert::1::sample::00001,trial_001::insert::1,trial_001,0.5,1::1,Push,insertion",
                "trial_001::insert::1::sample::00002,trial_001::insert::1,trial_001,1.0,1::1,Push,insertion",
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        f"""
dataset:
  raw_root: {raw_root.as_posix()}
  pose_root: {raw_root.as_posix()}
  processed_root: {processed_root.as_posix()}
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

    output_path = run_contact_label_pipeline(load_config(config_path))
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["y_contact"] for row in rows] == ["free", "insertion_contact", "jam_or_abnormal"]
