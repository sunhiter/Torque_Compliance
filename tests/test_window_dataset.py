import csv
import json
from pathlib import Path

import h5py

from reassemble_minexp.dataset.window_dataset import run_window_export
from reassemble_minexp.utils.config import load_config


def test_run_window_export_writes_windows_and_splits(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    h5_path = raw_root / "trial_001.h5"
    with h5py.File(h5_path, "w") as handle:
        robot_state = handle.create_group("robot_state")
        robot_state.create_dataset("pose", data=[[0.0] * 7, [1.0] * 7, [2.0] * 7, [3.0] * 7])
        robot_state.create_dataset("measured_force", data=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        segments = handle.create_group("segments_info")
        insert_segment = segments.create_group("1")
        insert_segment.create_dataset("success", data=True)

    (processed_root / "insert_index.csv").write_text(
        "\n".join(
            [
                "insert_id,trial_id,high_segment_id,segment_name,object_name,start_time,end_time,num_low_level_segments,low_level_sequence,trial_object_names,trial_success_all_actions,trial_success_last_action,h5_path,pose_path",
                f"trial_001::insert::1,trial_001,1,Insert Ethernet.,Ethernet,0.0,0.15,2,Approach|Push,Ethernet,True,True,{h5_path.as_posix()},/tmp/trial_001_poses.json",
            ]
        ),
        encoding="utf-8",
    )
    (processed_root / "aligned_samples.csv").write_text(
        "\n".join(
            [
                "sample_id,insert_id,trial_id,sample_index,aligned_time,pose_source_key,pose_index_0,pose_index_1,pose_weight_0,pose_weight_1,ft_source_key,ft_index_0,ft_index_1,ft_weight_0,ft_weight_1,rgb_hama1_index,h5_path,pose_path",
                f"trial_001::insert::1::sample::00000,trial_001::insert::1,trial_001,0,0.0,pose,0,0,1.0,0.0,measured_force,0,0,1.0,0.0,10,{h5_path.as_posix()},/tmp/trial_001_poses.json",
                f"trial_001::insert::1::sample::00001,trial_001::insert::1,trial_001,1,0.05,pose,1,1,1.0,0.0,measured_force,1,1,1.0,0.0,11,{h5_path.as_posix()},/tmp/trial_001_poses.json",
                f"trial_001::insert::1::sample::00002,trial_001::insert::1,trial_001,2,0.10,pose,2,2,1.0,0.0,measured_force,2,2,1.0,0.0,12,{h5_path.as_posix()},/tmp/trial_001_poses.json",
                f"trial_001::insert::1::sample::00003,trial_001::insert::1,trial_001,3,0.15,pose,3,3,1.0,0.0,measured_force,3,3,1.0,0.0,13,{h5_path.as_posix()},/tmp/trial_001_poses.json",
            ]
        ),
        encoding="utf-8",
    )
    (processed_root / "phase_labels.csv").write_text(
        "\n".join(
            [
                "sample_id,insert_id,trial_id,aligned_time,active_low_level_segment_id,active_low_level_skill,y_phase",
                "trial_001::insert::1::sample::00000,trial_001::insert::1,trial_001,0.0,1::0,Approach,search",
                "trial_001::insert::1::sample::00001,trial_001::insert::1,trial_001,0.05,1::0,Approach,search",
                "trial_001::insert::1::sample::00002,trial_001::insert::1,trial_001,0.10,1::1,Push,insertion",
                "trial_001::insert::1::sample::00003,trial_001::insert::1,trial_001,0.15,1::1,Push,insertion",
            ]
        ),
        encoding="utf-8",
    )
    (processed_root / "contact_labels.csv").write_text(
        "\n".join(
            [
                "sample_id,insert_id,trial_id,aligned_time,active_low_level_skill,ft_source_key,ft_value_norm,y_contact,rule_reason",
                "trial_001::insert::1::sample::00000,trial_001::insert::1,trial_001,0.0,Approach,measured_force,0.0,free,ft_norm<touch_threshold",
                "trial_001::insert::1::sample::00001,trial_001::insert::1,trial_001,0.05,Approach,measured_force,1.0,free,ft_norm<touch_threshold",
                "trial_001::insert::1::sample::00002,trial_001::insert::1,trial_001,0.10,Push,measured_force,2.0,insertion_contact,contact_during_insertion_skill",
                "trial_001::insert::1::sample::00003,trial_001::insert::1,trial_001,0.15,Push,measured_force,3.0,insertion_contact,contact_during_insertion_skill",
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
    search_skills: [Approach]
    insertion_skills: [Push]
windows:
  history_length: 2
  history_step: 1
  export_stride: 1
  window_index_name: windows.csv
  structure_tokens_path: ""
  split_source_path: ""
  split_ratios:
    train: 0.7
    val: 0.2
    test: 0.1
  split_output_names:
    train: train_windows.csv
    val: val_windows.csv
    test: test_windows.csv
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

    output_path = run_window_export(load_config(config_path))
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["y_phase"] == "search"
    assert rows[1]["y_phase"] == "insertion"
    assert rows[0]["y_contact"] == "free"
    assert rows[1]["y_contact"] == "insertion_contact"
    assert rows[0]["y_success"] == "True"
    assert json.loads(rows[0]["rgb_hama1_history_json"]) == ["10", "11"]
    assert json.loads(rows[0]["y_next_delta_json"]) == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    assert (processed_root / "train_windows.csv").exists()
    assert (processed_root / "val_windows.csv").exists()
    assert (processed_root / "test_windows.csv").exists()
