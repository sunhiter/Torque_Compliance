import csv
from pathlib import Path

import h5py

from reassemble_minexp.io.timestamp_aligner import (
    align_linear_indices,
    align_nearest_indices,
    build_uniform_timeline,
    run_alignment_pipeline,
)
from reassemble_minexp.utils.config import load_config


def test_build_uniform_timeline_includes_endpoints() -> None:
    timeline = build_uniform_timeline(0.0, 1.0, 2.0)
    assert timeline == [0.0, 0.5, 1.0]


def test_alignment_helpers_cover_linear_and_nearest_modes() -> None:
    source_times = [0.0, 1.0, 2.0]
    target_times = [0.25, 1.75]

    nearest = align_nearest_indices(source_times, target_times)
    linear = align_linear_indices(source_times, target_times)

    assert nearest == [0, 2]
    assert linear[0] == {"index_0": 0, "index_1": 1, "weight_0": 0.75, "weight_1": 0.25}
    assert linear[1] == {"index_0": 1, "index_1": 2, "weight_0": 0.25, "weight_1": 0.75}


def test_run_alignment_pipeline_writes_aligned_sample_index(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    h5_path = raw_root / "trial_001.h5"
    with h5py.File(h5_path, "w") as handle:
        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset("pose", data=[0.0, 0.5, 1.0])
        timestamps.create_dataset("measured_force", data=[0.0, 0.5, 1.0])
        timestamps.create_dataset("hama1", data=[0.0, 0.4, 0.8, 1.2])

    insert_index_path = processed_root / "insert_index.csv"
    insert_index_path.write_text(
        "\n".join(
            [
                "insert_id,trial_id,high_segment_id,segment_name,object_name,start_time,end_time,num_low_level_segments,low_level_sequence,trial_object_names,trial_success_all_actions,trial_success_last_action,h5_path,pose_path",
                f"trial_001::insert::1,trial_001,1,Insert Ethernet.,Ethernet,0.0,1.0,2,Approach|Align,Ethernet,True,True,{h5_path.as_posix()},/tmp/trial_001_poses.json",
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
  target_frequency_hz: 2.0
  pose_mode: linear
  ft_mode: linear
  rgb_mode: nearest
  pose_source_keys: [pose]
  ft_source_keys: [measured_force]
  rgb_source_keys: [hama1]
  aligned_index_name: aligned_samples.csv
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

    output_path = run_alignment_pipeline(load_config(config_path))
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert rows[0]["sample_id"] == "trial_001::insert::1::sample::00000"
    assert rows[1]["pose_index_0"] == "1"
    assert rows[1]["pose_index_1"] == "1"
    assert rows[2]["rgb_hama1_index"] == "2"
