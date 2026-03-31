import csv
from pathlib import Path

from reassemble_minexp.dataset.insert_index import extract_insert_segments, run_insert_extraction
from reassemble_minexp.utils.config import load_config


def test_extract_insert_segments_keeps_only_insert_high_level_rows() -> None:
    trial_rows = [
        {
            "trial_id": "trial_001",
            "object_names": "Ethernet",
            "trial_success_all_actions": "True",
            "trial_success_last_action": "True",
            "h5_path": "/tmp/trial_001.h5",
            "pose_path": "/tmp/trial_001_poses.json",
        }
    ]
    segment_rows = [
        {
            "trial_id": "trial_001",
            "segment_level": "high",
            "segment_id": "0",
            "parent_segment_id": "",
            "segment_name": "Pick Ethernet.",
            "start_time": "0.0",
            "end_time": "1.0",
        },
        {
            "trial_id": "trial_001",
            "segment_level": "high",
            "segment_id": "1",
            "parent_segment_id": "",
            "segment_name": "Insert Ethernet.",
            "start_time": "1.0",
            "end_time": "2.0",
        },
        {
            "trial_id": "trial_001",
            "segment_level": "low",
            "segment_id": "1::0",
            "parent_segment_id": "1",
            "segment_name": "Approach",
            "start_time": "1.0",
            "end_time": "1.5",
        },
        {
            "trial_id": "trial_001",
            "segment_level": "low",
            "segment_id": "1::1",
            "parent_segment_id": "1",
            "segment_name": "Align",
            "start_time": "1.5",
            "end_time": "2.0",
        },
    ]

    rows = extract_insert_segments(trial_rows, segment_rows)

    assert len(rows) == 1
    assert rows[0]["insert_id"] == "trial_001::insert::1"
    assert rows[0]["object_name"] == "Ethernet"
    assert rows[0]["num_low_level_segments"] == 2
    assert rows[0]["low_level_sequence"] == "Approach|Align"


def test_run_insert_extraction_writes_insert_index(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()

    trial_index_path = processed_root / "trial_index.csv"
    trial_index_path.write_text(
        "\n".join(
            [
                "trial_id,h5_path,pose_path,object_names,trial_success_all_actions,trial_success_last_action,has_ft,has_pose,has_rgb,pose_start_time,pose_end_time,segment_start_time,segment_end_time,num_high_level_segments,num_low_level_segments,high_level_sequence,low_level_sequence",
                "trial_001,/tmp/trial_001.h5,/tmp/trial_001_poses.json,Ethernet,True,True,True,True,True,0.0,2.0,0.0,2.0,2,2,Pick Ethernet.|Insert Ethernet.,Approach|Align",
            ]
        ),
        encoding="utf-8",
    )

    segment_index_path = processed_root / "segment_index.csv"
    segment_index_path.write_text(
        "\n".join(
            [
                "trial_id,segment_level,segment_id,parent_segment_id,segment_name,start_time,end_time",
                "trial_001,high,0,,Pick Ethernet.,0.0,1.0",
                "trial_001,high,1,,Insert Ethernet.,1.0,2.0",
                "trial_001,low,1::0,1,Approach,1.0,1.5",
                "trial_001,low,1::1,1,Align,1.5,2.0",
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        f"""
dataset:
  raw_root: {tmp_path.as_posix()}
  pose_root: {tmp_path.as_posix()}
  processed_root: {processed_root.as_posix()}
  h5_glob: "*.h5"
  pose_suffix: "_poses.json"
  manifest_name: file_manifest.csv
  trial_index_name: trial_index.csv
  segment_index_name: segment_index.csv
  insert_index_name: insert_index.csv
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

    output_path = run_insert_extraction(load_config(config_path))
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["insert_id"] == "trial_001::insert::1"
    assert rows[0]["trial_object_names"] == "Ethernet"
    assert rows[0]["low_level_sequence"] == "Approach|Align"
