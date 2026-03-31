import csv
import json
from pathlib import Path

import h5py

from reassemble_minexp.dataset.trial_index import build_trial_index, run_scan_pipeline, scan_raw_files
from reassemble_minexp.io.pose_json_reader import read_pose_metadata
from reassemble_minexp.io.segment_parser import parse_segment_rows
from reassemble_minexp.utils.config import load_config


def _write_dataset_config(config_path: Path, raw_root: Path, pose_root: Path, processed_root: Path) -> None:
    config_path.write_text(
        f"""
dataset:
  raw_root: {raw_root.as_posix()}
  pose_root: {pose_root.as_posix()}
  processed_root: {processed_root.as_posix()}
  h5_glob: "*.h5"
  pose_suffix: "_poses.json"
  manifest_name: file_manifest.csv
  trial_index_name: trial_index.csv
  segment_index_name: segment_index.csv
schema:
  segments_key: segments_info
  result_keys: [success]
  object_keys: [object_name]
  timestamp_keys:
    start: [start_time, start]
    end: [end_time, end]
  modality_keys:
    ft: [measured_force, measured_torque]
    pose: [pose, robot_state/pose]
    rgb: [hama1, hand]
""",
        encoding="utf-8",
    )


def test_scan_and_build_trial_index_with_separate_pose_root(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "data"
    raw_root.mkdir(parents=True)
    pose_root = tmp_path / "raw" / "poses"
    pose_root.mkdir(parents=True)
    processed_root = tmp_path / "processed"

    h5_path = raw_root / "trial_001.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["object_name"] = "peg"
        handle.attrs["success"] = True
        handle.create_dataset("hama1", data=[1])

        robot_state = handle.create_group("robot_state")
        robot_state.create_dataset("pose", data=[[0.0] * 7, [1.0] * 7])
        robot_state.create_dataset("measured_force", data=[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])
        robot_state.create_dataset("measured_torque", data=[[0.0] * 3, [0.1] * 3])

        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset("pose", data=[0.1, 0.6, 1.1])
        timestamps.create_dataset("measured_force", data=[0.1, 0.6])

        segments = handle.create_group("segments_info")
        high = segments.create_group("0")
        high.create_dataset("start", data=0.1)
        high.create_dataset("end", data=1.0)
        high.create_dataset("text", data="insert")
        low_level = high.create_group("Low_level")
        low = low_level.create_group("0")
        low.create_dataset("start", data=0.1)
        low.create_dataset("end", data=0.5)
        low.create_dataset("text", data="approach")

    pose_path = pose_root / "trial_001_poses.json"
    pose_path.write_text(
        json.dumps(
            {
                "Hama1": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                "NIST_Board1": [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "dataset.yaml"
    _write_dataset_config(config_path, raw_root, pose_root, processed_root)
    config = load_config(config_path)

    manifest_rows = scan_raw_files(config)
    assert manifest_rows == [
        {
            "trial_id": "trial_001",
            "h5_path": str(h5_path),
            "pose_path": str(pose_path),
            "has_h5": True,
            "has_pose_json": True,
        }
    ]

    trial_rows, segment_rows = build_trial_index(config, manifest_rows)
    assert len(trial_rows) == 1
    assert trial_rows[0]["object_name"] == "peg"
    assert trial_rows[0]["success"] is True
    assert trial_rows[0]["has_ft"] is True
    assert trial_rows[0]["has_pose"] is True
    assert trial_rows[0]["has_rgb"] is True
    assert trial_rows[0]["pose_start_time"] == 0.1
    assert trial_rows[0]["pose_end_time"] == 1.1
    assert trial_rows[0]["high_level_sequence"] == "insert"
    assert trial_rows[0]["low_level_sequence"] == "approach"
    assert len(segment_rows) == 2

    artifacts = run_scan_pipeline(config)
    assert artifacts.manifest_path.exists()
    assert artifacts.trial_index_path.exists()
    assert artifacts.segment_index_path.exists()

    with artifacts.trial_index_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["trial_id"] == "trial_001"
    assert rows[0]["pose_path"] == str(pose_path)


def test_parse_segment_rows_supports_documented_group_payload(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    _write_dataset_config(config_path, tmp_path / "data", tmp_path / "poses", tmp_path / "processed")
    config = load_config(config_path)

    segments_info = {
        "0": {
            "start": 0.0,
            "end": 1.0,
            "text": "insert",
            "Low_level": {
                "0": {
                    "start": 0.0,
                    "end": 0.4,
                    "text": "approach",
                }
            },
        }
    }

    rows = parse_segment_rows(segments_info, "trial_group", config.schema)

    assert [row["segment_level"] for row in rows] == ["high", "low"]
    assert rows[0]["segment_name"] == "insert"
    assert rows[0]["start_time"] == 0.0
    assert rows[1]["segment_name"] == "approach"
    assert rows[1]["end_time"] == 0.4


def test_parse_segment_rows_reads_group_like_attrs(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    _write_dataset_config(config_path, tmp_path / "data", tmp_path / "poses", tmp_path / "processed")
    config = load_config(config_path)

    segments_info = {
        "segment_0": {
            "attrs": {
                "id": "high_0",
                "text": "insert",
                "start": 0.0,
                "end": 1.0,
            },
            "Low_level": {
                "child_0": {
                    "attrs": {
                        "id": "low_0",
                        "text": "approach",
                        "start": 0.0,
                        "end": 0.4,
                    }
                }
            },
        }
    }

    rows = parse_segment_rows(segments_info, "trial_group_attrs", config.schema)

    assert [row["segment_level"] for row in rows] == ["high", "low"]
    assert rows[0]["segment_name"] == "insert"
    assert rows[0]["start_time"] == 0.0
    assert rows[1]["segment_id"] == "high_0::low_0"
    assert rows[1]["end_time"] == 0.4


def test_read_pose_metadata_reads_static_pose_manifest(tmp_path: Path) -> None:
    pose_path = tmp_path / "trial_001_poses.json"
    pose_path.write_text(
        json.dumps(
            {
                "Hama1": [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]],
                "NIST_Board1": [[4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )

    metadata = read_pose_metadata(pose_path)

    assert metadata["start_time"] is None
    assert metadata["end_time"] is None
    assert metadata["num_pose_samples"] == 0
    assert metadata["num_pose_entities"] == 2
