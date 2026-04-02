"""Rule-based weak contact labeling for aligned insert samples."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import h5py

from reassemble_minexp.utils.paths import ensure_directory


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_skill_name(skill_name: str) -> str:
    return skill_name.strip().lower().replace(" ", "_")


def _dataset_path_candidates(source_key: str) -> list[str]:
    candidates = [source_key]
    if "/" not in source_key:
        candidates.extend(
            [
                f"robot_state/{source_key}",
                f"timestamps/{source_key}",
            ]
        )
    return candidates


def _read_numeric_dataset(handle: h5py.File, source_key: str) -> list[list[float]]:
    for candidate in _dataset_path_candidates(source_key):
        if candidate not in handle:
            continue
        raw = handle[candidate][()]
        values = raw.tolist() if hasattr(raw, "tolist") else raw
        if not isinstance(values, list):
            values = [values]
        rows: list[list[float]] = []
        for item in values:
            if isinstance(item, list):
                rows.append([float(value) for value in item])
            else:
                rows.append([float(item)])
        return rows
    raise KeyError(f"Could not find dataset for source key: {source_key}")


def _interpolate_vector(samples: list[list[float]], index_0: int, index_1: int, weight_0: float, weight_1: float) -> list[float]:
    left = samples[index_0]
    right = samples[index_1]
    width = min(len(left), len(right))
    return [(left[i] * weight_0) + (right[i] * weight_1) for i in range(width)]


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def classify_contact_label(skill_name: str, ft_norm: float, config: Any) -> tuple[str, str]:
    """Classify a weak contact label from active skill and interpolated F/T norm."""

    normalized_skill = _normalize_skill_name(skill_name)
    touch_threshold = float(config.labels.contact.touch_ft_norm_threshold)
    jam_threshold = float(config.labels.contact.jam_ft_norm_threshold)
    search_skills = {_normalize_skill_name(skill) for skill in config.labels.contact.search_skills}
    insertion_skills = {_normalize_skill_name(skill) for skill in config.labels.contact.insertion_skills}

    if ft_norm >= jam_threshold:
        return "jam_or_abnormal", "ft_norm>=jam_threshold"
    if ft_norm < touch_threshold:
        return "free", "ft_norm<touch_threshold"
    if normalized_skill in search_skills:
        return "search_contact", "contact_during_search_skill"
    if normalized_skill in insertion_skills:
        return "insertion_contact", "contact_during_insertion_skill"
    return "touch", "contact_without_specific_skill_group"


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a quantile of an empty list")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]

    position = (len(sorted_values) - 1) * q
    left_index = int(math.floor(position))
    right_index = int(math.ceil(position))
    if left_index == right_index:
        return sorted_values[left_index]

    weight = position - left_index
    return sorted_values[left_index] * (1.0 - weight) + sorted_values[right_index] * weight


def summarize_ft_value_norms(contact_rows: list[dict[str, Any]], config: Any) -> dict[str, Any]:
    """Summarize F/T norm distribution and suggest threshold candidates."""

    all_values = sorted(float(row["ft_value_norm"]) for row in contact_rows)
    if not all_values:
        raise ValueError("contact_rows must not be empty")

    search_skills = {_normalize_skill_name(skill) for skill in config.labels.contact.search_skills}
    insertion_skills = {_normalize_skill_name(skill) for skill in config.labels.contact.insertion_skills}

    search_values = sorted(
        float(row["ft_value_norm"])
        for row in contact_rows
        if _normalize_skill_name(str(row.get("active_low_level_skill", ""))) in search_skills
    )
    insertion_values = sorted(
        float(row["ft_value_norm"])
        for row in contact_rows
        if _normalize_skill_name(str(row.get("active_low_level_skill", ""))) in insertion_skills
    )

    summary = {
        "count": len(all_values),
        "min": _quantile(all_values, 0.0),
        "p05": _quantile(all_values, 0.05),
        "p10": _quantile(all_values, 0.10),
        "p25": _quantile(all_values, 0.25),
        "p50": _quantile(all_values, 0.50),
        "p75": _quantile(all_values, 0.75),
        "p90": _quantile(all_values, 0.90),
        "p95": _quantile(all_values, 0.95),
        "p99": _quantile(all_values, 0.99),
        "max": _quantile(all_values, 1.0),
        "search_count": len(search_values),
        "insertion_count": len(insertion_values),
    }

    touch_source = search_values if search_values else all_values
    jam_source = insertion_values if insertion_values else all_values
    touch_suggestion = _quantile(touch_source, 0.10)
    jam_suggestion = max(_quantile(jam_source, 0.95), _quantile(all_values, 0.99))
    if jam_suggestion <= touch_suggestion:
        jam_suggestion = max(touch_suggestion * 1.25, _quantile(all_values, 0.99))

    summary["suggested_touch_ft_norm_threshold"] = touch_suggestion
    summary["suggested_jam_ft_norm_threshold"] = jam_suggestion
    return summary


def format_threshold_suggestion(summary: dict[str, Any]) -> str:
    """Format a readable threshold suggestion block for CLI output."""

    return "\n".join(
        [
            "F/T norm summary:",
            f"  count={summary['count']}",
            f"  min={summary['min']:.3f} p05={summary['p05']:.3f} p10={summary['p10']:.3f} p25={summary['p25']:.3f}",
            f"  p50={summary['p50']:.3f} p75={summary['p75']:.3f} p90={summary['p90']:.3f} p95={summary['p95']:.3f} p99={summary['p99']:.3f} max={summary['max']:.3f}",
            f"  search_count={summary['search_count']} insertion_count={summary['insertion_count']}",
            "Suggested overrides:",
            f"  labels.contact.touch_ft_norm_threshold={summary['suggested_touch_ft_norm_threshold']:.3f}",
            f"  labels.contact.jam_ft_norm_threshold={summary['suggested_jam_ft_norm_threshold']:.3f}",
        ]
    )


def build_contact_visualization_payload(contact_rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Prepare a lightweight plotting payload for label quality checks."""

    return {
        "aligned_time": [row["aligned_time"] for row in contact_rows],
        "ft_value_norm": [row["ft_value_norm"] for row in contact_rows],
        "y_contact": [row["y_contact"] for row in contact_rows],
        "active_low_level_skill": [row["active_low_level_skill"] for row in contact_rows],
    }


def make_contact_labels(aligned_rows: list[dict[str, str]], phase_rows: list[dict[str, str]], config: Any) -> list[dict[str, Any]]:
    """Generate rule-based contact labels for aligned samples."""

    phase_by_sample = {row["sample_id"]: row for row in phase_rows}
    h5_cache: dict[str, dict[str, list[list[float]]]] = {}
    contact_rows: list[dict[str, Any]] = []

    for row in aligned_rows:
        sample_id = row["sample_id"]
        phase_row = phase_by_sample.get(sample_id, {})
        h5_path = row["h5_path"]
        ft_source_key = row["ft_source_key"]
        if h5_path not in h5_cache:
            h5_cache[h5_path] = {}
        if ft_source_key and ft_source_key not in h5_cache[h5_path]:
            with h5py.File(h5_path, "r") as handle:
                h5_cache[h5_path][ft_source_key] = _read_numeric_dataset(handle, ft_source_key)

        ft_samples = h5_cache[h5_path].get(ft_source_key, [])
        ft_norm = 0.0
        if ft_samples and row["ft_index_0"] != "":
            vector = _interpolate_vector(
                ft_samples,
                int(row["ft_index_0"]),
                int(row["ft_index_1"]),
                float(row["ft_weight_0"]),
                float(row["ft_weight_1"]),
            )
            ft_norm = _vector_norm(vector)

        active_skill = phase_row.get("active_low_level_skill", "")
        contact_label, reason = classify_contact_label(active_skill, ft_norm, config)
        contact_rows.append(
            {
                "sample_id": sample_id,
                "insert_id": row["insert_id"],
                "trial_id": row["trial_id"],
                "aligned_time": row["aligned_time"],
                "active_low_level_skill": active_skill,
                "ft_source_key": ft_source_key,
                "ft_value_norm": ft_norm,
                "y_contact": contact_label,
                "rule_reason": reason,
            }
        )

    return contact_rows


def build_contact_labels(config: Any) -> list[dict[str, Any]]:
    """Load aligned samples and phase labels, then generate weak contact labels."""

    processed_root = Path(config.dataset.processed_root)
    aligned_path = processed_root / str(config.alignment.aligned_index_name)
    phase_path = processed_root / str(config.labels.phase_index_name)

    if not aligned_path.exists():
        raise FileNotFoundError(f"Aligned sample index not found: {aligned_path}")
    if not phase_path.exists():
        raise FileNotFoundError(f"Phase labels not found: {phase_path}")

    aligned_rows = _read_csv(aligned_path)
    phase_rows = _read_csv(phase_path)
    return make_contact_labels(aligned_rows, phase_rows, config)


def write_contact_labels(config: Any, contact_rows: list[dict[str, Any]]) -> Path:
    """Persist contact labels."""

    processed_root = ensure_directory(config.dataset.processed_root)
    output_path = processed_root / str(config.labels.contact_index_name)
    _write_csv(
        output_path,
        contact_rows,
        [
            "sample_id",
            "insert_id",
            "trial_id",
            "aligned_time",
            "active_low_level_skill",
            "ft_source_key",
            "ft_value_norm",
            "y_contact",
            "rule_reason",
        ],
    )
    return output_path


def run_contact_label_pipeline(config: Any) -> Path:
    """Run Milestone 5 contact labeling end to end."""

    contact_rows = build_contact_labels(config)
    return write_contact_labels(config, contact_rows)
