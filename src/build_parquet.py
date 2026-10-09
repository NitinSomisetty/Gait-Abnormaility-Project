"""Build balanced, per-gait Parquet files from the pathological gait CSVs."""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("build_parquet")
WALK_PATTERN = re.compile(r"^human(?P<subject>\d+)_(?P<gait>.+?)(?P<walk>\d+)$")
FILE_PATTERN = re.compile(
    r"^human(?P<subject>\d+)_(?P<gait>.+?)(?P<walk>\d+)"
    r"_SkeletonData(?P<sensor>\d+)\.csv$"
)
EXPECTED_COLUMNS = 101
GAIT_ORDER = (
    "normal",
    "antalgic",
    "stiff_legged",
    "lurch",
    "steppage",
    "trendelenburg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/Pathological_Gaits"),
        help="Directory containing one folder per walk set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/parquet"),
        help="Directory for Parquet files and the selection manifest.",
    )
    parser.add_argument(
        "--sets-per-subject",
        type=int,
        default=8,
        help="Number of walk sets to retain per subject and gait.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for reproducible walk-set selection.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def discover_walks(input_dir: Path) -> dict[tuple[str, str], list[tuple[int, Path]]]:
    walks: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for directory in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        match = WALK_PATTERN.fullmatch(directory.name)
        if match is None:
            LOGGER.warning("Ignoring unrecognized walk directory: %s", directory)
            continue
        key = (match["subject"], match["gait"])
        walks.setdefault(key, []).append((int(match["walk"]), directory))
    return walks


def select_walks(
    walks: dict[tuple[str, str], list[tuple[int, Path]]],
    sets_per_subject: int,
    seed: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    rng = random.Random(seed)
    for (subject, gait), candidates in sorted(walks.items()):
        if len(candidates) < sets_per_subject:
            raise ValueError(
                f"{subject}/{gait} has {len(candidates)} walk sets; "
                f"{sets_per_subject} requested"
            )
        chosen = sorted(rng.sample(candidates, sets_per_subject))
        for walk_id, directory in chosen:
            selected.append(
                {
                    "subject_id": subject,
                    "gait_label": gait,
                    "walk_id": walk_id,
                    "walk_directory": directory,
                }
            )
    return selected


def read_sensor_file(
    path: Path,
    selection: dict[str, object],
    input_dir: Path,
) -> tuple[pd.DataFrame | None, str | None]:
    match = FILE_PATTERN.fullmatch(path.name)
    if match is None:
        return None, f"unrecognized CSV filename: {path}"
    if (
        match["subject"] != selection["subject_id"]
        or match["gait"] != selection["gait_label"]
        or int(match["walk"]) != selection["walk_id"]
    ):
        return None, f"filename does not match its parent directory: {path}"

    try:
        raw = pd.read_csv(path, sep="\t", header=None)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return None, f"could not read {path}: {exc}"
    if raw.shape[1] != EXPECTED_COLUMNS:
        return None, f"{path} has {raw.shape[1]} columns; expected {EXPECTED_COLUMNS}"

    coordinate_names = ["time"]
    coordinate_columns: list[int] = [0]
    for joint in range(25):
        joint_id_column = 1 + (joint * 4)
        joint_id_values = raw.iloc[:, joint_id_column].dropna().unique()
        if len(joint_id_values) != 1 or joint_id_values[0] != joint:
            return None, f"{path} has invalid joint IDs for joint {joint}"
        coordinate_columns.extend(
            [joint_id_column + 1, joint_id_column + 2, joint_id_column + 3]
        )
        coordinate_names.extend(
            [f"joint{joint}_x", f"joint{joint}_y", f"joint{joint}_z"]
        )

    data = raw.iloc[:, coordinate_columns].copy()
    data.columns = coordinate_names
    coordinate_columns = [column for column in coordinate_names if column != "time"]
    with np.errstate(over="ignore", invalid="ignore"):
        float32_coordinates = data[coordinate_columns].astype("float32")
    invalid_conversion = (
        data[coordinate_columns].notna()
        & ~np.isfinite(float32_coordinates)
    ).any().any()
    if invalid_conversion:
        return None, f"{path} contains coordinate values outside float32 range"
    data[coordinate_columns] = float32_coordinates
    data["subject_id"] = str(selection["subject_id"])
    data["gait_label"] = str(selection["gait_label"])
    data["walk_id"] = int(selection["walk_id"])
    data["sensor_id"] = int(match["sensor"])
    data["frame_index"] = range(len(data))
    data["source_file"] = path.relative_to(input_dir).as_posix()
    data["n_frames"] = len(data)
    metadata = [
        "subject_id",
        "gait_label",
        "walk_id",
        "sensor_id",
        "frame_index",
        "source_file",
        "n_frames",
    ]
    return data[metadata + coordinate_names], None


def build_category(
    gait: str,
    selections: list[dict[str, object]],
    input_dir: Path,
    output_dir: Path,
    skipped: list[str],
) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    for selection in selections:
        if selection["gait_label"] != gait:
            continue
        directory = selection["walk_directory"]
        assert isinstance(directory, Path)
        for path in sorted(directory.glob("*.csv")):
            data, error = read_sensor_file(path, selection, input_dir)
            if error is not None:
                LOGGER.error(error)
                skipped.append(error)
                continue
            assert data is not None
            frames.append(data)

    category_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path = output_dir / f"{gait}.parquet"
    category_data.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    coordinate_columns = [column for column in category_data if column.startswith("joint")]
    nan_counts = category_data[coordinate_columns].isna().sum()
    nonzero_nan_counts = {column: int(count) for column, count in nan_counts.items() if count}
    round_trip_rows = len(pd.read_parquet(output_path))
    summary = {
        "category": gait,
        "walk_sets": category_data[["subject_id", "walk_id"]].drop_duplicates().shape[0]
        if not category_data.empty
        else 0,
        "subjects": category_data["subject_id"].nunique() if not category_data.empty else 0,
        "source_files": category_data["source_file"].nunique() if not category_data.empty else 0,
        "total_rows": len(category_data),
        "columns": len(category_data.columns),
        "file_size_bytes": output_path.stat().st_size,
        "round_trip_rows": round_trip_rows,
        "coordinate_nan_counts": nonzero_nan_counts,
    }
    LOGGER.info(
        "%s: %d walk sets, %d rows, %d source files, %.1f KiB",
        gait,
        summary["walk_sets"],
        summary["total_rows"],
        summary["source_files"],
        summary["file_size_bytes"] / 1024,
    )
    return summary


def write_manifest(selections: list[dict[str, object]], output_dir: Path) -> None:
    manifest = pd.DataFrame(
        {
            "subject_id": item["subject_id"],
            "gait_label": item["gait_label"],
            "walk_id": item["walk_id"],
            "walk_directory": Path(item["walk_directory"]).as_posix(),
        }
        for item in selections
    )
    manifest.sort_values(["gait_label", "subject_id", "walk_id"]).to_csv(
        output_dir / "selection_manifest.csv", index=False
    )


def main() -> int:
    args = parse_args()
    configure_logging()
    if args.sets_per_subject < 1:
        LOGGER.error("--sets-per-subject must be positive")
        return 2
    if not args.input_dir.is_dir():
        LOGGER.error("Input directory does not exist: %s", args.input_dir)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    walks = discover_walks(args.input_dir)
    selections = select_walks(walks, args.sets_per_subject, args.seed)
    write_manifest(selections, args.output_dir)
    LOGGER.info(
        "Selected %d walk sets (%d per subject/gait) with seed %d",
        len(selections),
        args.sets_per_subject,
        args.seed,
    )

    skipped: list[str] = []
    summaries = [
        build_category(gait, selections, args.input_dir, args.output_dir, skipped)
        for gait in GAIT_ORDER
        if any(item["gait_label"] == gait for item in selections)
    ]
    pd.DataFrame(summaries).drop(columns=["coordinate_nan_counts"]).to_csv(
        args.output_dir / "summary.csv", index=False
    )
    for summary in summaries:
        if summary["coordinate_nan_counts"]:
            LOGGER.warning(
                "%s coordinate NaN counts: %s",
                summary["category"],
                summary["coordinate_nan_counts"],
            )
    if skipped:
        LOGGER.error("Skipped %d malformed files:", len(skipped))
        for error in skipped:
            LOGGER.error("  %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
