"""Input and output formats shared by detection and plotting."""

from pathlib import Path
import csv
import numpy as np
import pandas as pd

BOX_COLUMNS = [
    "cluster_id",
    "t0",
    "t1",
    "duration_s",
    "f0",
    "f1",
    "bandwidth_hz",
    "peak_idx",
    "support_left",
    "support_right",
    "members",
]
AUDIO_SUFFIXES = {".wav", ".flac"}
AUDIO_DIRS = {"recordings", "recording", "audio", "audios", "wav", "wavs", "sounds"}
GT_DIRS = {
    "annotations",
    "annotation",
    "raven",
    "raven_annotations",
    "raven pro annotations",
    "raven_pro",
    "selection_tables",
    "selection tables",
    "gt",
    "ground_truth",
}


def read_annotations(path):
    """Input: Raven TXT/TSV or CSV. Process: validate explicit units. Output: boxes + labels.

    A header-only file is a valid empty annotation set. A missing file is represented
    by None by the caller. Malformed or ambiguous columns raise an actionable error.
    """
    path = Path(path)
    first = path.read_text(encoding="utf-8-sig").splitlines()
    if not first:
        raise ValueError(f"Empty annotation file: {path}; include a header row.")
    delimiter = "\t" if "\t" in first[0] else ","
    df = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")
    normalized = {str(c).strip().lower(): c for c in df.columns}
    aliases = {
        "t0": ["t0", "begin time (s)", "begin time", "start time", "start time (s)"],
        "t1": ["t1", "end time (s)", "end time", "stop time"],
        "f0": ["f0", "low freq (hz)", "low freq", "low frequency"],
        "f1": ["f1", "high freq (hz)", "high freq", "high frequency"],
    }
    selected = {}
    for target, names in aliases.items():
        matches = [normalized[n] for n in names if n in normalized]
        if len(matches) != 1:
            raise ValueError(
                f"{path.name}: supply exactly one column for {target}; accepted: {names}"
            )
        selected[target] = matches[0]
    label_col = next(
        (
            normalized[n]
            for n in [
                "species",
                "label",
                "annotation",
                "class",
                "species scientific name",
            ]
            if n in normalized
        ),
        None,
    )
    boxes = []
    for index, row in df.iterrows():
        try:
            b = {k: float(row[c]) for k, c in selected.items()}
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{path.name}, row {index+2}: time/frequency values must be numeric."
            ) from exc
        if not all(np.isfinite(v) for v in b.values()) or not (
            0 <= b["t0"] < b["t1"] and 0 <= b["f0"] < b["f1"]
        ):
            raise ValueError(
                f"{path.name}, row {index+2}: require 0 <= start < end, 0 <= low < high, and finite values."
            )
        b["label"] = (
            str(row[label_col])
            if label_col and pd.notna(row[label_col])
            else "Reference event"
        )
        boxes.append(b)
    return boxes


def find_annotations(audio, folder):
    """Input: audio path + optional annotation folder. Process: unambiguous stem matching. Output: path/None."""
    if folder is None:
        return None
    audio, folder = Path(audio), Path(folder)
    candidates = [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".tsv", ".csv"}
    ]
    exact = [p for p in candidates if p.stem.lower() == audio.stem.lower()]
    matches = exact or [
        p
        for p in candidates
        if p.stem.lower().startswith(audio.stem.lower() + "_")
        or p.stem.lower().startswith(audio.stem.lower() + ".")
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple annotation files match {audio.name}: {matches}; keep one or use annotation_file."
        )
    return matches[0] if matches else None


def discover(data_root):
    """Input: WABAD root or site folder. Process: find site/audio/reference folders. Output: file records."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data folder does not exist: {root}; edit data_root in your config."
        )
    records, seen = [], set()
    for site in [root] + sorted(p for p in root.iterdir() if p.is_dir()):
        children = [p for p in site.iterdir() if p.is_dir()]
        audio_dirs = [p for p in children if p.name.lower() in AUDIO_DIRS]
        gt_dirs = [p for p in children if p.name.lower() in GT_DIRS]
        if len(audio_dirs) > 1 or len(gt_dirs) > 1:
            raise ValueError(
                f"Ambiguous folders in {site}: keep one audio folder and at most one annotation folder."
            )
        if not audio_dirs:
            continue
        audio_dir = audio_dirs[0]
        if audio_dir.resolve() in seen:
            continue
        seen.add(audio_dir.resolve())
        for audio in sorted(
            p
            for p in audio_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
        ):
            records.append(
                {
                    "site": site.name,
                    "audio": audio,
                    "annotation": find_annotations(
                        audio, gt_dirs[0] if gt_dirs else None
                    ),
                }
            )
    if not records:
        raise ValueError(
            f"No WAV/FLAC files found under {root}. Expected SITE/recordings/*.wav or use audio_file."
        )
    keys = [(r["site"], r["audio"].stem) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "Repeated audio stems within a site would overwrite outputs. Rename the files uniquely."
        )
    return records


def save_boxes(boxes, path):
    """Input: candidate boxes. Process: add units/derived fields. Output: CSV, with headers even for zero events."""
    rows = []
    for b in boxes:
        row = {k: b.get(k) for k in BOX_COLUMNS}
        row.update(
            duration_s=b["t1"] - b["t0"],
            bandwidth_hz=b["f1"] - b["f0"],
            members=",".join(map(str, b.get("members", []))),
        )
        rows.append(row)
    pd.DataFrame(rows, columns=BOX_COLUMNS).to_csv(path, index=False)
