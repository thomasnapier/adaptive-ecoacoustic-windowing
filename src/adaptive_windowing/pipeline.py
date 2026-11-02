"""User-facing serial runner around the notebook's WABAD scientific functions."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import traceback
import warnings
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from . import research
from .config import load_config
from .io import discover, read_annotations, save_boxes
from .evaluation import conditional_metrics


def digest(path):
    """Input: file path. Process: stream SHA-256. Output: content fingerprint."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records_for(cfg):
    """Input: resolved settings. Process: select batch or explicit file. Output: recording records."""
    if cfg.get("audio_file"):
        audio = Path(cfg["audio_file"])
        if not audio.is_file() or audio.suffix.lower() not in {".wav", ".flac"}:
            raise ValueError(f"audio_file must be an existing WAV or FLAC: {audio}")
        annotation = (
            Path(cfg["annotation_file"]) if cfg.get("annotation_file") else None
        )
        return [{"site": "single_file", "audio": audio, "annotation": annotation}]
    return discover(cfg["data_root"])


def validate_record(record, parameters):
    """Input: recording record. Process: check duration and reference extents. Output: boxes/None + info."""
    info = sf.info(record["audio"])
    if info.frames / info.samplerate < parameters["WINDOW_S"]:
        raise ValueError(
            f"{record['audio'].name}: recording shorter than WINDOW_S; use a longer file."
        )
    if info.samplerate / 2 < parameters["FMAX"]:
        warnings.warn(
            f"{record['audio'].name}: original Nyquist {info.samplerate/2:g} Hz is below FMAX; resampling cannot restore missing frequencies."
        )
    refs = (
        read_annotations(record["annotation"])
        if record["annotation"] is not None
        else None
    )
    for b in refs or []:
        if b["t1"] > info.duration + 0.02:
            raise ValueError(
                f"{record['audio'].name}: reference end {b['t1']} s exceeds audio duration {info.duration:.3f} s. Check clip-relative times."
            )
        if b["f1"] > info.samplerate / 2 + 1:
            warnings.warn(
                f"{record['audio'].name}: reference extends above original Nyquist; check frequency units."
            )
    return refs, info


def run(config_path, limit=None, dry_run=False, force=False):
    """Input: JSON config. Process: validate, detect, evaluate, save. Output: batch summary DataFrame.

    References are loaded for validation, plotting and evaluation only. They never
    enter compute_pipeline_for_file or GMM proposal generation.
    """
    # INPUT: read the user settings and identify the requested recordings.
    cfg = load_config(config_path)
    records = records_for(cfg)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        records = records[:limit]
    print(f"Found {len(records)} recording(s). Output: {cfg['output_dir']}")
    if dry_run:
        for r in records:
            # INPUT CHECK: references are inspected here, but never passed into detection.
            refs, info = validate_record(r, cfg["parameters"])
            print(
                f"{r['site']}/{r['audio'].name}: {info.duration:.2f} s, {info.samplerate} Hz; references: {len(refs) if refs is not None else 'not supplied'}"
            )
        return pd.DataFrame()
    research.configure(cfg["parameters"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "scipy",
            "pandas",
            "librosa",
            "matplotlib",
            "scikit-learn",
            "soundfile",
        ]
    }
    code_hashes = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    manifest = {
        "configuration": cfg,
        "python": platform.python_version(),
        "dependencies": versions,
        "source_hashes": code_hashes,
        "scope": "limited run" if limit else "all discovered recordings",
        "limit": limit,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    rows, failures = [], []
    for index, r in enumerate(records, 1):
        folder = out / r["site"] / r["audio"].stem
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / "done.json"
        summary_file = folder / "summary.json"
        print(f"[{index}/{len(records)}] {r['audio'].name}")
        try:
            # INPUT CHECK: references are inspected here, but never passed into detection.
            refs, info = validate_record(r, cfg["parameters"])
            # RESUME: only reuse a completed run with the same data and environment.
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "config": cfg,
                        "code": code_hashes,
                        "dependencies": versions,
                        "python": platform.python_version(),
                        "audio": digest(r["audio"]),
                        "annotation": (
                            digest(r["annotation"]) if r["annotation"] else None
                        ),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            required = [folder / "predicted_boxes.csv", summary_file]
            if cfg["save_plots"]:
                required.append(folder / "spectrogram_boxes.png")
            if cfg["save_drift_plots"]:
                required.append(folder / "cluster_drifts.png")
            if (
                marker.exists()
                and cfg["resume"]
                and not force
                and all(p.is_file() for p in required)
            ):
                try:
                    saved = json.loads(marker.read_text())
                    if saved.get("fingerprint") == fingerprint:
                        row = json.loads(summary_file.read_text())
                        row["skipped_existing"] = True
                        rows.append(row)
                        continue
                except (ValueError, OSError):
                    pass
            marker.unlink(missing_ok=True)
            # PROCESSING A-C: waveform preparation, local spectral change, and band grouping.
            pipe = research.compute_pipeline_for_file(str(r["audio"]))
            p = cfg["parameters"]
            # PROCESSING D: fit active/inactive states per group and construct event boxes.
            boxes, states = research.gmm_boxes_from_cluster_drifts(
                pipe["cluster_drifts"],
                pipe["cluster_members"],
                pipe["uniq_clusters"],
                pipe["edges_hz"],
                pipe["win_starts"],
                pipe["win_ends"],
                min_event_posterior=p["GMM_MIN_EVENT_POSTERIOR"],
                smooth_labels_k=p["GMM_SMOOTH_LABELS_K"],
                merge_gap_windows=p["GMM_MERGE_GAP_WINDOWS"],
                min_event_windows=p["GMM_MIN_EVENT_WINDOWS"],
                min_event_area_q=p["GMM_MIN_EVENT_AREA_Q"],
                use_log_drift=p["GMM_USE_LOG_DRIFT"],
                random_state=p["GMM_RANDOM_STATE"],
            )
            # OUTPUT: always save raw candidate coordinates, including a valid empty table.
            save_boxes(boxes, folder / "predicted_boxes.csv")
            row = {
                "dataset": r["site"],
                "file": r["audio"].name,
                "audio_path": str(r["audio"]),
                "annotation_path": str(r["annotation"]) if r["annotation"] else None,
                "duration_s": info.duration,
                "n_windows": len(pipe["win_starts"]),
                "n_clusters": len(pipe["uniq_clusters"]),
                "n_pred_boxes": len(boxes),
                "n_gt_boxes": len(refs) if refs is not None else None,
                "reference_status": (
                    "missing" if refs is None else ("empty" if not refs else "present")
                ),
                "skipped_existing": False,
                "output_dir": str(folder),
            }
            # EVALUATION ONLY: compare fixed predictions with optional expert references.
            if refs:
                legacy = research.evaluate_predicted_boxes(
                    boxes,
                    refs,
                    iou_thr=p["GT_IOU_THRESHOLD"],
                    win_starts=pipe["win_starts"],
                    win_ends=pipe["win_ends"],
                    duration_s=info.duration,
                    temporal_eval_hop_s=p["TEMPORAL_EVAL_HOP_S"],
                )
                row.update({"notebook_" + k: v for k, v in legacy.items()})
            if refs is not None:
                row.update(
                    {
                        "conditional_" + k: v
                        for k, v in conditional_metrics(boxes, refs).items()
                    }
                )
            if cfg["save_plots"]:
                research.save_spectrogram_with_boxes(
                    pipe["y"],
                    pipe["sr"],
                    boxes,
                    refs,
                    folder / "spectrogram_boxes.png",
                    title=f"{r['site']} / {r['audio'].stem}: candidates (cyan), references (green dashed)",
                )
            if cfg["save_drift_plots"]:
                research.save_cluster_drifts_plot(
                    pipe["cluster_drifts"],
                    pipe["cluster_members"],
                    pipe["uniq_clusters"],
                    pipe["win_starts"],
                    pipe["edges_hz"],
                    folder / "cluster_drifts.png",
                    title="Grouped acoustic-change trajectories",
                )
            # CHECKPOINT: mark success only after all requested outputs have been written.
            summary_file.write_text(json.dumps(row, indent=2, allow_nan=False))
            marker.write_text(json.dumps({"fingerprint": fingerprint}))
            rows.append(row)
        except Exception as exc:
            failures.append(
                {"dataset": r["site"], "file": str(r["audio"]), "error": str(exc)}
            )
            print(f"FAILED: {exc}")
            traceback.print_exc()
        finally:
            plt.close("all")
        pd.DataFrame(rows).to_csv(out / "batch_summary.csv", index=False)
        pd.DataFrame(failures, columns=["dataset", "file", "error"]).to_csv(
            out / "batch_failures.csv", index=False
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "batch_summary.csv", index=False)
    pd.DataFrame(failures, columns=["dataset", "file", "error"]).to_csv(
        out / "batch_failures.csv", index=False
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} recording(s) failed; see {out/'batch_failures.csv'}. Successful files can be resumed."
        )
    print(f"Finished. Summary: {out/'batch_summary.csv'}")
    return df
