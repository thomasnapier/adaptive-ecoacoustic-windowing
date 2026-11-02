import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from adaptive_windowing.io import (
    read_annotations,
    find_annotations,
    discover,
    save_boxes,
)
from adaptive_windowing.pipeline import run
from adaptive_windowing.config import load_config
from adaptive_windowing.evaluation import conditional_metrics
from adaptive_windowing import research


def config(tmp_path, **extra):
    payload = {
        "data_root": "data",
        "output_dir": "outputs",
        "save_plots": False,
        "save_drift_plots": False,
        "resume": True,
        "parameters": {},
    }
    payload.update(extra)
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(payload))
    return p


def test_annotations_units_and_ambiguity(tmp_path):
    p = tmp_path / "bird.txt"
    p.write_text(
        "Begin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tSpecies\n1\t2\t1000\t2000\tBird\n"
    )
    assert read_annotations(p)[0] == {
        "t0": 1.0,
        "t1": 2.0,
        "f0": 1000.0,
        "f1": 2000.0,
        "label": "Bird",
    }
    (tmp_path / "bird.csv").write_text("t0,t1,f0,f1\n")
    assert read_annotations(tmp_path / "bird.csv") == []
    with pytest.raises(ValueError, match="Multiple"):
        find_annotations(tmp_path / "bird.wav", tmp_path)
    p.write_text("t0,t1,f0,f1\n2,1,1000,2000\n")
    with pytest.raises(ValueError, match="require"):
        read_annotations(p)


def test_unknown_settings_and_relative_paths(tmp_path):
    p = config(tmp_path)
    assert load_config(p)["data_root"] == str(tmp_path / "data")
    p = config(tmp_path, parameters={"FMAX": 20000})
    with pytest.raises(ValueError, match="FMIN"):
        load_config(p)
    p = config(tmp_path, parameters={"DRIFT_SMOOTH_K": 4})
    with pytest.raises(ValueError, match="odd"):
        load_config(p)
    p = config(tmp_path, parameters={"TYPO": 1})
    with pytest.raises(ValueError, match="Unknown"):
        load_config(p)


def test_duplicate_audio_stems_are_rejected(tmp_path):
    a = tmp_path / "SITE" / "recordings"
    a.mkdir(parents=True)
    (a / "same.wav").touch()
    (a / "same.flac").touch()
    with pytest.raises(ValueError, match="Repeated"):
        discover(tmp_path)


def test_empty_prediction_csv_has_schema(tmp_path):
    save_boxes([], tmp_path / "boxes.csv")
    df = pd.read_csv(tmp_path / "boxes.csv")
    assert len(df) == 0 and {"t0", "t1", "f0", "f1"}.issubset(df.columns)


def test_region_timing_and_cross_group_overlap():
    research.configure()
    # Two steps span 0.375 s, not 0.25 s, because micro-windows overlap.
    starts, ends = research.make_fixed_windows(1.0, 0.25, 0.125)
    assert ends[1] - starts[0] == pytest.approx(0.375)
    boxes = [
        {"cluster_id": 1, "t0": 0.0, "t1": 0.5},
        {"cluster_id": 2, "t0": 0.2, "t1": 0.7},
        {"cluster_id": 1, "t0": 0.4, "t1": 0.8},
    ]
    merged = research.merge_overlapping_time_boxes(boxes)
    assert len(merged) == 2 and merged[0]["t1"] == 0.8
    assert research.binary_labels_to_regions([0, 1, 1, 0, 0, 1, 1, 0], 2, 2) == [[1, 6]]


def test_conditional_matching_uses_one_to_one_and_explicit_frequency_filter():
    ref = {"t0": 0.0, "t1": 2.0, "f0": 1000.0, "f1": 2000.0}
    # One perfect proposal cannot be matched twice to duplicate references.
    result = conditional_metrics([ref], [ref, ref])
    assert result["iou50_tp"] == 1 and result["iou50_recall"] == 0.5
    assert result["mean_best_temporal_iomin"] == 1.0
    distant = dict(ref, f0=5000.0, f1=6000.0)
    result = conditional_metrics([ref, distant], [ref])
    assert (
        result["raw_prediction_count"] == 2
        and result["evaluable_prediction_count"] == 1
    )
    assert result["t_gated_f1"] == 1.0
    # Nested box: IoMin=1 despite IoU=.1. Keep these criteria distinct.
    nested = dict(ref, t1=0.2)
    result = conditional_metrics([nested], [ref])
    assert result["iou50_tp"] == 0 and result["iomin50_tp"] == 1
    assert result["temporal_iomin50_tp"] == 1
    assert conditional_metrics([], [])["reference_count"] == 0
    assert conditional_metrics([], [ref])["mean_best_temporal_iomin"] == 0.0


@pytest.mark.skipif(
    not os.environ.get("WABAD_TEST_CONFIG"),
    reason="Set WABAD_TEST_CONFIG to a real study-data config after downloading the Zenodo files.",
)
def test_end_to_end_resume_and_reference_independence(tmp_path):
    from adaptive_windowing.pipeline import records_for

    source_config = load_config(Path(os.environ["WABAD_TEST_CONFIG"]).resolve())
    selected = records_for(source_config)[0]
    assert (
        selected["annotation"] is not None
    ), "The real-data check requires expert references."
    # Copy reference geometry into the temporary test folder; never alter study files.
    reference_copy = tmp_path / "references.csv"
    pd.DataFrame(read_annotations(selected["annotation"])).to_csv(
        reference_copy, index=False
    )
    payload = {
        "audio_file": str(selected["audio"].resolve()),
        "annotation_file": str(reference_copy),
        "output_dir": str(tmp_path / "outputs"),
        "save_plots": False,
        "save_drift_plots": False,
        "resume": True,
        "parameters": source_config["parameters"],
    }
    c = tmp_path / "study_test.json"
    c.write_text(json.dumps(payload))
    assert run(c, dry_run=True).empty
    assert not (tmp_path / "outputs").exists()
    first = run(c)
    directory = tmp_path / "outputs/single_file" / selected["audio"].stem
    original = (directory / "predicted_boxes.csv").read_bytes()
    assert first.iloc[0]["n_pred_boxes"] >= 0
    assert bool(run(c).iloc[0]["skipped_existing"])
    # Change only display labels: it invalidates the input hash, not detection.
    table = pd.read_csv(reference_copy)
    table["label"] = "Reference label changed for software check"
    table.to_csv(reference_copy, index=False)
    changed = run(c)
    assert not bool(changed.iloc[0]["skipped_existing"])
    assert (directory / "predicted_boxes.csv").read_bytes() == original
    payload.pop("annotation_file")
    c.write_text(json.dumps(payload))
    unlabelled = run(c)
    assert unlabelled.iloc[0]["reference_status"] == "missing"
    assert (directory / "predicted_boxes.csv").read_bytes() == original
    assert not any(k.startswith("conditional_") for k in unlabelled.columns)
    payload["parameters"]["GMM_MIN_EVENT_POSTERIOR"] = 0.7
    c.write_text(json.dumps(payload))
    assert not bool(run(c).iloc[0]["skipped_existing"])


def test_silence_and_failure_reporting(tmp_path):
    audio = tmp_path / "data/SILENCE/recordings"
    audio.mkdir(parents=True)
    sf.write(audio / "silence.wav", np.zeros(22050), 22050)
    c = config(tmp_path)
    result = run(c)
    assert result.iloc[0]["n_pred_boxes"] == 0
    assert pd.read_csv(tmp_path / "outputs/SILENCE/silence/predicted_boxes.csv").empty
    # A second file that is too short is reported while successful outputs persist.
    sf.write(audio / "short.wav", np.zeros(100), 22050)
    with pytest.raises(RuntimeError, match="failed"):
        run(c)
    assert len(pd.read_csv(tmp_path / "outputs/batch_failures.csv")) == 1
    assert (tmp_path / "outputs/SILENCE/silence/done.json").is_file()
