"""Read the one user-editable settings file before loading audio."""

import json
import math
from pathlib import Path
from importlib.resources import files

DEFAULTS = json.loads(files(__package__).joinpath("defaults.json").read_text())


def load_config(path):
    """Input: JSON path. Process: resolve paths and validate settings. Output: resolved config dict."""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    allowed = {
        "data_root",
        "output_dir",
        "audio_file",
        "annotation_file",
        "save_plots",
        "save_drift_plots",
        "resume",
        "parameters",
    }
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    for k in ["data_root", "output_dir", "audio_file", "annotation_file"]:
        if cfg.get(k) is not None:
            p = Path(cfg[k]).expanduser()
            cfg[k] = str(
                (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
            )
    if "output_dir" not in cfg:
        raise ValueError("Set output_dir in your JSON config.")
    if not cfg.get("audio_file") and not cfg.get("data_root"):
        raise ValueError("Set data_root (batch) or audio_file (one recording).")
    if cfg.get("annotation_file") and not cfg.get("audio_file"):
        raise ValueError("annotation_file is only used with audio_file.")
    for k in ["save_plots", "save_drift_plots", "resume"]:
        cfg.setdefault(k, True)
        if not isinstance(cfg[k], bool):
            raise ValueError(f"{k} must be true or false.")
    overrides = cfg.get("parameters", {})
    if not isinstance(overrides, dict):
        raise ValueError("parameters must be a JSON object.")
    if set(overrides) - set(DEFAULTS):
        raise ValueError(f"Unknown parameters: {sorted(set(overrides)-set(DEFAULTS))}")
    p = DEFAULTS | overrides
    for k, v in overrides.items():
        if (
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and not math.isfinite(v)
        ):
            raise ValueError(f"{k} must be finite.")
        if type(DEFAULTS[k]) is int and type(v) is not int:
            raise ValueError(f"{k} must be an integer (not a boolean or decimal).")
        default = DEFAULTS[k]
        if (
            default is not None
            and not isinstance(v, type(default))
            and not (isinstance(default, float) and type(v) in (int, float))
            and not (isinstance(default, list) and isinstance(v, list))
        ):
            raise ValueError(
                f"{k} should have the same type as its documented default."
            )
    positive = [
        "TARGET_SR",
        "WINDOW_S",
        "HOP_S",
        "N_FFT",
        "HOP_LENGTH",
        "N_MELS_FULL",
        "N_FINE_BANDS",
        "ADJ_TAU",
        "GMM_MIN_EVENT_WINDOWS",
    ]
    if any(p[k] <= 0 for k in positive):
        raise ValueError(
            "Sample rate, window/hop sizes, band count, and minimum run length must be positive."
        )
    for k in [
        "DRIFT_SMOOTH_K",
        "EVENT_SMOOTH_K",
        "BOX_SMOOTH_ACTIVITY_K",
        "GMM_SMOOTH_LABELS_K",
    ]:
        if type(p[k]) is not int or p[k] < 1 or p[k] % 2 == 0:
            raise ValueError(f"{k} must be a positive odd integer.")
    if not 0 <= p["FMIN"] < p["FMAX"] <= p["TARGET_SR"] / 2:
        raise ValueError("Require 0 <= FMIN < FMAX <= TARGET_SR/2.")
    if p["HOP_S"] > p["WINDOW_S"]:
        raise ValueError("HOP_S must not exceed WINDOW_S (would leave gaps).")
    for k in [
        "GMM_MIN_EVENT_POSTERIOR",
        "GT_IOU_THRESHOLD",
        "GT_TEMPORAL_HIT_IOU_THRESHOLD",
        "GT_FREQUENCY_HIT_IOU_THRESHOLD",
        "DISTANCE_THRESHOLD",
    ]:
        if not 0 < p[k] <= 1:
            raise ValueError(f"{k} must be above 0 and at most 1.")
    if p["N_FINE_BANDS"] > p["N_MELS_FULL"] or p["N_FINE_BANDS"] < 2:
        raise ValueError("Use at least 2 fine bands and no more than N_MELS_FULL.")
    if (
        min(p["ALPHA_TIME"], p["BETA_FREQ"], p["GAMMA_ADJ"]) < 0
        or p["ALPHA_TIME"] + p["BETA_FREQ"] + p["GAMMA_ADJ"] <= 0
    ):
        raise ValueError(
            "Similarity weights must be nonnegative and sum to more than zero."
        )
    if p["DENOISE_METHOD"] not in [
        "none",
        "spectral_gate",
        "hpss_percussive",
        "hpss_residual",
        "median_specsub",
    ]:
        raise ValueError("Unknown DENOISE_METHOD.")
    if p["SIMILARITY_MODE"] not in ["lagged_corr", "jaccard_binary"]:
        raise ValueError("Unknown SIMILARITY_MODE.")
    if p["LINKAGE_METHOD"] not in ["average", "single", "complete", "weighted"]:
        raise ValueError(
            "Use a linkage method compatible with precomputed dissimilarities."
        )
    if p["CLUSTER_AGG"] not in ["median", "mean", "max"]:
        raise ValueError("Unknown CLUSTER_AGG.")
    if p["BAND_SCALE"] not in ["mel", "linear"]:
        raise ValueError("BAND_SCALE must be mel or linear.")
    if p["N_GROUPS"] is not None and (
        type(p["N_GROUPS"]) is not int or not 1 <= p["N_GROUPS"] <= p["N_FINE_BANDS"]
    ):
        raise ValueError("N_GROUPS must be null or an integer from 1 to N_FINE_BANDS.")
    for k in [
        "GMM_MERGE_GAP_WINDOWS",
        "MAX_LAG_WINDOWS",
        "PEAK_MIN_SEP_WINDOWS",
        "EVENT_PEAK_MIN_SEP_WINDOWS",
        "PEAK_MATCH_TOL_WINDOWS",
    ]:
        if type(p[k]) is not int or p[k] < 0:
            raise ValueError(f"{k} must be a nonnegative integer.")
    if p["GMM_MIN_EVENT_AREA_Q"] is not None and (
        type(p["GMM_MIN_EVENT_AREA_Q"]) not in (int, float)
        or not 0 <= p["GMM_MIN_EVENT_AREA_Q"] <= 1
    ):
        raise ValueError("GMM_MIN_EVENT_AREA_Q must be null or a number in [0,1].")
    if p["SPEC_FLOOR_MODE"] not in ["median", "mean"]:
        raise ValueError("SPEC_FLOOR_MODE must be median or mean.")
    if (
        p["HPSS_KERNEL_SIZE"] < 1
        or p["HPSS_POWER"] <= 0
        or p["NOISE_PROFILE_SEC"] <= 0
        or p["FIG_DPI"] < 1
    ):
        raise ValueError(
            "HPSS controls, noise profile length, and figure DPI must be positive."
        )
    margin = p["HPSS_MARGIN"]
    if (
        not isinstance(margin, list)
        or len(margin) != 2
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 1 for v in margin
        )
    ):
        raise ValueError("HPSS_MARGIN must contain two numbers, each at least 1.")
    if not 0 <= p["SPEC_SUB_FLOOR"] <= 1 or p["SPEC_FLOOR_STRENGTH"] < 0:
        raise ValueError(
            "Require SPEC_SUB_FLOOR in [0,1] and nonnegative SPEC_FLOOR_STRENGTH."
        )
    cfg["parameters"] = p
    return cfg
