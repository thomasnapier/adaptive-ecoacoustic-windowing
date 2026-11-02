"""WABAD scientific implementation extracted from batch notebook cell 2 (one-based).

Inputs: waveform paths or numeric arrays, depending on the function.
Processing: fine-band drift grouping and per-group GMM event proposal generation.
Outputs: candidate boxes and diagnostics, never species labels.

Configure once per serial run with configure(); globals mirror the supplied notebook.
This module is not safe for concurrent runs with different settings in one process.
See docs/METHOD.md and docs/API_REFERENCE.md for the processing map.
"""

import os
import glob
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.mixture import GaussianMixture

import json
from importlib.resources import files

DEFAULTS = json.loads(files(__package__).joinpath("defaults.json").read_text())
globals().update(DEFAULTS)


def configure(parameters=None):
    """Input: parameter overrides. Processing: reset defaults then apply overrides. Output: none."""
    unknown = set(parameters or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    globals().update(DEFAULTS)
    globals().update(parameters or {})


# ========================================================================
# Numerical helpers
# INPUT: arrays and scalar measurements.
# PROCESSING: stabilise, scale, or compare numeric values.
# OUTPUT: numeric arrays or scalar statistics.
# ========================================================================


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Input: arrays and scalar measurements; see the signature for specific arguments.
    Processing: stabilise, scale, or compare numeric values.
    Output: numeric arrays or scalar statistics.
    """
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    w = np.ones(k, dtype=float) / k
    return np.convolve(xp, w, mode="valid")


def robust_mad(x: np.ndarray) -> float:
    """Input: arrays and scalar measurements; see the signature for specific arguments.
    Processing: stabilise, scale, or compare numeric values.
    Output: numeric arrays or scalar statistics.
    """
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Input: arrays and scalar measurements; see the signature for specific arguments.
    Processing: stabilise, scale, or compare numeric values.
    Output: numeric arrays or scalar statistics.
    """
    med = np.median(x)
    mad = robust_mad(x)
    return (x - med) / mad


def minmax01(x: np.ndarray) -> np.ndarray:
    """Input: arrays and scalar measurements; see the signature for specific arguments.
    Processing: stabilise, scale, or compare numeric values.
    Output: numeric arrays or scalar statistics.
    """
    x = np.asarray(x, dtype=float)
    mn = np.min(x)
    mx = np.max(x)
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def safe_pearson(x, y):
    """Input: arrays and scalar measurements; see the signature for specific arguments.
    Processing: stabilise, scale, or compare numeric values.
    Output: numeric arrays or scalar statistics.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    r = pearsonr(x, y)[0]
    if np.isnan(r):
        return 0.0
    return float(r)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Input: arrays and scalar measurements; see the signature for specific arguments.
    Processing: stabilise, scale, or compare numeric values.
    Output: numeric arrays or scalar statistics.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    s = float(np.dot(a, b) / (na * nb))
    return max(0.0, min(1.0, 0.5 * (s + 1.0)))


def make_fixed_windows(duration_s: float, window_s: float, hop_s: float):
    """Input: arrays and scalar measurements; see the signature for specific arguments.
    Processing: stabilise, scale, or compare numeric values.
    Output: numeric arrays or scalar statistics.
    """
    starts = []
    ends = []
    t = 0.0
    while t + window_s <= duration_s + 1e-12:
        starts.append(t)
        ends.append(t + window_s)
        t += hop_s
    return np.array(starts, dtype=float), np.array(ends, dtype=float)


def nonmax_suppress_peaks(peaks, scores, radius):
    """Input: arrays and scalar measurements; see the signature for specific arguments.
    Processing: stabilise, scale, or compare numeric values.
    Output: numeric arrays or scalar statistics.
    """
    if radius <= 0 or len(peaks) <= 1:
        return sorted(peaks)
    peaks_sorted = sorted(peaks, key=lambda i: scores[i], reverse=True)
    blocked = np.zeros(len(scores), dtype=bool)
    kept = []
    for p in peaks_sorted:
        if blocked[p]:
            continue
        kept.append(p)
        lo = max(0, p - radius)
        hi = min(len(scores), p + radius + 1)
        blocked[lo:hi] = True
    return sorted(kept)


def detect_peaks_from_drift(drift: np.ndarray, k_mad: float, min_sep_windows: int):
    """Input: arrays and scalar measurements; see the signature for specific arguments.
    Processing: stabilise, scale, or compare numeric values.
    Output: numeric arrays or scalar statistics.
    """
    z = robust_zscore(drift)
    thr = np.median(z) + k_mad * robust_mad(z)

    peaks = []
    for i in range(1, len(z) - 1):
        if z[i] >= thr and z[i] >= z[i - 1] and z[i] >= z[i + 1]:
            peaks.append(i)

    peaks = nonmax_suppress_peaks(peaks, z, radius=min_sep_windows)

    evidence = np.zeros(len(z), dtype=int)
    evidence[peaks] = 1
    return z, evidence, peaks, thr


# ========================================================================
# Frequency bands and preprocessing
# INPUT: waveform samples and frequency settings.
# PROCESSING: define band edges or suppress selected background components.
# OUTPUT: band edges or a preprocessed waveform.
# ========================================================================


def band_edges_hz(
    fmin: float, fmax: float, n_bands: int, scale: str = "mel"
) -> np.ndarray:
    """Input: waveform samples and frequency settings; see the signature for specific arguments.
    Processing: define band edges or suppress selected background components.
    Output: band edges or a preprocessed waveform.
    """
    if scale == "mel":
        m0 = librosa.hz_to_mel(fmin)
        m1 = librosa.hz_to_mel(fmax)
        m_edges = np.linspace(m0, m1, n_bands + 1)
        return librosa.mel_to_hz(m_edges)
    elif scale == "linear":
        return np.linspace(fmin, fmax, n_bands + 1)
    else:
        raise ValueError("scale must be 'mel' or 'linear'")


def denoise_spectral_gate(
    y, sr, n_fft=1024, hop_length=256, noise_profile_sec=2.0, floor=0.1
):
    """Input: waveform samples and frequency settings; see the signature for specific arguments.
    Processing: define band edges or suppress selected background components.
    Output: band edges or a preprocessed waveform.
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    phase = np.angle(S)

    noise_frames = max(1, int(round((noise_profile_sec * sr) / hop_length)))
    noise_frames = min(noise_frames, mag.shape[1])

    noise_mag = np.median(mag[:, :noise_frames], axis=1, keepdims=True)
    mag_denoised = mag - noise_mag
    mag_denoised = np.maximum(mag_denoised, floor * mag)

    S_denoised = mag_denoised * np.exp(1j * phase)
    y_denoised = librosa.istft(S_denoised, hop_length=hop_length, length=len(y))
    return y_denoised


def denoise_hpss_percussive(y, kernel_size=31, power=2.0, margin=(1.0, 5.0)):
    """Input: waveform samples and frequency settings; see the signature for specific arguments.
    Processing: define band edges or suppress selected background components.
    Output: band edges or a preprocessed waveform.
    """
    y_h, y_p = librosa.effects.hpss(
        y, kernel_size=kernel_size, power=power, margin=margin
    )
    return y_p


def denoise_hpss_residual(y, kernel_size=31, power=2.0, margin=(1.0, 5.0)):
    """Input: waveform samples and frequency settings; see the signature for specific arguments.
    Processing: define band edges or suppress selected background components.
    Output: band edges or a preprocessed waveform.
    """
    y_h, y_p = librosa.effects.hpss(
        y, kernel_size=kernel_size, power=power, margin=margin
    )
    residual = y - y_h
    return residual


def subtract_stationary_band_floor(X, strength=1.0, floor_mode="median"):
    """Input: waveform samples and frequency settings; see the signature for specific arguments.
    Processing: define band edges or suppress selected background components.
    Output: band edges or a preprocessed waveform.
    """
    X = np.asarray(X, dtype=float)

    if floor_mode == "median":
        bg = np.median(X, axis=0, keepdims=True)
    elif floor_mode == "mean":
        bg = np.mean(X, axis=0, keepdims=True)
    else:
        raise ValueError("floor_mode must be 'median' or 'mean'")

    out = X - strength * bg
    return np.maximum(out, 0.0)


def apply_waveform_preprocessing(y, sr):
    """Input: waveform samples and frequency settings; see the signature for specific arguments.
    Processing: define band edges or suppress selected background components.
    Output: band edges or a preprocessed waveform.
    """
    if DENOISE_METHOD == "none":
        return y

    if DENOISE_METHOD == "spectral_gate":
        print("Applying spectral gate denoising...")
        return denoise_spectral_gate(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            noise_profile_sec=NOISE_PROFILE_SEC,
            floor=SPEC_SUB_FLOOR,
        )

    if DENOISE_METHOD == "hpss_percussive":
        print("Applying HPSS percussive emphasis...")
        return denoise_hpss_percussive(
            y, kernel_size=HPSS_KERNEL_SIZE, power=HPSS_POWER, margin=HPSS_MARGIN
        )

    if DENOISE_METHOD == "hpss_residual":
        print("Applying HPSS residual emphasis...")
        return denoise_hpss_residual(
            y, kernel_size=HPSS_KERNEL_SIZE, power=HPSS_POWER, margin=HPSS_MARGIN
        )

    if DENOISE_METHOD == "median_specsub":
        return y

    raise ValueError(f"Unsupported DENOISE_METHOD: {DENOISE_METHOD}")


# ========================================================================
# Micro-window representations and drift
# INPUT: mono waveform, window times, and band vectors.
# PROCESSING: compute PCEN/log-Mel summaries, adjacent-window drift, and summary descriptors.
# OUTPUT: spectral vectors, band drift arrays, or descriptor matrix.
# ========================================================================


def compute_window_repr_vectors(
    y, sr, win_starts, win_ends, n_mels, fmin, fmax, use_pcen=False
):
    """Input: mono waveform, window times, and band vectors; see the signature for specific arguments.
    Processing: compute PCEN/log-Mel summaries, adjacent-window drift, and summary descriptors.
    Output: spectral vectors, band drift arrays, or descriptor matrix.
    """
    out = []

    for t0, t1 in zip(win_starts, win_ends):
        s0 = int(round(t0 * sr))
        s1 = int(round(t1 * sr))
        seg = y[s0:s1]

        if len(seg) < 16:
            out.append(np.zeros(n_mels, dtype=np.float32))
            continue

        n_fft_eff = min(N_FFT, 2 ** int(np.floor(np.log2(len(seg)))))
        n_fft_eff = max(n_fft_eff, 128)
        hop_eff = min(HOP_LENGTH, max(32, n_fft_eff // 4))

        S = librosa.feature.melspectrogram(
            y=seg,
            sr=sr,
            n_fft=n_fft_eff,
            hop_length=hop_eff,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=2.0,
        )

        if use_pcen:
            rep = librosa.pcen(S * (2.0**16), sr=sr, hop_length=hop_eff)
        else:
            rep = librosa.power_to_db(S, ref=np.max)

        vec = rep.mean(axis=1)
        out.append(vec.astype(np.float32))

    return np.stack(out, axis=0)


def split_mel_vectors_into_bands(logmel_vectors, mel_freqs, edges_hz):
    """Input: mono waveform, window times, and band vectors; see the signature for specific arguments.
    Processing: compute PCEN/log-Mel summaries, adjacent-window drift, and summary descriptors.
    Output: spectral vectors, band drift arrays, or descriptor matrix.
    """
    band_vectors = []
    for bi in range(len(edges_hz) - 1):
        lo, hi = edges_hz[bi], edges_hz[bi + 1]
        if bi < len(edges_hz) - 2:
            mask = (mel_freqs >= lo) & (mel_freqs < hi)
        else:
            mask = (mel_freqs >= lo) & (mel_freqs <= hi)

        if np.sum(mask) == 0:
            band_vectors.append(
                np.zeros((logmel_vectors.shape[0], 1), dtype=np.float32)
            )
        else:
            band_vectors.append(logmel_vectors[:, mask])
    return band_vectors


def compute_direct_band_drifts(band_vectors, smooth_k):
    """Input: mono waveform, window times, and band vectors; see the signature for specific arguments.
    Processing: compute PCEN/log-Mel summaries, adjacent-window drift, and summary descriptors.
    Output: spectral vectors, band drift arrays, or descriptor matrix.
    """
    drifts = []
    for Xb in band_vectors:
        d = np.zeros(Xb.shape[0], dtype=float)
        for t in range(1, Xb.shape[0]):
            d[t] = np.linalg.norm(Xb[t] - Xb[t - 1])
        ds = moving_average(d, smooth_k)
        drifts.append(ds)
    return drifts


def compute_band_frequency_descriptors(
    band_vectors, k_mad_activity=2.0, normalize_subband=True, use_activity=True
):
    """Input: mono waveform, window times, and band vectors; see the signature for specific arguments.
    Processing: compute PCEN/log-Mel summaries, adjacent-window drift, and summary descriptors.
    Output: spectral vectors, band drift arrays, or descriptor matrix.

    Stable per-band descriptor.

    Each band gets a compact summary vector built from:
    - median energy per sub-bin
    - variability per sub-bin (MAD)
    - global stats: median, MAD, max, flatness proxy, activity proportion,
      burstiness, temporal sparsity, mean pairwise sub-bin correlation
    """
    descs = []

    for Xb in band_vectors:
        X = np.asarray(Xb, dtype=float)
        if X.ndim == 1:
            X = X[:, None]

        if normalize_subband:
            Xn = np.zeros_like(X)
            for j in range(X.shape[1]):
                col = X[:, j]
                Xn[:, j] = robust_zscore(col)
            X_work = Xn
        else:
            X_work = X

        med_sub = np.median(X_work, axis=0)
        mad_sub = np.median(np.abs(X_work - med_sub[None, :]), axis=0)

        collapsed = np.mean(X_work, axis=1)
        collapsed_z = robust_zscore(collapsed)
        thr = np.median(collapsed_z) + k_mad_activity * robust_mad(collapsed_z)
        active = (collapsed_z >= thr).astype(float)

        if use_activity:
            activity_prop = float(np.mean(active))
            runs = np.diff(np.pad(active.astype(int), (1, 1), mode="constant"))
            starts = np.where(runs == 1)[0]
            ends = np.where(runs == -1)[0]
            durations = ends - starts
            mean_run = float(np.mean(durations)) if len(durations) > 0 else 0.0
            n_runs = float(len(durations))
        else:
            activity_prop = 0.0
            mean_run = 0.0
            n_runs = 0.0

        global_median = float(np.median(collapsed))
        global_mad = float(robust_mad(collapsed))
        global_max = float(np.max(collapsed))
        global_mean = float(np.mean(collapsed))
        global_std = float(np.std(collapsed))
        burstiness = float(global_std / (abs(global_mean) + 1e-12))
        sparsity = float(np.mean(np.abs(collapsed_z) < 0.5))

        # Flatness-like ratio over positive-shifted data
        xp = collapsed - np.min(collapsed) + 1e-6
        gmean = float(np.exp(np.mean(np.log(xp))))
        amean = float(np.mean(xp))
        flatness = float(gmean / (amean + 1e-12))

        # Correlation between sub-bins inside the band
        if X_work.shape[1] >= 2:
            C = np.corrcoef(X_work.T)
            vals = C[np.triu_indices_from(C, k=1)]
            vals = vals[np.isfinite(vals)]
            mean_subcorr = float(np.mean(vals)) if len(vals) > 0 else 0.0
        else:
            mean_subcorr = 1.0

        desc = np.concatenate(
            [
                med_sub,
                mad_sub,
                np.array(
                    [
                        global_median,
                        global_mad,
                        global_max,
                        burstiness,
                        sparsity,
                        flatness,
                        activity_prop,
                        mean_run,
                        n_runs,
                        mean_subcorr,
                    ],
                    dtype=float,
                ),
            ]
        )
        descs.append(desc)

    max_len = max(len(d) for d in descs)
    out = []
    for d in descs:
        if len(d) < max_len:
            pad = np.zeros(max_len - len(d), dtype=float)
            d = np.concatenate([d, pad])
        out.append(d)
    return np.stack(out, axis=0)


# ========================================================================
# Hybrid similarity and grouping
# INPUT: band drift trajectories and spectral descriptors.
# PROCESSING: compare timing, profile, and adjacency, then group related bands.
# OUTPUT: similarity matrices or band-group labels.
# ========================================================================


def jaccard_similarity_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Input: band drift trajectories and spectral descriptors; see the signature for specific arguments.
    Processing: compare timing, profile, and adjacency, then group related bands.
    Output: similarity matrices or band-group labels.
    """
    inter = np.sum((a > 0) & (b > 0))
    union = np.sum((a > 0) | (b > 0))
    if union == 0:
        return 0.0
    return float(inter / union)


def lagged_corr_similarity(a: np.ndarray, b: np.ndarray, max_lag: int = 2) -> float:
    """Input: band drift trajectories and spectral descriptors; see the signature for specific arguments.
    Processing: compare timing, profile, and adjacency, then group related bands.
    Output: similarity matrices or band-group labels.
    """
    a = robust_zscore(a)
    b = robust_zscore(b)
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: len(aa)]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b

        if len(aa) < 3:
            continue

        c = np.corrcoef(aa, bb)[0, 1]
        if np.isnan(c):
            c = 0.0
        best = max(best, c)

    return float(max(best, 0.0))


def build_time_similarity_matrix(drifts, evidences, mode="jaccard_binary", max_lag=2):
    """Input: band drift trajectories and spectral descriptors; see the signature for specific arguments.
    Processing: compare timing, profile, and adjacency, then group related bands.
    Output: similarity matrices or band-group labels.
    """
    n = len(drifts)
    S = np.eye(n, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if mode == "jaccard_binary":
                sim = jaccard_similarity_binary(evidences[i], evidences[j])
            elif mode == "lagged_corr":
                sim = lagged_corr_similarity(drifts[i], drifts[j], max_lag=max_lag)
            else:
                raise ValueError("Unsupported similarity mode")

            S[i, j] = sim
            S[j, i] = sim

    return S


def build_frequency_similarity_matrix(band_descs):
    """Input: band drift trajectories and spectral descriptors; see the signature for specific arguments.
    Processing: compare timing, profile, and adjacency, then group related bands.
    Output: similarity matrices or band-group labels.
    """
    n = band_descs.shape[0]
    S = np.eye(n, dtype=float)

    # robust standardization across bands
    X = np.asarray(band_descs, dtype=float)
    Xs = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        Xs[:, j] = robust_zscore(col)

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_sim(Xs[i], Xs[j])
            S[i, j] = sim
            S[j, i] = sim
    return S


def build_adjacency_similarity_matrix(n_bands, tau=2.5):
    """Input: band drift trajectories and spectral descriptors; see the signature for specific arguments.
    Processing: compare timing, profile, and adjacency, then group related bands.
    Output: similarity matrices or band-group labels.
    """
    S = np.eye(n_bands, dtype=float)
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            d = abs(i - j)
            sim = float(np.exp(-d / tau))
            S[i, j] = sim
            S[j, i] = sim
    return S


def combine_similarity_matrices(S_time, S_freq, S_adj, alpha=0.7, beta=0.2, gamma=0.1):
    """Input: band drift trajectories and spectral descriptors; see the signature for specific arguments.
    Processing: compare timing, profile, and adjacency, then group related bands.
    Output: similarity matrices or band-group labels.
    """
    wsum = alpha + beta + gamma
    if wsum <= 0:
        raise ValueError("alpha + beta + gamma must be > 0")
    St = alpha / wsum
    Sf = beta / wsum
    Sa = gamma / wsum
    S = St * S_time + Sf * S_freq + Sa * S_adj
    S = np.clip(S, 0.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S


def cluster_bands_from_similarity(
    S, linkage_method="average", n_groups=None, distance_threshold=0.55
):
    """Input: band drift trajectories and spectral descriptors; see the signature for specific arguments.
    Processing: compare timing, profile, and adjacency, then group related bands.
    Output: similarity matrices or band-group labels.
    """
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=linkage_method)

    if n_groups is not None:
        labels = fcluster(Z, n_groups, criterion="maxclust")
    else:
        labels = fcluster(Z, distance_threshold, criterion="distance")

    return Z, labels


def aggregate_cluster_drifts(drifts, cluster_labels, agg="median"):
    """Input: band drift trajectories and spectral descriptors; see the signature for specific arguments.
    Processing: compare timing, profile, and adjacency, then group related bands.
    Output: similarity matrices or band-group labels.
    """
    uniq = sorted(np.unique(cluster_labels))
    cluster_drifts = []
    cluster_members = []

    for c in uniq:
        idx = np.where(cluster_labels == c)[0]
        M = np.stack([drifts[i] for i in idx], axis=0)

        if agg == "median":
            cd = np.median(M, axis=0)
        elif agg == "mean":
            cd = np.mean(M, axis=0)
        elif agg == "max":
            cd = np.max(M, axis=0)
        else:
            raise ValueError("agg must be median, mean, or max")

        cluster_drifts.append(cd)
        cluster_members.append(idx.tolist())

    return uniq, cluster_drifts, cluster_members


def reorder_by_dendrogram(Z):
    """Input: band drift trajectories and spectral descriptors; see the signature for specific arguments.
    Processing: compare timing, profile, and adjacency, then group related bands.
    Output: similarity matrices or band-group labels.
    """
    dend = dendrogram(Z, no_plot=True)
    return dend["leaves"]


def cluster_band_ranges(edges_hz, cluster_labels):
    """Input: band drift trajectories and spectral descriptors; see the signature for specific arguments.
    Processing: compare timing, profile, and adjacency, then group related bands.
    Output: similarity matrices or band-group labels.
    """
    uniq = sorted(np.unique(cluster_labels))
    out = {}
    for c in uniq:
        idx = np.where(cluster_labels == c)[0]
        lo = edges_hz[np.min(idx)]
        hi = edges_hz[np.max(idx) + 1]
        out[c] = (lo, hi, idx.tolist())
    return out


# ========================================================================
# Diagnostic event curve
# INPUT: representations, drift curves, or peaks.
# PROCESSING: compare candidate activity with a signal-derived diagnostic curve.
# OUTPUT: diagnostic curves or agreement statistics.
# ========================================================================


def compute_global_repr_vectors(y, sr, win_starts, win_ends, n_mels, fmin, fmax):
    """Input: representations, drift curves, or peaks; see the signature for specific arguments.
    Processing: compare candidate activity with a signal-derived diagnostic curve.
    Output: diagnostic curves or agreement statistics.
    """
    return compute_window_repr_vectors(
        y=y,
        sr=sr,
        win_starts=win_starts,
        win_ends=win_ends,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        use_pcen=USE_PCEN,
    )


def compute_spectrogram_event_curve(global_repr_vectors, smooth_k=7):
    """Input: representations, drift curves, or peaks; see the signature for specific arguments.
    Processing: compare candidate activity with a signal-derived diagnostic curve.
    Output: diagnostic curves or agreement statistics.
    """
    d = np.zeros(global_repr_vectors.shape[0], dtype=float)
    for t in range(1, global_repr_vectors.shape[0]):
        diff = global_repr_vectors[t] - global_repr_vectors[t - 1]
        d[t] = np.linalg.norm(diff)
    d = moving_average(d, smooth_k)
    return d


def match_peaks_with_tolerance(ref_peaks, test_peaks, tol=2):
    """Input: representations, drift curves, or peaks; see the signature for specific arguments.
    Processing: compare candidate activity with a signal-derived diagnostic curve.
    Output: diagnostic curves or agreement statistics.
    """
    ref_peaks = sorted(ref_peaks)
    test_peaks = sorted(test_peaks)

    matched_ref = set()
    matched_test = set()

    for i, rp in enumerate(ref_peaks):
        best_j = None
        best_dist = None
        for j, tp in enumerate(test_peaks):
            if j in matched_test:
                continue
            dist = abs(rp - tp)
            if dist <= tol and (best_dist is None or dist < best_dist):
                best_j = j
                best_dist = dist
        if best_j is not None:
            matched_ref.add(i)
            matched_test.add(best_j)

    tp = len(matched_test)
    fp = len(test_peaks) - tp
    fn = len(ref_peaks) - tp

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def best_lagged_corr(a, b, max_lag=2):
    """Input: representations, drift curves, or peaks; see the signature for specific arguments.
    Processing: compare candidate activity with a signal-derived diagnostic curve.
    Output: diagnostic curves or agreement statistics.
    """
    a = robust_zscore(a)
    b = robust_zscore(b)

    best_corr = -1.0
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: len(aa)]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b

        if len(aa) < 3:
            continue

        c = np.corrcoef(aa, bb)[0, 1]
        if np.isnan(c):
            c = 0.0

        if c > best_corr:
            best_corr = c
            best_lag = lag

    return float(best_corr), int(best_lag)


def compute_cluster_vs_event_metrics(
    cluster_drifts,
    event_curve,
    max_lag=2,
    drift_peak_k_mad=2.5,
    drift_peak_sep=2,
    event_peak_k_mad=2.0,
    event_peak_sep=2,
    peak_match_tol=2,
):
    """Input: representations, drift curves, or peaks; see the signature for specific arguments.
    Processing: compare candidate activity with a signal-derived diagnostic curve.
    Output: diagnostic curves or agreement statistics.
    """
    metrics = []

    ev_z, ev_bin, ev_peaks, ev_thr = detect_peaks_from_drift(
        event_curve, k_mad=event_peak_k_mad, min_sep_windows=event_peak_sep
    )

    for i, cd in enumerate(cluster_drifts):
        x = robust_zscore(cd)
        y = robust_zscore(event_curve)

        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            pear = 0.0
            spear = 0.0
        else:
            pear = pearsonr(x, y)[0]
            spear = spearmanr(x, y).statistic
            if np.isnan(pear):
                pear = 0.0
            if np.isnan(spear):
                spear = 0.0

        lagcorr, bestlag = best_lagged_corr(cd, event_curve, max_lag=max_lag)

        cd_z, cd_bin, cd_peaks, cd_thr = detect_peaks_from_drift(
            cd, k_mad=drift_peak_k_mad, min_sep_windows=drift_peak_sep
        )

        peak_scores = match_peaks_with_tolerance(
            ref_peaks=ev_peaks, test_peaks=cd_peaks, tol=peak_match_tol
        )

        metrics.append(
            {
                "cluster_index": i,
                "pearson_r": float(pear),
                "spearman_rho": float(spear),
                "max_lag_corr": float(lagcorr),
                "best_lag_windows": int(bestlag),
                "n_cluster_peaks": int(len(cd_peaks)),
                "n_event_peaks": int(len(ev_peaks)),
                "peak_precision": float(peak_scores["precision"]),
                "peak_recall": float(peak_scores["recall"]),
                "peak_f1": float(peak_scores["f1"]),
                "cluster_peaks": cd_peaks,
                "event_peaks": ev_peaks,
                "cluster_drift_z": cd_z,
                "event_curve_z": ev_z,
            }
        )

    return metrics


# ========================================================================
# Event segmentation and boxes
# INPUT: group drift arrays, window indices, and frequency edges.
# PROCESSING: find active runs and merge time-overlapping proposals within a group.
# OUTPUT: state information, event regions, or candidate box dictionaries.
# ========================================================================


def find_peak_support(z, peak_idx, thr, expand_windows=0):
    """Input: group drift arrays, window indices, and frequency edges; see the signature for specific arguments.
    Processing: find active runs and merge time-overlapping proposals within a group.
    Output: state information, event regions, or candidate box dictionaries.
    """
    n = len(z)

    left = peak_idx
    while left > 0 and z[left] >= thr:
        left -= 1
    if z[left] < thr and left < peak_idx:
        left += 1

    right = peak_idx
    while right < n - 1 and z[right] >= thr:
        right += 1
    if z[right] < thr and right > peak_idx:
        right -= 1

    left = max(0, left - expand_windows)
    right = min(n - 1, right + expand_windows)

    if right < left:
        left = max(0, peak_idx - 1)
        right = min(n - 1, peak_idx + 1)

    return left, right


def merge_overlapping_time_boxes(boxes):
    """Input: group drift arrays, window indices, and frequency edges; see the signature for specific arguments.
    Processing: find active runs and merge time-overlapping proposals within a group.
    Output: state information, event regions, or candidate box dictionaries.
    """
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: (b["cluster_id"], b["t0"], b["t1"]))
    merged = [boxes[0].copy()]

    for b in boxes[1:]:
        prev = merged[-1]
        same_cluster = prev["cluster_id"] == b["cluster_id"]
        overlapping = b["t0"] <= prev["t1"]

        if same_cluster and overlapping:
            prev["t1"] = max(prev["t1"], b["t1"])
        else:
            merged.append(b.copy())

    return merged


def smooth_binary_labels(labels, k=3):
    """Input: group drift arrays, window indices, and frequency edges; see the signature for specific arguments.
    Processing: find active runs and merge time-overlapping proposals within a group.
    Output: state information, event regions, or candidate box dictionaries.

    Simple majority filter for binary labels.
    """
    labels = np.asarray(labels, dtype=int)
    if k <= 1:
        return labels.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(labels, (pad, pad), mode="edge")
    out = np.zeros_like(labels)
    for i in range(len(labels)):
        window = padded[i : i + k]
        out[i] = 1 if np.mean(window) >= 0.5 else 0
    return out


def binary_labels_to_regions(labels, merge_gap_windows=2, min_event_windows=2):
    """Input: group drift arrays, window indices, and frequency edges; see the signature for specific arguments.
    Processing: find active runs and merge time-overlapping proposals within a group.
    Output: state information, event regions, or candidate box dictionaries.

    Convert 0/1 labels into contiguous event regions.
    """
    labels = np.asarray(labels, dtype=int)
    regions = []
    in_event = False
    start = None

    for i, lab in enumerate(labels):
        if lab == 1 and not in_event:
            in_event = True
            start = i
        elif lab == 0 and in_event:
            regions.append([start, i - 1])
            in_event = False

    if in_event:
        regions.append([start, len(labels) - 1])

    # Merge nearby regions
    if merge_gap_windows is not None and merge_gap_windows >= 0 and len(regions) > 1:
        merged = [regions[0]]
        for st, en in regions[1:]:
            prev = merged[-1]
            gap = st - prev[1] - 1
            if gap <= merge_gap_windows:
                prev[1] = en
            else:
                merged.append([st, en])
        regions = merged

    # Drop very short regions
    regions = [[st, en] for st, en in regions if (en - st + 1) >= min_event_windows]
    return regions


def fit_gmm_event_state(
    drift,
    min_event_posterior=0.50,
    smooth_labels_k=3,
    merge_gap_windows=2,
    min_event_windows=2,
    min_event_area_q=None,
    use_log_drift=False,
    random_state=0,
):
    """Input: group drift arrays, window indices, and frequency edges; see the signature for specific arguments.
    Processing: find active runs and merge time-overlapping proposals within a group.
    Output: state information, event regions, or candidate box dictionaries.

    Per-cluster 2-state segmentation using a Gaussian mixture model.

    The lower-mean component is interpreted as background and the higher-mean
    component as event. This estimates the non-event/event state from each
    post-cluster drift curve rather than using a manually chosen drift threshold.
    """
    raw = np.asarray(drift, dtype=float)

    # Optional compression for highly spiky clusters.
    if use_log_drift:
        shift = np.min(raw)
        x = np.log1p(raw - shift)
    else:
        x = raw.copy()

    if len(x) < 5 or np.std(x) < 1e-12:
        labels = np.zeros(len(x), dtype=int)
        return {
            "labels": labels,
            "regions": [],
            "threshold": float(np.inf),
            "means": np.array([float(np.mean(x)), float(np.mean(x))]),
            "weights": np.array([1.0, 0.0]),
            "event_component": 1,
            "posterior": np.zeros(len(x), dtype=float),
            "raw_event_areas": [],
        }

    X = x.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=random_state,
        reg_covar=1e-6,
        n_init=5,
    )
    gmm.fit(X)

    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()
    event_component = int(np.argmax(means))
    bg_component = 1 - event_component

    posterior = gmm.predict_proba(X)[:, event_component]
    labels = (posterior >= min_event_posterior).astype(int)
    labels = smooth_binary_labels(labels, k=smooth_labels_k)

    regions = binary_labels_to_regions(
        labels,
        merge_gap_windows=merge_gap_windows,
        min_event_windows=min_event_windows,
    )

    # Optional region-level area filtering. This is intentionally weak by default.
    raw_event_areas = []
    for st, en in regions:
        local_base = min(raw[st], raw[en]) if en > st else raw[st]
        area = float(np.sum(np.maximum(raw[st : en + 1] - local_base, 0.0)))
        raw_event_areas.append(area)

    if min_event_area_q is not None and len(regions) > 0:
        area_gate = float(np.quantile(raw_event_areas, min_event_area_q))
        kept_regions = []
        kept_areas = []
        for region, area in zip(regions, raw_event_areas):
            if area >= area_gate:
                kept_regions.append(region)
                kept_areas.append(area)
        regions = kept_regions
        raw_event_areas = kept_areas

        labels2 = np.zeros_like(labels)
        for st, en in regions:
            labels2[st : en + 1] = 1
        labels = labels2

    # Approximate threshold in the same space used by the GMM.
    # This is diagnostic only; boxes are built from posterior labels/regions.
    threshold = float(np.mean(means))
    if use_log_drift:
        threshold = float(np.expm1(threshold) + np.min(raw))

    return {
        "labels": labels,
        "regions": regions,
        "threshold": threshold,
        "means": means,
        "weights": weights,
        "event_component": event_component,
        "bg_component": bg_component,
        "posterior": posterior,
        "raw_event_areas": raw_event_areas,
    }


def cluster_peaks_to_boxes(
    cluster_drifts,
    cluster_members,
    cluster_labels_unique,
    edges_hz,
    win_starts,
    win_ends,
    peak_k_mad,
    peak_min_sep_windows,
    expand_windows=0,
    min_duration_windows=1,
):
    """Input: group drift arrays, window indices, and frequency edges; see the signature for specific arguments.
    Processing: find active runs and merge time-overlapping proposals within a group.
    Output: state information, event regions, or candidate box dictionaries.
    """
    all_boxes = []
    peak_info = []

    print("\n=== POST-CLUSTER GMM STATE SEGMENTATION ===")

    for cd, members, c in zip(cluster_drifts, cluster_members, cluster_labels_unique):
        f0 = float(edges_hz[min(members)])
        f1 = float(edges_hz[max(members) + 1])

        res = fit_gmm_event_state(
            cd,
            min_event_posterior=GMM_MIN_EVENT_POSTERIOR,
            smooth_labels_k=GMM_SMOOTH_LABELS_K,
            merge_gap_windows=GMM_MERGE_GAP_WINDOWS,
            min_event_windows=max(GMM_MIN_EVENT_WINDOWS, min_duration_windows),
            min_event_area_q=GMM_MIN_EVENT_AREA_Q,
            use_log_drift=GMM_USE_LOG_DRIFT,
            random_state=GMM_RANDOM_STATE,
        )

        labels = res["labels"]
        regions = res["regions"]

        cluster_boxes = []
        peaks = []
        for st, en in regions:
            segment = cd[st : en + 1]
            peak_idx = int(st + np.argmax(segment))
            peaks.append(peak_idx)

            box = {
                "cluster_id": int(c),
                "t0": float(win_starts[st]),
                "t1": float(win_ends[en]),
                "f0": f0,
                "f1": f1,
                "peak_idx": int(peak_idx),
                "support_left": int(st),
                "support_right": int(en),
                "members": members,
            }
            cluster_boxes.append(box)

        cluster_boxes = merge_overlapping_time_boxes(cluster_boxes)
        all_boxes.extend(cluster_boxes)

        means_str = np.array2string(res["means"], precision=3, separator=", ")
        weights_str = np.array2string(res["weights"], precision=3, separator=", ")
        event_fraction = float(np.mean(labels)) if len(labels) else 0.0
        print(
            f"Cluster {c} | {int(f0):4d}-{int(f1):5d} Hz | "
            f"mode=gmm_state | thr≈{res['threshold']:.3f} | "
            f"means={means_str} | weights={weights_str} | "
            f"event_fraction={event_fraction:.3f} | regions={len(regions)} | boxes={len(cluster_boxes)}"
        )

        peak_info.append(
            {
                "cluster_id": int(c),
                "freq_lo": f0,
                "freq_hi": f1,
                "n_peaks": int(len(peaks)),
                "n_boxes": int(len(cluster_boxes)),
                "peaks": peaks,
                "z": labels.astype(float),
                "thr": res["threshold"],
                "labels": labels,
                "posterior": res["posterior"],
                "regions": regions,
                "means": res["means"],
                "weights": res["weights"],
            }
        )

    return all_boxes, peak_info


def boxes_to_global_activity_curve(boxes, n_windows, win_starts, win_ends, smooth_k=1):
    """Input: group drift arrays, window indices, and frequency edges; see the signature for specific arguments.
    Processing: find active runs and merge time-overlapping proposals within a group.
    Output: state information, event regions, or candidate box dictionaries.
    """
    activity = np.zeros(n_windows, dtype=float)

    for b in boxes:
        overlap = (win_starts < b["t1"]) & (win_ends > b["t0"])
        activity[overlap] = 1.0

    if smooth_k > 1:
        activity = moving_average(activity, smooth_k)

    return activity


# ========================================================================
# Notebook evaluation metrics
# INPUT: predicted boxes and reference boxes in seconds and Hz.
# PROCESSING: measure overlap, greedily match pairs, and calculate coverage.
# OUTPUT: IoU, coverage, or temporal activity statistics.
# ========================================================================


def interval_iou(a0, a1, b0, b1):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.
    """
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def box_iou(a, b):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.
    """
    it = max(0.0, min(a["t1"], b["t1"]) - max(a["t0"], b["t0"]))
    ifr = max(0.0, min(a["f1"], b["f1"]) - max(a["f0"], b["f0"]))
    inter = it * ifr
    area_a = max(0.0, a["t1"] - a["t0"]) * max(0.0, a["f1"] - a["f0"])
    area_b = max(0.0, b["t1"] - b["t0"]) * max(0.0, b["f1"] - b["f0"])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _safe_div(n, d):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.
    """
    return float(n / d) if d not in (0, 0.0, None) and not pd.isna(d) else 0.0


def _box_duration(b):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.
    """
    return max(0.0, float(b.get("t1", 0.0)) - float(b.get("t0", 0.0)))


def _box_bandwidth(b):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.
    """
    return max(0.0, float(b.get("f1", 0.0)) - float(b.get("f0", 0.0)))


def _box_area(b):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.
    """
    return _box_duration(b) * _box_bandwidth(b)


def _intersection_length(a0, a1, b0, b1):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.
    """
    return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))


def _union_length_intervals(intervals):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.

    Union length of [start, end] intervals.
    """
    clean = []
    for lo, hi in intervals:
        lo = float(lo)
        hi = float(hi)
        if hi > lo:
            clean.append((lo, hi))
    if not clean:
        return 0.0
    clean.sort()
    total = 0.0
    cur_lo, cur_hi = clean[0]
    for lo, hi in clean[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    total += cur_hi - cur_lo
    return float(total)


def _pair_alignment_metrics(pred_box, gt_box):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.

    Returns time, frequency, and 2D box alignment for one prediction/GT pair.
    """
    t_inter = _intersection_length(
        pred_box["t0"], pred_box["t1"], gt_box["t0"], gt_box["t1"]
    )
    f_inter = _intersection_length(
        pred_box["f0"], pred_box["f1"], gt_box["f0"], gt_box["f1"]
    )

    pred_t = _box_duration(pred_box)
    gt_t = _box_duration(gt_box)
    pred_f = _box_bandwidth(pred_box)
    gt_f = _box_bandwidth(gt_box)

    t_union = pred_t + gt_t - t_inter
    f_union = pred_f + gt_f - f_inter

    inter_area = t_inter * f_inter
    pred_area = _box_area(pred_box)
    gt_area = _box_area(gt_box)
    union_area = pred_area + gt_area - inter_area

    pred_centre_t = 0.5 * (float(pred_box["t0"]) + float(pred_box["t1"]))
    gt_centre_t = 0.5 * (float(gt_box["t0"]) + float(gt_box["t1"]))
    pred_centre_f = 0.5 * (float(pred_box["f0"]) + float(pred_box["f1"]))
    gt_centre_f = 0.5 * (float(gt_box["f0"]) + float(gt_box["f1"]))

    return {
        "temporal_iou": _safe_div(t_inter, t_union),
        "frequency_iou": _safe_div(f_inter, f_union),
        "iou": _safe_div(inter_area, union_area),
        "temporal_intersection_s": float(t_inter),
        "frequency_intersection_hz": float(f_inter),
        "temporal_recall_coverage": _safe_div(t_inter, gt_t),
        "frequency_recall_coverage": _safe_div(f_inter, gt_f),
        "area_recall_coverage": _safe_div(inter_area, gt_area),
        "temporal_centre_error_s": abs(pred_centre_t - gt_centre_t),
        "frequency_centre_error_hz": abs(pred_centre_f - gt_centre_f),
    }


def _make_temporal_activity_curve(
    boxes, win_starts=None, win_ends=None, duration_s=None, hop_s=None
):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.

    Converts boxes to a 0/1 temporal activity curve.
    This is used for time-only accuracy/precision/recall. It ignores frequency.
    """
    if win_starts is None or win_ends is None:
        if duration_s is None or duration_s <= 0:
            max_t = 0.0
            for b in boxes:
                max_t = max(max_t, float(b.get("t1", 0.0)))
            duration_s = max_t
        hop_s = hop_s or TEMPORAL_EVAL_HOP_S
        if duration_s <= 0 or hop_s <= 0:
            return np.zeros(0, dtype=int)
        win_starts = np.arange(0.0, duration_s + 1e-9, hop_s)
        win_ends = win_starts + hop_s

    activity = np.zeros(len(win_starts), dtype=int)
    for b in boxes:
        overlap = (win_starts < float(b["t1"])) & (win_ends > float(b["t0"]))
        activity[overlap] = 1
    return activity


def _binary_curve_metrics(pred_curve, gt_curve):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.
    """
    n = int(min(len(pred_curve), len(gt_curve)))
    if n <= 0:
        return {
            "temporal_n_windows": 0,
            "temporal_tp_windows": 0,
            "temporal_fp_windows": 0,
            "temporal_fn_windows": 0,
            "temporal_tn_windows": 0,
            "temporal_accuracy": None,
            "temporal_precision": None,
            "temporal_recall": None,
            "temporal_hit_rate": None,
            "temporal_f1": None,
            "temporal_iou": None,
            "temporal_balanced_accuracy": None,
        }

    pred = np.asarray(pred_curve[:n], dtype=bool)
    gt = np.asarray(gt_curve[:n], dtype=bool)

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    jaccard = _safe_div(tp, tp + fp + fn)
    acc = _safe_div(tp + tn, n)
    bal_acc = 0.5 * (recall + specificity)

    return {
        "temporal_n_windows": n,
        "temporal_tp_windows": tp,
        "temporal_fp_windows": fp,
        "temporal_fn_windows": fn,
        "temporal_tn_windows": tn,
        "temporal_accuracy": acc,
        "temporal_precision": precision,
        "temporal_recall": recall,
        "temporal_hit_rate": recall,
        "temporal_f1": f1,
        "temporal_iou": jaccard,
        "temporal_balanced_accuracy": bal_acc,
    }


def evaluate_predicted_boxes(
    pred_boxes,
    gt_boxes,
    iou_thr=0.3,
    win_starts=None,
    win_ends=None,
    duration_s=None,
    temporal_eval_hop_s=None,
    temporal_hit_iou_thr=None,
    frequency_hit_iou_thr=None,
):
    """Input: predicted boxes and reference boxes in seconds and Hz; see the signature for specific arguments.
    Processing: measure overlap, greedily match pairs, and calculate coverage.
    Output: IoU, coverage, or temporal activity statistics.

    Evaluates predicted time-frequency boxes against Raven GT boxes.

    Important interpretation:
    - strict precision/recall/F1 use one-to-one IoU matching and count unmatched
      predictions as false positives.
    - GT coverage/hit metrics ask whether annotated GT boxes were found and are
      less punitive when the detector produces additional boxes not present in GT.
    - temporal_* metrics are time-only binary activity metrics over analysis windows.
      They are useful for temporal alignment but can be biased by incomplete GT.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_gt == 0:
        return None

    temporal_hit_iou_thr = (
        GT_TEMPORAL_HIT_IOU_THRESHOLD
        if temporal_hit_iou_thr is None
        else temporal_hit_iou_thr
    )
    frequency_hit_iou_thr = (
        GT_FREQUENCY_HIT_IOU_THRESHOLD
        if frequency_hit_iou_thr is None
        else frequency_hit_iou_thr
    )
    temporal_eval_hop_s = (
        TEMPORAL_EVAL_HOP_S if temporal_eval_hop_s is None else temporal_eval_hop_s
    )

    # Pairwise metrics for every prediction/GT pair.
    pair_rows = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            m = _pair_alignment_metrics(pb, gb)
            pair_rows.append({"pred_idx": pi, "gt_idx": gi, **m})

    # Strict one-to-one matching by descending 2D IoU.
    matched_pred = set()
    matched_gt = set()
    matched_rows = []
    for r in sorted(pair_rows, key=lambda x: x["iou"], reverse=True):
        if r["iou"] < iou_thr:
            break
        if r["pred_idx"] in matched_pred or r["gt_idx"] in matched_gt:
            continue
        matched_pred.add(r["pred_idx"])
        matched_gt.add(r["gt_idx"])
        matched_rows.append(r)

    tp = len(matched_rows)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    def mean_from(rows, key):
        vals = [
            float(r[key])
            for r in rows
            if r.get(key) is not None and not pd.isna(r.get(key))
        ]
        return float(np.mean(vals)) if vals else 0.0

    def median_from(rows, key):
        vals = [
            float(r[key])
            for r in rows
            if r.get(key) is not None and not pd.isna(r.get(key))
        ]
        return float(np.median(vals)) if vals else 0.0

    # Best prediction per GT. This is deliberately not one-to-one, because many
    # predicted boxes may map to the same GT or split one GT into smaller regions.
    best_by_gt = []
    for gi, gb in enumerate(gt_boxes):
        rows = [r for r in pair_rows if r["gt_idx"] == gi]
        if rows:
            best_iou_row = max(rows, key=lambda x: x["iou"])
            best_temporal_row = max(rows, key=lambda x: x["temporal_iou"])
            best_frequency_row = max(rows, key=lambda x: x["frequency_iou"])
        else:
            best_iou_row = {
                "iou": 0.0,
                "temporal_iou": 0.0,
                "frequency_iou": 0.0,
                "area_recall_coverage": 0.0,
                "temporal_centre_error_s": None,
                "frequency_centre_error_hz": None,
            }
            best_temporal_row = {"temporal_iou": 0.0}
            best_frequency_row = {"frequency_iou": 0.0}

        # Union-based GT coverage: this is better when one GT is split into
        # several predicted boxes.
        t_intervals = []
        f_intervals = []
        for pb in pred_boxes:
            t0 = max(float(pb["t0"]), float(gb["t0"]))
            t1 = min(float(pb["t1"]), float(gb["t1"]))
            f0 = max(float(pb["f0"]), float(gb["f0"]))
            f1 = min(float(pb["f1"]), float(gb["f1"]))
            if t1 > t0:
                t_intervals.append((t0, t1))
            if t1 > t0 and f1 > f0:
                f_intervals.append((f0, f1))

        temporal_coverage = _safe_div(
            _union_length_intervals(t_intervals), _box_duration(gb)
        )
        frequency_coverage = _safe_div(
            _union_length_intervals(f_intervals), _box_bandwidth(gb)
        )

        any_overlap_hit = any(
            (r["temporal_intersection_s"] > 0 and r["frequency_intersection_hz"] > 0)
            for r in rows
        )

        best_by_gt.append(
            {
                "gt_idx": gi,
                "best_iou": float(best_iou_row.get("iou", 0.0)),
                "best_temporal_iou": float(best_temporal_row.get("temporal_iou", 0.0)),
                "best_frequency_iou": float(
                    best_frequency_row.get("frequency_iou", 0.0)
                ),
                "best_area_recall_coverage": float(
                    best_iou_row.get("area_recall_coverage", 0.0)
                ),
                "temporal_coverage": float(temporal_coverage),
                "frequency_coverage": float(frequency_coverage),
                "temporal_centre_error_s": best_iou_row.get("temporal_centre_error_s"),
                "frequency_centre_error_hz": best_iou_row.get(
                    "frequency_centre_error_hz"
                ),
                "hit_iou": float(best_iou_row.get("iou", 0.0)) >= iou_thr,
                "hit_temporal": float(best_temporal_row.get("temporal_iou", 0.0))
                >= temporal_hit_iou_thr,
                "hit_frequency": float(best_frequency_row.get("frequency_iou", 0.0))
                >= frequency_hit_iou_thr,
                "hit_any_overlap": bool(any_overlap_hit),
            }
        )

    pred_curve = _make_temporal_activity_curve(
        pred_boxes,
        win_starts=win_starts,
        win_ends=win_ends,
        duration_s=duration_s,
        hop_s=temporal_eval_hop_s,
    )
    gt_curve = _make_temporal_activity_curve(
        gt_boxes,
        win_starts=win_starts,
        win_ends=win_ends,
        duration_s=duration_s,
        hop_s=temporal_eval_hop_s,
    )
    temporal_metrics = _binary_curve_metrics(pred_curve, gt_curve)

    # Conservative precision penalises all extra predicted boxes. This is often
    # useful, but can be harsh when GT is incomplete.
    strict_precision = precision

    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "hit_rate": recall,
        "f1": f1,
        "mean_iou": mean_from(matched_rows, "iou"),
        "median_iou": median_from(matched_rows, "iou"),
        "mean_temporal_iou": mean_from(matched_rows, "temporal_iou"),
        "mean_frequency_iou": mean_from(matched_rows, "frequency_iou"),
        "mean_temporal_centre_error_s": mean_from(
            matched_rows, "temporal_centre_error_s"
        ),
        "mean_frequency_centre_error_hz": mean_from(
            matched_rows, "frequency_centre_error_hz"
        ),
        "strict_precision": strict_precision,
        "pred_to_gt_box_ratio": _safe_div(n_pred, n_gt),
        "n_unmatched_predictions": fp,
        "n_unmatched_gt": fn,
        "best_iou_mean": mean_from(best_by_gt, "best_iou"),
        "best_temporal_iou_mean": mean_from(best_by_gt, "best_temporal_iou"),
        "best_frequency_iou_mean": mean_from(best_by_gt, "best_frequency_iou"),
        "temporal_coverage_mean": mean_from(best_by_gt, "temporal_coverage"),
        "frequency_coverage_mean": mean_from(best_by_gt, "frequency_coverage"),
        "area_recall_coverage_mean": mean_from(best_by_gt, "best_area_recall_coverage"),
        "iou_hit_rate": _safe_div(sum(bool(x["hit_iou"]) for x in best_by_gt), n_gt),
        "box_temporal_hit_rate": _safe_div(
            sum(bool(x["hit_temporal"]) for x in best_by_gt), n_gt
        ),
        "box_frequency_hit_rate": _safe_div(
            sum(bool(x["hit_frequency"]) for x in best_by_gt), n_gt
        ),
        "any_overlap_hit_rate": _safe_div(
            sum(bool(x["hit_any_overlap"]) for x in best_by_gt), n_gt
        ),
        **temporal_metrics,
    }

    return out


# ========================================================================
# Inspection plots
# INPUT: audio, boxes, similarity matrices, or group drift arrays.
# PROCESSING: draw spectrograms and diagnostic plots.
# OUTPUT: Matplotlib figures or saved PNG files.
# ========================================================================


def maybe_limit_time(y, sr, sec_limit):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    if sec_limit is None:
        return y
    n = int(round(sec_limit * sr))
    return y[:n]


def get_time_mask(win_starts, sec_limit):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    if sec_limit is None:
        return np.ones_like(win_starts, dtype=bool)
    return win_starts <= sec_limit


def plot_spectrogram(y, sr, sec_limit=None):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    y2 = maybe_limit_time(y, sr, sec_limit)

    S = librosa.stft(y2, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=FIG_DPI)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="hz",
        fmin=0,
        fmax=FMAX,
        cmap="magma",
        ax=ax,
    )
    ax.set_title("Spectrogram")
    ax.set_ylim(0, FMAX)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


def plot_similarity_matrix(S, title="Similarity Matrix"):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=FIG_DPI)
    im = ax.imshow(S, origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Band index")
    ax.set_ylabel("Band index")
    plt.colorbar(im, ax=ax, label="Similarity")
    plt.tight_layout()
    plt.show()


def plot_similarity_matrix_reordered(S, order, title="Reordered Similarity Matrix"):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    Sr = S[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=FIG_DPI)
    im = ax.imshow(Sr, origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Reordered band index")
    ax.set_ylabel("Reordered band index")
    plt.colorbar(im, ax=ax, label="Similarity")
    plt.tight_layout()
    plt.show()


def plot_dendrogram(Z):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=FIG_DPI)
    dendrogram(Z, ax=ax)
    ax.set_title("Hierarchical Clustering of Bands")
    ax.set_xlabel("Band index")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.show()


def plot_cluster_bandmap(edges_hz, cluster_labels):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    uniq = sorted(np.unique(cluster_labels))
    fig, ax = plt.subplots(figsize=(10, 2.2), dpi=FIG_DPI)

    for bi in range(len(edges_hz) - 1):
        lo = edges_hz[bi]
        hi = edges_hz[bi + 1]
        c = cluster_labels[bi]
        ax.fill_between(
            [lo, hi],
            0,
            1,
            alpha=0.7,
            label=f"C{c}" if bi == np.where(cluster_labels == c)[0][0] else None,
        )

    ax.set_xlim(edges_hz[0], edges_hz[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title("Discovered Band Grouping")
    ax.legend(ncol=min(len(uniq), 8), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.show()


def plot_cluster_drifts(
    cluster_drifts,
    cluster_members,
    cluster_labels_unique,
    win_starts,
    edges_hz,
    sec_limit=None,
):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    t = win_starts.copy()
    mask = get_time_mask(t, sec_limit)

    n = len(cluster_drifts)
    fig, axes = plt.subplots(
        n, 1, figsize=(14, max(2.5 * n, 6)), sharex=True, dpi=FIG_DPI
    )

    if n == 1:
        axes = [axes]

    for ax, cd, members, c in zip(
        axes, cluster_drifts, cluster_members, cluster_labels_unique
    ):
        tt = t[mask]
        yy = cd[mask]

        lo = edges_hz[min(members)]
        hi = edges_hz[max(members) + 1]

        ax.plot(tt, yy, linewidth=1.4)
        ax.set_ylabel("Drift")
        ax.set_title(f"Cluster {c}: {int(lo)} - {int(hi)} Hz | members={members}")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Merged Cluster Drifts ({CLUSTER_AGG})", y=1.0)
    plt.tight_layout()
    plt.show()


def plot_all_cluster_drifts_overlay(
    cluster_drifts,
    cluster_members,
    cluster_labels_unique,
    win_starts,
    edges_hz,
    sec_limit=None,
    normalize=True,
    event_curve=None,
    activity_curve=None,
):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    t = win_starts.copy()
    mask = get_time_mask(t, sec_limit)
    tt = t[mask]

    fig, ax = plt.subplots(figsize=(14, 5), dpi=FIG_DPI)

    for cd, members, c in zip(cluster_drifts, cluster_members, cluster_labels_unique):
        lo = edges_hz[min(members)]
        hi = edges_hz[max(members) + 1]
        yy = cd[mask]
        if normalize:
            yy = minmax01(yy)
        ax.plot(tt, yy, linewidth=1.6, label=f"C{c}: {int(lo)}-{int(hi)} Hz")

    if event_curve is not None:
        ev = event_curve[mask]
        if normalize:
            ev = minmax01(ev)
        ax.plot(tt, ev, linewidth=2.5, linestyle="--", label="Spectrogram event curve")

    if activity_curve is not None:
        ac = activity_curve[mask]
        if normalize:
            ac = minmax01(ac)
        ax.plot(tt, ac, linewidth=2.0, linestyle=":", label="Box-derived activity")

    ax.set_title("Merged Cluster Drifts + Event Curve + Box Activity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized magnitude" if normalize else "Magnitude")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


def plot_spectrogram_with_cluster_drifts(
    y,
    sr,
    win_starts,
    cluster_drifts,
    cluster_members,
    cluster_labels_unique,
    edges_hz,
    event_curve=None,
    activity_curve=None,
    sec_limit=None,
):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    y2 = maybe_limit_time(y, sr, sec_limit)
    S = librosa.stft(y2, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    t = win_starts.copy()
    mask = get_time_mask(t, sec_limit)
    tt = t[mask]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
        dpi=FIG_DPI,
    )

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="hz",
        fmin=0,
        fmax=FMAX,
        cmap="magma",
        ax=axes[0],
    )
    axes[0].set_ylim(0, FMAX)
    axes[0].set_title("Spectrogram")
    plt.colorbar(img, ax=axes[0], format="%+2.0f dB")

    for cd, members, c in zip(cluster_drifts, cluster_members, cluster_labels_unique):
        lo = edges_hz[min(members)]
        hi = edges_hz[max(members) + 1]
        yy = minmax01(cd[mask])
        axes[1].plot(tt, yy, linewidth=1.5, label=f"C{c}: {int(lo)}-{int(hi)} Hz")

    if event_curve is not None:
        ev = minmax01(event_curve[mask])
        axes[1].plot(
            tt, ev, linewidth=2.5, linestyle="--", label="Spectrogram event curve"
        )

    if activity_curve is not None:
        ac = minmax01(activity_curve[mask])
        axes[1].plot(tt, ac, linewidth=2.0, linestyle=":", label="Box-derived activity")

    axes[1].set_title("Merged Cluster Drifts vs Event Curve vs Box Activity")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Normalized magnitude")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()


def plot_spectrogram_with_discovered_boxes(y, sr, boxes, gt_boxes=None, sec_limit=None):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    y2 = maybe_limit_time(y, sr, sec_limit)
    S = librosa.stft(y2, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=FIG_DPI)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="hz",
        fmin=0,
        fmax=FMAX,
        cmap="magma",
        ax=ax,
    )
    ax.set_ylim(0, FMAX)
    ax.set_title("Spectrogram with Discovered Peak-Derived Boxes")

    for b in boxes:
        if sec_limit is not None and b["t0"] > sec_limit:
            continue

        rect = plt.Rectangle(
            (b["t0"], b["f0"]),
            b["t1"] - b["t0"],
            b["f1"] - b["f0"],
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
        )
        ax.add_patch(rect)

    if gt_boxes is not None:
        for b in gt_boxes:
            if sec_limit is not None and b["t0"] > sec_limit:
                continue
            rect = plt.Rectangle(
                (b["t0"], b["f0"]),
                b["t1"] - b["t0"],
                b["f1"] - b["f0"],
                fill=False,
                edgecolor="lime",
                linewidth=1.2,
                linestyle="--",
            )
            ax.add_patch(rect)

    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


def print_cluster_event_metrics(
    metrics, cluster_members, cluster_labels_unique, edges_hz
):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    print("\n=== CLUSTER vs SPECTROGRAM EVENT METRICS ===")
    for m, members, c in zip(metrics, cluster_members, cluster_labels_unique):
        lo = edges_hz[min(members)]
        hi = edges_hz[max(members) + 1]

        print(
            f"Cluster {c} | {int(lo):4d}-{int(hi):5d} Hz | members={members}\n"
            f"  Pearson r       : {m['pearson_r']:.3f}\n"
            f"  Spearman rho    : {m['spearman_rho']:.3f}\n"
            f"  Max lag corr    : {m['max_lag_corr']:.3f}  (lag={m['best_lag_windows']} windows)\n"
            f"  Peak precision  : {m['peak_precision']:.3f}\n"
            f"  Peak recall     : {m['peak_recall']:.3f}\n"
            f"  Peak F1         : {m['peak_f1']:.3f}\n"
            f"  Cluster peaks   : {m['n_cluster_peaks']}\n"
            f"  Event peaks     : {m['n_event_peaks']}"
        )


def print_similarity_diagnostics(S_time, S_freq, S_adj, S_total):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    print("\n=== SIMILARITY MATRIX DIAGNOSTICS ===")
    for name, S in [
        ("Time", S_time),
        ("Freq", S_freq),
        ("Adj", S_adj),
        ("Total", S_total),
    ]:
        off = S[~np.eye(S.shape[0], dtype=bool)]
        print(
            f"{name:>5s} | min={off.min():.3f} mean={off.mean():.3f} max={off.max():.3f}"
        )


def save_spectrogram_with_boxes(y, sr, boxes, gt_boxes, out_png, title, sec_limit=None):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    y2 = maybe_limit_time(y, sr, sec_limit)
    S = librosa.stft(y2, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=FIG_DPI)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="hz",
        fmin=0,
        fmax=FMAX,
        cmap="magma",
        ax=ax,
    )
    ax.set_ylim(0, FMAX)
    ax.set_title(title)

    for b in boxes:
        if sec_limit is not None and b["t0"] > sec_limit:
            continue
        ax.add_patch(
            plt.Rectangle(
                (b["t0"], b["f0"]),
                b["t1"] - b["t0"],
                b["f1"] - b["f0"],
                fill=False,
                edgecolor="cyan",
                linewidth=1.3,
            )
        )

    if gt_boxes is not None:
        for b in gt_boxes:
            if sec_limit is not None and b["t0"] > sec_limit:
                continue
            ax.add_patch(
                plt.Rectangle(
                    (b["t0"], b["f0"]),
                    b["t1"] - b["t0"],
                    b["f1"] - b["f0"],
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.1,
                    linestyle="--",
                )
            )

    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_cluster_drifts_plot(
    cluster_drifts,
    cluster_members,
    cluster_labels_unique,
    win_starts,
    edges_hz,
    out_png,
    title,
    sec_limit=None,
):
    """Input: audio, boxes, similarity matrices, or group drift arrays; see the signature for specific arguments.
    Processing: draw spectrograms and diagnostic plots.
    Output: Matplotlib figures or saved PNG files.
    """
    t = win_starts.copy()
    mask = get_time_mask(t, sec_limit)
    n = len(cluster_drifts)
    fig, axes = plt.subplots(
        n, 1, figsize=(14, max(2.3 * n, 6)), sharex=True, dpi=FIG_DPI
    )
    if n == 1:
        axes = [axes]

    for ax, cd, members, c in zip(
        axes, cluster_drifts, cluster_members, cluster_labels_unique
    ):
        lo = edges_hz[min(members)]
        hi = edges_hz[max(members) + 1]
        ax.plot(t[mask], cd[mask], linewidth=1.2)
        ax.set_ylabel("Drift")
        ax.set_title(f"Cluster {c}: {int(lo)}-{int(hi)} Hz | members={members}")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, y=1.0)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ========================================================================
# GMM proposals and optional search
# INPUT: group drift trajectories and micro-window times.
# PROCESSING: fit event states, construct boxes, or rank exploratory settings.
# OUTPUT: candidate boxes, model diagnostics, or search results.
# ========================================================================


def gmm_boxes_from_cluster_drifts(
    cluster_drifts,
    cluster_members,
    cluster_labels_unique,
    edges_hz,
    win_starts,
    win_ends,
    min_event_posterior=0.50,
    smooth_labels_k=3,
    merge_gap_windows=2,
    min_event_windows=2,
    min_event_area_q=None,
    use_log_drift=False,
    random_state=0,
    verbose=False,
):
    """Input: group drift trajectories and micro-window times; see the signature for specific arguments.
    Processing: fit event states, construct boxes, or rank exploratory settings.
    Output: candidate boxes, model diagnostics, or search results.
    """
    all_boxes = []
    peak_info = []

    for cd, members, c in zip(cluster_drifts, cluster_members, cluster_labels_unique):
        f0 = float(edges_hz[min(members)])
        f1 = float(edges_hz[max(members) + 1])

        res = fit_gmm_event_state(
            cd,
            min_event_posterior=min_event_posterior,
            smooth_labels_k=smooth_labels_k,
            merge_gap_windows=merge_gap_windows,
            min_event_windows=min_event_windows,
            min_event_area_q=min_event_area_q,
            use_log_drift=use_log_drift,
            random_state=random_state,
        )

        labels = res["labels"]
        regions = res["regions"]
        cluster_boxes = []
        peaks = []

        for st, en in regions:
            segment = cd[st : en + 1]
            peak_idx = int(st + np.argmax(segment))
            peaks.append(peak_idx)
            cluster_boxes.append(
                {
                    "cluster_id": int(c),
                    "t0": float(win_starts[st]),
                    "t1": float(win_ends[en]),
                    "f0": f0,
                    "f1": f1,
                    "peak_idx": int(peak_idx),
                    "support_left": int(st),
                    "support_right": int(en),
                    "members": members,
                }
            )

        cluster_boxes = merge_overlapping_time_boxes(cluster_boxes)
        all_boxes.extend(cluster_boxes)

        if verbose:
            means_str = np.array2string(res["means"], precision=3, separator=", ")
            weights_str = np.array2string(res["weights"], precision=3, separator=", ")
            event_fraction = float(np.mean(labels)) if len(labels) else 0.0
            print(
                f"Cluster {c} | {int(f0):4d}-{int(f1):5d} Hz | "
                f"thr≈{res['threshold']:.3f} | means={means_str} | "
                f"weights={weights_str} | event_fraction={event_fraction:.3f} | "
                f"regions={len(regions)} | boxes={len(cluster_boxes)}"
            )

        peak_info.append(
            {
                "cluster_id": int(c),
                "freq_lo": f0,
                "freq_hi": f1,
                "n_peaks": int(len(peaks)),
                "n_boxes": int(len(cluster_boxes)),
                "peaks": peaks,
                "labels": labels,
                "posterior": res["posterior"],
                "regions": regions,
                "means": res["means"],
                "weights": res["weights"],
                "threshold": res["threshold"],
            }
        )

    return all_boxes, peak_info


def score_boxes_against_event_curve(boxes, event_curve, win_starts, win_ends):
    """Input: group drift trajectories and micro-window times; see the signature for specific arguments.
    Processing: fit event states, construct boxes, or rank exploratory settings.
    Output: candidate boxes, model diagnostics, or search results.
    """
    activity = boxes_to_global_activity_curve(
        boxes=boxes,
        n_windows=len(win_starts),
        win_starts=win_starts,
        win_ends=win_ends,
        smooth_k=BOX_SMOOTH_ACTIVITY_K,
    )
    corr = safe_pearson(robust_zscore(activity), robust_zscore(event_curve))

    _, _, event_peaks, _ = detect_peaks_from_drift(
        event_curve,
        k_mad=EVENT_PEAK_K_MAD,
        min_sep_windows=EVENT_PEAK_MIN_SEP_WINDOWS,
    )

    pred_peaks = []
    for b in boxes:
        centre_t = 0.5 * (b["t0"] + b["t1"])
        pred_peaks.append(int(np.argmin(np.abs(win_starts - centre_t))))

    peak_scores = match_peaks_with_tolerance(
        event_peaks, pred_peaks, tol=PEAK_MATCH_TOL_WINDOWS
    )
    box_density = len(boxes) / max(1, len(win_starts))
    proxy = (
        PROXY_CORR_WEIGHT * corr
        + PROXY_PEAK_F1_WEIGHT * peak_scores["f1"]
        - PROXY_BOX_DENSITY_PENALTY * box_density
    )
    return {
        "proxy_score": float(proxy),
        "event_corr": float(corr),
        "event_peak_f1": float(peak_scores["f1"]),
        "event_peak_precision": float(peak_scores["precision"]),
        "event_peak_recall": float(peak_scores["recall"]),
        "box_density": float(box_density),
    }


def run_gmm_grid_search(
    cluster_drifts,
    cluster_members,
    uniq_clusters,
    edges_hz,
    win_starts,
    win_ends,
    event_curve,
    gt_boxes,
):
    """Input: group drift trajectories and micro-window times; see the signature for specific arguments.
    Processing: fit event states, construct boxes, or rank exploratory settings.
    Output: candidate boxes, model diagnostics, or search results.
    """
    rows = []
    best = None

    for posterior in GRID_GMM_MIN_EVENT_POSTERIOR:
        for smooth_k in GRID_GMM_SMOOTH_LABELS_K:
            for gap in GRID_GMM_MERGE_GAP_WINDOWS:
                for min_w in GRID_GMM_MIN_EVENT_WINDOWS:
                    for use_log in GRID_GMM_USE_LOG_DRIFT:
                        for area_q in GRID_GMM_MIN_EVENT_AREA_Q:
                            boxes, _ = gmm_boxes_from_cluster_drifts(
                                cluster_drifts,
                                cluster_members,
                                uniq_clusters,
                                edges_hz,
                                win_starts,
                                win_ends,
                                min_event_posterior=posterior,
                                smooth_labels_k=smooth_k,
                                merge_gap_windows=gap,
                                min_event_windows=min_w,
                                min_event_area_q=area_q,
                                use_log_drift=use_log,
                                random_state=GMM_RANDOM_STATE,
                                verbose=False,
                            )

                            proxy = score_boxes_against_event_curve(
                                boxes, event_curve, win_starts, win_ends
                            )
                            gt_metrics = (
                                evaluate_predicted_boxes(
                                    boxes, gt_boxes, iou_thr=GT_IOU_THRESHOLD
                                )
                                if len(gt_boxes)
                                else None
                            )

                            if gt_metrics is not None:
                                score = (
                                    0.65 * gt_metrics["f1"]
                                    + 0.20 * gt_metrics["mean_iou"]
                                    + 0.15 * proxy["event_corr"]
                                )
                            else:
                                score = proxy["proxy_score"]

                            row = {
                                "score": float(score),
                                "posterior": posterior,
                                "smooth_k": smooth_k,
                                "merge_gap": gap,
                                "min_event_windows": min_w,
                                "use_log_drift": use_log,
                                "min_event_area_q": area_q,
                                "n_boxes": len(boxes),
                                **proxy,
                            }
                            if gt_metrics is not None:
                                row.update(
                                    {f"gt_{k}": v for k, v in gt_metrics.items()}
                                )
                            rows.append(row)

                            if best is None or row["score"] > best["score"]:
                                best = row

    return best, pd.DataFrame(rows).sort_values("score", ascending=False)


# ========================================================================
# Pipeline orchestration
# INPUT: one audio-file path and configured parameters.
# PROCESSING: run preprocessing, representations, grouping, and diagnostics.
# OUTPUT: dictionary of audio, window times, frequency groups, and drift curves.
# ========================================================================


def compute_pipeline_for_file(wav_path):
    """Input: one audio-file path and configured parameters; see the signature for specific arguments.
    Processing: run preprocessing, representations, grouping, and diagnostics.
    Output: dictionary of audio, window times, frequency groups, and drift curves.
    """
    print(f"\n=== Processing {Path(wav_path).name} ===")
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    y = apply_waveform_preprocessing(y, sr)
    duration_s = len(y) / sr

    win_starts, win_ends = make_fixed_windows(duration_s, WINDOW_S, HOP_S)

    repr_vectors = compute_window_repr_vectors(
        y=y,
        sr=sr,
        win_starts=win_starts,
        win_ends=win_ends,
        n_mels=N_MELS_FULL,
        fmin=FMIN,
        fmax=FMAX,
        use_pcen=USE_PCEN,
    )

    if DENOISE_METHOD == "median_specsub":
        repr_vectors = subtract_stationary_band_floor(
            repr_vectors, strength=SPEC_FLOOR_STRENGTH, floor_mode=SPEC_FLOOR_MODE
        )

    edges_hz = band_edges_hz(FMIN, FMAX, N_FINE_BANDS, scale=BAND_SCALE)
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS_FULL, fmin=FMIN, fmax=FMAX)
    band_vectors = split_mel_vectors_into_bands(repr_vectors, mel_freqs, edges_hz)
    band_drifts = compute_direct_band_drifts(band_vectors, smooth_k=DRIFT_SMOOTH_K)

    band_drifts_z, band_evidences = [], []
    for d in band_drifts:
        dz, ev, _, _ = detect_peaks_from_drift(
            d, k_mad=PEAK_K_MAD, min_sep_windows=PEAK_MIN_SEP_WINDOWS
        )
        band_drifts_z.append(dz)
        band_evidences.append(ev)

    band_descs = compute_band_frequency_descriptors(
        band_vectors,
        k_mad_activity=FREQ_DESC_ACTIVITY_K_MAD,
        normalize_subband=FREQ_DESC_NORMALIZE_SUBBAND,
        use_activity=FREQ_DESC_USE_ACTIVITY,
    )

    S_time = build_time_similarity_matrix(
        drifts=band_drifts_z,
        evidences=band_evidences,
        mode=SIMILARITY_MODE,
        max_lag=MAX_LAG_WINDOWS,
    )
    S_freq = build_frequency_similarity_matrix(band_descs)
    S_adj = build_adjacency_similarity_matrix(N_FINE_BANDS, tau=ADJ_TAU)
    S_total = combine_similarity_matrices(
        S_time=S_time,
        S_freq=S_freq,
        S_adj=S_adj,
        alpha=ALPHA_TIME,
        beta=BETA_FREQ,
        gamma=GAMMA_ADJ,
    )

    _, cluster_labels = cluster_bands_from_similarity(
        S_total,
        linkage_method=LINKAGE_METHOD,
        n_groups=N_GROUPS,
        distance_threshold=DISTANCE_THRESHOLD,
    )

    uniq_clusters, cluster_drifts, cluster_members = aggregate_cluster_drifts(
        drifts=band_drifts_z,
        cluster_labels=cluster_labels,
        agg=CLUSTER_AGG,
    )

    global_repr_vectors = compute_global_repr_vectors(
        y=y,
        sr=sr,
        win_starts=win_starts,
        win_ends=win_ends,
        n_mels=N_MELS_FULL,
        fmin=FMIN,
        fmax=FMAX,
    )

    if DENOISE_METHOD == "median_specsub":
        global_repr_vectors = subtract_stationary_band_floor(
            global_repr_vectors,
            strength=SPEC_FLOOR_STRENGTH,
            floor_mode=SPEC_FLOOR_MODE,
        )

    event_curve = compute_spectrogram_event_curve(
        global_repr_vectors, smooth_k=EVENT_SMOOTH_K
    )

    return {
        "y": y,
        "sr": sr,
        "duration_s": duration_s,
        "win_starts": win_starts,
        "win_ends": win_ends,
        "edges_hz": edges_hz,
        "cluster_labels": cluster_labels,
        "uniq_clusters": uniq_clusters,
        "cluster_drifts": cluster_drifts,
        "cluster_members": cluster_members,
        "event_curve": event_curve,
    }
