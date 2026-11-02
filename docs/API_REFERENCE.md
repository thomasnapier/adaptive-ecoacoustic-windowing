# Function reference

Scientific functions below come from WABAD cell 2 (one-based) of the supplied batch notebook. Their bodies are retained. Each reference provides the Input–Processing–Output context; function-specific original docstrings are also preserved. `research.configure` resets all defaults before each run. The runner is serial.


## Numerical helpers

### `moving_average`

```python
moving_average(x: np.ndarray, k: int)
```

**Input:** arrays and scalar measurements. **Processing:** stabilise, scale, or compare numeric values. **Output:** numeric arrays or scalar statistics.


### `robust_mad`

```python
robust_mad(x: np.ndarray)
```

**Input:** arrays and scalar measurements. **Processing:** stabilise, scale, or compare numeric values. **Output:** numeric arrays or scalar statistics.


### `robust_zscore`

```python
robust_zscore(x: np.ndarray)
```

**Input:** arrays and scalar measurements. **Processing:** stabilise, scale, or compare numeric values. **Output:** numeric arrays or scalar statistics.


### `minmax01`

```python
minmax01(x: np.ndarray)
```

**Input:** arrays and scalar measurements. **Processing:** stabilise, scale, or compare numeric values. **Output:** numeric arrays or scalar statistics.


### `safe_pearson`

```python
safe_pearson(x, y)
```

**Input:** arrays and scalar measurements. **Processing:** stabilise, scale, or compare numeric values. **Output:** numeric arrays or scalar statistics.


### `cosine_sim`

```python
cosine_sim(a: np.ndarray, b: np.ndarray)
```

**Input:** arrays and scalar measurements. **Processing:** stabilise, scale, or compare numeric values. **Output:** numeric arrays or scalar statistics.


### `make_fixed_windows`

```python
make_fixed_windows(duration_s: float, window_s: float, hop_s: float)
```

**Input:** arrays and scalar measurements. **Processing:** stabilise, scale, or compare numeric values. **Output:** numeric arrays or scalar statistics.


### `nonmax_suppress_peaks`

```python
nonmax_suppress_peaks(peaks, scores, radius)
```

**Input:** arrays and scalar measurements. **Processing:** stabilise, scale, or compare numeric values. **Output:** numeric arrays or scalar statistics.


### `detect_peaks_from_drift`

```python
detect_peaks_from_drift(drift: np.ndarray, k_mad: float, min_sep_windows: int)
```

**Input:** arrays and scalar measurements. **Processing:** stabilise, scale, or compare numeric values. **Output:** numeric arrays or scalar statistics.



## Frequency bands and preprocessing

### `band_edges_hz`

```python
band_edges_hz(fmin: float, fmax: float, n_bands: int, scale: str='mel')
```

**Input:** waveform samples and frequency settings. **Processing:** define band edges or suppress selected background components. **Output:** band edges or a preprocessed waveform.


### `denoise_spectral_gate`

```python
denoise_spectral_gate(y, sr, n_fft=1024, hop_length=256, noise_profile_sec=2.0, floor=0.1)
```

**Input:** waveform samples and frequency settings. **Processing:** define band edges or suppress selected background components. **Output:** band edges or a preprocessed waveform.


### `denoise_hpss_percussive`

```python
denoise_hpss_percussive(y, kernel_size=31, power=2.0, margin=(1.0, 5.0))
```

**Input:** waveform samples and frequency settings. **Processing:** define band edges or suppress selected background components. **Output:** band edges or a preprocessed waveform.


### `denoise_hpss_residual`

```python
denoise_hpss_residual(y, kernel_size=31, power=2.0, margin=(1.0, 5.0))
```

**Input:** waveform samples and frequency settings. **Processing:** define band edges or suppress selected background components. **Output:** band edges or a preprocessed waveform.


### `subtract_stationary_band_floor`

```python
subtract_stationary_band_floor(X, strength=1.0, floor_mode='median')
```

**Input:** waveform samples and frequency settings. **Processing:** define band edges or suppress selected background components. **Output:** band edges or a preprocessed waveform.


### `apply_waveform_preprocessing`

```python
apply_waveform_preprocessing(y, sr)
```

**Input:** waveform samples and frequency settings. **Processing:** define band edges or suppress selected background components. **Output:** band edges or a preprocessed waveform.



## Micro-window representations and drift

### `compute_window_repr_vectors`

```python
compute_window_repr_vectors(y, sr, win_starts, win_ends, n_mels, fmin, fmax, use_pcen=False)
```

**Input:** mono waveform, window times, and band vectors. **Processing:** compute PCEN/log-Mel summaries, adjacent-window drift, and summary descriptors. **Output:** spectral vectors, band drift arrays, or descriptor matrix.


### `split_mel_vectors_into_bands`

```python
split_mel_vectors_into_bands(logmel_vectors, mel_freqs, edges_hz)
```

**Input:** mono waveform, window times, and band vectors. **Processing:** compute PCEN/log-Mel summaries, adjacent-window drift, and summary descriptors. **Output:** spectral vectors, band drift arrays, or descriptor matrix.


### `compute_direct_band_drifts`

```python
compute_direct_band_drifts(band_vectors, smooth_k)
```

**Input:** mono waveform, window times, and band vectors. **Processing:** compute PCEN/log-Mel summaries, adjacent-window drift, and summary descriptors. **Output:** spectral vectors, band drift arrays, or descriptor matrix.


### `compute_band_frequency_descriptors`

```python
compute_band_frequency_descriptors(band_vectors, k_mad_activity=2.0, normalize_subband=True, use_activity=True)
```

**Input:** mono waveform, window times, and band vectors. **Processing:** compute PCEN/log-Mel summaries, adjacent-window drift, and summary descriptors. **Output:** spectral vectors, band drift arrays, or descriptor matrix.

Stable per-band descriptor.

Each band gets a compact summary vector built from:
- median energy per sub-bin
- variability per sub-bin (MAD)
- global stats: median, MAD, max, flatness proxy, activity proportion,
  burstiness, temporal sparsity, mean pairwise sub-bin correlation


## Hybrid similarity and grouping

### `jaccard_similarity_binary`

```python
jaccard_similarity_binary(a: np.ndarray, b: np.ndarray)
```

**Input:** band drift trajectories and spectral descriptors. **Processing:** compare timing, profile, and adjacency, then group related bands. **Output:** similarity matrices or band-group labels.


### `lagged_corr_similarity`

```python
lagged_corr_similarity(a: np.ndarray, b: np.ndarray, max_lag: int=2)
```

**Input:** band drift trajectories and spectral descriptors. **Processing:** compare timing, profile, and adjacency, then group related bands. **Output:** similarity matrices or band-group labels.


### `build_time_similarity_matrix`

```python
build_time_similarity_matrix(drifts, evidences, mode='jaccard_binary', max_lag=2)
```

**Input:** band drift trajectories and spectral descriptors. **Processing:** compare timing, profile, and adjacency, then group related bands. **Output:** similarity matrices or band-group labels.


### `build_frequency_similarity_matrix`

```python
build_frequency_similarity_matrix(band_descs)
```

**Input:** band drift trajectories and spectral descriptors. **Processing:** compare timing, profile, and adjacency, then group related bands. **Output:** similarity matrices or band-group labels.


### `build_adjacency_similarity_matrix`

```python
build_adjacency_similarity_matrix(n_bands, tau=2.5)
```

**Input:** band drift trajectories and spectral descriptors. **Processing:** compare timing, profile, and adjacency, then group related bands. **Output:** similarity matrices or band-group labels.


### `combine_similarity_matrices`

```python
combine_similarity_matrices(S_time, S_freq, S_adj, alpha=0.7, beta=0.2, gamma=0.1)
```

**Input:** band drift trajectories and spectral descriptors. **Processing:** compare timing, profile, and adjacency, then group related bands. **Output:** similarity matrices or band-group labels.


### `cluster_bands_from_similarity`

```python
cluster_bands_from_similarity(S, linkage_method='average', n_groups=None, distance_threshold=0.55)
```

**Input:** band drift trajectories and spectral descriptors. **Processing:** compare timing, profile, and adjacency, then group related bands. **Output:** similarity matrices or band-group labels.


### `aggregate_cluster_drifts`

```python
aggregate_cluster_drifts(drifts, cluster_labels, agg='median')
```

**Input:** band drift trajectories and spectral descriptors. **Processing:** compare timing, profile, and adjacency, then group related bands. **Output:** similarity matrices or band-group labels.


### `reorder_by_dendrogram`

```python
reorder_by_dendrogram(Z)
```

**Input:** band drift trajectories and spectral descriptors. **Processing:** compare timing, profile, and adjacency, then group related bands. **Output:** similarity matrices or band-group labels.


### `cluster_band_ranges`

```python
cluster_band_ranges(edges_hz, cluster_labels)
```

**Input:** band drift trajectories and spectral descriptors. **Processing:** compare timing, profile, and adjacency, then group related bands. **Output:** similarity matrices or band-group labels.



## Diagnostic event curve

### `compute_global_repr_vectors`

```python
compute_global_repr_vectors(y, sr, win_starts, win_ends, n_mels, fmin, fmax)
```

**Input:** representations, drift curves, or peaks. **Processing:** compare candidate activity with a signal-derived diagnostic curve. **Output:** diagnostic curves or agreement statistics.


### `compute_spectrogram_event_curve`

```python
compute_spectrogram_event_curve(global_repr_vectors, smooth_k=7)
```

**Input:** representations, drift curves, or peaks. **Processing:** compare candidate activity with a signal-derived diagnostic curve. **Output:** diagnostic curves or agreement statistics.


### `match_peaks_with_tolerance`

```python
match_peaks_with_tolerance(ref_peaks, test_peaks, tol=2)
```

**Input:** representations, drift curves, or peaks. **Processing:** compare candidate activity with a signal-derived diagnostic curve. **Output:** diagnostic curves or agreement statistics.


### `best_lagged_corr`

```python
best_lagged_corr(a, b, max_lag=2)
```

**Input:** representations, drift curves, or peaks. **Processing:** compare candidate activity with a signal-derived diagnostic curve. **Output:** diagnostic curves or agreement statistics.


### `compute_cluster_vs_event_metrics`

```python
compute_cluster_vs_event_metrics(cluster_drifts, event_curve, max_lag=2, drift_peak_k_mad=2.5, drift_peak_sep=2, event_peak_k_mad=2.0, event_peak_sep=2, peak_match_tol=2)
```

**Input:** representations, drift curves, or peaks. **Processing:** compare candidate activity with a signal-derived diagnostic curve. **Output:** diagnostic curves or agreement statistics.



## Event segmentation and boxes

### `find_peak_support`

```python
find_peak_support(z, peak_idx, thr, expand_windows=0)
```

**Input:** group drift arrays, window indices, and frequency edges. **Processing:** find active runs and merge time-overlapping proposals within a group. **Output:** state information, event regions, or candidate box dictionaries.


### `merge_overlapping_time_boxes`

```python
merge_overlapping_time_boxes(boxes)
```

**Input:** group drift arrays, window indices, and frequency edges. **Processing:** find active runs and merge time-overlapping proposals within a group. **Output:** state information, event regions, or candidate box dictionaries.


### `smooth_binary_labels`

```python
smooth_binary_labels(labels, k=3)
```

**Input:** group drift arrays, window indices, and frequency edges. **Processing:** find active runs and merge time-overlapping proposals within a group. **Output:** state information, event regions, or candidate box dictionaries.

Simple majority filter for binary labels.

### `binary_labels_to_regions`

```python
binary_labels_to_regions(labels, merge_gap_windows=2, min_event_windows=2)
```

**Input:** group drift arrays, window indices, and frequency edges. **Processing:** find active runs and merge time-overlapping proposals within a group. **Output:** state information, event regions, or candidate box dictionaries.

Convert 0/1 labels into contiguous event regions.

### `fit_gmm_event_state`

```python
fit_gmm_event_state(drift, min_event_posterior=0.5, smooth_labels_k=3, merge_gap_windows=2, min_event_windows=2, min_event_area_q=None, use_log_drift=False, random_state=0)
```

**Input:** group drift arrays, window indices, and frequency edges. **Processing:** find active runs and merge time-overlapping proposals within a group. **Output:** state information, event regions, or candidate box dictionaries.

Per-cluster 2-state segmentation using a Gaussian mixture model.

The lower-mean component is interpreted as background and the higher-mean
component as event. This estimates the non-event/event state from each
post-cluster drift curve rather than using a manually chosen drift threshold.

### `cluster_peaks_to_boxes`

```python
cluster_peaks_to_boxes(cluster_drifts, cluster_members, cluster_labels_unique, edges_hz, win_starts, win_ends, peak_k_mad, peak_min_sep_windows, expand_windows=0, min_duration_windows=1)
```

**Input:** group drift arrays, window indices, and frequency edges. **Processing:** find active runs and merge time-overlapping proposals within a group. **Output:** state information, event regions, or candidate box dictionaries.


### `boxes_to_global_activity_curve`

```python
boxes_to_global_activity_curve(boxes, n_windows, win_starts, win_ends, smooth_k=1)
```

**Input:** group drift arrays, window indices, and frequency edges. **Processing:** find active runs and merge time-overlapping proposals within a group. **Output:** state information, event regions, or candidate box dictionaries.



## Notebook evaluation metrics

### `interval_iou`

```python
interval_iou(a0, a1, b0, b1)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.


### `box_iou`

```python
box_iou(a, b)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.


### `_safe_div`

```python
_safe_div(n, d)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.


### `_box_duration`

```python
_box_duration(b)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.


### `_box_bandwidth`

```python
_box_bandwidth(b)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.


### `_box_area`

```python
_box_area(b)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.


### `_intersection_length`

```python
_intersection_length(a0, a1, b0, b1)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.


### `_union_length_intervals`

```python
_union_length_intervals(intervals)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.

Union length of [start, end] intervals.

### `_pair_alignment_metrics`

```python
_pair_alignment_metrics(pred_box, gt_box)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.

Returns time, frequency, and 2D box alignment for one prediction/GT pair.

### `_make_temporal_activity_curve`

```python
_make_temporal_activity_curve(boxes, win_starts=None, win_ends=None, duration_s=None, hop_s=None)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.

Converts boxes to a 0/1 temporal activity curve.
This is used for time-only accuracy/precision/recall. It ignores frequency.

### `_binary_curve_metrics`

```python
_binary_curve_metrics(pred_curve, gt_curve)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.


### `evaluate_predicted_boxes`

```python
evaluate_predicted_boxes(pred_boxes, gt_boxes, iou_thr=0.3, win_starts=None, win_ends=None, duration_s=None, temporal_eval_hop_s=None, temporal_hit_iou_thr=None, frequency_hit_iou_thr=None)
```

**Input:** predicted boxes and reference boxes in seconds and Hz. **Processing:** measure overlap, greedily match pairs, and calculate coverage. **Output:** IoU, coverage, or temporal activity statistics.

Evaluates predicted time-frequency boxes against Raven GT boxes.

Important interpretation:
- strict precision/recall/F1 use one-to-one IoU matching and count unmatched
  predictions as false positives.
- GT coverage/hit metrics ask whether annotated GT boxes were found and are
  less punitive when the detector produces additional boxes not present in GT.
- temporal_* metrics are time-only binary activity metrics over analysis windows.
  They are useful for temporal alignment but can be biased by incomplete GT.


## Inspection plots

### `maybe_limit_time`

```python
maybe_limit_time(y, sr, sec_limit)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `get_time_mask`

```python
get_time_mask(win_starts, sec_limit)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `plot_spectrogram`

```python
plot_spectrogram(y, sr, sec_limit=None)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `plot_similarity_matrix`

```python
plot_similarity_matrix(S, title='Similarity Matrix')
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `plot_similarity_matrix_reordered`

```python
plot_similarity_matrix_reordered(S, order, title='Reordered Similarity Matrix')
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `plot_dendrogram`

```python
plot_dendrogram(Z)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `plot_cluster_bandmap`

```python
plot_cluster_bandmap(edges_hz, cluster_labels)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `plot_cluster_drifts`

```python
plot_cluster_drifts(cluster_drifts, cluster_members, cluster_labels_unique, win_starts, edges_hz, sec_limit=None)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `plot_all_cluster_drifts_overlay`

```python
plot_all_cluster_drifts_overlay(cluster_drifts, cluster_members, cluster_labels_unique, win_starts, edges_hz, sec_limit=None, normalize=True, event_curve=None, activity_curve=None)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `plot_spectrogram_with_cluster_drifts`

```python
plot_spectrogram_with_cluster_drifts(y, sr, win_starts, cluster_drifts, cluster_members, cluster_labels_unique, edges_hz, event_curve=None, activity_curve=None, sec_limit=None)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `plot_spectrogram_with_discovered_boxes`

```python
plot_spectrogram_with_discovered_boxes(y, sr, boxes, gt_boxes=None, sec_limit=None)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `print_cluster_event_metrics`

```python
print_cluster_event_metrics(metrics, cluster_members, cluster_labels_unique, edges_hz)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `print_similarity_diagnostics`

```python
print_similarity_diagnostics(S_time, S_freq, S_adj, S_total)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `save_spectrogram_with_boxes`

```python
save_spectrogram_with_boxes(y, sr, boxes, gt_boxes, out_png, title, sec_limit=None)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.


### `save_cluster_drifts_plot`

```python
save_cluster_drifts_plot(cluster_drifts, cluster_members, cluster_labels_unique, win_starts, edges_hz, out_png, title, sec_limit=None)
```

**Input:** audio, boxes, similarity matrices, or group drift arrays. **Processing:** draw spectrograms and diagnostic plots. **Output:** Matplotlib figures or saved PNG files.



## GMM proposals and optional search

### `gmm_boxes_from_cluster_drifts`

```python
gmm_boxes_from_cluster_drifts(cluster_drifts, cluster_members, cluster_labels_unique, edges_hz, win_starts, win_ends, min_event_posterior=0.5, smooth_labels_k=3, merge_gap_windows=2, min_event_windows=2, min_event_area_q=None, use_log_drift=False, random_state=0, verbose=False)
```

**Input:** group drift trajectories and micro-window times. **Processing:** fit event states, construct boxes, or rank exploratory settings. **Output:** candidate boxes, model diagnostics, or search results.


### `score_boxes_against_event_curve`

```python
score_boxes_against_event_curve(boxes, event_curve, win_starts, win_ends)
```

**Input:** group drift trajectories and micro-window times. **Processing:** fit event states, construct boxes, or rank exploratory settings. **Output:** candidate boxes, model diagnostics, or search results.


### `run_gmm_grid_search`

```python
run_gmm_grid_search(cluster_drifts, cluster_members, uniq_clusters, edges_hz, win_starts, win_ends, event_curve, gt_boxes)
```

**Input:** group drift trajectories and micro-window times. **Processing:** fit event states, construct boxes, or rank exploratory settings. **Output:** candidate boxes, model diagnostics, or search results.



## Pipeline orchestration

### `compute_pipeline_for_file`

```python
compute_pipeline_for_file(wav_path)
```

**Input:** one audio-file path and configured parameters. **Processing:** run preprocessing, representations, grouping, and diagnostics. **Output:** dictionary of audio, window times, frequency groups, and drift curves.

