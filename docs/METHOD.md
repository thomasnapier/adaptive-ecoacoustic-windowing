# Method and code map

The four scientific stages follow sections 3.1–3.4 and Algorithm 1 of the supplied v5 manuscript. Input/output handling is separated so non-experts do not need to edit mathematical code.

| Code section | Input | Processing | Output | Usually change? |
| --- | --- | --- | --- | --- |
| `config.py` | User JSON | Resolve paths relative to the settings file; check values and option names | Complete settings dictionary | Edit the JSON, not this code |
| `io.py` | Site folders and Raven/CSV tables | Discover recordings, match annotation stems, validate explicit time/frequency columns | Recording records and optional references | Use the supported layout |
| `research.apply_waveform_preprocessing` | Mono resampled waveform | Default HPSS, retaining the percussive component | Preprocessed waveform | Advanced: denoising method |
| `research.make_fixed_windows` | Recording duration | Create complete 0.25-s windows with 0.125-s steps | Start/end arrays, seconds | Advanced: duration/step |
| `research.compute_window_repr_vectors` | Waveform and windows | 128-bin Mel power spectra, PCEN per micro-window, frame averaging | Matrix shaped `(windows, 128)` | Advanced: frequency range and representation |
| `research.split_mel_vectors_into_bands` | Spectral matrix and Mel-space band edges | Select bins into 20 sub-bands | List of `(windows, bins_in_band)` matrices | Advanced: band count |
| `research.compute_direct_band_drifts` | Band representations | Euclidean distance between consecutive vectors; seven-step moving average | One drift trajectory per band | Advanced: smoothing |
| `research.detect_peaks_from_drift` | Smoothed drift | Median/MAD scaling and binary peak evidence | Standardised drift, evidence and diagnostic peaks | Rarely; peak evidence affects the Jaccard alternative |
| `research.compute_band_frequency_descriptors` | Within-band spectral vectors | Median/MAD, burstiness, flatness proxy, activity and bin correlation summaries | Padded band-descriptor matrix | Research changes |
| `research.build_*_similarity_matrix` | Drift, descriptors, band indices | Lagged timing correlation, rescaled cosine profile similarity, exponential adjacency | Three `(bands, bands)` affinity matrices | Research changes |
| `research.combine_similarity_matrices` | Three affinity matrices | Weighted combination, default 0.70 / 0.20 / 0.10 | Hybrid affinity | Research changes |
| `research.cluster_bands_from_similarity` | Hybrid affinity | Average-linkage HCA on `1 - affinity`, cut at distance 0.55 | Band-group IDs | Research changes |
| `research.aggregate_cluster_drifts` | Standardised band drift and group IDs | Median across member bands | Group trajectories and memberships | Research changes |
| `research.fit_gmm_event_state` | One group trajectory | Two Gaussian states, higher-mean state active, posterior threshold, majority filter, gap joining, short-run removal | Labels, active regions, posterior and model diagnostics | Posterior is an optional user control |
| `research.gmm_boxes_from_cluster_drifts` | Group trajectories, times and frequency edges | Convert active runs into boxes; merge overlapping same-group boxes | Candidate boxes and state diagnostics | Normally leave unchanged |
| `evaluation.py` | Predictions + references | New manuscript-oriented conditional matching | Conditional metrics | Evaluation design only |
| `pipeline.py` | Complete config | Run serial detection, save candidates, optionally plot and evaluate, checkpoint each recording | Per-recording outputs and batch summary | Normally leave unchanged |
| `visualize.py` | Audio + labelled references | Original reference-spectrogram plotting approach | Labelled PNG with legend | Plotting only |

## Details that affect interpretation

- HPSS uses librosa's waveform helper with the supplied kernel, power and margin values. Percussive retention is an acoustic preprocessing choice, not identification of a biological source. Sustained tonal calls can be weakened.
- PCEN is computed independently inside each micro-window in the supplied implementation. PCEN state is not carried across the whole recording. The power spectrum is multiplied by `2**16` before PCEN, matching the notebook.
- Mel band allocation follows `librosa.mel_frequencies` plus the configured edges. Descriptor vectors of different lengths are padded to a shared length, exactly as in the supplied code. This is not a learned embedding.
- Median/MAD normalisation uses a small additive epsilon; degenerate or constant signals can produce no candidate events.
- Similarity weights are internally normalised by their sum. The frequency-descriptor cosine similarity is rescaled into `[0,1]`. Negative timing correlations are clipped to zero.
- The seven-step moving average spans 0.875 s of step positions; its associated 0.25-s windows cover approximately 1 s. It uses edge padding and centred smoothing. This is an offline method, not a causal streaming detector.
- Only complete micro-windows are analysed. Audio shorter than one window is rejected. A trailing fraction shorter than the step may remain outside the final complete window.
- GMM states are fitted independently per group and recording, using two components, full covariance, `reg_covar=1e-6`, `n_init=5`, and seed 0. A high-state posterior is a local statistical activity score, not a calibrated probability of an animal or species.
- With two minimum active steps, the shortest retained run spans 0.375 s at default settings: `WINDOW_S + (2-1)*HOP_S`. Gap joining uses inactive step counts, not exported box duration.
- Different groups can retain simultaneous proposals. Frequency bounds use the outermost group edges, and may cover noncontiguous member-band gaps.
- The signal-derived global event curve is a diagnostic; it is not ground truth. Optional grid search exists in the expert API only and is not used by the command-line runner. It can use reference labels when supplied, so it changes the evaluation design.

## Expert usage

```python
from adaptive_windowing import research
research.configure({"GMM_MIN_EVENT_POSTERIOR": 0.7})
pipe = research.compute_pipeline_for_file("recording.wav")
```

`pipe` holds preprocessed audio, sample rate, duration, window times, group labels, group drift and membership, and a diagnostic event curve. For complete box generation use the guided notebook or `pipeline.run(config_path)`.

The research module retains notebook-style global settings. Configure once and process serially. Use separate processes/output folders for experiments with different settings; concurrent threads with different configurations are not supported. See [the full function reference](API_REFERENCE.md) for every retained research helper.
