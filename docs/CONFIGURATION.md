# Settings guide

Edit the JSON file under `configs/`, not `src/adaptive_windowing/defaults.json`. Top-level paths are relative to the settings file; absolute paths are also supported. Use `/` in Windows paths. Use JSON `true`/`false` and `null`, not Python `True`/`False`/`None`.

Start with `configs/paper_example.json` for the original ARD example, or `configs/wabad.json` for the full study dataset. Download the real study files using [DATA_RELEASE.md](DATA_RELEASE.md). For a first run change only the data/output paths. Detection defaults are taken from the WABAD source notebook. Put an override inside `parameters`; omitted parameters retain their defaults. `run_manifest.json` records the complete resolved configuration.

```json
{
  "data_root": "D:/WABAD",
  "output_dir": "D:/experiments/wabad-default",
  "save_plots": true,
  "save_drift_plots": true,
  "resume": true,
  "parameters": {
    "GMM_MIN_EVENT_POSTERIOR": 0.50
  }
}
```

## Top-level controls

| Key | Default/requirement | What it does |
| --- | --- | --- |
| `data_root` | Required for batch mode | Root containing sites, or one site directory |
| `output_dir` | Required | Results directory; use different directories for experiments |
| `audio_file` | Optional | Analyse one explicit WAV/FLAC; takes precedence over data_root |
| `annotation_file` | Optional, single-file only | Explicit matching table for audio_file |
| `save_plots` | true | Save prediction/reference spectrogram |
| `save_drift_plots` | true | Save grouped drift diagnostics |
| `resume` | true | Reuse completed results only with matching fingerprints and required files |
| `parameters` | {} | Scientific overrides; unknown names are rejected |

## Scientific parameters

Changing scientific parameters can change both the proposals and their interpretation. Keep them fixed across a benchmark. Threshold changes should be evaluated on a separate calibration population if used to select a final benchmark setting.

The command-line runner always uses GMM segmentation with fixed supplied settings. Grid-search and proxy parameters below are retained for expert inspection; changing those values does not enable grid search. The reconstructed conditional evaluator has fixed manuscript thresholds and is separate from the notebook GT thresholds.

| Parameter | Supplied default | Meaning and constraints |
| --- | --- | --- |
| `GT_IOU_THRESHOLD` | `0.3` | Original notebook strict 2D IoU threshold. Evaluation only; default 0.30 differs from manuscript conditional IoU@0.50. |
| `GT_TEMPORAL_HIT_IOU_THRESHOLD` | `0.1` | Original notebook temporal IoU hit threshold; evaluation only. |
| `GT_FREQUENCY_HIT_IOU_THRESHOLD` | `0.1` | Original notebook frequency IoU hit threshold; evaluation only. |
| `TEMPORAL_EVAL_HOP_S` | `0.125` | Time-grid step for notebook evaluation when explicit windows are not supplied. Runner supplies micro-window starts/ends, so this fallback does not control its time grid. |
| `GRID_GMM_MIN_EVENT_POSTERIOR` | `[0.35, 0.4, 0.45, 0.5, 0.55]` | Candidate values used only by research.run_gmm_grid_search; the normal runner never searches this grid. Per-file GT-guided selection is exploratory. |
| `GRID_GMM_SMOOTH_LABELS_K` | `[1, 3, 5]` | Candidate values used only by research.run_gmm_grid_search; the normal runner never searches this grid. Per-file GT-guided selection is exploratory. |
| `GRID_GMM_MERGE_GAP_WINDOWS` | `[1, 2, 3, 5]` | Candidate values used only by research.run_gmm_grid_search; the normal runner never searches this grid. Per-file GT-guided selection is exploratory. |
| `GRID_GMM_MIN_EVENT_WINDOWS` | `[1, 2]` | Candidate values used only by research.run_gmm_grid_search; the normal runner never searches this grid. Per-file GT-guided selection is exploratory. |
| `GRID_GMM_USE_LOG_DRIFT` | `[false, true]` | Candidate values used only by research.run_gmm_grid_search; the normal runner never searches this grid. Per-file GT-guided selection is exploratory. |
| `GRID_GMM_MIN_EVENT_AREA_Q` | `[null, 0.05, 0.1]` | Candidate values used only by research.run_gmm_grid_search; the normal runner never searches this grid. Per-file GT-guided selection is exploratory. |
| `PROXY_CORR_WEIGHT` | `0.8` | Weight on signal-derived correlation in exploratory proxy scoring. Expert diagnostic/API only. |
| `PROXY_PEAK_F1_WEIGHT` | `0.2` | Weight on signal-derived peak agreement in proxy scoring. Expert diagnostic/API only. |
| `PROXY_BOX_DENSITY_PENALTY` | `0.03` | Penalty on excessive proposal density in proxy scoring. Expert diagnostic/API only. |
| `TARGET_SR` | `22050` | Detection sampling rate, Hz. Default supports frequencies up to 11,025 Hz. Original low-rate bandwidth cannot be restored by resampling. |
| `WINDOW_S` | `0.25` | Micro-window duration in seconds; must be positive. Changes resolution and minimum event duration. |
| `HOP_S` | `0.125` | Step between micro-windows, seconds; positive and no larger than WINDOW_S. Smaller steps increase compute. |
| `N_FFT` | `1024` | Maximum FFT length within a micro-window, samples; the implementation adapts it to short windows. Use a positive integer, normally a power of two. |
| `HOP_LENGTH` | `256` | Nominal spectrogram frame hop, samples; the implementation may reduce it for short windows. |
| `N_MELS_FULL` | `128` | Number of Mel bins in each window representation; positive integer, at least N_FINE_BANDS. |
| `FMIN` | `200.0` | Lower analysis frequency in Hz. Use 0 <= FMIN < FMAX. |
| `FMAX` | `10000.0` | Upper analysis frequency in Hz. Must be at most TARGET_SR/2; check original input sampling rate too. |
| `N_FINE_BANDS` | `20` | Number of fine frequency bands. Integer from 2 through N_MELS_FULL. Changes spectral granularity. |
| `BAND_SCALE` | `"mel"` | Frequency band edge spacing: "mel" (default) or "linear". |
| `DRIFT_SMOOTH_K` | `7` | Centred moving-average length for band drift; positive odd step count. Default 7. |
| `PEAK_K_MAD` | `2.5` | MAD multiplier for peak evidence. With default lagged-correlation similarity, this peak threshold does not determine final GMM box boundaries. |
| `PEAK_MIN_SEP_WINDOWS` | `2` | Minimum separation for diagnostic/binary peak evidence, in micro-window steps. |
| `SIMILARITY_MODE` | `"lagged_corr"` | "lagged_corr" (default) or "jaccard_binary". The latter compares peak evidence rather than continuous drift. |
| `MAX_LAG_WINDOWS` | `2` | Maximum allowed timing lag for band similarity, in step counts. Default 2 = 0.25 s. |
| `ALPHA_TIME` | `0.7` | Nonnegative temporal-coherence weight, default 0.70. Weights are normalised by their sum. |
| `BETA_FREQ` | `0.2` | Nonnegative summary-profile weight, default 0.20. Research change, not a label prior. |
| `GAMMA_ADJ` | `0.1` | Nonnegative adjacency weight, default 0.10. All three weights must sum to more than zero. |
| `ADJ_TAU` | `2.5` | Positive exponential adjacency decay constant in band-index units. |
| `FREQ_DESC_USE_ACTIVITY` | `true` | Include activity duty-cycle/run descriptors when true; keeps the descriptor fields zero when false. |
| `FREQ_DESC_ACTIVITY_K_MAD` | `2.0` | MAD multiplier used to define descriptor activity; not the final event threshold. |
| `FREQ_DESC_NORMALIZE_SUBBAND` | `true` | Median/MAD-normalise each sub-bin before building descriptors when true. |
| `LINKAGE_METHOD` | `"average"` | Hierarchical linkage: "average" (default), "single", "complete", or "weighted". Ward/centroid/median are disallowed for the precomputed dissimilarity. |
| `N_GROUPS` | `null` | null for the default distance cut; otherwise integer from 1 through N_FINE_BANDS for a requested maximum group count. |
| `DISTANCE_THRESHOLD` | `0.55` | Distance cut on 1-similarity, in (0,1]. Larger cuts generally allow more merging. Ignored when N_GROUPS is set. |
| `CLUSTER_AGG` | `"median"` | Group drift aggregation: "median" (default), "mean", or "max". |
| `FIG_DPI` | `120` | PNG resolution for prediction/drift plots. Reference-only plots retain the source plotting function's 200 DPI. |
| `EVENT_SMOOTH_K` | `7` | Positive odd smoothing length for the signal-derived diagnostic event curve. |
| `EVENT_PEAK_K_MAD` | `2.0` | MAD multiplier for diagnostic event-curve peaks. Expert proxy scoring only, not final GMM segmentation. |
| `EVENT_PEAK_MIN_SEP_WINDOWS` | `2` | Minimum diagnostic event-peak separation, steps. Expert proxy scoring only. |
| `PEAK_MATCH_TOL_WINDOWS` | `2` | Allowed peak difference, steps, for expert proxy agreement metrics. |
| `BOX_SMOOTH_ACTIVITY_K` | `5` | Positive odd smoothing of box activity in expert proxy scoring. Does not alter final GMM boxes. |
| `GMM_RANDOM_STATE` | `0` | Integer random seed for per-group two-component mixtures. Default 0. |
| `GMM_MIN_EVENT_POSTERIOR` | `0.5` | Event-state posterior threshold in (0,1]. Default 0.50; try 0.6/0.7 for stricter local state decisions, then inspect outputs. |
| `GMM_SMOOTH_LABELS_K` | `3` | Majority-filter length in steps; positive odd integer. 1 disables label smoothing. |
| `GMM_MERGE_GAP_WINDOWS` | `2` | Nonnegative count of inactive micro-window steps to bridge. Default 2 = 0.25 s of step positions. |
| `GMM_MIN_EVENT_WINDOWS` | `2` | Minimum active-region length after gap merging, in steps. Default 2 spans 0.375 s with the default window/step. |
| `GMM_MIN_EVENT_AREA_Q` | `null` | null disables region-strength filtering. A number in [0,1] filters by the within-group event-area quantile; this is drift area, not time-frequency box area. |
| `GMM_USE_LOG_DRIFT` | `false` | Apply log1p to shifted group drift before GMM fitting when true; default false. |
| `DENOISE_METHOD` | `"hpss_percussive"` | "hpss_percussive" (default), "none", "spectral_gate", "hpss_residual", or "median_specsub". Acoustic assumption that may affect sustained calls. |
| `USE_PCEN` | `true` | true uses per-window PCEN; false uses per-window log-power dB. Changes representations and should be validated as a research change. |
| `NOISE_PROFILE_SEC` | `2.0` | Initial seconds used by spectral_gate as a noise profile. Only applies to that denoising mode. |
| `SPEC_SUB_FLOOR` | `0.1` | Floor fraction for spectral_gate. Only applies to that mode. |
| `HPSS_KERNEL_SIZE` | `31` | Median-filter kernel size for HPSS; positive integer. Used by HPSS denoising modes. |
| `HPSS_POWER` | `2.0` | Positive masking power for HPSS. |
| `HPSS_MARGIN` | `[1.0, 5.0]` | Two-element JSON array [harmonic_margin, percussive_margin], each at least 1. Default [1.0,5.0]. |
| `SPEC_FLOOR_MODE` | `"median"` | "median" or "mean" stationary floor for median_specsub only. |
| `SPEC_FLOOR_STRENGTH` | `1.0` | Nonnegative stationary-floor subtraction strength for median_specsub only. |

The spectral hop HOP_LENGTH is measured in audio samples; HOP_S is measured in seconds. Plot dimensions/legends belong to visualization code, not the detection settings. HPSS and spectral-subtraction modes are alternatives: parameters for inactive modes are ignored.
