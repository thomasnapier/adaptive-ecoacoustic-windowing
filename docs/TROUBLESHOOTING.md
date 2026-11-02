# Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `acoustic-windowing` not found | Environment inactive or package not installed | Activate `.venv`, rerun `python -m pip install -e .`; alternatively use `python -m adaptive_windowing.cli` |
| `No module named ...` in Jupyter | Notebook kernel differs from installation environment | Select the environment used for installation; install the notebooks extra there |
| No recordings found | `data_root` points at the audio folder instead of site/root, or folder names differ | Follow `SITE/recordings`; or use `audio_file` explicitly |
| JSON cannot be read | Comments, trailing comma, or Windows backslashes | Use the supplied JSON as a template; use forward-slash paths |
| Multiple matching annotations | Several tables share the audio stem | Keep one authoritative table or use explicit single-file mode |
| Missing column error | Table lacks explicit time/frequency fields | Use the Raven headers or compact `t0,t1,f0,f1` schema |
| References lie beyond audio | Times refer to the full recording rather than the supplied clip | Convert to clip-relative seconds before running |
| Frequencies or boxes look wrong | Frequencies entered as kHz or input bandwidth differs | Use Hz; inspect original audio sampling rate and `FMIN`/`FMAX` |
| Many candidate boxes | Broad/noisy activity, overlapping sources, or overly permissive state decisions | Inspect examples; try posterior 0.6 or 0.7 in a separate output folder; do not assume every box is biological |
| Weak or missing sustained calls | HPSS percussive retention or drift changes are weak | Compare `DENOISE_METHOD=none` as a documented research experiment; validate before generalising |
| Empty prediction CSV | No retained active events | Header-only output is valid; check waveform and frequency range; compare settings on inspected examples |
| Run is slow | Per-window spectral extraction and plotting, long recordings, or CPU thread oversubscription | Start with `--limit 1`, disable plots; if needed set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` in your shell |
| Process stopped halfway | Recording error or interruption | Read `batch_failures.csv`; rerun the command; only matching completed outputs are reused |
| Previously completed files rerun | Settings, data, dependencies, or code changed | Expected; separate experiment folders help track comparisons |
| Two evaluations disagree | Different thresholds, filtering, or matching rules | Compare the same prefix and criterion; see EVALUATION.md |

The command exits with a nonzero status if any recording fails, while saving successful per-recording outputs. Install/runtime problems that occur before discovery may not produce a failure CSV. Constant-input correlation warnings may occur on silence; these do not establish a failed recording. Inspect its summary and marker.
