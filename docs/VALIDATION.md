# Validation performed

The public walkthrough now requires the real WABAD study recording and annotations. The three original ARD figures were supplied as PNG files and are included unchanged; the corresponding waveform and annotation table were not supplied with this revision and are intended for the author’s Zenodo data release.

Local checks on Python 3.12.14 / Linux:

- **7 software tests passed.** These cover explicit annotation columns and coordinate validity, ambiguous matching, duplicate stems, empty-output schema, run-length timing, cross-group overlap, conditional-metric geometry, and a silence/failure-handling edge case.
- **1 real-WABAD integration test was skipped** because `WABAD_TEST_CONFIG` was not set to installed study data. It checks end-to-end completion, resume invalidation and independence from references when those real files are available; see CONTRIBUTING.md.
- Both revised notebooks passed notebook-schema and Python-cell syntax checks. Their data-loading step stops with an actionable missing-file error until the real recording/table are installed. Their full numerical execution on WABAD is not claimed here.
- The supplied ARD image files were copied byte-for-byte, and their hashes are recorded in `examples/paper/ARD_20211027_072000/figures.json`. They were not regenerated during this revision.
- The software’s scientific implementation is unchanged by this revision. The example configuration and user-facing routes now require the real study data.
- The silence test emits a NumPy constant-correlation warning from the retained descriptor code; it still produces zero proposals and a valid empty CSV.

The full WABAD study analysis, exact recovery of the supplied example outputs, reconstructed conditional evaluator's dataset-level agreement, original ablations and all manuscript figures remain to be checked against the released inputs and original run settings. The original ARD run manifest was not supplied, so the included figures are visual references rather than a claim of verified identical results.

The configured GitHub Actions matrix targets Python 3.11 and 3.12. It has not run remotely as part of this revision. Its real-data integration test is skipped unless study data are provisioned.

## Direct runtime versions

| Package | Version |
| --- | --- |
| numpy | 2.3.5 |
| scipy | 1.17.0 |
| pandas | 2.2.3 |
| librosa | 1.0.0 |
| matplotlib | 3.10.8 |
| scikit-learn | 1.8.0 |
| soundfile | 0.14.0 |

`requirements-tested.txt` records these direct pins. It is not a complete transitive environment lock. `run_manifest.json` records the environment and source hashes for each user run.
