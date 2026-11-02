# ARD paper example

These are the three original WABAD example figures supplied by Thomas Napier for `ARD_20211027_072000.wav`. The PNG files are copied unchanged. They are provided as viewable paper examples, independently of the audio download.

| File | Input | Processing represented | Output shown |
| --- | --- | --- | --- |
| `ground_truth.png` | Original WABAD audio and expert annotation table | Spectrogram plotting with labelled reference regions | Expert boxes and species legend |
| `cluster_drifts.png` | Local spectral representations | Fine-band change measurement, band grouping, and aggregation | Five grouped drift trajectories |
| `predicted_boxes.png` | Group drift trajectories | Two-state GMM segmentation and interval post-processing | Candidate time–frequency boxes in cyan |

The labels in the reference plot come from expert annotations. The numeric group IDs in the drift plot are recording-local acoustic groups, not species labels. The supplied prediction figure contains cyan proposals; it does not display the expert boxes in green. Newly generated pipeline plots also include dashed green references when an annotation table is supplied.

The actual waveform, annotation table and machine-readable predictions cannot be recovered exactly from PNGs and are not reconstructed here. The study audio and reference tables are to be supplied through the author's planned Zenodo release. See [data download instructions](../../../docs/DATA_RELEASE.md) and [the configured paper example](../../../configs/paper_example.json). The metadata in `figures.json` record the original filenames and hashes.

Re-running uses the repository's WABAD settings. The historical figures are illustrative references; their original complete run manifest was not supplied, so numerical or pixel-identical reproduction is not asserted.
