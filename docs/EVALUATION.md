# Evaluation and reproduction limits

## Three questions require different measurements

1. **Does a proposal recover some of an annotated event's time?** Temporal IoMin can be permissive when one interval is nested in another.
2. **Does it also cover a compatible frequency range?** Two-dimensional and frequency-gated criteria add spectral agreement.
3. **Do the boundaries agree tightly?** Two-dimensional IoU penalises both overextension and fragmentation.

These are event-proposal measurements, not species-classification accuracy. Extra boxes may reflect non-bird sounds or unannotated calls, but their validity requires inspection; incomplete labels do not prove that the detector is correct.

## Two explicit namespaces

| Output prefix | Origin | Filtering and matching | Default threshold |
| --- | --- | --- | --- |
| `notebook_` | Supplied WABAD notebook, cell 2 | All predictions, greedy one-to-one 2D IoU; additional GT-centred coverage and time-activity metrics | IoU 0.30 |
| `conditional_` | New implementation from manuscript v5 §4.2 | Keep predictions whose frequency interval overlaps any reference frequency interval in the recording; match separately per criterion | Fixed criteria below |

The reference-frequency filter is used **only after detection** for conditional evaluation. It uses ground truth and cannot be applied to unlabelled recordings as an operational detection filter. `n_pred_boxes` always counts raw proposals; `conditional_evaluable_prediction_count` counts the retained evaluation subset.

## Manuscript-oriented conditional criteria

For time–frequency intersection area `I`, predicted area `P`, and reference area `G`:

- `IoU = I / (P + G - I)`.
- `IoMin = I / min(P, G)`.
- Temporal/frequency IoMin apply the same definition to their respective interval lengths.
- Prediction purity is `I / P`. It is geometric box occupancy, not species purity.

| Name in output | Eligible pair | Matching |
| --- | --- | --- |
| `conditional_iou50_*` | 2D IoU ≥ 0.50 | Greedy one-to-one, ordered by IoU |
| `conditional_iomin50_*` | 2D IoMin ≥ 0.50 | Greedy one-to-one, ordered by 2D IoMin |
| `conditional_temporal_iomin50_*` | Temporal IoMin ≥ 0.50 | Greedy one-to-one, ordered by temporal IoMin, irrespective of pair frequency agreement |
| `conditional_t_gated_*` | Temporal IoMin ≥ 0.50, frequency IoMin ≥ 0.20, prediction purity ≥ 0.25 | Greedy one-to-one, ordered by temporal IoMin |
| `conditional_mean_best_temporal_iomin` | Best temporal IoMin for each reference | No one-to-one restriction; proposals can be reused |

Ties are broken by 2D IoMin, prediction purity and IoU, then input order. For T-Gated the primary ranking score is interpreted as temporal IoMin: the manuscript does not explicitly define a separate composite score. This implementation decision needs comparison with the original evaluation script before claiming exact reproduction.

All thresholded criteria export TP/FP/FN, precision, recall and F1, although the manuscript reports selected measures. Scores are fractions between 0 and 1, not percentages. Mean-best temporal IoMin is zero for a reference with no overlapping temporal proposal. Conditional metrics are undefined for an empty reference set and are omitted, with a status field explaining why. Missing reference files also produce no evaluation scores.

Notebook temporal union coverage can combine several proposals, and its time-only coverage does not require frequency agreement. It differs from one-to-one IoMin recall. Means across recordings also differ from pooled TP/FP/FN metrics. `batch_summary.csv` contains per-recording values; it does not automatically pool or reproduce the paper's site tables. For a pooled precision/recall/F1 calculation, sum TP/FP/FN first within the same criterion and population. Average mean-best scores weighted by reference count.

## What is and is not reproduced

The main scientific code was extracted from WABAD cell 2 (one-based) of the supplied batch notebook. It follows the manuscript's preprocessing, 20-band drift representation, hybrid grouping, and two-state GMM proposal generation with fixed defaults. The reference plotting function comes from cell 1 of the spectrogram notebook.

The manuscript's Table 3 refers to a reference-frequency-filtered population of proposals; the ablation table uses a different analysis population. Do not compare their proposal totals without reproducing those population rules.

This repository adds validated input/output handling, a command-line interface, educational notebooks, safe resume checks, original author-supplied example figures, and conditional criteria. **The conditional evaluator is a new reconstruction, not a recovered original script.** Its geometry is tested but dataset-level numerical agreement has not been established. The original post-hoc evaluation, bootstrap, full ablation, species/overlap stratification, and figure-generation workflows were not supplied as standalone scripts. They are not recreated or claimed here.

The supplied ARD PNGs document the original example visually; the waveform and annotation table are to be supplied through the study Zenodo release. The current checks cover file handling and metric geometry; real-data integration is conditional on those files being installed. Do not describe this release as a verified reproduction of the manuscript's percentages or all figures. To establish that, retain the exact WABAD version and eight-site selection, compare saved predictions and matching rules with the original analyses, then publish the resulting manifests and validated scripts.

Keep parameters fixed across evaluation recordings. The optional research grid-search helper can use GT to select per-recording settings; those exploratory results are not equivalent to the fixed unsupervised pipeline and are deliberately absent from the beginner runner.
