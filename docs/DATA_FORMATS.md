# Required files and formats

## Study data source

The worked example and batch walkthrough use the author’s WABAD study files supplied through the planned Zenodo release. See [DATA_RELEASE.md](DATA_RELEASE.md). Three original ARD figures are included for viewing; they do not replace the waveform and annotation tables.

## Minimum needed to detect events

One readable `.wav` or `.flac` audio file. Both lowercase and uppercase extensions are accepted. Mono and multichannel audio are accepted; detection averages to mono and resamples to the configured sample rate. Prefer original lossless audio. Upsampling a lower-rate recording cannot restore high frequencies.

No annotation file, label list, trained model, image, or GPU is needed for detection. Spectrogram images are outputs, not inputs. Start with the 60-second WABAD recordings used by the supplied workflow. Long audio is accepted but loaded and processed as a whole; memory use and processing time grow with duration. There is no automatic long-recording chunking.

## Optional references for inspection and evaluation

Supply one annotation table per recording, using one of these explicit schemas:

| Meaning | Compact column | Raven column | Unit |
| --- | --- | --- | --- |
| Event start | `t0` | `Begin Time (s)` | Seconds relative to the start of this audio file |
| Event end | `t1` | `End Time (s)` | Seconds relative to the start of this audio file |
| Lower frequency | `f0` | `Low Freq (Hz)` | Hertz |
| Upper frequency | `f1` | `High Freq (Hz)` | Hertz |
| Optional display label | `label` | `Species` | Text; used for reference plots only |

Use comma-separated `.csv`, tab-separated `.tsv`, or tab-separated Raven `.txt`. The parser detects tabs/commas from the header. Column names are case-insensitive and surrounding spaces are ignored. Additional Raven columns such as Selection, View, and Channel can remain. Channel is not used: the analysis is mono. Do not supply both compact and Raven aliases for the same field, since their intended precedence would be ambiguous.

For the worked example, use the actual `ARD_20211027_072000` annotation table supplied in the study data release. Do not estimate its coordinates or labels from the figure. A compact CSV uses the header `t0,t1,f0,f1,label`; a Raven table uses the time/frequency headers above. The original table's data rows should be retained unchanged.

Every row requires finite numbers with `0 <= t0 < t1` and `0 <= f0 < f1`. Use Hz, not kHz; decimal seconds, not timestamps or milliseconds. Event end times must fit the audio duration (a 0.02-s rounding tolerance is allowed). For a cropped recording, subtract its original start offset from the annotation times. References outside the configured analysis range remain unchanged; the pipeline does not silently clip or filter them. Inspect these before comparing results.

**Missing file** means no references were supplied and evaluation is unavailable. A **header-only file** means a deliberately empty reference set. Incomplete or missing annotations do not establish that extra proposals are incorrect. Conditional scores are undefined when the reference set is empty. The raw candidate count is still recorded.

## Matching audio to annotation tables

1. Prefer identical stems: `recording.wav` pairs with `recording.txt`, `.tsv`, or `.csv`.
2. If no exact match exists, a stem followed by `_` or `.` is allowed, for example `recording_SelectionTable.txt`.
3. More than one matching file is an error. The program does not choose one silently.
4. No matching annotation is allowed for detection and recorded as `reference_status=missing`.

Folder names accepted for audio are `recordings`, `recording`, `audio`, `audios`, `wav`, `wavs`, and `sounds`. Annotation folder names include `annotations`, `annotation`, `raven`, `raven_annotations`, `Raven Pro annotations`, `raven_pro`, `selection_tables`, `selection tables`, `gt`, and `ground_truth`. Matching is case-insensitive. Keep one audio folder and at most one annotation folder per site. Site folders are discovered at the root and one level below it; file discovery inside the matched audio/reference folders is recursive.

## Single-file mode

Create `configs/my_recording.json`:

```json
{
  "audio_file": "D:/my_audio/recording.wav",
  "annotation_file": "D:/my_annotations/recording.txt",
  "output_dir": "../outputs/my_recording",
  "parameters": {}
}
```

Omit `annotation_file` for unlabelled audio. `audio_file` takes precedence over `data_root` if both are present. The output site name is `single_file`. Run:

```bash
acoustic-windowing run --config configs/my_recording.json --dry-run
acoustic-windowing run --config configs/my_recording.json
```

## Prediction table

| Column | Meaning |
| --- | --- |
| `cluster_id` | Recording-local band-group identifier, not a species class |
| `t0`, `t1` | Candidate start and end, in seconds from audio-file start |
| `duration_s` | `t1 - t0` |
| `f0`, `f1` | Lower and upper candidate frequencies, in Hz |
| `bandwidth_hz` | `f1 - f0` |
| `peak_idx` | Micro-window index with greatest drift in the original active region |
| `support_left`, `support_right` | Inclusive micro-window indices of the original active region |
| `members` | Comma-separated zero-based fine-band indices, quoted within CSV when necessary |

Zero predictions produce a header-only CSV. Group boxes span the minimum-to-maximum member-band frequencies, including intervening nonmember bands. When overlapping same-group boxes merge, `t1` is updated; support/peak diagnostic indices are retained from the first box, as in the original code. Treat `t0`/`t1` as the final exported bounds and these indices as diagnostics. No calibrated confidence or species probability is exported.
