# WABAD study data on Zenodo

## Download status

Thomas Napier plans to supply the WABAD study recordings and accompanying tables through a dedicated Zenodo release. **The study record URL, DOI, archive name and checksum have not yet been assigned in this repository.** Their authoritative metadata belong in [data_release.json](../data_release.json); fields remain `null` until known.

The original ARD paper figures are already included under [examples/paper/ARD_20211027_072000](../examples/paper/ARD_20211027_072000/README.md). To execute the walkthrough, download the actual study waveform and reference table. A PNG alone cannot supply the audio or exact coordinates.

## Download and extract once the record is available

1. Open the study Zenodo record identified in `data_release.json` and download its WABAD study-data archive.
2. Check the downloaded archive's SHA-256 against the value recorded for that release. On PowerShell use `Get-FileHash WABAD.zip -Algorithm SHA256`; on Linux use `sha256sum WABAD.zip`; on macOS use `shasum -a 256 WABAD.zip`. Replace `WABAD.zip` with the actual release filename.
3. Extract the archive so its site folders sit directly inside this repository's `data/WABAD/` directory. Avoid an extra nested `WABAD/WABAD/` directory.
4. Check the file paths in `configs/paper_example.json`, then run the `--dry-run` command below. Batch users should also check `data_root` in `configs/wabad.json`.

The intended paper-example paths are:

| Required item | Path after extraction | Purpose |
| --- | --- | --- |
| Real recording | `data/WABAD/ARD/recordings/ARD_20211027_072000.wav` | Input to feature extraction and event detection |
| Matching expert table | `data/WABAD/ARD/annotations/ARD_20211027_072000.txt` | Reference plot and evaluation |

The `.txt` path is the intended standardised release path, not a claim about a table received in this session. If original folder names or Raven suffixes are retained in the final archive, update the config and manifest to those exact paths. The annotation file must contain the real exported rows, with seconds relative to the supplied waveform and frequencies in Hz.

```bash
acoustic-windowing run --config configs/paper_example.json --dry-run
acoustic-windowing run --config configs/paper_example.json
acoustic-windowing plot-gt --config configs/paper_example.json
```

There is no automatic download fallback to another dataset or example. If the files are missing, the workflow stops and identifies the required path.

## Suggested contents for the study data release

| Content | Why include it |
| --- | --- |
| Exact study WAV/FLAC recordings, including the ARD worked example | Makes the paper example and study input population available |
| Original corresponding Raven/CSV reference tables | Preserves annotation coordinates and expert labels |
| Recording manifest with site, filename, duration, sample rate, split/inclusion status and hashes | Identifies the precise analysis population and input bytes |
| Dataset version, upstream attribution, applicable data terms and permitted redistribution details | Connects the study subset to its original WABAD source |
| Original predicted-box CSVs and run settings, if available | Allows numeric comparison with the supplied figures and manuscript analyses |

The last item is useful reference output, not an additional input needed for detection. The repository currently includes the original figure PNGs, not invented machine-readable boxes. If reference predictions are added, keep them separate from newly computed `outputs/`.

## For the maintainer when publishing

Fill in the null metadata fields with the real record URL, DOI, version, archive name and archive checksum. Check that the release's extracted paths match the example and batch configs. Add the study data DOI to the README and citation metadata while retaining the original WABAD citation. Verify the ARD run and the intended full-data evaluation using the published archive before reporting reproduction results.
