# Development and validation

Use the environment from the installation guide:

```bash
python -m pip install -e ".[dev,notebooks]"
python -m pytest -q
```

Unit checks exercise annotation validation/matching, duplicate filenames, event-region timing, cross-group overlap, conditional evaluation geometry, empty output handling and a silence edge case. These are software checks, not public example data or ecological benchmarks. The end-to-end integration test requires the real study recording and reference table; it is skipped when `WABAD_TEST_CONFIG` is unset.

After obtaining the study data, run the real-recording check:

| Shell | Command to select the test data |
| --- | --- |
| Linux/macOS | `export WABAD_TEST_CONFIG=configs/paper_example.json` |
| Windows PowerShell | `$env:WABAD_TEST_CONFIG="configs/paper_example.json"` |
| Windows Command Prompt | `set WABAD_TEST_CONFIG=configs/paper_example.json` |

Then run `python -m pytest -q` from the repository root. The integration test writes results and annotation copies to a temporary test directory; it does not alter the study input files. It checks completion, resume reuse, invalidation after reference/config changes, and that references do not influence candidate generation.

The GitHub Actions workflow targets Python 3.11 and 3.12. Without separately provisioned study files, its integration test is skipped. A configured CI job does not establish real-data reproduction. Local checks are recorded in docs/VALIDATION.md.

Keep user-facing settings in JSON, maintain the original ARD figures, and document scientific changes in PROVENANCE.md. Keep audio and full result archives in the study Zenodo release and outside Git. Use separate output directories for experiments.
