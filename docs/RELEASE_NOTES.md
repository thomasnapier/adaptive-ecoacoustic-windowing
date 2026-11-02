# Release preparation

Suggested repository name: `adaptive-ecoacoustic-windowing`, matching the URL stated in the supplied manuscript.

The ZIP is ready to extract into a local Git repository or upload through GitHub's **Add file → Upload files** interface. It has not been published or pushed to GitHub by this preparation step. Upload the contents of the extracted folder, including the `.github` folder, rather than the ZIP itself. Use a Git client to preserve dotfiles if your file picker hides them.

Before a public research release:

- Select a software licence with the relevant rights holders; none was specified in the attachments.
- Confirm manuscript title, publication status and DOI in README/CITATION.cff.
- Complete data_release.json with the study Zenodo record/DOI, version and archive checksum, then verify the configured ARD input paths.
- Establish full-dataset agreement with the original analysis before making exact table/figure reproduction claims; real-recording execution requires the study audio and annotation tables.

The unpublished manuscript PDF and original notebook outputs are intentionally not bundled. The study WABAD files will be supplied through the author’s Zenodo release, recorded in data_release.json. The original ARD figures are included with this repository.
