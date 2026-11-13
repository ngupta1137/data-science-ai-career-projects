# Releasing and Zenodo

One concept DOI for this repository. One version DOI per release.

## One-time setup

1. Create the GitHub repository `ngupta1137/data-science-ai-career-projects` (public).
2. In Zenodo, open GitHub settings and switch the repository on.
3. Confirm `.zenodo.json` carries the ORCID and the license you want on the record.

## Per-release checklist

1. Replace `LICENSE` with the full official CC-BY-4.0 text. The release gate fails until this is done.
2. Run `make check` and `python tools/check_release.py` (the release mode). Both must pass.
3. For each project in the release, run its pipeline and commit `results/`. Set its `STATUS` file to `released`.
4. Update `CHANGELOG.md`.
5. Tag and create the GitHub release (`v0.1.0` and so on). Zenodo archives it.
6. Read the concept DOI and the version DOI from the Zenodo record.
7. Put the concept DOI in the README badge and in `CITATION.cff`. Run `python tools/check_release.py --post`.
8. Add the new version DOI under Related works on the sibling Zenodo records.
9. Add a row to the Zenodo artifact tracker. Log downloads, views, and citations monthly.

## Planned releases

| Version | Content |
|---|---|
| v0.1.0 | Fourteen reference projects (01 to 14): 01 to 04 on real public data, 05 to 14 on generated or simulated data. Gates failed: 03, 07, 08, 09, 10, 11, 14 (reported as is). No LLM called in 09 and 10. |
| v0.2.0 | Domain projects (15 onward). |
| v1.0.0 | Post-review stabilization. |

## Do not

- Write DOIs by hand before Zenodo mints them. Placeholders in README or CITATION.cff fail the gate.
- Describe roadmap projects as part of the claim until they exist.
