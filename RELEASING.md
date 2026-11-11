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
| v0.1.0 | Scaffold, project 01, project 02. Proves the template. |
| v0.2.0 | Projects 03 to 05. |
| v0.3.0 | Projects 06 to 08. |
| v1.0.0 | After a hostile review of projects 01 to 08. |
| v1.1.0 | Roadmap projects (recommender), only if the projects above passed the bar. |

## Do not

- Write DOIs by hand before Zenodo mints them. Placeholders in README or CITATION.cff fail the gate.
- Describe roadmap projects as part of the claim until they exist.
