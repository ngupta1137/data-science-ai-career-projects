# License layout

| Content | License | File |
|---|---|---|
| Source code (`*.py`, workflows, tooling) | MIT | `LICENSE-CODE` |
| Documentation, reports, model cards, decision memos, and any data authored here | CC-BY-4.0 | `LICENSE` |
| Third-party datasets | Their own terms | Not redistributed. Each project's data card names the source and terms, and a script downloads the data. |

## Why third-party data is not committed

A dataset's license is set by its owner, not by this repository. Where terms are
restrictive, unclear, or require acceptance, this repo ships a download script
with a checksum instead of a copy. Small synthetic data that this repo generates
itself is covered by CC-BY-4.0.

## Zenodo record license

A Zenodo record carries one license field. The record uses `cc-by-4.0`, matching
the tracker standard for this set of repositories. Code remains MIT under
`LICENSE-CODE`. Confirm this choice at the first release.
