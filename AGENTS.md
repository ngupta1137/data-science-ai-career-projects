# Rules for coding agents in this repository

These rules apply to any agent working here, including Antigravity, Claude Code and Codex. They protect one thing: every number in this repository must survive a hostile follow-up question. The current task list is in [docs/HANDOFF.md](docs/HANDOFF.md).

## What this repository is

End-to-end data science and AI reference projects for career learners, archived on Zenodo under the author's name. Read [docs/evidence-standard.md](docs/evidence-standard.md) and [docs/repo-layout.md](docs/repo-layout.md) before changing anything. Copy the structure of `13-promotion-lift` or `14-out-of-stock-detection` for a new project.

## Non-negotiable rules

1. **Set the gate before results.** Write the decision gate (`GATE_RULE` and the checks in `config.py`) and commit it before the first full run. Never change a gate, a threshold, a headline model or a metric after seeing results. If you must change process, keep the old verdict and explain it in the project README under "Process note (read this)".
2. **No hand-typed numbers.** Every number in `reports/` is rendered by `report.py` from `results/metrics.json` and `results/error_analysis.json`. Keep result numbers out of READMEs: describe the direction and point to the report. If a README needs one, it must match the rendered report and name it. Configuration values (sizes, assumptions) are fine.
3. **Report failures as they are.** A failed gate is a valid result. Write "The decision gate FAILED, and that is reported as is." in the README status line. Never tune until it passes.
4. **Label what exists.** Use exactly these words: built (code runs, results committed), designed (written down, not run), proposed (an idea). Never describe a designed thing as built.
5. **Never invent numbers, sources or citations.** If a dataset reference is unverified, write "Reference to verify before release".
6. **No em dashes anywhere** (the character U+2014). The release gate fails on them. Use commas, colons or full stops.
7. **Plain, direct language.** Short sentences. No marketing words.
8. **No data or model files in git.** Third-party data is downloaded by script into `data/raw/` or `data/cache/`, which are ignored. Allowed in `data/`: `DATA_CARD.md`, `CHECKSUMS.txt`, `raw/`, `cache/`, `corpus.jsonl`, `questions.jsonl`. No CSV, parquet, pickle or model weights.
9. **Simulations must say so.** A simulated project sets `DATA_SOURCE = "simulated_known_truth"`, says "simulated" in the README status line and the data card, and names what does not transfer to real systems.
10. **Do not touch other projects** unless the task names them.

## Project layout (enforced by `tools/check_release.py`)

```text
NN-short-name/
  README.md        status line, domain link, decision context, what you learn, traps,
                   process note, not done, run commands, files table
  STATUS           planned | in_progress | built | released
  <package>/       unique package name; must contain __init__.py, config.py, data.py,
                   metrics.py, train.py, report.py
  tests/           conftest.py plus one file per concern
  data/            DATA_CARD.md, CHECKSUMS.txt
  results/         metrics.json, error_analysis.json, one plot (written by train.py)
  reports/         model_card.md, decision_memo.md (written by report.py)
```

Folder names match `[0-9][0-9]-*`. Every project must appear in the root README table and in at least one guide in `domains/`.

## What train.py must write

- `metrics.json` with: `project`, `data_source`, `fast_mode`, `seed`, `versions`, `data` (including `sha256` from the generator fingerprint or download checksum), `assumptions`, a `headline` block with value, 95% interval and interval method, `gate` (`rule`, `checks`, `passed` as the conjunction of checks), `traps`, `caveats`.
- `error_analysis.json` with segment tables and any post hoc study, each post hoc block carrying a `status` string that says it was added after results and does not change the verdict.
- A `--fast` flag that sets `fast_mode: true`. Reports from fast runs must show the banner "Not a release result." The release gate rejects them.
- A `--write-checksum` flag and a fingerprint check that refuses to run if the data generator or download changed.

## Tests every project needs

- Metrics checked against hand-computed values.
- Data generator or loader mechanisms (shapes, validation errors, the mechanism each trap relies on).
- A fast pipeline test: files written, fast banner present, deterministic rerun, gate equals the conjunction of its checks, each trap shows the direction the README claims.
- A reproduction test that reruns the full pipeline and compares with committed `results/metrics.json`, skipped when no full results are committed. Keep a full run under about 30 seconds so CI stays fast.

## Workflow

1. Create a branch per task: `git checkout -b <task-name>`. Never push straight to `main`.
2. Commit the config and gate first, then the first full results in a later commit, so the history shows the gate came first.
3. Run, from the project folder: `python -m <package>.train --write-checksum` (first time), `python -m <package>.train`, `python -m <package>.report`.
4. Wire a new project: add the package to `known-first-party`, the tests folder to `testpaths` and the folder to `pythonpath` in `pyproject.toml`. Then run `python tools/wire_domain_project.py ...` for the root README, CHANGELOG and domain guide.
5. From the repo root run `make check` (ruff check, ruff format check, `tools/check_release.py --ci`, pytest). All must pass. The only accepted release-gate failure outside `--ci` mode is the LICENSE text, until the owner adds it.
6. Re-run `train` and `report` after any code change that can move a number, then commit the regenerated `results/` and `reports/`.
7. Open a pull request. Say in it what was built, the gate verdict, and anything designed but not built. Wait for green CI. The repository owner merges.

## Writing the README and reports

- Status line first, in bold, with the gate verdict.
- "Process note (read this)" lists every change of plan, post hoc study and known weakness of the gate.
- "Not done" lists what a reader might assume exists but does not.
- Decision memo sections: The question, The rule (set before results), The call (PASS or FAIL with each check), What this means, Where this is fragile, What a wrong call costs, Not done.
- Before you finish, reread every claim and ask: what would a sceptical expert one level up ask, and does the committed evidence answer it? Remove or qualify anything that does not survive that question.

## Commit messages

Imperative subject under 72 characters, a short body saying what changed and the gate verdict. Do not add attribution lines for tools you are not.
