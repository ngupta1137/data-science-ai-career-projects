# Repository layout

Every project follows the same shape, so you can open any folder and know where to look.

```text
data-science-ai-career-projects/
├── README.md                 project index, how to run, status of every project
├── pyproject.toml            ruff and pytest settings for the whole repo
├── requirements.txt          pinned runtime dependencies
├── requirements-dev.txt      runtime plus lint tooling
├── Makefile                  make check runs everything CI runs
├── docs/                     repo-wide documents (evidence standard, this file)
├── domains/                  index of the projects by domain (guides only, no code)
├── templates/                skeleton for a new project
├── tools/                    release gate (layout, metadata, style checks)
├── .github/                  CI, issue and PR templates, Dependabot, CODEOWNERS
│
├── 01-house-price-regression/
│   ├── README.md             decision context, what you learn, how to run, what is not done
│   ├── STATUS                planned | in_progress | built | released
│   ├── house_price/          the Python package (unique name per project)
│   │   ├── config.py         constants, paths, the decision gate
│   │   ├── data.py           download, checksum, validation
│   │   ├── features.py       feature construction
│   │   ├── splits.py         split design and leakage checks
│   │   ├── models.py         models and search grids
│   │   ├── metrics.py        metrics, uncertainty, segment tables
│   │   ├── train.py          fit, evaluate, write results/
│   │   ├── report.py         render reports/ from results/
│   │   └── synthetic.py      synthetic data for tests only
│   ├── tests/                one file per concern, hand-computed expected values
│   ├── data/
│   │   ├── DATA_CARD.md      source, terms, quirks, what was not checked
│   │   ├── CHECKSUMS.txt     sha256 of the downloaded file (committed)
│   │   ├── raw/              downloaded data (never committed)
│   │   └── cache/            download cache (never committed)
│   ├── results/              metrics.json, error_analysis.json, plots (written by code)
│   └── reports/              model_card.md, decision_memo.md (rendered from results/)
│
├── 02-fraud-detection/       same shape, package name fraud_detection
└── 03-... to 05-...          same shape
```

## Rules the release gate enforces

- Project folders are named `NN-name` and sit at the repo root.
- Each project has exactly one Python package, and its name is unique across the repo.
- Only the entries shown above are allowed in a project folder and in its `data/` folder.
- No file over 1 MB. No `.csv`, `.parquet`, `.zip`, `.joblib`, or other data or model artifact is committed.
- Reports are generated from `results/`, never edited by hand.
- Every project is listed in the root README.
- No em dashes anywhere.

## Why this shape

- **One folder per project.** A learner can copy or read a project without the rest.
- **A package per project, not a loose `src/`.** Unique package names let one `pytest` run cover the whole repo with no path tricks.
- **Data is a script and a checksum, not a file.** Third-party licenses are respected and the repo stays small.
- **Results are committed, reports are rendered.** A number in a report can always be traced to a file written by code.
