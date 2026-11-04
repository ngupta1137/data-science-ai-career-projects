"""Repo gate. Fails on layout drift and on the defects found in earlier repos in this set.

Modes
  --ci     layout, metadata, style. Safe to run on every push.
  (none)   release gate: --ci plus license text present and committed results.
  --post   after Zenodo mints the DOI: README and CITATION.cff hold a real concept DOI.

Exit code 0 means pass, 1 means fail. Every failure is printed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EM_DASH = chr(0x2014)
ORCID = "0009-0003-2501-7378"
MAX_FILE_BYTES = 1_000_000
STATUSES = {"planned", "in_progress", "built", "released"}

TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".cff", ".yml", ".yaml", ".toml", ".csv", ""}
BLOCKED_SUFFIXES = {".csv", ".parquet", ".zip", ".gz", ".joblib", ".pkl", ".pickle", ".h5", ".npy"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv", "raw", "cache"}

ROOT_REQUIRED = [
    "README.md",
    "CITATION.cff",
    ".zenodo.json",
    "LICENSE",
    "LICENSE-CODE",
    "LICENSE-NOTE.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "RELEASING.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "Makefile",
    ".editorconfig",
    ".github/workflows/ci.yml",
    ".github/dependabot.yml",
    ".github/CODEOWNERS",
    ".github/pull_request_template.md",
    "docs/evidence-standard.md",
    "docs/repo-layout.md",
]
ROOT_ALLOWED_DIRS = {"docs", "templates", "tools", ".github"}

PROJECT_REQUIRED = [
    "README.md",
    "STATUS",
    "data/DATA_CARD.md",
    "tests",
    "results",
    "reports/model_card.md",
    "reports/decision_memo.md",
]
PROJECT_ALLOWED = {"README.md", "STATUS", "data", "tests", "results", "reports"}
PACKAGE_REQUIRED = ["__init__.py", "config.py", "data.py", "metrics.py", "train.py", "report.py"]
DATA_ALLOWED = {"DATA_CARD.md", "CHECKSUMS.txt", "raw", "cache"}

PLACEHOLDER_PATTERNS = [
    re.compile(r"X{4,}"),
    re.compile(r"10\.5281/zenodo\.(?!\d)"),
    re.compile(r"LICENSE_TEXT_PENDING"),
]
DOI_RE = re.compile(r"10\.5281/zenodo\.\d{6,}")


def project_dirs() -> list[Path]:
    return sorted(p for p in ROOT.glob("[0-9][0-9]-*") if p.is_dir())


def iter_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            yield path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def package_dirs(project: Path) -> list[Path]:
    return sorted(
        p
        for p in project.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and p.name not in {"tests"}
    )


def check_root_layout(errors: list[str]) -> None:
    for rel in ROOT_REQUIRED:
        if not (ROOT / rel).exists():
            errors.append(f"missing root file: {rel}")
    for entry in ROOT.iterdir():
        if entry.name in SKIP_DIRS or (entry.name.startswith(".") and entry.is_file()):
            continue
        if entry.is_dir() and entry.name not in ROOT_ALLOWED_DIRS and entry not in project_dirs():
            errors.append(f"unexpected top-level folder: {entry.name} (projects are NN-name)")
    if not project_dirs():
        errors.append("no project folders (NN-name) found at repo root")
    readme = read(ROOT / "README.md") if (ROOT / "README.md").exists() else ""
    for project in project_dirs():
        if project.name not in readme:
            errors.append(f"root README does not list {project.name}")


def check_project_layout(errors: list[str]) -> None:
    for project in project_dirs():
        name = project.name
        for rel in PROJECT_REQUIRED:
            if not (project / rel).exists():
                errors.append(f"{name}: missing {rel}")
        packages = package_dirs(project)
        if len(packages) != 1:
            errors.append(f"{name}: expected exactly one Python package, found {len(packages)}")
        else:
            for rel in PACKAGE_REQUIRED:
                if not (packages[0] / rel).exists():
                    errors.append(f"{name}: package {packages[0].name} is missing {rel}")
        allowed = PROJECT_ALLOWED | {p.name for p in packages}
        for entry in project.iterdir():
            if entry.name in SKIP_DIRS:
                continue
            if entry.name not in allowed:
                errors.append(f"{name}: unexpected entry {entry.name}")
        data_dir = project / "data"
        if data_dir.is_dir():
            for entry in data_dir.iterdir():
                if entry.name not in DATA_ALLOWED:
                    errors.append(f"{name}: unexpected entry in data/: {entry.name}")
        status_file = project / "STATUS"
        if status_file.exists() and read(status_file).strip() not in STATUSES:
            errors.append(f"{name}: STATUS must be one of {sorted(STATUSES)}")


def check_files(errors: list[str]) -> None:
    for path in iter_files():
        rel = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file over {MAX_FILE_BYTES:,} bytes: {rel}")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            errors.append(f"data or model artifact must not be committed: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES and EM_DASH in read(path):
            errors.append(f"em dash in {rel}")


def check_metadata(errors: list[str]) -> None:
    zen = ROOT / ".zenodo.json"
    if zen.exists():
        try:
            data = json.loads(read(zen))
        except json.JSONDecodeError as exc:
            errors.append(f".zenodo.json is not valid JSON: {exc}")
        else:
            if not any(c.get("orcid") == ORCID for c in data.get("creators", [])):
                errors.append(".zenodo.json: ORCID missing from creators")
            for key in ("title", "description", "license", "upload_type"):
                if not data.get(key):
                    errors.append(f".zenodo.json: missing {key}")
    cff = ROOT / "CITATION.cff"
    if cff.exists() and ORCID not in read(cff):
        errors.append("CITATION.cff: ORCID missing")


def check_placeholders(errors: list[str], allow_license_marker: bool) -> None:
    for path in iter_files():
        if path.name == "check_release.py" or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(ROOT)
        text = read(path)
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.pattern == "LICENSE_TEXT_PENDING" and allow_license_marker:
                continue
            if pattern.search(text):
                errors.append(f"placeholder {pattern.pattern!r} in {rel}")


def check_release_only(errors: list[str]) -> None:
    license_text = read(ROOT / "LICENSE") if (ROOT / "LICENSE").exists() else ""
    if "Creative Commons" not in license_text or len(license_text) < 5000:
        errors.append("LICENSE does not hold the full CC-BY-4.0 text")
    for project in project_dirs():
        status_file = project / "STATUS"
        status = read(status_file).strip() if status_file.exists() else "planned"
        if status not in {"built", "released"}:
            continue
        metrics = project / "results" / "metrics.json"
        if not metrics.exists():
            errors.append(f"{project.name}: STATUS is {status} but results/metrics.json is missing")
        else:
            try:
                saved = json.loads(read(metrics))
            except json.JSONDecodeError as exc:
                errors.append(f"{project.name}: metrics.json is not valid JSON: {exc}")
            else:
                if saved.get("fast_mode"):
                    errors.append(f"{project.name}: metrics.json comes from a fast run")
                if str(saved.get("data_source", "")).startswith("synthetic"):
                    errors.append(f"{project.name}: metrics.json comes from synthetic data")
                if not saved.get("data", {}).get("sha256"):
                    errors.append(f"{project.name}: metrics.json has no data checksum")
        for report in ("model_card.md", "decision_memo.md"):
            path = project / "reports" / report
            if not path.exists():
                continue
            if "NOT_YET_GENERATED" in read(path):
                errors.append(f"{project.name}: reports/{report} has not been generated")
            if "Not a release result" in read(path):
                errors.append(f"{project.name}: reports/{report} comes from a non-release run")


def check_post(errors: list[str]) -> None:
    readme_dois = set(DOI_RE.findall(read(ROOT / "README.md")))
    cff_dois = set(DOI_RE.findall(read(ROOT / "CITATION.cff")))
    if not readme_dois:
        errors.append("README: no Zenodo DOI found")
    if not cff_dois:
        errors.append("CITATION.cff: no Zenodo DOI found")
    if readme_dois and cff_dois and not (readme_dois & cff_dois):
        errors.append("README and CITATION.cff hold different DOIs")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ci", action="store_true", help="layout, metadata, style only")
    parser.add_argument("--post", action="store_true", help="check DOIs after Zenodo release")
    args = parser.parse_args()

    errors: list[str] = []
    check_root_layout(errors)
    check_project_layout(errors)
    check_files(errors)
    check_metadata(errors)
    check_placeholders(errors, allow_license_marker=args.ci)
    if not args.ci:
        check_release_only(errors)
    if args.post:
        check_post(errors)

    if errors:
        print(f"FAIL ({len(errors)} problems)")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
