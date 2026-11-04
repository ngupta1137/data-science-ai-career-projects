"""Repo gate. Fails on the defects that were found in earlier repos in this set.

Modes
  --ci     structure, metadata, style. Safe to run on every push.
  (none)   release gate: --ci plus license text present and committed results.
  --post   after Zenodo mints the DOI: README badge and CITATION.cff hold a real concept DOI.

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
TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".cff", ".yml", ".yaml", ".csv", ".toml", ""}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "raw", "cache"}

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
    "requirements.txt",
    ".github/workflows/ci.yml",
    "docs/evidence-standard.md",
]

PROJECT_REQUIRED = [
    "README.md",
    "data/DATA_CARD.md",
    "src",
    "tests",
    "reports/model_card.md",
    "reports/decision_memo.md",
]

PLACEHOLDER_PATTERNS = [
    re.compile(r"X{4,}"),
    re.compile(r"10\.5281/zenodo\.(?!\d)"),
    re.compile(r"LICENSE_TEXT_PENDING"),
]

DOI_RE = re.compile(r"10\.5281/zenodo\.\d{6,}")


def iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def check_structure(errors: list[str]) -> None:
    for rel in ROOT_REQUIRED:
        if not (ROOT / rel).exists():
            errors.append(f"missing root file: {rel}")
    projects = sorted(p for p in (ROOT / "projects").glob("[0-9][0-9]-*") if p.is_dir())
    if not projects:
        errors.append("no projects found under projects/")
    for project in projects:
        for rel in PROJECT_REQUIRED:
            if not (project / rel).exists():
                errors.append(f"{project.name}: missing {rel}")


def check_style(errors: list[str]) -> None:
    for path in iter_text_files():
        rel = path.relative_to(ROOT)
        text = read(path)
        if EM_DASH in text:
            errors.append(f"em dash in {rel}")


def check_metadata(errors: list[str]) -> None:
    zen = ROOT / ".zenodo.json"
    if zen.exists():
        try:
            data = json.loads(read(zen))
        except json.JSONDecodeError as exc:
            errors.append(f".zenodo.json is not valid JSON: {exc}")
        else:
            creators = data.get("creators", [])
            if not any(c.get("orcid") == ORCID for c in creators):
                errors.append(".zenodo.json: ORCID missing from creators")
            for key in ("title", "description", "license", "upload_type"):
                if not data.get(key):
                    errors.append(f".zenodo.json: missing {key}")
    cff = ROOT / "CITATION.cff"
    if cff.exists() and ORCID not in read(cff):
        errors.append("CITATION.cff: ORCID missing")


def check_placeholders(errors: list[str], allow_license_marker: bool) -> None:
    for path in iter_text_files():
        if path.name == "check_release.py":
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
    for project in sorted((ROOT / "projects").glob("[0-9][0-9]-*")):
        status_file = project / "STATUS"
        status = read(status_file).strip() if status_file.exists() else "planned"
        if status != "released":
            continue
        metrics = project / "results" / "metrics.json"
        if not metrics.exists():
            errors.append(f"{project.name}: STATUS is released but results/metrics.json is missing")
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
            if path.exists() and "NOT_YET_GENERATED" in read(path):
                errors.append(f"{project.name}: reports/{report} has not been generated")
            if path.exists() and "Not a release result" in read(path):
                errors.append(f"{project.name}: reports/{report} comes from a non-release run")


def check_post(errors: list[str]) -> None:
    readme = read(ROOT / "README.md")
    cff = read(ROOT / "CITATION.cff")
    readme_dois = set(DOI_RE.findall(readme))
    cff_dois = set(DOI_RE.findall(cff))
    if not readme_dois:
        errors.append("README: no Zenodo DOI found")
    if not cff_dois:
        errors.append("CITATION.cff: no Zenodo DOI found")
    if readme_dois and cff_dois and not (readme_dois & cff_dois):
        errors.append("README and CITATION.cff hold different DOIs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ci", action="store_true", help="structure, metadata, style only")
    parser.add_argument("--post", action="store_true", help="check DOIs after Zenodo release")
    args = parser.parse_args()

    errors: list[str] = []
    check_structure(errors)
    check_style(errors)
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
