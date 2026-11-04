"""Run each project's tests in its own process.

Every project has a package named `src`. Running them in one pytest process would
make them collide, so each gets a separate run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    projects = sorted(p for p in (ROOT / "projects").glob("[0-9][0-9]-*") if (p / "tests").is_dir())
    if not projects:
        print("no projects with tests found")
        return 1
    failed = []
    for project in projects:
        print(f"== {project.name}")
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=project)
        if result.returncode != 0:
            failed.append(project.name)
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("all project tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
