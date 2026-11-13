"""Wire a new domain project into the root README, CHANGELOG and domain guides.

Run from the repo root after the project is built:

    python tools/wire_domain_project.py NUM FOLDER TITLE SKILL STATUS DOMAIN BLURB CHANGELOG

DOMAIN is a file name in domains/ without .md, such as retail-cpg. Quote every argument.

pyproject.toml is not edited here: add the package to known-first-party, the tests folder to
testpaths and the project folder to pythonpath by hand, then run `make check`.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if len(sys.argv) != 9:
    sys.exit(__doc__)
num, folder, title, skill, status, slug, blurb, change = sys.argv[1:9]
WORDS = dict(
    enumerate(
        [
            "Thirteen",
            "Fourteen",
            "Fifteen",
            "Sixteen",
            "Seventeen",
            "Eighteen",
            "Nineteen",
            "Twenty",
            "Twenty-one",
            "Twenty-two",
            "Twenty-three",
            "Twenty-four",
            "Twenty-five",
            "Twenty-six",
        ],
        start=13,
    )
)

r = (ROOT / "README.md").read_text()
if f"[{title}]({folder})" not in r:
    rows = list(re.finditer(r"^\| \d\d \| .*$", r, re.M))
    last = rows[-1]
    r = (
        r[: last.end()]
        + f"\n| {num} | [{title}]({folder}) | {skill} | {status} |"
        + r[last.end() :]
    )
n = int(num)
r = re.sub(r"- (\w[\w-]*) reference projects", f"- {WORDS[n]} reference projects", r)
r = re.sub(r"Projects 05 to \d\d are built", f"Projects 05 to {num} are built", r)
r = re.sub(r"Projects 05 to \d\d run offline", f"Projects 05 to {num} run offline", r)
(ROOT / "README.md").write_text(r)

c = (ROOT / "CHANGELOG.md").read_text()
if change not in c:
    lines = c.splitlines()
    idx = max(i for i, line in enumerate(lines) if line.startswith("- Project "))
    lines.insert(idx + 1, f"- {change}")
    (ROOT / "CHANGELOG.md").write_text("\n".join(lines) + "\n")

g = ROOT / "domains" / f"{slug}.md"
t = g.read_text()
entry = f"- [{title}](../{folder}): {blurb}"
if entry not in t:
    if "## Projects built for this domain" not in t:
        t = t.replace(
            "## Start here",
            "## Projects built for this domain\n\nThese projects use a simulation shaped around this domain's own decision.\n\n"
            + entry
            + "\n\n## Projects that carry over from the core set",
            1,
        )
    else:
        t = t.replace(
            "\n\n## Projects that carry over from the core set",
            "\n" + entry + "\n\n## Projects that carry over from the core set",
            1,
        )
    g.write_text(t)

ix = ROOT / "domains" / "README.md"
t = ix.read_text()
header = next(line for line in t.splitlines() if line.startswith("| Project |"))
cols = [c.strip() for c in header.strip("|").split("|")][1:]
names = {
    "retail-cpg": "Retail and consumer goods",
    "banking-insurance": "Banking, fintech and insurance",
    "marketing-growth": "Marketing and growth",
    "ecommerce-media": "E-commerce, media and consumer apps",
    "healthcare": "Healthcare and life sciences",
    "operations-supply-chain": "Operations and supply chain",
    "enterprise-genai": "Enterprise generative AI",
}
row = (
    f"| [{num} {title}](../{folder}) | "
    + " | ".join("domain project" if c == names[slug] else "" for c in cols)
    + " |"
)
if f"](../{folder})" not in t:
    lines = t.splitlines()
    idx = max(i for i, line in enumerate(lines) if line.startswith("| ["))
    lines.insert(idx + 1, row)
    t = "\n".join(lines) + "\n"
    t = t.replace(
        "Key: **main** means start here for this domain, **also** means useful second.",
        "Key: **main** means start here for this domain, **also** means useful second, **domain project** means built for that domain.",
    )
    ix.write_text(t)
print("wired", folder)
