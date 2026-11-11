"""Generated invoices with known field values and known distractors.

Three templates are 'seen' (used for training) and two are 'unseen' (test only). Each template has
label variants, optional lines and distractor values (subtotal, due date, purchase order, the
customer's name, an exchange-rate note). Because the code writes every document, the true value of
every field and the identity of every distractor are known.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from . import config
from .config import CHECKSUM_FILE
from .normalize import MONTHS, MONTHS_LONG, normalize


class DataValidationError(ValueError):
    """Raised on a schema problem or a fingerprint mismatch."""


@dataclass
class Doc:
    text: str
    truth: dict  # canonical values: vendor, invoice_number, date (ISO), total (float), currency
    template: str
    distractors: dict = field(default_factory=dict)  # field -> {role: raw string}


_SYL = (
    "bar",
    "ton",
    "del",
    "mer",
    "kro",
    "van",
    "lex",
    "nor",
    "sav",
    "tri",
    "ola",
    "pex",
    "quin",
    "ruda",
    "zel",
)
_SUFFIX = ("Ltd", "GmbH", "Inc", "LLC", "BV", "Corp")
_SYMBOL = {"EUR": "€", "USD": "$", "GBP": "£"}


def _company(rng: np.random.Generator) -> str:
    parts = rng.choice(_SYL, size=int(rng.integers(2, 4)))
    return "".join(parts).capitalize() + " " + str(rng.choice(_SUFFIX))


def _invoice_number(rng: np.random.Generator, year: int) -> str:
    style = int(rng.integers(0, 3))
    n = int(rng.integers(1, 99999))
    if style == 0:
        return f"INV-{year}-{n:05d}"
    if style == 1:
        return f"{'ABCDEFGHJK'[int(rng.integers(0, 10))]}{n:06d}"
    return f"{year}/{n % 10000:04d}"


def _fmt_date(d: date, fmt: str) -> str:
    if fmt == "iso":
        return d.isoformat()
    if fmt == "d_mon_y":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if fmt == "dmy_slash":
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    if fmt == "month_d_y":
        return f"{MONTHS_LONG[d.month - 1]} {d.day}, {d.year}"
    return f"{d.day:02d}.{d.month:02d}.{d.year}"  # dmy_dot


def _fmt_money(x: float, style: str) -> str:
    us = f"{x:,.2f}"
    return us if style == "us" else us.replace(",", "X").replace(".", ",").replace("X", ".")


def _pick(rng: np.random.Generator, options: tuple[tuple[str, float], ...]) -> str:
    labels, weights = zip(*options, strict=True)
    return str(rng.choice(labels, p=np.array(weights) / sum(weights)))


# Label variants and their weights, per template. The rules in extractors.py cover only variants
# with weight 0.3 or more, which is how a hand-written rule set usually ends up.
LABELS = {
    "A": {
        "inv": (("Invoice No:", 0.6), ("Invoice Number:", 0.3), ("Inv No:", 0.1)),
        "date": (("Date:", 0.7), ("Invoice Date:", 0.3)),
        "total": (("Total:", 0.7), ("Total due:", 0.2), ("Balance due:", 0.1)),
    },
    "B": {
        "inv": (("INVOICE #", 0.8), ("INVOICE NO.", 0.2)),
        "date": (("Issued", 0.8), ("Dated", 0.2)),
        "total": (("Amount Due:", 0.8), ("Total Due:", 0.2)),
    },
    "C": {
        "inv": (("Ref", 0.7), ("Invoice ref", 0.3)),
        "date": (("Invoice date", 0.7), ("Date issued", 0.3)),
        "total": (("TOTAL", 0.8), ("Total payable", 0.2)),
    },
}


def _amounts(rng: np.random.Generator) -> tuple[list[float], float, float, float]:
    items = [round(float(np.exp(rng.normal(4.5, 1.0))), 2) for _ in range(int(rng.integers(2, 5)))]
    sub = round(sum(items), 2)
    tax = round(sub * float(rng.choice([0.07, 0.19, 0.2])), 2)
    return items, sub, tax, round(sub + tax, 2)


def generate_doc(rng: np.random.Generator, template: str) -> Doc:
    d0 = date(2022, 1, 1) + timedelta(days=int(rng.integers(0, 1460)))
    due = d0 + timedelta(days=int(rng.integers(14, 61)))
    vendor, customer = _company(rng), _company(rng)
    while customer == vendor:
        customer = _company(rng)
    inv = _invoice_number(rng, d0.year)
    po = f"PO-{int(rng.integers(10000, 99999))}"
    cur = str(rng.choice(["EUR", "USD", "GBP"]))
    items, sub, tax, total = _amounts(rng)
    fx = bool(rng.random() < 0.25)
    prev_balance = bool(rng.random() < 0.15)
    prev_val = round(float(np.exp(rng.normal(4.5, 1.0))), 2)
    has_po, has_tax = bool(rng.random() < 0.7), bool(rng.random() < 0.8)
    swap_dates = bool(rng.random() < 0.2)  # the due line comes before the issue line
    net_total_label = bool(rng.random() < 0.3)  # the net line is worded like a total
    other = str(rng.choice([c for c in ("EUR", "USD", "GBP") if c != cur]))
    fx_line = f"Exchange rate: 1 {cur} = {1 + float(rng.random()):.2f} {other}"

    style = "eu" if template in ("C", "G") else "us"
    datefmt = {"A": "iso", "B": "d_mon_y", "C": "dmy_slash", "E": "month_d_y", "G": "dmy_dot"}[
        template
    ]
    ds, dd = _fmt_date(d0, datefmt), _fmt_date(due, datefmt)
    money = lambda x: _fmt_money(x, style)  # noqa: E731
    sym = _SYMBOL[cur]
    item_lines = [f"Service {i + 1}  {money(x)}" for i, x in enumerate(items)]
    lines: list[str]

    if template == "A":
        lab = LABELS["A"]
        lines = [
            vendor,
            f"{_pick(rng, lab['inv'])} {inv}",
            f"{_pick(rng, lab['date'])} {ds}",
            f"Due: {dd}",
        ]
        if has_po:
            lines.append(f"PO: {po}")
        lines.append(f"Bill to: {customer}")
        lines += item_lines
        lines.append(f"{'Total (excl. tax):' if net_total_label else 'Subtotal:'} {money(sub)}")
        if has_tax:
            lines.append(f"Tax: {money(tax)}")
        if prev_balance:
            lines.append(f"Previous balance: {money(prev_val)}")
        lines.append(f"{_pick(rng, lab['total'])} {cur} {money(total)}")
    elif template == "B":
        lab = LABELS["B"]
        lines = [
            f"{_pick(rng, lab['inv'])}{inv}",
            vendor,
            f"{_pick(rng, lab['date'])} {ds} | Payment due {dd}",
        ]
        lines.append(f"Customer: {customer}")
        if has_po:
            lines.append(f"Order ref {po}")
        lines += [f"{ln.split('  ')[0]}  {sym}{ln.split('  ')[1]}" for ln in item_lines]
        lines.append(f"{'Total net' if net_total_label else 'Net'} {sym}{money(sub)}")
        if has_tax:
            lines.append(f"VAT {sym}{money(tax)}")
        if prev_balance:
            lines.append(f"Previous balance {sym}{money(prev_val)}")
        lines.append(f"{_pick(rng, lab['total'])} {sym}{money(total)}")
    elif template == "C":
        lab = LABELS["C"]
        lines = [
            f"Bill from | {vendor}",
            f"{_pick(rng, lab['inv'])} | {inv}",
            f"{_pick(rng, lab['date'])} | {ds}",
            f"Due date | {dd}",
        ]
        if has_po:
            lines.append(f"Purchase order | {po}")
        lines.append(f"Bill to | {customer}")
        lines += [ln.replace("  ", " | ") for ln in item_lines]
        lines.append(f"{'Total net' if net_total_label else 'Sub-total'} | {money(sub)}")
        if has_tax:
            lines.append(f"Tax | {money(tax)}")
        if prev_balance:
            lines.append(f"Previous balance | {money(prev_val)}")
        lines.append(f"{_pick(rng, lab['total'])} | {money(total)} {cur}")
    elif template == "E":
        lines = [
            f"{customer} - Billing Statement",
            f"Document ref   {inv}",
            f"Billing date   {ds}",
            f"Pay by         {dd}",
            f"Supplier       {vendor}",
        ]
        if has_po:
            lines.append(f"Customer PO    {po}")
        lines += [f"{ln.split('  ')[0]}  {sym}{ln.split('  ')[1]}" for ln in item_lines]
        lines.append(
            f"{'Total before tax' if net_total_label else 'Net amount'}  {sym}{money(sub)}"
        )
        if has_tax:
            lines.append(f"Sales tax   {sym}{money(tax)}")
        if prev_balance:
            lines.append(f"Carried forward  {sym}{money(prev_val)}")
        lines.append(f"Grand total {sym}{money(total)}")
    else:  # G
        lines = [f"Rechnung Nr. {inv}", vendor, f"Datum: {ds}", f"Fällig am: {dd}"]
        if has_po:
            lines.append(f"Bestellnummer: {po}")
        lines.append(f"Kunde: {customer}")
        lines += item_lines
        lines.append(f"Zwischensumme: {money(sub)}")
        if has_tax:
            lines.append(f"MwSt: {money(tax)}")
        if prev_balance:
            lines.append(f"Vortrag: {money(prev_val)}")
        lines.append(f"Gesamtbetrag: {money(total)} {cur}")
    if swap_dates and template != "B":
        lines[2], lines[3] = lines[3], lines[2]
    if fx:
        lines.append(fx_line)

    total_dist = {"subtotal": money(sub)}
    if has_tax:
        total_dist["tax"] = money(tax)
    for i, x in enumerate(items):
        total_dist[f"item{i + 1}"] = money(x)
    if prev_balance:
        total_dist["previous_balance"] = money(prev_val)
    distractors = {
        "vendor": {"customer": customer},
        "invoice_number": {"po_number": po} if has_po else {},
        "date": {"due_date": dd},
        "total": total_dist,
        "currency": {"fx_note": other} if fx else {},
    }
    truth = {
        "vendor": vendor,
        "invoice_number": inv,
        "date": d0.isoformat(),
        "total": total,
        "currency": cur,
    }
    return Doc("\n".join(lines), truth, template, distractors)


def generate(rng: np.random.Generator, n: int, templates: tuple[str, ...]) -> list[Doc]:
    return [generate_doc(rng, str(rng.choice(templates))) for _ in range(n)]


_CONFUSE = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5"}


def add_ocr_noise(rng: np.random.Generator, text: str, rate: float) -> str:
    """Swap look-alike characters with probability `rate` each. A crude stand-in for OCR error."""
    if rate <= 0:
        return text
    chars = list(text)
    hit = rng.random(len(chars)) < rate
    return "".join(_CONFUSE.get(c, c) if h else c for c, h in zip(chars, hit, strict=True))


def make_splits(seed: int = config.SEED) -> dict[str, list[Doc]]:
    rng = np.random.default_rng([seed, 10])
    return {
        "train": generate(rng, config.N_TRAIN, config.SEEN_TEMPLATES),
        "dev": generate(rng, config.N_DEV, config.SEEN_TEMPLATES),
        "test_seen": generate(rng, config.N_TEST_SEEN, config.SEEN_TEMPLATES),
        **{
            f"test_unseen_{t}": generate(rng, config.N_TEST_UNSEEN, (t,))
            for t in config.UNSEEN_TEMPLATES
        },
    }


def validate(docs: list[Doc]) -> None:
    problems = []
    for i, d in enumerate(docs):
        for f in config.FIELDS:
            if f not in d.truth:
                problems.append(f"doc {i} missing truth for {f}")
        if normalize("date", d.truth.get("date", "")) is None:
            problems.append(f"doc {i} truth date is not ISO")
    if problems:
        raise DataValidationError("; ".join(problems[:5]))


def fingerprint(seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    docs = generate(rng, 60, ("A", "B", "C", "E", "G"))
    blob = "\n==\n".join(d.text + "|" + repr(sorted(d.truth.items())) for d in docs)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    path.write_text(f"{digest}  generator_probe(seed=0)\n", encoding="utf-8")
    return digest


def verify_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    if path.exists():
        expected = path.read_text(encoding="utf-8").split()[0]
        if expected != digest:
            raise DataValidationError(
                f"generator fingerprint mismatch: expected {expected}, got {digest}"
            )
    return digest
