"""Turn raw extracted strings into canonical values, so 'March 15, 2024' equals '2024-03-15'.

Every function returns None when the string cannot be parsed. That is a schema failure, which the
evaluation counts separately from a wrong value.
"""

from __future__ import annotations

import re
from datetime import date

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MONTHS_LONG = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP"}
CURRENCY_CODES = ("EUR", "USD", "GBP")

_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DMY_SLASH = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")
_DMY_DOT = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")
_D_MON_Y = re.compile(r"^(\d{1,2}) ([A-Z][a-z]{2}) (\d{4})$")
_MONTH_D_Y = re.compile(r"^([A-Z][a-z]+) (\d{1,2}), (\d{4})$")


def _make(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(raw: str) -> str | None:
    """ISO date from one of the five formats. Day first for slashes; never month first."""
    s = raw.strip()
    if m := _ISO.match(s):
        return _make(int(m[1]), int(m[2]), int(m[3]))
    if m := _DMY_SLASH.match(s):
        return _make(int(m[3]), int(m[2]), int(m[1]))
    if m := _DMY_DOT.match(s):
        return _make(int(m[3]), int(m[2]), int(m[1]))
    if m := _D_MON_Y.match(s):
        if m[2] in MONTHS:
            return _make(int(m[3]), MONTHS.index(m[2]) + 1, int(m[1]))
        return None
    if (m := _MONTH_D_Y.match(s)) and m[1] in MONTHS_LONG:
        return _make(int(m[3]), MONTHS_LONG.index(m[1]) + 1, int(m[2]))
    return None


def parse_money(raw: str) -> float | None:
    """Amount from '1,234.50' or '1.234,50'. The decimal mark is the last . or , followed by two digits."""
    s = raw.strip()
    m = re.fullmatch(r"(\d[\d.,]*)([.,])(\d{2})", s)
    if not m:
        return None
    whole = re.sub(r"[.,]", "", m[1])
    if not whole.isdigit():
        return None
    return float(f"{whole}.{m[3]}")


def parse_currency(raw: str) -> str | None:
    s = raw.strip()
    if s in CURRENCY_SYMBOLS:
        return CURRENCY_SYMBOLS[s]
    return s if s in CURRENCY_CODES else None


def normalize(field: str, raw: str | None):
    """Canonical value for a field, or None if missing or unparseable."""
    if raw is None:
        return None
    if field == "date":
        return parse_date(raw)
    if field == "total":
        return parse_money(raw)
    if field == "currency":
        return parse_currency(raw)
    value = " ".join(raw.split())
    return value or None
