import pytest

from extraction_eval import normalize as nz


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,234.50", 1234.5),
        ("1.234,50", 1234.5),
        ("12.50", 12.5),
        ("12,50", 12.5),
        ("1,234,567.89", 1234567.89),
    ],
)
def test_parse_money(raw, expected):
    assert nz.parse_money(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["1,234", "abc", "", "12.5", "1.2.3"])
def test_parse_money_rejects(raw):
    assert nz.parse_money(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-03-15", "2024-03-15"),
        ("15 Mar 2024", "2024-03-15"),
        ("15/03/2024", "2024-03-15"),
        ("March 5, 2024", "2024-03-05"),
        ("15.03.2024", "2024-03-15"),
    ],
)
def test_parse_date(raw, expected):
    assert nz.parse_date(raw) == expected


@pytest.mark.parametrize(
    "raw", ["31/02/2024", "03/15/2024", "2024-13-01", "15 Foo 2024", "yesterday"]
)
def test_parse_date_rejects(raw):
    assert nz.parse_date(raw) is None


def test_day_first_is_assumed_for_slashes():
    assert nz.parse_date("04/05/2024") == "2024-05-04"


def test_currency():
    assert (
        nz.parse_currency("€") == "EUR"
        and nz.parse_currency("$") == "USD"
        and nz.parse_currency("£") == "GBP"
    )
    assert nz.parse_currency("GBP") == "GBP" and nz.parse_currency("CHF") is None


def test_normalize_dispatch():
    assert nz.normalize("vendor", "  Acme   Ltd ") == "Acme Ltd"
    assert nz.normalize("vendor", "   ") is None
    assert nz.normalize("total", None) is None
    assert nz.normalize("date", "March 15, 2024") == nz.normalize("date", "2024-03-15")
