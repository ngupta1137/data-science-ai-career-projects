import numpy as np
import pytest

from extraction_eval import metrics
from extraction_eval.data import Doc

TRUTH = {
    "vendor": "Acme Ltd",
    "invoice_number": "INV-1",
    "date": "2024-03-15",
    "total": 360.0,
    "currency": "EUR",
}
DIST = {
    "date": {"due_date": "20/04/2024"},
    "total": {"subtotal": "300.00", "item1": "100.00"},
    "vendor": {"customer": "Beta GmbH"},
}
TEXT = "Acme Ltd INV-1 15/03/2024 20/04/2024 300.00 360.00 EUR"


def doc():
    return Doc(TEXT, dict(TRUTH), "A", DIST)


def test_wilson_matches_hand_value():
    lo, hi = metrics.wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)


def test_prf_hand_example():
    out = metrics.prf(np.array([1, 0, 1, 0], bool), np.array([1, 1, 1, 0], bool))
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == 0.5
    assert out["f1"] == pytest.approx(4 / 7)  # 2 * (2/3 * 1/2) / (2/3 + 1/2)


def test_prf_with_nothing_extracted():
    out = metrics.prf(np.zeros(3, bool), np.zeros(3, bool))
    assert np.isnan(out["precision"]) and out["recall"] == 0.0 and np.isnan(out["f1"])


def test_score_counts_a_format_difference_as_correct():
    pred = {
        "vendor": "Acme  Ltd",
        "invoice_number": "INV-1",
        "date": "15/03/2024",
        "total": "360.00",
        "currency": "€",
    }
    sc = metrics.score([doc()], [pred])
    assert sc["doc_correct"].all()


def test_score_one_wrong_field_fails_the_document():
    pred = {
        "vendor": "Acme Ltd",
        "invoice_number": "INV-1",
        "date": "20/04/2024",
        "total": "360.00",
        "currency": "EUR",
    }
    sc = metrics.score([doc()], [pred])
    assert (
        not sc["doc_correct"].any()
        and sc["correct"]["total"].all()
        and not sc["correct"]["date"].any()
    )


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("total", None, "missing"),
        ("total", "360.00", None),
        ("total", "12,5", "unparseable"),
        ("total", "300.00", "distractor:subtotal"),
        ("total", "100.00", "distractor:item"),
        ("date", "20/04/2024", "distractor:due_date"),
        ("vendor", "Beta GmbH", "distractor:customer"),
        ("total", "999.00", "not_in_text"),
        ("currency", "USD", "not_in_text"),
    ],
)
def test_classify_error(field, raw, expected):
    assert metrics.classify_error(field, raw, TRUTH[field], DIST, TEXT) == expected


def test_taxonomy_counts():
    pred = {
        "vendor": "Beta GmbH",
        "invoice_number": None,
        "date": "15/03/2024",
        "total": "300.00",
        "currency": "EUR",
    }
    tax = metrics.taxonomy([doc()], [pred])
    assert tax["vendor"] == {"distractor:customer": 1}
    assert tax["invoice_number"] == {"missing": 1}
    assert tax["total"] == {"distractor:subtotal": 1}
    assert tax["date"] == {}


def test_silent_error_share():
    preds = [
        {
            "vendor": "Beta GmbH",
            "invoice_number": None,
            "date": "x",
            "total": "300.00",
            "currency": "EUR",
        },
    ]
    sc = metrics.score([doc()], preds)
    out = metrics.silent_error_share(sc)
    # wrong and extracted: vendor (parses), date 'x' (does not parse), total (parses)
    assert out == {"wrong_values": 3, "schema_valid_but_wrong": 2}


def test_recoverable():
    d = doc()
    readable = "Acme Ltd\nINV-1\n15/03/2024\n360.00\nEUR"
    assert metrics.recoverable([d], [readable])[0]
    garbled = readable.replace("360.00", "36O.00")  # an amount an OCR error made unreadable
    assert not metrics.recoverable([d], [garbled])[0]


def test_boot_diff_of_identical_arrays_is_zero():
    x = np.random.default_rng(0).random(100) < 0.5
    assert metrics.boot_diff_ci(x, x, np.random.default_rng(1), 50) == (0.0, 0.0)
