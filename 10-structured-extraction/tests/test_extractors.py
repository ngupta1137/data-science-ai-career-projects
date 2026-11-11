import numpy as np

from extraction_eval import config, data, extractors

DOC_A = """Acme Ltd
Invoice No: INV-2024-00123
Date: 2024-03-15
Due: 2024-04-20
PO: PO-48213
Bill to: Beta GmbH
Service 1  100.00
Service 2  200.00
Subtotal: 300.00
Tax: 60.00
Total: EUR 360.00"""


def texts(field, text=DOC_A):
    return [c.text for c in extractors.candidates(field, text)]


def test_candidates_by_type():
    assert texts("date") == ["2024-03-15", "2024-04-20"]
    assert texts("total") == ["100.00", "200.00", "300.00", "60.00", "360.00"]
    assert texts("currency") == ["EUR"]
    assert "INV-2024-00123" in texts("invoice_number") and "PO-48213" in texts("invoice_number")


def test_id_pattern_skips_dates_and_plain_numbers():
    assert (
        extractors.candidates("invoice_number", "Date: 2024-03-15\nAmount 360.00\nCode 12345") == []
    )


def test_vendor_candidates_exclude_lines_with_digits():
    assert "Acme Ltd" in texts("vendor") and "Beta GmbH" in texts("vendor")
    assert not any(any(ch.isdigit() for ch in t) for t in texts("vendor"))


def test_first_match_takes_the_first_candidate_and_gets_total_wrong():
    out = extractors.first_match(DOC_A)
    assert out["total"] == "100.00"  # the first amount is a line item
    assert out["date"] == "2024-03-15" and out["vendor"] == "Acme Ltd"


def test_template_regex_on_a_known_document():
    out = extractors.template_regex(DOC_A)
    assert out == {
        "vendor": "Acme Ltd",
        "invoice_number": "INV-2024-00123",
        "date": "2024-03-15",
        "total": "360.00",
        "currency": "EUR",
    }


def test_template_regex_abstains_on_an_unknown_template():
    out = extractors.template_regex(
        "Document ref   J084454\nBilling date   June 11, 2025\nGrand total £646.47"
    )
    assert all(v is None for v in out.values())


def test_template_regex_misses_uncovered_label_variants():
    text = DOC_A.replace("Total: EUR", "Balance due: EUR")
    assert extractors.template_regex(text)["total"] is None


def test_span_ranker_only_returns_text_that_is_in_the_document():
    rng = np.random.default_rng(0)
    ranker = extractors.SpanRanker().fit(data.generate(rng, 200, config.SEEN_TEMPLATES))
    for d in data.generate(rng, 50, config.SEEN_TEMPLATES + config.UNSEEN_TEMPLATES):
        for v in ranker.extract(d.text).values():
            assert v is None or v in d.text


def test_span_ranker_abstains_when_threshold_is_unreachable():
    rng = np.random.default_rng(0)
    ranker = extractors.SpanRanker(abstain_below=1.01).fit(data.generate(rng, 100, ("A",)))
    assert all(v is None for v in ranker.extract(DOC_A).values())


def test_span_ranker_learns_the_total_from_context():
    rng = np.random.default_rng(1)
    ranker = extractors.SpanRanker().fit(data.generate(rng, 400, config.SEEN_TEMPLATES))
    assert ranker.extract(DOC_A)["total"] == "360.00"
