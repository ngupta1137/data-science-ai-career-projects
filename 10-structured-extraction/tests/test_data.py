import numpy as np
import pytest

from extraction_eval import config, data, extractors
from extraction_eval.normalize import normalize, parse_money


@pytest.mark.parametrize("template", ["A", "B", "C", "E", "G"])
def test_every_true_value_can_be_read_from_the_text(template):
    docs = data.generate(np.random.default_rng(1), 150, (template,))
    for d in docs:
        for f in config.FIELDS:
            assert any(
                normalize(f, c.text) == d.truth[f] for c in extractors.candidates(f, d.text)
            ), (template, f, d.text)


def test_total_is_subtotal_plus_tax_when_tax_is_shown():
    docs = data.generate(np.random.default_rng(2), 300, ("A", "B", "C", "E", "G"))
    checked = 0
    for d in docs:
        dist = d.distractors["total"]
        if "tax" in dist:
            parts = parse_money(dist["subtotal"]) + parse_money(dist["tax"])
            assert parts == pytest.approx(d.truth["total"], abs=0.011)
            checked += 1
    assert checked > 100


def test_distractors_differ_from_truth():
    for d in data.generate(np.random.default_rng(3), 200, ("A", "B", "C", "E", "G")):
        assert normalize("date", d.distractors["date"]["due_date"]) != d.truth["date"]
        assert normalize("vendor", d.distractors["vendor"]["customer"]) != d.truth["vendor"]
        for raw in d.distractors["total"].values():
            assert parse_money(raw) != d.truth["total"]


def test_templates_use_their_own_formats():
    rng = np.random.default_rng(4)
    assert "Rechnung" in data.generate_doc(rng, "G").text
    assert "Billing Statement" in data.generate_doc(rng, "E").text
    assert "Bill from |" in data.generate_doc(rng, "C").text


def test_deterministic_and_splits_independent():
    a = data.generate(np.random.default_rng(5), 20, ("A",))
    b = data.generate(np.random.default_rng(5), 20, ("A",))
    assert [d.text for d in a] == [d.text for d in b]
    s = data.make_splits(1)
    assert not ({d.text for d in s["train"]} & {d.text for d in s["test_seen"]})
    assert {d.template for d in s["test_seen"]} == set(config.SEEN_TEMPLATES)
    assert {d.template for d in s["test_unseen_E"]} == {"E"}


def test_ocr_noise():
    rng = np.random.default_rng(6)
    assert data.add_ocr_noise(rng, "Total 150", 0.0) == "Total 150"
    assert data.add_ocr_noise(np.random.default_rng(6), "0101", 1.0) == "O1O1".replace("1", "l")
    noisy = data.add_ocr_noise(np.random.default_rng(7), "INV-2024-00105 " * 40, 0.1)
    assert noisy != "INV-2024-00105 " * 40 and len(noisy) == len("INV-2024-00105 " * 40)


def test_validate_rejects_missing_truth():
    d = data.generate(np.random.default_rng(8), 2, ("A",))
    del d[0].truth["total"]
    with pytest.raises(data.DataValidationError):
        data.validate(d)


def test_fingerprint_roundtrip_and_mismatch(tmp_path):
    f = tmp_path / "sum.txt"
    digest = data.write_fingerprint(f)
    assert data.verify_fingerprint(f) == digest
    f.write_text("0" * 64 + "  x\n")
    with pytest.raises(data.DataValidationError):
        data.verify_fingerprint(f)


def test_committed_fingerprint_matches_generator():
    assert data.verify_fingerprint() == data.fingerprint()
