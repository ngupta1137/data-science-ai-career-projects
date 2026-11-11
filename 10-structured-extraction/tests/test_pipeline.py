import json

import pytest

from extraction_eval import config, data, report, train


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return train.run(out, fast=True), out


def test_fast_run_is_marked_and_writes_files(fast):
    metrics, out = fast
    assert metrics["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "doc_level.png"):
        assert (out / name).exists()
    analysis = json.loads((out / "error_analysis.json").read_text())
    assert "Not a release result" in report.render_model_card(metrics, analysis)
    assert "Not a release result" in report.render_decision_memo(metrics, analysis)


def test_run_is_deterministic(fast, tmp_path):
    again = train.run(tmp_path, fast=True)
    assert train._clean(again["extractors"]) == train._clean(fast[0]["extractors"])


def test_first_match_gets_the_total_wrong(fast):
    f = fast[0]["extractors"]["first_match"]["seen"]["fields"]["total"]
    assert f["accuracy"]["value"] == 0.0


def test_rules_abstain_on_unseen_templates_instead_of_guessing(fast):
    e = fast[0]["extractors"]["template_regex"]["unseen_pooled"]
    assert e["abstain_rate"] == 1.0 and e["silent"]["wrong_values"] == 0


def test_learned_extractor_drops_on_unseen_templates(fast):
    t = fast[0]["traps"]["unseen_template_drop"]["span_ranker"]
    assert t["seen"] > 0.95 and t["unseen_E"] < 0.5


def test_unseen_failures_are_silent(fast):
    s = fast[0]["traps"]["silent_errors"]["span_ranker"]["unseen"]
    assert s["wrong_values"] > 0 and s["schema_valid_but_wrong"] == s["wrong_values"]


def test_held_out_template_is_worse_than_random_split(fast):
    t = fast[0]["traps"]["random_split_vs_held_out_template"]
    assert t["mean_leave_one_template_out"] < t["random_split_doc_level_seen"] - 0.1


def test_ocr_noise_lowers_the_ceiling_and_the_ranker_stays_under_it(fast):
    o = fast[0]["traps"]["ocr_noise"]
    assert o["0"]["ceiling_all_values_still_readable"] == 1.0
    assert o["0.01"]["ceiling_all_values_still_readable"] < 0.97
    assert o["0.01"]["span_ranker"] <= o["0.01"]["ceiling_all_values_still_readable"] + 1e-9


def test_field_scores_hide_document_errors(fast):
    t = fast[0]["traps"]["field_scores_hide_document_errors"]
    assert t["micro_f1_unseen"] > t["doc_level_unseen"] + 0.2


def test_gate_verdict_is_the_conjunction_of_its_checks(fast):
    g = fast[0]["gate"]
    assert g["passed"] == all(g["checks"].values()) and len(g["checks"]) == 3


def test_headline_is_picked_on_dev(fast):
    analysis = json.loads((fast[1] / "error_analysis.json").read_text())
    dev = {k: v for k, v in analysis["dev_doc_level"].items() if k in config.EXTRACTORS}
    assert dev[analysis["headline_extractor"]] == max(dev.values())


def test_new_template_labels_help(fast):
    rows = json.loads((fast[1] / "error_analysis.json").read_text())[
        "post_hoc_new_template_labels"
    ]["rows"]
    e = {
        r["labelled_documents_from_new_template"]: r["doc_level"]["value"]
        for r in rows
        if r["template"] == "E"
    }
    assert e[30] > e[0] + 0.5


def test_full_run_reproduces_committed_results(tmp_path):
    committed = config.RESULTS_DIR / "metrics.json"
    if not committed.exists() or json.loads(committed.read_text())["fast_mode"]:
        pytest.skip("no committed full-run results yet")
    again = train.run(tmp_path, fast=False, data_sha256=data.fingerprint())
    saved = json.loads(committed.read_text())
    cleaned = train._clean(again)
    assert cleaned["extractors"] == saved["extractors"]
    assert cleaned["gate"]["checks"] == saved["gate"]["checks"]
