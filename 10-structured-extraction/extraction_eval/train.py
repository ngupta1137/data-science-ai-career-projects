"""Fit the learned extractor, score three extractors on seen and unseen templates, write results.

    python -m extraction_eval.train          # full run
    python -m extraction_eval.train --fast   # smaller sets, marked

Scoring is by canonical value: 'March 15, 2024' equals '2024-03-15'. A document counts as correct
only if all five fields are right.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

from . import config, data, extractors
from .data import Doc
from .metrics import (
    boot_diff_ci,
    micro_f1,
    prf,
    recoverable,
    score,
    silent_error_share,
    taxonomy,
    wilson_ci,
)

CAVEATS = [
    "All documents are generated. They are far cleaner and more regular than real invoices, so the near-perfect results on seen templates say the task is easy to learn in this setting, not that extraction is solved.",
    "The unseen templates were written by the same author as the seen ones, and the rules were written knowing the seen templates only. The author was not blind to the unseen ones. A real shift is usually less tidy.",
    "No LLM was used. An LLM extractor with a schema is designed, not built; the scoring code here applies to its output unchanged.",
    "OCR noise is a crude character swap, not a real recogniser. Layout errors, skewed scans and tables spanning pages are not modelled.",
    "Dates are assumed day first for slashes. A month-first source would be silently misread.",
    "One currency per document, two-decimal amounts, five fields, no line items extracted.",
    "The 0.90 and 0.70 bars are assumptions. Change them and the verdict can change.",
]


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _rng(seed: int, label: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(label.encode("utf-8"))])


def _wilson(correct: np.ndarray) -> dict:
    k, n = int(correct.sum()), len(correct)
    return {"value": k / n, "ci95": list(wilson_ci(k, n)), "n": n}


def predict(fn, docs: list[Doc]) -> list[dict]:
    return [fn(d.text) for d in docs]


def block(docs: list[Doc], preds: list[dict]) -> dict:
    sc = score(docs, preds)
    return {
        "doc_level": _wilson(sc["doc_correct"]),
        "fields": {
            f: {"accuracy": _wilson(sc["correct"][f]), **prf(sc["correct"][f], sc["extracted"][f])}
            for f in config.FIELDS
        },
        "micro": micro_f1(sc),
        "abstain_rate": float(1.0 - np.mean([sc["extracted"][f].mean() for f in config.FIELDS])),
        "silent": silent_error_share(sc),
    }


def noisy_docs(docs: list[Doc], rate: float, rng: np.random.Generator) -> list[Doc]:
    return [dataclasses.replace(d, text=data.add_ocr_noise(rng, d.text, rate)) for d in docs]


def _save_plot(table: dict, results_dir: Path) -> None:
    names = list(table)
    groups = ("seen", "unseen_E", "unseen_G")
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, g in enumerate(groups):
        vals = [table[n][g]["doc_level"]["value"] for n in names]
        err = np.maximum(
            0.0,
            [
                [v - table[n][g]["doc_level"]["ci95"][0] for n, v in zip(names, vals, strict=True)],
                [table[n][g]["doc_level"]["ci95"][1] - v for n, v in zip(names, vals, strict=True)],
            ],
        )
        ax.bar(np.arange(len(names)) + (j - 1) * width, vals, width, yerr=err, capsize=3, label=g)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("documents with all five fields correct")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "doc_level.png", dpi=120)
    plt.close(fig)


def run(
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_sha256: str | None = None,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    boots = 100 if fast else config.N_BOOT
    n_train, n_dev, n_seen, n_unseen = (
        (400, 200, 300, 150)
        if fast
        else (config.N_TRAIN, config.N_DEV, config.N_TEST_SEEN, config.N_TEST_UNSEEN)
    )
    ocr_rates = (0.0, 0.01) if fast else config.OCR_RATES
    train_sizes = (50, 200) if fast else config.TRAIN_SIZES

    train = data.generate(_rng(seed, "train"), n_train, config.SEEN_TEMPLATES)
    dev = data.generate(_rng(seed, "dev"), n_dev, config.SEEN_TEMPLATES)
    test_seen = data.generate(_rng(seed, "test_seen"), n_seen, config.SEEN_TEMPLATES)
    test_unseen = {
        t: data.generate(_rng(seed, f"test_unseen|{t}"), n_unseen, (t,))
        for t in config.UNSEEN_TEMPLATES
    }
    pooled = [d for t in config.UNSEEN_TEMPLATES for d in test_unseen[t]]
    data.validate(train + dev + test_seen + pooled)

    ranker = extractors.SpanRanker().fit(train)
    fns = {
        "first_match": extractors.first_match,
        "template_regex": extractors.template_regex,
        "span_ranker": ranker.extract,
    }
    names = (*config.REFERENCES, *config.EXTRACTORS)

    preds: dict[str, dict[str, list[dict]]] = {n: {} for n in names}
    table: dict[str, dict] = {}
    for n in names:
        fn = fns[n]
        preds[n]["dev"] = predict(fn, dev)
        preds[n]["seen"] = predict(fn, test_seen)
        for t in config.UNSEEN_TEMPLATES:
            preds[n][f"unseen_{t}"] = predict(fn, test_unseen[t])
        preds[n]["pooled"] = preds[n]["unseen_E"] + preds[n]["unseen_G"]
        per_template = {
            t: _wilson(
                score(
                    [d for d in test_seen if d.template == t],
                    [
                        p
                        for d, p in zip(test_seen, preds[n]["seen"], strict=True)
                        if d.template == t
                    ],
                )["doc_correct"]
            )
            for t in config.SEEN_TEMPLATES
        }
        table[n] = {
            "dev_doc_level": float(score(dev, preds[n]["dev"])["doc_correct"].mean()),
            "seen": block(test_seen, preds[n]["seen"]),
            "unseen_E": block(test_unseen["E"], preds[n]["unseen_E"]),
            "unseen_G": block(test_unseen["G"], preds[n]["unseen_G"]),
            "unseen_pooled": block(pooled, preds[n]["pooled"]),
            "seen_by_template": per_template,
        }

    headline = max(
        config.EXTRACTORS, key=lambda k: (table[k]["dev_doc_level"], -config.EXTRACTORS.index(k))
    )
    reason = "higher share of documents with all five fields correct on dev"
    seen_ok = score(test_seen, preds[headline]["seen"])["doc_correct"]
    base_ok = score(test_seen, preds["first_match"]["seen"])["doc_correct"]
    diff_lo, diff_hi = boot_diff_ci(seen_ok, base_ok, _rng(seed, "gate-diff"), boots)
    checks = {
        "seen_doc_level_lower_bound_at_least_floor": bool(
            table[headline]["seen"]["doc_level"]["ci95"][0] >= config.DOC_FLOOR_SEEN
        ),
        "unseen_doc_level_lower_bound_at_least_floor": bool(
            table[headline]["unseen_pooled"]["doc_level"]["ci95"][0] >= config.DOC_FLOOR_UNSEEN
        ),
        "beats_first_match_on_seen": bool(diff_lo > 0),
    }

    # Traps, measured.
    h = table[headline]
    indep = float(np.prod([h["seen"]["fields"][f]["accuracy"]["value"] for f in config.FIELDS]))
    ocr = {}
    for rate in ocr_rates:
        nd = noisy_docs(test_seen, rate, _rng(seed, f"ocr|{rate}"))
        row = {
            "ceiling_all_values_still_readable": float(
                recoverable(test_seen, [d.text for d in nd]).mean()
            )
        }
        for n in names:
            row[n] = float(score(nd, predict(fns[n], nd))["doc_correct"].mean())
        ocr[f"{rate:g}"] = row

    lto = {}
    for held in config.SEEN_TEMPLATES:
        sub = extractors.SpanRanker().fit([d for d in train if d.template != held])
        fresh = data.generate(_rng(seed, f"lto|{held}"), max(n_seen // 3, 100), (held,))
        lto[held] = float(score(fresh, predict(sub.extract, fresh))["doc_correct"].mean())

    curve = []
    for size in train_sizes:
        sub = extractors.SpanRanker().fit(train[:size])
        curve.append(
            {
                "train_docs": size,
                "seen_doc_level": float(
                    score(test_seen, predict(sub.extract, test_seen))["doc_correct"].mean()
                ),
                "unseen_doc_level": float(
                    score(pooled, predict(sub.extract, pooled))["doc_correct"].mean()
                ),
            }
        )

    traps = {
        "field_scores_hide_document_errors": {
            "headline": headline,
            "micro_f1_seen": h["seen"]["micro"]["f1"],
            "doc_level_seen": h["seen"]["doc_level"]["value"],
            "product_of_field_accuracies_seen": indep,
            "micro_f1_unseen": h["unseen_pooled"]["micro"]["f1"],
            "doc_level_unseen": h["unseen_pooled"]["doc_level"]["value"],
            "note": "Five fields at 0.9 each give about 0.59 of documents fully right if errors are independent. Report both numbers.",
        },
        "unseen_template_drop": {
            n: {
                "seen": table[n]["seen"]["doc_level"]["value"],
                "unseen_E": table[n]["unseen_E"]["doc_level"]["value"],
                "unseen_G": table[n]["unseen_G"]["doc_level"]["value"],
            }
            for n in names
        },
        "random_split_vs_held_out_template": {
            "random_split_doc_level_seen": table["span_ranker"]["seen"]["doc_level"]["value"],
            "leave_one_template_out": lto,
            "mean_leave_one_template_out": float(np.mean(list(lto.values()))),
            "note": "Training on two templates and testing on the third is the honest estimate for a new customer's format. A random split of documents from the same templates is not.",
        },
        "ocr_noise": ocr,
        "silent_errors": {
            n: {
                "seen": table[n]["seen"]["silent"],
                "unseen": table[n]["unseen_pooled"]["silent"],
                "abstain_rate_seen": table[n]["seen"]["abstain_rate"],
                "abstain_rate_unseen": table[n]["unseen_pooled"]["abstain_rate"],
            }
            for n in names
        },
        "labelled_documents_needed": curve,
    }

    new_template = []
    for t in config.UNSEEN_TEMPLATES:
        pool = data.generate(_rng(seed, f"pool|{t}"), 100, (t,))
        for k in (0, 30) if fast else config.NEW_TEMPLATE_LABELS:
            sub = extractors.SpanRanker().fit(train + pool[:k])
            sc = score(test_unseen[t], predict(sub.extract, test_unseen[t]))
            new_template.append(
                {
                    "template": t,
                    "labelled_documents_from_new_template": k,
                    "doc_level": _wilson(sc["doc_correct"]),
                }
            )

    analysis = {
        "post_hoc_new_template_labels": {
            "status": "added after the learning curve showed more labelled documents from seen templates did not help unseen ones; not part of the gate",
            "rows": new_template,
        },
        "headline_extractor": headline,
        "selection_reason": reason,
        "dev_doc_level": {n: table[n]["dev_doc_level"] for n in names},
        "error_taxonomy": {
            n: {
                "seen": taxonomy(test_seen, preds[n]["seen"]),
                "unseen_E": taxonomy(test_unseen["E"], preds[n]["unseen_E"]),
                "unseen_G": taxonomy(test_unseen["G"], preds[n]["unseen_G"]),
            }
            for n in names
        },
        "field_accuracy_headline": {
            k: {f: table[headline][k]["fields"][f]["accuracy"]["value"] for f in config.FIELDS}
            for k in ("seen", "unseen_E", "unseen_G")
        },
    }

    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "train": len(train),
            "dev": len(dev),
            "test_seen": len(test_seen),
            "test_unseen_per_template": n_unseen,
            "seen_templates": list(config.SEEN_TEMPLATES),
            "unseen_templates": list(config.UNSEEN_TEMPLATES),
            "sha256": data_sha256,
        },
        "assumptions": {
            "abstain_below": config.ABSTAIN_BELOW,
            "doc_floor_seen": config.DOC_FLOOR_SEEN,
            "doc_floor_unseen": config.DOC_FLOOR_UNSEEN,
        },
        "extractors": table,
        "headline": {
            "extractor": headline,
            "selected_by": reason,
            "metric": "documents_with_all_fields_correct",
            "value": h["seen"]["doc_level"]["value"],
            "ci95": h["seen"]["doc_level"]["ci95"],
            "ci_method": "Wilson interval over test documents",
            "unseen_value": h["unseen_pooled"]["doc_level"]["value"],
            "unseen_ci95": h["unseen_pooled"]["doc_level"]["ci95"],
            "minus_first_match_seen": {
                "value": float(seen_ok.mean() - base_ok.mean()),
                "ci95": [diff_lo, diff_hi],
            },
        },
        "gate": {"rule": config.GATE_RULE, "checks": checks, "passed": bool(all(checks.values()))},
        "traps": traps,
        "caveats": CAVEATS,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(analysis), indent=2) + "\n", encoding="utf-8"
    )
    _save_plot(table, results_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate invoice field extractors.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    parser.add_argument(
        "--write-checksum", action="store_true", help="write the generator fingerprint"
    )
    args = parser.parse_args()
    if args.write_checksum:
        data.write_fingerprint()
    digest = data.verify_fingerprint()
    m = run(config.RESULTS_DIR, fast=args.fast, data_sha256=digest)
    h = m["headline"]
    print(
        f"headline {h['extractor']}: seen {h['value']:.3f} {h['ci95']}, unseen {h['unseen_value']:.3f}"
    )
    print(f"gate: {m['gate']['checks']} passed: {m['gate']['passed']}")


if __name__ == "__main__":
    main()
