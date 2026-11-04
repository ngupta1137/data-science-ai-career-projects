"""Train, evaluate, and write every result file.

    python -m text_classification.train            # full run on the downloaded data
    python -m text_classification.train --fast     # one config per model, smoke tests only

Order of use of each split:
  fit   : models are fitted here
  val   : model choice (macro-F1), a stratified share of the official training set
  test  : the official test set, later in time, scored once after every choice is made
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from . import config, data
from .metrics import (
    accuracy,
    bootstrap_ci,
    confusion,
    length_buckets,
    macro_f1,
    per_class,
    top_confused_pairs,
    top_features,
    word_counts,
)
from .models import BASELINE, SIMPLEST, TUNED, make_model, param_grid
from .splits import assert_disjoint, fit_val_split

CAVEATS = [
    "The interval resamples documents as if independent. Crossposted and quoted posts are not independent, so it is somewhat too narrow.",
    "Posts are from 1990s Usenet. Vocabulary and topics do not transfer to current text.",
    "Models are fitted on the fit rows only, not refitted on fit plus validation, so test scores are slightly conservative.",
    "A transformer comparison is not built in this version. See the project README.",
    "The metadata trap result is for the headline model and these hyperparameters. Other models may leak more or less.",
]


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
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


def _save_confusion_plot(matrix, classes, results_dir: Path) -> None:
    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix, row-normalised (test)")
    fig.colorbar(image, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def run(
    df: pd.DataFrame,
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_source: str = "newsgroups_sklearn",
    data_sha256: str | None = None,
    expected_rows: int | None = config.EXPECTED_ROWS,
    expected_classes: int | None = config.EXPECTED_CLASSES,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data.validate_frame(df, expected_rows, expected_classes)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    classes = sorted(df["label"].unique().tolist())
    fit_idx, val_idx = fit_val_split(train_df["label"], config.VAL_FRAC, seed)
    assert_disjoint(fit_idx, val_idx)

    def texts(frame, idx, col="text"):
        return frame[col].iloc[idx].tolist()

    X_fit, y_fit = texts(train_df, fit_idx), train_df["label"].iloc[fit_idx].to_numpy()
    X_val, y_val = texts(train_df, val_idx), train_df["label"].iloc[val_idx].to_numpy()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].to_numpy()
    n_boot = 50 if fast else config.N_BOOT

    fitted, model_results = {}, {}
    baseline = make_model(BASELINE, seed).fit(X_fit, y_fit)
    fitted[BASELINE] = baseline
    model_results[BASELINE] = {
        "val_macro_f1": macro_f1(y_val, baseline.predict(X_val)),
        "test_accuracy": accuracy(y_test, baseline.predict(X_test)),
        "test_macro_f1": macro_f1(y_test, baseline.predict(X_test)),
        "best_params": {},
    }
    for name in TUNED:
        best = None
        for params in param_grid(name, fast):
            candidate = make_model(name, seed).set_params(**params).fit(X_fit, y_fit)
            val_f1 = macro_f1(y_val, candidate.predict(X_val))
            if best is None or val_f1 > best[0]:
                best = (val_f1, params, candidate)
        val_f1, params, model = best
        pred = model.predict(X_test)
        fitted[name] = model
        model_results[name] = {
            "val_macro_f1": val_f1,
            "test_accuracy": accuracy(y_test, pred),
            "test_macro_f1": macro_f1(y_test, pred),
            "best_params": _clean(params),
        }

    headline = max(TUNED, key=lambda n: model_results[n]["val_macro_f1"])
    pred_test = fitted[headline].predict(X_test)
    pred_simple = fitted[SIMPLEST].predict(X_test)
    head_f1 = model_results[headline]["test_macro_f1"]
    base_f1 = model_results[BASELINE]["test_macro_f1"]

    f1_ci = bootstrap_ci(lambda i: macro_f1(y_test[i], pred_test[i]), len(y_test), n_boot, seed)
    gate1 = bool(f1_ci[0] > base_f1)
    if headline == SIMPLEST:
        gain_ci, gate2 = None, True
    else:
        gain_ci = bootstrap_ci(
            lambda i: macro_f1(y_test[i], pred_test[i]) - macro_f1(y_test[i], pred_simple[i]),
            len(y_test),
            n_boot,
            seed,
        )
        gate2 = bool(gain_ci[0] > 0)
    gate_passed = gate1 and gate2

    # Trap 1: metadata kept. Same model family and hyperparameters, raw text.
    head_params = model_results[headline]["best_params"]
    raw_model = make_model(headline, seed).set_params(**head_params)
    raw_model.fit(texts(train_df, fit_idx, "text_raw"), y_fit)
    raw_f1 = macro_f1(y_test, raw_model.predict(test_df["text_raw"].tolist()))
    clean_vocab = set(fitted[headline].named_steps["tfidf"].get_feature_names_out().tolist())
    raw_top = top_features(raw_model, classes, config.TOP_FEATURES)
    flat = [tok for toks in raw_top.values() for tok in toks]
    absent = [tok for tok in flat if tok not in clean_vocab]
    metadata_trap = {
        "test_macro_f1_metadata_kept": raw_f1,
        "test_macro_f1_metadata_removed": head_f1,
        "difference": raw_f1 - head_f1,
        "top_features_metadata_kept": raw_top,
        "share_of_top_features_absent_from_clean_vocabulary": len(absent) / max(len(flat), 1),
        "note": "Headers, footers, and quoted replies identify the poster and thread, not the topic.",
    }

    # Trap 2: the vectorizer is fitted on test text as well as training text.
    leaky_tfidf = clone(fitted[headline].named_steps["tfidf"]).fit(X_fit + X_test)
    leaky_clf = clone(fitted[headline].named_steps["clf"]).fit(leaky_tfidf.transform(X_fit), y_fit)
    leaky_f1 = macro_f1(y_test, leaky_clf.predict(leaky_tfidf.transform(X_test)))
    vocab_trap = {
        "test_macro_f1_vectorizer_fit_on_test": leaky_f1,
        "test_macro_f1_honest": head_f1,
        "difference": leaky_f1 - head_f1,
        "note": "Fitting vocabulary and IDF on test text lets test statistics shape the features.",
    }

    # Error analysis, headline model, test set.
    matrix = confusion(y_test, pred_test, classes)
    classes_f1 = per_class(y_test, pred_test, classes)
    weakest = sorted(classes_f1.items(), key=lambda kv: kv[1]["f1"])[:5]
    fit_counts = word_counts(X_fit)
    edges = np.quantile(fit_counts, [1 / 3, 2 / 3])
    test_counts = word_counts(X_test)
    bucket = length_buckets(test_counts, edges)
    by_length = {}
    for name in ("short", "medium", "long"):
        mask = bucket == name
        by_length[name] = {
            "n": int(mask.sum()),
            "accuracy": accuracy(y_test[mask], pred_test[mask]) if mask.any() else None,
            "macro_f1": macro_f1(y_test[mask], pred_test[mask]) if mask.any() else None,
        }
    empty = test_counts == 0
    error_analysis = {
        "model": headline,
        "per_class": classes_f1,
        "weakest_five_classes": [{"class": k, "f1": v["f1"]} for k, v in weakest],
        "top_confused_pairs": top_confused_pairs(matrix, classes, 10),
        "by_length": by_length,
        "length_edges_words_from_fit": [float(e) for e in edges],
        "empty_documents": {
            "n": int(empty.sum()),
            "share": float(empty.mean()),
            "accuracy": accuracy(y_test[empty], pred_test[empty]) if empty.any() else None,
        },
    }

    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": data_source,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "rows": len(df),
            "sha256": data_sha256,
            "classes": len(classes),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        "split": {
            "fit_rows": len(fit_idx),
            "val_rows": len(val_idx),
            "test_rows": len(test_df),
            "test_selection": "official test set, later in time than the training set",
        },
        "models": model_results,
        "headline": {
            "model": headline,
            "selected_by": "highest macro-F1 on validation",
            "metric": "macro_f1",
            "value": head_f1,
            "ci95": [f1_ci[0], f1_ci[1]],
            "ci_method": f"document bootstrap, {n_boot} resamples",
        },
        "baseline": {"model": BASELINE, "test_macro_f1": base_f1},
        "gain_over_naive_bayes_ci95": None if gain_ci is None else [gain_ci[0], gain_ci[1]],
        "gate": {
            "rule": config.GATE_RULE,
            "interval_above_majority_baseline": gate1,
            "gain_over_naive_bayes_interval_above_zero_or_headline_is_naive_bayes": gate2,
            "passed": gate_passed,
        },
        "traps": {"metadata_kept": metadata_trap, "vectorizer_fit_on_test": vocab_trap},
        "caveats": CAVEATS,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(error_analysis), indent=2) + "\n", encoding="utf-8"
    )
    _save_confusion_plot(matrix, classes, results_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate. Writes results/.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    args = parser.parse_args()

    frame = data.load(config.RAW_CSV)
    digest = data.verify_checksum(config.RAW_CSV)
    metrics = run(frame, config.RESULTS_DIR, fast=args.fast, data_sha256=digest)
    head = metrics["headline"]
    print(
        f"headline {head['model']}: test macro-F1 {head['value']:.4f}, 95% interval {head['ci95']}"
    )
    print(f"majority baseline macro-F1 {metrics['baseline']['test_macro_f1']:.4f}")
    print(f"gate passed: {metrics['gate']['passed']}")
    print(f"wrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
