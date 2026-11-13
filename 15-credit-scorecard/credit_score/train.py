"""Fit four default models on past applications and score them on fresh applicants with known outcomes.

python -m credit_score.train          # full run
python -m credit_score.train --fast   # smaller samples, marked
"""

from __future__ import annotations

import argparse
import json
import platform
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from . import config, data, metrics, models

CAVEATS = [
    "Everything is simulated. The default model, the unrecorded trait, the old policy and the economics are chosen by the author, so the sizes describe this setup, not any lender.",
    "The test set knows whether every applicant would have defaulted. No lender has that. It is how this project can score the methods; in practice only the random sample gives a partial view.",
    "Loans are the same size and term. No pricing by risk, no recoveries over time, no prepayment, no economic cycle.",
    "The random sample approves applicants the old policy would have declined. That costs money and needs governance and, in many places, legal review. This project prices the cost and nothing else.",
    "Nothing here is a fair-lending or regulatory method. See project 08 for a group audit, and take legal questions to counsel.",
    "Models are untuned with settings fixed before results.",
]


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
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


def _rng(seed: int, tag: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(tag.encode())])


def run(
    results_dir: Path, fast: bool = False, seed: int = config.SEED, data_sha256: str | None = None
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    n_hist = 10_000 if fast else config.N_HISTORY
    n_test = 10_000 if fast else config.N_TEST
    n_boot = 200 if fast else config.N_BOOT
    hist = data.generate(_rng(seed, "history"), n_hist)
    test = data.generate(_rng(seed, "test"), n_test)
    data.validate(hist)
    data.validate(test)
    thr = config.APPROVE_BELOW

    fitted = {n: cls().fit(hist, seed) for n, cls in models.MODELS.items()}
    preds = {n: m.predict(test.x) for n, m in fitted.items()}
    old = test.old_approve
    terms = {
        "old_policy": metrics.profit_terms(old, test.y),
        "oracle": metrics.profit_terms(test.pd < thr, test.y),
    }
    for n, p in preds.items():
        terms[n] = metrics.profit_terms(p < thr, test.y)

    table = {}
    for n in ("old_policy", *config.MODELS, "oracle"):
        appr = (
            old == 1 if n == "old_policy" else (test.pd < thr if n == "oracle" else preds[n] < thr)
        )
        row = {
            "approval_rate": float(appr.mean()),
            "default_rate_approved": float(test.y[appr].mean()) if appr.any() else None,
            "profit_per_applicant": metrics.boot_ci(terms[n], _rng(seed, f"boot|{n}"), n_boot),
        }
        if n in preds:
            p = preds[n]
            row.update(
                {
                    "gini_all_applicants": metrics.gini(test.y, p),
                    "gini_old_approved_only": metrics.gini(test.y[old == 1], p[old == 1]),
                    "calibration_all": metrics.calibration_ratio(p, test.y),
                    "calibration_approved": metrics.calibration_ratio(p[appr], test.y[appr])
                    if appr.any()
                    else None,
                    "calibration_old_approved": metrics.calibration_ratio(
                        p[old == 1], test.y[old == 1]
                    ),
                    "calibration_old_declined": metrics.calibration_ratio(
                        p[old == 0], test.y[old == 0]
                    ),
                    "minus_old_policy": metrics.boot_ci(
                        terms[n] - terms["old_policy"], _rng(seed, f"diff|{n}"), n_boot
                    ),
                }
            )
        table[n] = row
    h = config.HEADLINE
    vs_acc = metrics.boot_ci(terms[h] - terms["accepted_only"], _rng(seed, "diff|acc"), n_boot)
    lo, hi = config.CALIBRATION_BAND
    cal = table[h]["calibration_approved"]
    checks = {
        "beats_old_policy": table[h]["minus_old_policy"]["ci95"][0] > 0,
        "beats_accepted_only_model": vs_acc["ci95"][0] > 0,
        "calibrated_on_approved": cal is not None and lo <= cal <= hi,
    }

    rand_only = (hist.random_approve == 1) & (hist.old_approve == 0)
    sample_cost = {
        "random_approvals_in_history": int((hist.random_approve == 1).sum()),
        "of_which_old_policy_declined": int(rand_only.sum()),
        "default_rate_of_those": float(hist.y[rand_only].mean()),
        "profit_on_those_per_loan": float(
            metrics.profit_terms(np.ones(rand_only.sum()), hist.y[rand_only]).mean()
        ),
        "cost_per_history_applicant": float(
            -metrics.profit_terms(rand_only, hist.y).sum() / n_hist
        ),
    }

    size_rows = []
    shares = (0.01, 0.05) if fast else config.RANDOM_SHARES
    reps = 1 if fast else config.RANDOM_SHARE_REPS
    for share in shares:
        prof, calr = [], []
        for r in range(reps):
            hs = data.generate(_rng(seed, f"share|{share}|{r}"), n_hist, share)
            p = models.Recalibrated().fit(hs, seed).predict(test.x)
            a = p < thr
            prof.append(metrics.profit_terms(a, test.y).mean())
            calr.append(metrics.calibration_ratio(p[a], test.y[a]))
        size_rows.append(
            {
                "random_share": share,
                "replications": reps,
                "random_loans": round(share * n_hist),
                "mean_profit_per_applicant": float(np.mean(prof)),
                "min_profit_per_applicant": float(np.min(prof)),
                "mean_calibration_approved": float(np.mean(calr)),
                "worst_calibration_approved": float(max(calr, key=lambda c: abs(np.log(c)))),
            }
        )

    dec_rows = []
    for n in ("accepted_only", h):
        p = preds[n]
        q = np.quantile(p, np.linspace(0.1, 0.9, 9))
        b = np.digitize(p, q)
        dec_rows.append(
            {
                "model": n,
                "rows": [
                    {
                        "decile": int(k) + 1,
                        "mean_predicted": float(p[b == k].mean()),
                        "true_rate": float(test.y[b == k].mean()),
                    }
                    for k in range(10)
                ],
            }
        )

    traps = {
        "evaluate_on_approved_only": {
            "gini_all_applicants": table["accepted_only"]["gini_all_applicants"],
            "gini_old_approved_only": table["accepted_only"]["gini_old_approved_only"],
            "calibration_old_approved": table["accepted_only"]["calibration_old_approved"],
            "calibration_old_declined": table["accepted_only"]["calibration_old_declined"],
            "note": "On the loans you can check, the model looks calibrated. On the applicants it would newly approve, it is not.",
        },
        "weighting_cannot_fix_what_was_not_recorded": {
            "calibration_old_declined_accepted_only": table["accepted_only"][
                "calibration_old_declined"
            ],
            "calibration_old_declined_ipw": table["ipw"]["calibration_old_declined"],
            "calibration_old_declined_recalibrated": table[h]["calibration_old_declined"],
            "note": "Weighting by the chance of approval given recorded features fixes selection on those features only.",
        },
        "small_unbiased_sample": {
            "random_sample_only_gini": table["random_sample_only"]["gini_all_applicants"],
            "recalibrated_gini": table[h]["gini_all_applicants"],
            "note": "The random sample alone is unbiased but too small to rank well; it is best used to recalibrate.",
        },
        "cost_of_learning": sample_cost,
    }
    analysis = {"calibration_by_decile": dec_rows, "random_sample_size": size_rows}
    out = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "history": n_hist,
            "history_approved": int(hist.approved.sum()),
            "history_random_approved": int((hist.random_approve == 1).sum()),
            "test": n_test,
            "test_default_rate_all": float(test.y.mean()),
            "test_old_approval_rate": float(old.mean()),
            "sha256": data_sha256,
        },
        "assumptions": {
            "margin": config.MARGIN,
            "loss": config.LOSS,
            "approve_below": thr,
            "random_approve_share": config.RANDOM_APPROVE_SHARE,
        },
        "models": table,
        "headline": {
            "model": h,
            "metric": "profit_per_applicant",
            "value": table[h]["profit_per_applicant"]["value"],
            "ci95": table[h]["profit_per_applicant"]["ci95"],
            "ci_method": f"percentile bootstrap over test applicants, {n_boot} resamples; differences paired",
            "minus_old_policy": table[h]["minus_old_policy"],
            "minus_accepted_only": vs_acc,
            "calibration_approved": cal,
        },
        "gate": {"rule": config.GATE_RULE, "checks": checks, "passed": bool(all(checks.values()))},
        "traps": traps,
        "caveats": CAVEATS,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(out), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(analysis), indent=2) + "\n", encoding="utf-8"
    )
    _plot(dec_rows, results_dir)
    return out


def _plot(dec_rows: list, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for r in dec_rows:
        ax.plot(
            [x["mean_predicted"] for x in r["rows"]],
            [x["true_rate"] for x in r["rows"]],
            "o-",
            label=r["model"],
        )
    m = max(max(x["true_rate"], x["mean_predicted"]) for r in dec_rows for x in r["rows"])
    ax.plot([0, m], [0, m], "--", color="grey", label="perfect")
    ax.set_xlabel("mean predicted default chance (decile)")
    ax.set_ylabel("true default rate, all applicants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "calibration.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit scorecard with reject bias.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    parser.add_argument(
        "--write-checksum", action="store_true", help="write the generator fingerprint"
    )
    args = parser.parse_args()
    if args.write_checksum:
        data.write_fingerprint()
    digest = data.verify_fingerprint()
    m = run(config.RESULTS_DIR, fast=args.fast, data_sha256=digest)
    for n, r in m["models"].items():
        print(
            n,
            {k: v for k, v in r.items() if k not in ("profit_per_applicant", "minus_old_policy")},
            r["profit_per_applicant"],
        )
    print("gate", m["gate"]["checks"], m["gate"]["passed"])


if __name__ == "__main__":
    main()
