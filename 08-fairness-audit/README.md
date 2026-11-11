# Project 08: Fairness and model risk audit

**Status: built on simulated applicants with known mechanisms. The decision gate FAILED, and that is reported as is.** Results are reproducible offline. They describe this simulated setup, not any real lender or employer. Nothing here is legal advice or a compliance determination.

## Decision context
A model risk reviewer must decide whether an approval policy can be signed off. The question is not "is the model accurate" but "who does it fail, by how much, what fixes it, and what does the fix cost". The decision gate is fixed in `fairness_audit/config.py` before results are seen.

## What you learn
- Group metrics: selection rate, true and false positive rates, precision, AUC, by group, with intervals.
- Why dropping the group column does not remove a gap.
- Why one overall AUC hides a group the model serves worse.
- Equal opportunity post-processing with group thresholds, and what it costs in utility.
- Why you cannot equalize every metric when base rates differ.
- How a small subgroup limits what an audit can prove.

## Data
Simulated. See `data/DATA_CARD.md`.

## The traps
Measured and written to `results/metrics.json` under `traps`: unawareness, one overall AUC, thresholds tuned on test, a gap of zero from rejecting everyone, equalizing one metric moving the others, and a small group's wide interval.

## Process note (read this)
- The pre-set rule picks the headline policy on dev and then tests its gap on test. The headline came out as `blind_single`. Its dev gap looked inside the tolerance, but on test the gap is larger, and the gate failed. I did not change the rule.
- Group thresholds tuned on dev reduced the test gap but not to zero. I then added a labelled post hoc study (in `results/error_analysis.json`) that repeats the dev draw and scores the thresholds on a large fresh set. On that set the average gap is close to zero and shrinks in spread as dev data grows. So the mitigation works on average, and the remaining test gap is the sampling error of one test set plus one dev draw. This study cannot rescue the gate. It explains it.
- The practical lesson: with about 575 group B positives in the test set, the audit cannot certify a gap below 0.10 unless the true gap is well below it.

## Not done
- Intersectional groups.
- Legal analysis or compliance mapping.
- In-processing, reweighting, or other mitigations.
- Causal analysis of the gap beyond the known simulator.
- Any real data.

## Run

```bash
pip install -r ../requirements.txt
python -m fairness_audit.train --write-checksum   # first time only
python -m fairness_audit.train                    # writes results/ (a few seconds)
python -m fairness_audit.report                   # renders reports/ from results/
python -m pytest                                  # tests; no network needed
```

`--fast` uses fewer bootstrap draws for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `fairness_audit/data.py` | Simulated applicants, splits, validation, generator fingerprint |
| `fairness_audit/models.py` | Scorers, utility, single and group thresholds |
| `fairness_audit/metrics.py` | Group report, disparities, AUC, stratified bootstrap |
| `fairness_audit/train.py` | Candidates, dev selection, test audit, traps, sensitivity, post hoc study, plot |
| `fairness_audit/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-computed metrics, simulator mechanisms, threshold logic, determinism, reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `tpr_by_group.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
