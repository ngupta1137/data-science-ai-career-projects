# Data card: credit card fraud

## Source
Credit card transactions made by European cardholders over two days in September 2013, published by the ULB Machine Learning Group (MLG-ULB) in collaboration with Worldline. This project fetches it from OpenML (dataset ID 42175) through scikit-learn.
- Source page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- OpenML entry: https://www.openml.org/d/42175

Reference: Dal Pozzolo, A., Caelen, O., Johnson, R. A., and Bontempi, G. "Calibrating Probability with Undersampling for Unbalanced Classification." IEEE Symposium Series on Computational Intelligence, 2015.

## Terms
Listed on Kaggle under Open Database License (ODbL) v1.0 and Database Contents License (DbCL) v1.0. Verify the exact terms on the source page before redistributing anything derived from it. The data is **not** committed here: `python -m fraud_detection.download_data` fetches it, and the code rejects the file if the committed checksum does not match.

## Shape
Documented by the source: 284,807 rows, 492 fraud. `fraud_detection/config.py` enforces both counts, and `fraud_detection/data.py` enforces the schema, the label values, and non-negative amounts.

| Column | Meaning |
|---|---|
| Time | Seconds since the first transaction in the dataset |
| V1 to V28 | Anonymised numeric features, outputs of a PCA transformation |
| Amount | Transaction amount |
| Class | 1 for fraud, 0 otherwise (target) |

## Known quirks
- **Extreme imbalance.** The fraud rate is about 0.17%. Accuracy is meaningless. Any split must keep enough fraud rows in each part, and the code stops if a part has fewer than `MIN_POSITIVES`.
- **PCA features computed on all rows.** V1 to V28 were produced before this project saw the data, using the whole dataset, test period included. A small amount of test information leaks into the features and cannot be undone here.
- **Short window.** Two days of data. A time split here tests only the next few hours, not months of drift.
- **Duplicate rows.** The data contains duplicates. They are kept, and the count is written to `results/metrics.json`.
- **No identifiers.** No card, customer, or merchant column exists, so correlated transactions cannot be grouped. Bootstrap intervals treat rows as independent and are therefore too narrow.
- **Time is not a clock.** It is an offset, so no hour-of-day feature is built.

## Not checked
- How the source labelled fraud, or how many frauds went undetected and are labelled legitimate.
- Whether the anonymisation or PCA hides group differences that matter for fairness.
- Whether amounts are in a single currency.
