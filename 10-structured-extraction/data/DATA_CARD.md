# Data card: generated invoices

## Source
None. Every document is written by `extraction_eval/data.py` from templates and random draws. No download, no real company, no personal data. Company names are made-up syllables.

## Why generated
Scoring an extractor needs the true value of every field, and the identity of every wrong-but-plausible value (the subtotal next to the total, the due date next to the invoice date, the customer next to the vendor). Real invoices come without that. The cost is realism, listed below. This is simulated data with a known truth, not test fixtures: tests use small draws, while `results/` come from the full run. The data source is named `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- Five fields: vendor, invoice number, issue date, total, currency.
- Three seen templates (A: labelled lines, B: compact lines with currency symbols, C: pipe-separated table with European number format). Two unseen templates (E: new English label words and a different layout, G: German labels, European numbers, dotted dates). Unseen templates are used only for testing.
- Each template has label variants with fixed weights (for example `Total:` 70%, `Total due:` 20%, `Balance due:` 10%), optional lines (purchase order 70%, tax 80%), an exchange-rate note naming another currency (25%), a previous-balance line (15%), a date and due line in swapped order (20%), and a net line worded like a total (30%).
- Total equals the sum of line items plus tax. Line items are distractors for the total.
- Splits: 2,000 train, 600 dev and 1,200 test from seen templates; 600 test documents from each unseen template.

## What is assumed, not observed
The templates, label variants, weights and distractor rates. They are choices to make the failure modes easy to read.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of 60 fixed documents. `python -m extraction_eval.train` refuses to run if the generator no longer produces it. It does not hash the full simulation.

## Known quirks
- Far cleaner than real invoices: no skew, no stamps, no multi-page tables, no handwriting, one currency per document.
- The author wrote the unseen templates and the hand-written rules. The rules were written for the seen templates only, but the author knew the unseen ones existed.
- Dates are assumed day first when written with slashes.
- OCR noise is a character swap (0/O, 1/l, 5/S), applied after generation to the text only.

## Not checked
- Any real invoice, receipt or form.
- Layout or image-based extraction.
