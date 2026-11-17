# What the tie response measures

Reproduction material for the manuscript *What the Tie Response Measures:
Respondent-side Measurement Bias in Crowdsourced Street-view Perception and Its
Imprint on the Released Place Pulse 2.0 Image Pool*.

This repository contains the scripts that generate every number in the paper, the
derived tables they produce, and a provenance audit of the Place Pulse platform
code. The raw comparison archive is not duplicated here; see
[`data/README.md`](data/README.md).

## What the paper claims, and how to check it

### 1. The platform retires an image when it ties

The evidence is in `provenance_audit.md`, which quotes the public platform source
with line numbers and content hashes, plus distributional tests any reader can
rerun on the released archive.

| Check | Script | Result |
|---|---|---|
| An image's last comparison is a tie far more often than its others | `scripts/diag_last_vote.py` | 35.6% against 9.6%, a ratio of 3.7 |
| Images with exactly two comparisons almost all have exactly one tie | `scripts/diag_deactivation.py` | 88.8% one tie, 11.0% two, 0.2% none, against 21.8%, 1.6% and 76.6% under independent ties |
| Retirement truncates the record rather than removing it from the released tables | `scripts/diag_deactivation.py` | 78.7% of the 111,389 released images took part in at least one tie |

### 2. Tie propensity is a respondent trait

| Quantity | Value | Script |
|---|---|---|
| Intraclass correlation of tie propensity, binomial null removed | 0.214 | `scripts/verify_rater_variance.py` |
| Intraclass correlation of left/right preference, same treatment | 0.020 | `scripts/verify_rater_variance.py` |
| AUC from respondent identity | 0.812 | `scripts/tie_sources.py` |
| AUC from image content | 0.532 | `scripts/tie_sources.py` |
| AUC from pair composition | 0.502 | `scripts/tie_sources.py` |
| Cross-city transfer of tie style | +0.51, positive in 56/56 | `scripts/rater_tie_transfer.py` |
| Rank correlation with binary-choice inconsistency | +0.006 (all) / -0.094 (n>=10) | `scripts/defense_checks.py` |
| Transfer of tie propensity vs binary consistency | 0.64 vs 0.19 | `scripts/transfer_tie_vs_consistency.py` |
| ICC on all six questions | 0.188 to 0.246 | `scripts/six_attributes.py` |

The ICC is computed as the between-respondent share of variance after subtracting
the binomial null implied by the votes per respondent. Without that subtraction
the same script returns 0.263 and 0.086; the null baselines are 0.062 and 0.067.

### 3. A respondent-aware model predicts better

Tables 4 and 5 in the paper are both derived from one per-vote prediction file,
`results/rater/clean/unified_votes.parquet`. The estimator is computed under
strict isolation: the test city contributes nothing to the priors, the respondent
histories or the calibration constant.

| Quantity | Value | Script |
|---|---|---|
| Three-class log loss, discard ties | 1.1665 | `scripts/unified_model.py` |
| Three-class log loss, global tie scale | 1.1004 | `scripts/unified_model.py` |
| Three-class log loss, respondent scales | 1.0893 | `scripts/unified_model.py` |
| Relative improvement, 15 of 15 cities | 6.62% | `scripts/unified_summary.py` |
| Share attributable to respondent heterogeneity | 1.00% (14 of 15) | `scripts/unified_summary.py` |
| Tie average precision | 0.089 to 0.217 | `scripts/unified_summary.py` |
| Estimator recovery on synthetic data | r = 0.93 (image), 0.94 (respondent) | `scripts/hier_davidson.py --selfcheck` |

### 4. The truncation does not move place scores

| Quantity | Value | Script |
|---|---|---|
| Rank correlation of item strengths, with and without respondent scales | 0.9989 | `scripts/city_score_consequence.py` |
| Change in a city's mean strength | at most 0.017 against a within-city SD of about 1.2 | `scripts/city_score_consequence.py` |

### 5. The truncation is uneven between cities, not within them

| Quantity | Value | Script |
|---|---|---|
| Mean Moran's I of evidence depth, within city | +0.004, 3 of 56 at p < 0.05 | `scripts/evidence_depth_spatial.py` |
| Share of images with three or fewer comparisons | 15.6% to 20.9% | `scripts/evidence_depth_spatial.py` |
| That share against visual redundancy | r = +0.52 | `scripts/evidence_depth_spatial.py` |
| That share against city tie rate | r = +0.75 | `scripts/evidence_depth_spatial.py` |
| Weighted correlation, city tie rate against visual redundancy | +0.51 | `scripts/city_rescue.py` |
| Pruning-free design (first comparison only) | +0.34 | `scripts/city_rescue.py` |

## Running the checks

```bash
pip install -r requirements.txt

# the distributional signatures (fast, no model fitting)
python scripts/diag_last_vote.py
python scripts/diag_deactivation.py

# respondent trait estimation (fits a visual model per fold)
python scripts/verify_rater_variance.py
python scripts/tie_sources.py
python scripts/rater_tie_transfer.py
python scripts/six_attributes.py

# the hierarchical model, with its synthetic self-check
python scripts/hier_davidson.py --selfcheck
python scripts/unified_model.py      # writes results/rater/clean/
python scripts/unified_summary.py    # derives Tables 4 and 5 from that file

# city-level analyses
python scripts/evidence_depth_spatial.py
python scripts/city_rescue.py
```

Scripts expect the archive at the path set in `DATA` near the top of each file;
adjust it to your local copy.

## Data

The raw Place Pulse 2.0 comparison archive is publicly available. See
[`data/README.md`](data/README.md) for the source, the fields used, and the
checksums of the files we audited.

## Licence

Analysis code is released under the MIT licence. The provenance audit quotes
short excerpts of the Place Pulse platform source for the purpose of documenting
the data-generating process.
