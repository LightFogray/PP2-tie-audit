# Data sources

## The comparison archive

Place Pulse 2.0, distributed by the MIT Media Lab through figshare:

- Record: <https://figshare.com/articles/dataset/Place_Pulse/11859993>
- File: `votes_clean.csv`, 481,254,165 bytes
- MD5 (as recorded against figshare): `55d2da3bafac0521722bab517fa715c9`
- SHA-256: `dcd938d8088378c4f667c7b9f727499bd9861d7ace905ad30e2ad9ee8e94f857`

The archive contains 1,565,723 rows across six evaluative questions. The
fields used in this paper are `left`, `right`, `study_id`, `choice`,
`voter_uniqueid`, and `timestamp`. Image locations and their city assignments
come from the platform export (`locations.tsv`, `places.tsv`), joined through
the place identifier.

## The derived comparison table

The analyses operate on `pairs.parquet`, a cleaned extraction of the safety
question with city labels resolved. The safety subset after image filtering
contains 506,165 comparisons over 111,389 images, 48,398 respondents, and 56
cities, with 12.47% of responses recorded as ties.

Because the archive is 481 MB and publicly hosted, it is not duplicated in this
repository. To rebuild `pairs.parquet`, download the archive and apply the
joins described in `scripts/six_attributes.py`, which performs the same
image-to-city resolution.

## Platform source code

Quoted in `provenance_audit.md`, with hashes:

- `PlacePulse/db.py`, 15,139 bytes,
  SHA-256 `9AEF5C0FDA0C30BC29EA73B1D64A86B95DF7DFACE8C7A97E72C648062A69554E`
- `PlacePulse/study.py`, 14,706 bytes,
  SHA-256 `2FBADD4F57ED6D8C3B6323A9D4E02A4F75AC351474FDDA728F0D8034E8566076`

Repository: <https://github.com/CenterForCollectiveLearning/Place-Pulse>

## Frozen visual features

The visual model uses ConvNeXt-Tiny (ImageNet-1k) features, 768 dimensions, for
all 110,987 archive images. Rebuild with `scripts/extract_features.py`. The
feature file is deliberately not included because it is 158 MB and is fully
determined by the archive and the frozen weights.
