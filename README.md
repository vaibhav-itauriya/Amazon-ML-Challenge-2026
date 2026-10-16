# Amazon ML Challenge 2026: Business Entity Resolution

The task is to match business records across three sources. For every entity in **Source 1** (the
reference), find all records in **Source 2** and **Source 3** (noisy and multilingual) that describe
the same business. Submissions are scored with **macro F0.5** over S1 entities, and S1 entities with no
match count too.

## Results

| Split | Macro F0.5 |
|---|---|
| Overall (out-of-fold, 1.77M train S1, 20% hidden as orphans) | **0.98654** |
| US | 0.98731 |
| India | 0.98537 |
| Singletons | 0.98733 |

The test output passes the official validator (`utils/validate_submission.py --check-ids`).

## Approach

A multi-stage CPU pipeline built mostly on LightGBM:

1. **Normalize**: clean names (junk prefixes, aliases, domains, leetspeak, legal suffixes), parse
   addresses into components (state, locality, postcode, house number, unit, street), and transliterate
   Indic scripts to Latin with a dictionary learned from training pairs plus an ITRANS fallback.
2. **Block**: sparse TF-IDF top-k search within each (country, state) on name, name + address, and
   address alone, plus a reverse pass and exact-key blocks. The blockers' results are unioned.
3. **Prune**: a streaming LightGBM model keeps about 20 candidates per S1 entity. These candidates
   become `candidate_pairs.tsv`.
4. **Stage 1**: a pairwise LightGBM model over about 75 features (set overlaps, rapidfuzz similarities,
   house numbers, name frequency).
5. **Stage 2 (context)**: features built from competing candidates, S2/S3 agreement, twins, and
   (optionally) cross-encoder scores, followed by a second LightGBM model and isotonic calibration.
6. **Decide**: one-to-one assignment plus, for each S1 entity, the match subset that maximizes expected
   F0.5 (numba Monte Carlo).

An optional GPU stage fine-tunes the cross-encoder `intfloat/multilingual-e5-small` and feeds its scores
into stage 2.

## Repository layout

```
.
├── code/business_entity_resolution/   # pipeline source, tests, requirements
│   ├── README.md                      # full technical docs: stages, config, timings
│   ├── requirements.txt
│   ├── gpu_steps.sh                   # cross-encoder round trip to a GPU box
│   ├── src/                           # run_pipeline.py and the stage modules
│   └── tests/
├── output/                            # submission files (stored with Git LFS)
│   ├── candidate_pairs.tsv            # blocking candidates per S1 entity
│   └── matching_results.tsv           # final matches (the scored file)
└── utils/validate_submission.py       # official submission validator
```

## Quick start

```bash
# Clone. Git LFS is needed to download the output files.
git lfs install
git clone https://github.com/vaibhav-itauriya/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026

# Environment (Python 3.11; on macOS, first run: brew install libomp)
python -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt

# Put the data in dataset/{train,test}/*.tsv (or set ER_DATA), then run:
cd code/business_entity_resolution/src
python run_pipeline.py                  # full pipeline -> output/*.tsv, validated
python run_pipeline.py --from prune     # resume from a stage
```

A full run takes about 3.5 hours and peaks at about 30 GB of RAM (measured on an M3 Max with 48 GB).
Stages: `io → splits → normalize → sets → blocking → prune → stage1 → context → output`.

To validate a submission on its own:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

See [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) for
configuration (`ER_*` environment variables), prune diagnostics, the cross-encoder workflow, and
leaderboard probes.

## Output format

Both files are TSVs with one row per test S1 entity (1,732,544 rows) and comma-separated S2/S3 IDs:

```
source1_entity_id	matched_entity_ids
S1-714132312	S2-187020300,S2-637340732,S3-625880872
```

## Models and licences

LightGBM (MIT); optional cross-encoder `intfloat/multilingual-e5-small` (MIT, 118M parameters);
`indic-transliteration` (MIT). No model exceeds 8B parameters.
