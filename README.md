# AIR26 — Many Outlets, One Story

Course project for **Advanced Information Retrieval (TU Graz, WS 2026/27)**.

News search returns many copies of the same wire story, so a ranked list can look
diverse while covering one story. This project builds a time-scoped test
collection from the Infini-News (CC-News) corpus, groups near-duplicate articles
into story clusters, judges relevance at the event level, and compares BM25,
bi-encoder, cross-encoder and hybrid retrieval on classic relevance **and**
coverage metrics. It then tests cheap rerankers (story cap, outlet cap,
earliness prior) that trade a little relevance for better coverage.

The submitted design document is in `docs/design_document.pdf` (add it before
submission).

## Status

- [x] Design document submitted (3 A4 pages)
- [x] Environment + GitHub repo
- [x] P1 preprocessing: **170,781 docs** in 24 monthly files
- [x] P2 story clustering: 124,409 clusters (7,939 cross-host)
- [x] P3 pilot queries (10) + seed helper
- [x] P4 BM25 + dense on the 3 smoke-test months (BM25 ~0.4 ms, dense ~12 ms)
- [x] P5 candidate pool + annotation sheets (**138 candidates**, 30 double)
- [x] P5 pilot judgments + κ: **κ = 0.935** (quadratic 0.977), 96.7 % agreement, 29/30 pairs
- [x] P5 pilot pass: **9/10 events** have ≥3 relevant stories (plan needs ≥8/10); P01 is the only failure
- [x] P6 S1-S4 on the 10 pilot queries: seed hit 9/10 BM25 & cross-encoder, 10/10 dense & hybrid
- [x] P6 tooling: S3/S4 + 24-month runner + richness seed shortlist (192 seeds)
- [x] P6 query allocation: 50 queries over 19 months, 25 entity / 25 descriptive (`queries/query_allocation.csv`)
- [x] P6 50 evaluation queries + 10 reserve written and checked (`scripts/check_queries.py`)
- [x] P6 S1–S4 on all 60 active queries: 6,000 rows/system, seed hit 95 % BM25/S3, 100 % dense/hybrid
- [x] P7 final pool: **718 candidates** (50 evaluation queries) + **150 double** (3 per query)
- [ ] P7 annotation: **0/718 graded** — window 24.11–07.12, at least half by 01.12 (G3)
- [ ] G4 collect + κ + story merge; P8 rerankers; P9 metrics; analysis and packaging

## Repository layout

```
config/default.yaml        paths, filters, clustering parameters
data/manifest.csv          24-shard sample (one shard per month, 2023-01–2024-12)
data/processed/            gitignored: docs_*, clusters_*, stats
src/data/common.py         manifest, paths, normalisation, shingles
src/data/preprocess.py     P1: filter shards into monthly docs parquet
src/data/cluster.py        P2: exact + MinHash/LSH clustering per month
src/queries/select_seeds.py P3 helper: ranked seed clusters for query authoring
src/queries/build_candidates.py P5: top-10 clusters per system, union, cap 15
src/queries/make_sheets.py P5: annotation sheets, collect judgments
src/queries/validate_judgments.py P5: structural/grade checks
src/queries/kappa.py       P5: Cohen's κ (unweighted + quadratic)
src/retrieval/bm25.py      S1: BM25 index per month (bm25s)
src/retrieval/dense.py     S2: MiniLM embeddings + cosine search
src/retrieval/crossencoder.py S3: cross-encoder reranking of the BM25 top-k
src/retrieval/hybrid.py    S4: Reciprocal Rank Fusion of BM25 + dense
src/retrieval/run_all.py   month-scoped runs for S1-S4, latency, seed check
scripts/00_setup.sh        check env, create folders
scripts/01_preprocess.sh   run P1
scripts/02_cluster.sh      run P2
scripts/03_seeds.sh        build queries/seed_candidates.csv
scripts/04_retrieval.sh    run P4 on the pilot queries
scripts/05_pool.sh         P5 candidate pool + sheets
scripts/05_kappa.sh        P5 validate + κ
scripts/06_seeds_full.sh   P6 richness-ranked seed shortlist (24 months)
scripts/07_retrieval_full.sh P6 S1-S4 on all active queries, 24 months
scripts/08_pool_final.sh    P7 final pool (S1-S4) + final annotation sheets
src/retrieval/embed_all.py  precompute MiniLM embeddings for all 24 months
tests/test_common.py       smoke tests
tests/test_retrieval.py    RRF fusion + seed richness tests
```

## Data

- Source: [ruggsea/infini-news-corpus](https://huggingface.co/datasets/ruggsea/infini-news-corpus)
  (CC-News derived), paper arXiv:2605.18337.
- Sample: 24 parquet shards, one per month, 2023-01–2024-12.
- Raw rows 565,515 → **170,781 English articles** after the filters
  (language `eng`, 300–20,000 chars, HTTP 200).
- **Article text is not redistributed.** The repo ships only the manifest and
  the download script. The derived metadata is used under the dataset terms.

## Setup

```bash
ssh artemis
conda activate air26          # torch 2.6+cu124, sentence-transformers, bm25s, datasketch, ir_measures
cd ~/air26-news-coverage
bash scripts/00_setup.sh
```

If the cache with the 24 shards is not present, P1 downloads missing shards from
Hugging Face (`--download`). Override the cache location with `CACHE_DIR=...`.

## Run the pipeline

```bash
# P1 — 24 monthly docs files (expect 170,781 rows)
bash scripts/01_preprocess.sh

# P2 — per-month story clusters (parallel over months)
JOBS=8 bash scripts/02_cluster.sh

# P3 — seed clusters for writing the 10 pilot + 50 evaluation queries
bash scripts/03_seeds.sh

# P4 — BM25 + dense on the pilot queries (3 months)
bash scripts/04_retrieval.sh --months 2023-01,2023-06,2024-01 --strict

# P5 — pool + sheets (refuses to overwrite existing annotations; FORCE=1 to reset)
bash scripts/05_pool.sh
# people fill queries/candidates.csv and queries/double_candidates.csv
# after annotation:
python -m src.queries.make_sheets --mode collect
bash scripts/05_kappa.sh

# P6 — richness-ranked seed shortlist for the 50 evaluation queries
bash scripts/06_seeds_full.sh
# write the 50 evaluation queries into queries/queries.csv (status=eval)
# then S1-S4 on all 60 active queries over all 24 months
bash scripts/07_retrieval_full.sh --strict

# P7 — final pool from S1-S4 for the 50 evaluation queries
bash scripts/08_pool_final.sh

# optional: precompute all embeddings (24 months, ~160 s on the L40S)
python -m src.retrieval.embed_all

# smoke tests
python -m unittest discover -s tests -v
```

Expected outputs:

```
data/processed/docs_YYYY-MM.parquet       # doc_id, month, host, title, text, dates, text_hash, cluster_id
data/processed/clusters_YYYY-MM.parquet   # cluster_id, size, hosts, earliest dates, representative
data/processed/corpus_stats.csv
data/processed/cluster_stats.csv
data/processed/emb_*.npy                  # cached MiniLM embeddings
results/runs/bm25__none.parquet           # qid, system, month, doc_id, rank, score
results/runs/dense__none.parquet
results/runs/latency.csv
results/runs/seed_check.csv
queries/candidates.csv                    # gitignored: 138 candidates, titles/snippets
queries/double_candidates.csv             # gitignored: 30 rows for the second annotator
queries/judgments.csv                     # grades (no article text)
queries/judgments_double.csv              # second-annotator grades
queries/candidates_final.csv              # gitignored: 718 candidates (final pool)
queries/final_double_candidates.csv       # gitignored: 150 rows (3 per query)
queries/final_judgments.csv               # grades after collect (no article text)
queries/final_judgments_double.csv        # second-annotator grades after collect
results/metrics/pilot_kappa.csv
```

P4/P6 pilot result on the updated queries (v2): the seed cluster is in the
top-100 for **9/10 queries** with BM25 and the cross-encoder, and **10/10**
with dense and hybrid. The miss is P09 (Belichick): the query is descriptive
("veteran head coach parts ways with his NFL team after more than two decades")
and the seed article never says "veteran" or "parts ways"; BM25 ranks it 305,
dense 31, hybrid 75. That is a real entity-vs-descriptive effect (RQ3), not a
bug, and it is why the seed-hit threshold is 80 %.

Median latency per query (L40S, 3 months, 100 candidates):

| System | Median |
|---|---|
| BM25 (S1) | 0.4 ms |
| Dense (S2) | 12.0 ms |
| Cross-encoder (S3) | 116.5 ms |
| Hybrid RRF (S4) | 12.7 ms |

The cross-encoder is ~10x denser retrieval because it scores 100 query-document
pairs per query; the hybrid number includes both base runs plus fusion.

## Pilot evaluation (P5)

Annotator 1 graded all 138 candidates; annotator 2 graded the 30-row double
sheet without seeing the first set.

| Metric | Value |
|---|---|
| Agreement | 96.7 % (29/30 pairs) |
| Cohen's κ | **0.935** (quadratic-weighted 0.977) |
| Confusion matrix | 7 (0,0), 3 (1,1), 1 (1,2), 19 (2,2) |
| Grade spread | 84 × 2, 17 × 1, 37 × 0 |
| Pilot pass | **9/10 events** with ≥3 relevant stories (plan: ≥8/10) |

The single disagreement is the expected 1-vs-2 borderline; nobody confused
different event (0) with same event (2). P01 (Hells Angels) is the only failure
with 1 relevant story. It is a tuning query and is kept out of the 50
evaluation queries.

**The counts are inflated by wire-story fragmentation.** P05 looks like 13
relevant clusters, but they are mostly rewrites of one report. Counting distinct
headlines after merging outlet suffixes gives:

| Query | Relevant clusters | Distinct headlines | Pass |
|---|---|---|---|
| P01 Hells Angels | 1 | 1 | no |
| P02 Poland tanks | 10 | 7 | yes |
| P03 Iraq central bank | 4 | 3 | yes |
| P04 Lunar New Year shooting | 11 | 11 | yes |
| P05 partygate report | 13 | 11 | yes |
| P06 Daniel Penny | 6 | 5 | yes |
| P07 Club Q plea | 4 | 3 | yes |
| P08 South Africa v. Israel | 8 | 8 | yes |
| P09 Belichick | 8 | 8 | yes |
| P10 spending deal | 9 | 8 | yes |

The pass is solid for P02, P04, P08, P09 and P10 (different developments) and
thin for P03, P05, P06 and P07 (one report rewritten). For the 50 evaluation
queries, seeds are therefore ranked by **distinct headlines per event**, not by
raw cluster size (`scripts/06_seeds_full.sh`).

P5 pool result: **138 candidates** (11–15 per query), **30 double-judged**
(3 per query). `05_pool.sh` refuses to overwrite the sheets once
`queries/candidates.csv` exists; use `FORCE=1` only to rebuild from scratch. The seed cluster was outside the top-10 clusters for P09, so it
was force-included (`seed_forced=True`). P04 shows the expected fragmentation:
the Monterey Park shooting appears as several single-article clusters plus one
8-copy seed cluster; the `same_as` field will merge the wire copies.

For a quick check on three months:

```bash
bash scripts/01_preprocess.sh --months 2023-01,2023-06,2024-01
JOBS=3 bash scripts/02_cluster.sh --months 2023-01,2023-06,2024-01
```

## Multi-month notes

Each query is retrieved **only from its own month's pool**. Clustering is done
per month; never merge clusters across months. The design document's
"3-shard pilot (22,613 articles)" is a separate feasibility sample of three
shards from one month; the 24-month run is the evaluation corpus.

## Full run (P6): 60 queries, 24 months

`scripts/07_retrieval_full.sh --strict` ran S1–S4 on the 10 pilot + 50
evaluation queries (60 active; the 10 reserves are not run), each query against
its own month's pool. Output: **6,000 rows per system** (60 × top-100).

| System | Seed in top-100 | Median latency | Median seed rank |
|---|---|---|---|
| BM25 (S1) | 95 % (57/60) | 0.86 ms | 3 |
| Dense (S2) | 100 % (60/60) | 12.0 ms | 5 |
| Cross-encoder (S3) | 95 % (57/60) | 117.1 ms | 3 |
| Hybrid RRF (S4) | 100 % (60/60) | 13.1 ms | 3 |

Three queries miss the seed for BM25 and the cross-encoder: P09 (Belichick,
BM25 rank 305) and two descriptive queries whose seeds sit outside the top-100.
That is the 80 % threshold doing its job, not a bug. Top-10 overlap between
systems is 35–67 %, so the four lists really differ.

The 0.4 ms BM25 figure in the pilot table above is the 3-month pilot; the full
24-month run is 0.86 ms (larger monthly pools).

## Final annotation (P7)

`scripts/08_pool_final.sh` pooled the top-10 clusters of all four systems for
the 50 evaluation queries, deduplicated and capped at 15:

| Sheet | Rows | Who fills it | Status |
|---|---|---|---|
| `queries/candidates_final.csv` | 718 | three annotators, one share of queries each | 0 graded |
| `queries/final_double_candidates.csv` | 150 | a different person per query, independently | 0 graded |

Columns to fill: `grade`, `same_as`, `annotator`, `notes`. Grades follow the
pilot guide (2 = same event, 1 = related but different development,
0 = different event). After annotation, `make_sheets --mode collect
--prefix final` writes the slim `final_judgments*.csv` files.

**Do not confuse the two pairs of files.** `candidates_final.csv` and
`final_double_candidates.csv` are the sheets people edit (they contain titles
and snippets, so they are gitignored). `final_judgments.csv` and
`final_judgments_double.csv` are the slim collect output (IDs and grades only,
tracked in git). `make_sheets --mode init` writes a blank placeholder
`final_judgments.csv`; it is overwritten by `collect` and should not be filled
by hand.

Annotation window: **24.11 – 07.12**, with at least half of the 718 rows done
by **01.12 (G3)**. G4 on 07.12 collects the sheets, recomputes κ and runs the
`story_id` merge before the qrels are built.

## Deviations and known limitations

Tracked in `projectIdeas/12_implementation_plan.md` §0 (reserve queries,
S1/S2-only pilot pool, 1,080 judgment decisions, midterm timing). Copy that list
here before the final submission.

- Some wire stories fragment into several single-article clusters (e.g. the
  Hunter Biden plea and the Fed explainer in the pilot). The annotation
  `same_as` field records these; resolving them into one canonical `story_id`
  is deferred to **G4 (07.12)**, before the evaluation qrels are built.
- The seed blocklist was extended after "Today in History" columns and
  "Playoffs and Super Bowl Odds" pages survived the first filter and scored
  high on richness (they share boilerplate titles, not an event). Re-run
  `scripts/03_seeds.sh`/`06_seeds_full.sh` after changing the blocklist.
- The pilot candidate pool is built from **S1/S2 only**. It is used only to
  tune the reranker settings; the final pool uses S1–S4 and is never used for
  the pilot κ.
- **Cross-encoder input (open).** The design document says S3 reranks the
  top-100 of **S2 (dense)**; the code reranks the top-100 of **BM25**
  (`src/retrieval/crossencoder.py`, `--base bm25`). Pick one and align code or
  document. The pilot numbers above were produced with the BM25 base.
- **Pool size (open).** The design document estimates ~900 cluster decisions
  (750 evaluation + 150 pilot). The actual final pool is **718 evaluation +
  150 double-judged = 868 decisions**, and the 10 pilot queries are not in the
  final pool. Update the document's estimate or cite the real count.
- **Reserve duplicates (open).** `R01` is an exact duplicate of `E38` (same seed
  cluster `2023-03-C00091`, text similarity 1.00) and `R10` is a near-duplicate
  of `E39` (0.81). Replace or re-seed them before using the reserves.
- **Query months.** The 50 evaluation queries cover **19** of the 24 months
  (not 22 as an earlier draft said); five months have no evaluation query.
  `queries/query_allocation.csv` lists the per-month target.

## Generative AI statement

Fill this before submission: which tools were used (code scaffolding, debugging,
documentation), what they contributed, and that the design document was written
without AI support. All AI-assisted code must be reviewed by the team.

## Licence and citations

Code: to be decided with the group (e.g. MIT). Data: see the Infini-News terms;
do not redistribute article text. Cite Infini-News (arXiv:2605.18337),
Sentence-BERT/MiniLM (arXiv:1908.10084), e5 (arXiv:2212.03533), the MS MARCO
cross-encoder (arXiv:1901.04085), bm25s (arXiv:2407.03618) and ir_measures
(ECIR 2022, doi:10.1007/978-3-030-99739-7_38) in the final README.
