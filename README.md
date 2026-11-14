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
- [x] P4 BM25 + dense on the 3 smoke-test months (seed hit 100 %, BM25 ~1 ms, dense ~12 ms)
- [x] P5 candidate pool + annotation sheets (**138 candidates**, 30 double)
- [ ] P5 pilot judgments + κ (people, then `05_kappa.sh`)
- [ ] P6–P11 retrieval, reranking, evaluation, analysis, packaging

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
src/retrieval/run_all.py   month-scoped runs, latency, seed check
scripts/00_setup.sh        check env, create folders
scripts/01_preprocess.sh   run P1
scripts/02_cluster.sh      run P2
scripts/03_seeds.sh        build queries/seed_candidates.csv
scripts/04_retrieval.sh    run P4 on the pilot queries
scripts/05_pool.sh         P5 candidate pool + sheets
scripts/05_kappa.sh        P5 validate + κ
tests/test_common.py       smoke tests
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

# P5 — pool + sheets; then people fill candidates.csv and double_candidates.csv
bash scripts/05_pool.sh
# after annotation:
python -m src.queries.make_sheets --mode collect
bash scripts/05_kappa.sh

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
results/metrics/pilot_kappa.csv
```

P4 pilot result: seed cluster in top-100 for **10/10 queries** with both systems.
BM25 median latency ≈1 ms/query; dense ≈12 ms/query (L40S, 3 months).

P5 pool result: **138 candidates** (11–15 per query), **30 double-judged**
(3 per query). The seed cluster was outside the top-10 clusters for P09, so it
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

## Deviations and known limitations

Tracked in `projectIdeas/12_implementation_plan.md` §0 (reserve queries,
S1/S2-only pilot pool, 1,080 judgment decisions, midterm timing). Copy that list
here before the final submission.

- Some wire stories fragment into several single-article clusters (e.g. the
  Hunter Biden plea and the Fed explainer in the pilot). The annotation
  `same_as` field records these; resolving them into one canonical `story_id`
  is deferred to **G4 (07.12)**, before the evaluation qrels are built.
- The pilot candidate pool is built from **S1/S2 only**. It is used only to
  tune the reranker settings; the final pool uses S1–S4 and is never used for
  the pilot κ.

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
