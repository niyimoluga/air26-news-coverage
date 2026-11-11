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
- [ ] P1 preprocessing (24 months → `docs_YYYY-MM.parquet`)
- [ ] P2 story clustering (MinHash/LSH per month)
- [ ] P3 queries + judgments
- [ ] P4–P11 retrieval, reranking, evaluation, analysis, packaging

## Repository layout

```
config/default.yaml        paths, filters, clustering parameters
data/manifest.csv          24-shard sample (one shard per month, 2023-01–2024-12)
data/processed/            gitignored: docs_*, clusters_*, stats
src/data/common.py         manifest, paths, normalisation, shingles
src/data/preprocess.py     P1: filter shards into monthly docs parquet
src/data/cluster.py        P2: exact + MinHash/LSH clustering per month
scripts/00_setup.sh        check env, create folders
scripts/01_preprocess.sh   run P1
scripts/02_cluster.sh      run P2
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

# smoke tests
python -m unittest discover -s tests -v
```

Expected outputs:

```
data/processed/docs_YYYY-MM.parquet       # doc_id, month, host, title, text, dates, text_hash, cluster_id
data/processed/clusters_YYYY-MM.parquet   # cluster_id, size, hosts, earliest dates, representative
data/processed/corpus_stats.csv
data/processed/cluster_stats.csv
```

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

## Deviations from the design document

Tracked in `projectIdeas/12_implementation_plan.md` §0 (reserve queries,
S1/S2-only pilot pool, 1,080 judgment decisions, midterm timing). Copy that list
here before the final submission.

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
