# Queries and judgments

## Files

| File | Tracked? | Content |
|---|---|---|
| `seed_candidates.csv` | no (gitignored) | candidate seed clusters with title, URL, snippet |
| `queries.csv` | yes | the 60 active queries + 10 reserve |
| `candidates.csv` | no (gitignored) | pooled clusters per query for annotation |
| `judgments.csv` | yes | raw annotator labels |
| `judgments_final.csv` | yes | adjudicated labels + canonical `story_id` |
| `qrels_doc.csv` | yes | doc-level qrels, `iteration=story_id` |

## Seed selection

```bash
bash scripts/03_seeds.sh
```

Filters: cluster size >= 5, >= 3 hostnames, size/hosts <= 10, blocklist for
price pages, live blogs, listicles, etc. Writes `queries/seed_candidates.csv`
(top 12 per month). Pick 3–5 per month for the 50 evaluation queries and 2–3 for
the 10 pilot queries. Prefer clusters whose snippet describes a concrete event.

## Query authoring protocol

1. Read the seed title + snippet (open the URL if needed).
2. Write what a reader would type after seeing a push notification.
3. Do not copy the headline. Do not look at system outputs.
4. `style=entity` for names/places/dates; `style=descriptive` without the main
   entity names.
5. Add a time hint (`time_hint=2024-03` or empty) for half of each style.
6. Record `qid, text, style, month, time_hint, seed_cluster, seed_host, author,
   status` (`pilot` / `eval` / `reserve`).

Suggested IDs: `P01–P10` (pilot), `E01–E50` (evaluation), `R01–R10` (reserve).

## Judgments

Annotate one row per candidate cluster:

```
qid, cluster_id, grade, same_as, annotator, double, notes
```

`grade`: 2 = same event, 1 = related/different development, 0 = different.
`same_as`: another cluster_id if the two are the same wire text under different
headlines. Resolve merges into one canonical `story_id` before building qrels;
`iteration=story_id` for alpha-nDCG.
