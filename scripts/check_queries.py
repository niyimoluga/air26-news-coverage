import pandas as pd, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

q = pd.read_csv("/home/larry/air26-news-coverage/queries/queries.csv", dtype=str).fillna("")
s = pd.read_csv("/home/larry/air26-news-coverage/queries/seed_shortlist.csv", dtype=str).fillna("")
a = pd.read_csv("/home/larry/air26-news-coverage/queries/query_allocation.csv", dtype=str).fillna("")
ev, rs, pi = (q[q.status == k] for k in ("eval", "reserve", "pilot"))
print(f"eval={len(ev)} (want 50)  reserve={len(rs)} (want 10)  pilot={len(pi)}")
print("eval styles:", ev["style"].value_counts().to_dict(), "(want 25/25)")
new = pd.concat([ev, rs])
bad = new[(new.text.str.strip() == "") | (new.seed_cluster == "")]
print("rows with empty text or seed:", len(bad))
print("duplicate qids:", q.qid[q.qid.duplicated()].tolist())
print("seed month != query month:", new[new.seed_cluster.str[:7] != new.month].qid.tolist())
print("unknown seed (not in shortlist):", new[~new.seed_cluster.isin(s.cluster_id)].qid.tolist())

tit = dict(zip(s.cluster_id, s.rep_title))
tok = lambda t: set(re.findall(r"[a-z0-9]+", t.lower()))
copy = [r.qid for r in new.itertuples() if r.seed_cluster in tit and
        len(tok(r.text) & tok(tit[r.seed_cluster])) / max(1, len(tok(r.text) | tok(tit[r.seed_cluster]))) > 0.6]
print("text too close to the seed headline (>0.6 word overlap):", copy)

# same event as a pilot query? compare seed headlines within the same month
flag = []
for r in new.itertuples():
    p = pi[pi.month == r.month]
    if r.seed_cluster in set(p.seed_cluster): flag.append((r.qid, "same seed as pilot"))
    elif len(p) and r.seed_cluster in tit:
        ts = [tit[c] for c in p.seed_cluster if c in tit]
        if ts:
            X = TfidfVectorizer(stop_words="english").fit_transform([tit[r.seed_cluster]] + ts)
            if cosine_similarity(X[0], X[1:]).max() > 0.3: flag.append((r.qid, "headline close to a pilot seed"))
print("possible overlap with pilot events:", flag)

per = new[new.status == "eval"].groupby(["month", "author"]).size().rename("have").reset_index()
al = a[["month", "author", "quota"]].astype({"quota": int})
chk = al.merge(per, on=["month", "author"], how="left").fillna({"have": 0})
chk["have"] = chk.have.astype(int)
print("months where count != quota:\n", chk[chk.have != chk.quota].to_string(index=False))
