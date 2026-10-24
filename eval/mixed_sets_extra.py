"""
Seed sweeps and baselines for D-IMG and D-MIX using the application's own embedding code (Tables 6, 8, 10).

    python eval/mixed_sets_extra.py --data data/dimg --name D-IMG --out results/dimg
    python eval/mixed_sets_extra.py --data data/dmix --truth_csv data/ground_truth/dmix_labels.csv --name D-MIX --out results/dmix
"""
import argparse, csv, os, re, sys
from pathlib import Path
from collections import Counter

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=str(Path(__file__).resolve().parents[1] / "Core"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--truth_csv", default=None)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--out", default="results_extra")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n_init", type=int, default=100)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, a.core)
    import FileGroup as FG
    from FileScanner import FileScanner
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
                                 homogeneity_score, completeness_score)
    from scipy import stats

    sc = FileScanner(); sc.search_files(a.data)
    nodes = sorted(sc.file_data(), key=lambda n: n.path)
    g = FG.FileGrouper()
    emb, ordered, unprocessed = g._build_embeddings(nodes)
    for nd, why in unprocessed:
        print(f"not processed: {nd.filename}: {why}")

    if a.truth_csv:
        tmap = {r["filename"]: r["ground_truth"] for r in csv.DictReader(open(a.truth_csv, encoding="utf-8-sig"))}
    else:
        tmap = {nd.filename: re.match(r"[A-Za-z]+", os.path.splitext(nd.filename)[0]).group(0).lower() for nd in ordered}
    drop = {"unknown", "other (xlsx)", "_extra", ""}
    keep = [i for i, nd in enumerate(ordered) if tmap.get(nd.filename) not in drop and nd.filename in tmap]
    X = np.asarray(emb, dtype=np.float64)[keep]
    nds = [ordered[i] for i in keep]
    truth = np.array([tmap[nd.filename] for nd in nds])
    n = len(nds); k_true = len(set(truth)); k_app = max(2, min(n // 5, 500))
    print(f"\n{a.name}: {n} labelled files, {k_true} classes, app rule k={k_app}, n_init={a.n_init}\n")

    def metrics(lab):
        lab = np.asarray(lab).copy(); nxt = lab.max() + 1
        for i in np.where(lab == -1)[0]:
            lab[i] = nxt; nxt += 1                                  # noise = singleton clusters
        ks = len(set(lab.tolist()))
        pur = sum(Counter(truth[lab == c]).most_common(1)[0][1] for c in set(lab.tolist())) / n
        sil = silhouette_score(X, lab, metric="cosine") if 2 <= ks <= n - 1 else float("nan")
        return dict(ari=adjusted_rand_score(truth, lab), nmi=normalized_mutual_info_score(truth, lab), purity=pur,
                    homogeneity=homogeneity_score(truth, lab), completeness=completeness_score(truth, lab),
                    silhouette=sil, k=ks), tuple(sorted(tuple(np.where(lab == c)[0]) for c in set(lab.tolist())))

    results = []

    def add(name, fn, seeds):
        ms, ps = [], []
        for s in seeds:
            m, p = metrics(fn(s)); ms.append(m); ps.append(p)
        row = {"method": name, "runs": len(ms), "distinct": len(set(ps))}
        for key in ("ari", "nmi", "purity", "homogeneity", "completeness", "silhouette", "k"):
            v = np.array([m[key] for m in ms], dtype=float); v = v[~np.isnan(v)]
            sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            if len(v) > 1 and sd > 0:
                lo, hi = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
            else:
                lo = hi = float(v.mean()) if len(v) else float("nan")
            row[key] = float(v.mean()) if len(v) else float("nan"); row[key + "_sd"] = sd
            row[key + "_lo"], row[key + "_hi"] = float(lo), float(hi)
        results.append(row)

    seeds = list(range(a.seeds))
    ext = [nd.extension for nd in nds]; ext_id = {e: i for i, e in enumerate(sorted(set(ext)))}
    add("Extension grouping", lambda s: [ext_id[e] for e in ext], [0])
    for thr in (0.80, 0.50, 0.30):
        def name_lab(s, thr=thr):
            FG.NAME_SIMILARITY_THRESHOLD = thr
            gg = FG.FileGrouper(); gg.group_by_name(nds)
            where = {nd: gi for gi, grp in enumerate(gg.group["name"].values()) for nd in grp}
            return [where[nd] for nd in nds]
        add(f"Filename similarity, threshold {thr:.2f}", name_lab, [0])
    add(f"Embeddings + K-Means, true k={k_true}", lambda s: KMeans(k_true, random_state=s, n_init=a.n_init).fit_predict(X), seeds)
    add(f"Embeddings + K-Means, app rule k={k_app}", lambda s: KMeans(k_app, random_state=s, n_init=a.n_init).fit_predict(X), seeds)
    add(f"Embeddings + ward, k={k_true}", lambda s: AgglomerativeClustering(k_true, linkage="ward").fit_predict(X), [0])
    add(f"Embeddings + ward, k={k_app}", lambda s: AgglomerativeClustering(k_app, linkage="ward").fit_predict(X), [0])
    for eps in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        add(f"Embeddings + DBSCAN (cosine, eps={eps}, min_samples=2)", lambda s, eps=eps: DBSCAN(eps=eps, min_samples=2, metric="cosine").fit_predict(X), [0])
    add("Embeddings + HDBSCAN (min_cluster_size=3)", lambda s: HDBSCAN(min_cluster_size=3).fit_predict(X), [0])

    f = lambda r, k: ("n/a" if np.isnan(r[k]) else f"{r[k]:.3f}") + (f" ± {r[k + '_sd']:.3f}" if r[k + "_sd"] > 0 else "")
    print(f"{'method':62s}{'runs':>5s}{'dist':>5s}{'ARI':>16s}{'NMI':>16s}{'purity':>14s}{'homog':>14s}{'compl':>14s}{'silh':>14s}{'k':>8s}")
    for r in results:
        print(f"{r['method']:62s}{r['runs']:5d}{r['distinct']:5d}{f(r,'ari'):>16s}{f(r,'nmi'):>16s}{f(r,'purity'):>14s}"
              f"{f(r,'homogeneity'):>14s}{f(r,'completeness'):>14s}{f(r,'silhouette'):>14s}{r['k']:8.1f}")
    for r in results:
        if r["runs"] > 1:
            print(f"95% CI of ARI, {r['method']}: [{r['ari_lo']:.4f}, {r['ari_hi']:.4f}]")
    with open(os.path.join(a.out, f"{a.name}_extra.csv"), "w", newline="") as fh:
        cols = ["method", "runs", "distinct"] + [f"{m}{s}" for m in ("ari", "nmi", "purity", "homogeneity", "completeness", "silhouette", "k") for s in ("", "_sd", "_lo", "_hi")]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(results)
    print("\nSaved", os.path.join(a.out, f"{a.name}_extra.csv"))


if __name__ == "__main__":
    main()
