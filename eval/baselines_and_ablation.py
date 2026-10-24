"""
D-TEXT baselines, clustering alternatives, k-sensitivity, n_init ablation and row-order ablation
(Tables 3, 5, 10 and 11; Figures 5 and 11).

    python eval/baselines_and_ablation.py --data data/dtext --out results/dtext
"""
import argparse, csv, json, os, sys
from pathlib import Path
from collections import Counter

import numpy as np

# 5-class ground truth (ML and software development are separate classes)
GT = {}
for cls, stems in {
    "ml": ["cnn_architecture_notes", "gradient_descent_explained", "ml_supervised_learning",
           "ml_unsupervised_learning", "neural_networks_intro", "transformer_attention"],
    "dev": ["python_tips", "debugging_checklist", "docker_basics", "git_commands",
            "rest_api_design", "sql_join_notes"],
    "recipe": ["recipe_banana_bread", "recipe_curry", "recipe_fried_rice", "recipe_lentil_soup",
               "recipe_omelette", "recipe_pasta"],
    "finance": ["budget_february", "budget_january", "expense_report_q1", "loan_repayment_plan",
                "tax_filing_notes", "investment_summary"],
    "personal": ["birthday_party_plan", "gym_routine", "travel_journal_goa", "weekend_plans",
                 "book_notes_atomic_habits"],
}.items():
    for s in stems:
        GT[s + ".txt"] = cls


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=str(Path(__file__).resolve().parents[1] / "Core"))
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "dtext"))
    ap.add_argument("--out", default="results_baselines")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n_init", type=int, default=100)
    return ap.parse_args()


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, a.core)
    import FileGroup as FG
    from FileScanner import FileScanner
    from file_node import extract_text
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 silhouette_score, homogeneity_score, completeness_score)
    from scipy import stats

    # ---------------- data (same code path as the app) ----------------
    sc = FileScanner(); sc.search_files(a.data)
    nodes = sorted(sc.file_data(), key=lambda n: n.path)
    missing = [n.filename for n in nodes if n.filename not in GT]
    if missing:
        sys.exit(f"Files not in the ground-truth table: {missing}")
    truth = np.array([GT[n.filename] for n in nodes])
    n = len(nodes)
    texts = [extract_text(n_.path) for n_ in nodes]
    grouper = FG.FileGrouper()
    emb = np.asarray(grouper._encode_texts(texts), dtype=np.float64)   # unit-norm MiniLM vectors
    seeds = list(range(a.seeds))
    NI = a.n_init
    print(f"{n} files, {len(set(truth))} true classes, embedding dim {emb.shape[1]}\n")

    # ---------------- helpers ----------------
    def fix_noise(lab):
        lab = np.asarray(lab).copy(); nxt = lab.max() + 1 if lab.size else 0
        for i in np.where(lab == -1)[0]:
            lab[i] = nxt; nxt += 1          # each noise point becomes its own cluster
        return lab

    def canon(lab):
        d = {}
        for i, l in enumerate(lab):
            d.setdefault(int(l), []).append(i)
        return tuple(sorted(tuple(v) for v in d.values()))

    def metrics(lab):
        lab = fix_noise(lab); ks = set(lab.tolist()); k = len(ks)
        purity = sum(Counter(truth[lab == c]).most_common(1)[0][1] for c in ks) / n
        sil = silhouette_score(emb, lab, metric="cosine") if 2 <= k <= n - 1 else float("nan")
        return dict(ari=adjusted_rand_score(truth, lab), nmi=normalized_mutual_info_score(truth, lab),
                    purity=purity, homogeneity=homogeneity_score(truth, lab),
                    completeness=completeness_score(truth, lab), silhouette=sil, k=k)

    def summarise(runs, parts):
        out = {"runs": len(runs), "distinct_partitions": len(set(parts))}
        for key in ("ari", "nmi", "purity", "silhouette", "k"):
            v = np.array([r[key] for r in runs], dtype=float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                out[key] = dict(mean=float("nan"), sd=0.0, lo=float("nan"), hi=float("nan"), min=float("nan"), max=float("nan")); continue
            sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            if len(v) > 1 and sd > 0:
                lo, hi = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
            else:
                lo = hi = float(v.mean())
            out[key] = dict(mean=float(v.mean()), sd=sd, lo=float(lo), hi=float(hi), min=float(v.min()), max=float(v.max()))
        return out

    results = {}

    def add(name, label_fn, stochastic):
        its = seeds if stochastic else [0]
        runs, parts = [], []
        for s in its:
            lab = label_fn(s)
            runs.append(metrics(lab)); parts.append(canon(fix_noise(lab)))
        results[name] = summarise(runs, parts)

    # ---------------- 1. main comparison ----------------
    ext = [n_.extension for n_ in nodes]
    add("Extension grouping", lambda s: [{e: i for i, e in enumerate(sorted(set(ext)))}[e] for e in ext], False)

    for thr in (0.80, 0.50, 0.30):
        def name_labels(s, thr=thr):
            FG.NAME_SIMILARITY_THRESHOLD = thr
            g = FG.FileGrouper(); g.group_by_name(nodes)
            where = {nd: gi for gi, grp in enumerate(g.group["name"].values()) for nd in grp}
            return [where[nd] for nd in nodes]
        add(f"Filename similarity (threshold {thr:.2f})", name_labels, False)

    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    add(f"TF-IDF + K-Means (k=5, n_init={NI})", lambda s: KMeans(5, random_state=s, n_init=NI).fit_predict(tfidf), True)

    counts = CountVectorizer(stop_words="english").fit_transform(texts)
    add("LDA (5 topics, argmax)", lambda s: LatentDirichletAllocation(
        n_components=5, random_state=s, max_iter=50, learning_method="batch").fit_transform(counts).argmax(1), True)

    add("LDA (5 topics, sparse priors 0.1/0.01, min_df=2)", lambda s: LatentDirichletAllocation(
        n_components=5, random_state=s, max_iter=200, learning_method="batch", doc_topic_prior=0.1,
        topic_word_prior=0.01).fit_transform(CountVectorizer(stop_words="english", min_df=2).fit_transform(texts)).argmax(1), True)

    add(f"MiniLM + K-Means (k=5, n_init={NI})  [IntelliSort]", lambda s: KMeans(5, random_state=s, n_init=NI).fit_predict(emb), True)
    add("MiniLM + Agglomerative (ward, k=5)", lambda s: AgglomerativeClustering(5, linkage="ward").fit_predict(emb), False)
    add("MiniLM + Agglomerative (average, cosine, k=5)",
        lambda s: AgglomerativeClustering(5, metric="cosine", linkage="average").fit_predict(emb), False)
    for eps in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        add(f"MiniLM + DBSCAN (cosine, eps={eps}, min_samples=2)",
            lambda s, eps=eps: DBSCAN(eps=eps, min_samples=2, metric="cosine").fit_predict(emb), False)
    add("MiniLM + HDBSCAN (min_cluster_size=3)", lambda s: HDBSCAN(min_cluster_size=3).fit_predict(emb), False)

    def silhouette_k(s):
        best, best_lab = -2, None
        for k in range(2, 11):
            lab = KMeans(k, random_state=s, n_init=NI).fit_predict(emb)
            sil = silhouette_score(emb, lab, metric="cosine")
            if sil > best:
                best, best_lab = sil, lab
        return best_lab
    add(f"MiniLM + K-Means, k by silhouette 2..10 (n_init={NI})", silhouette_k, True)

    # reference: the ground-truth partition itself
    _, truth_ids = np.unique(truth, return_inverse=True)
    add("Ground-truth labels (reference)", lambda s: truth_ids, False)

    # ---------------- print + save table ----------------
    def fmt(d, key, digits=3):
        m = d[key]
        if np.isnan(m["mean"]):
            return "n/a"
        return f"{m['mean']:.{digits}f}" + (f" ± {m['sd']:.{digits}f}" if m["sd"] > 0 else "")

    print("=" * 140)
    print(f"{'Method':58s}{'runs':>5s}{'distinct':>9s}{'ARI':>16s}{'NMI':>16s}{'purity':>14s}{'silhouette':>14s}{'#clusters':>11s}")
    for name, d in results.items():
        print(f"{name:58s}{d['runs']:5d}{d['distinct_partitions']:9d}{fmt(d,'ari'):>16s}{fmt(d,'nmi'):>16s}"
              f"{fmt(d,'purity'):>14s}{fmt(d,'silhouette'):>14s}{fmt(d,'k',1):>11s}")
    print("=" * 140)
    print("Silhouette is computed in the MiniLM cosine space for every method. Noise points (DBSCAN/HDBSCAN) are\n"
          "counted as singleton clusters. ± is the SD across seeds for stochastic methods; blank = deterministic.\n")

    with open(os.path.join(a.out, "baselines_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "runs", "distinct_partitions", "ari_mean", "ari_sd", "ari_ci_lo", "ari_ci_hi",
                    "ari_min", "ari_max", "nmi_mean", "nmi_sd", "purity_mean", "purity_sd",
                    "silhouette_mean", "silhouette_sd", "clusters_mean"])
        for name, d in results.items():
            w.writerow([name, d["runs"], d["distinct_partitions"], d["ari"]["mean"], d["ari"]["sd"], d["ari"]["lo"],
                        d["ari"]["hi"], d["ari"]["min"], d["ari"]["max"], d["nmi"]["mean"], d["nmi"]["sd"],
                        d["purity"]["mean"], d["purity"]["sd"], d["silhouette"]["mean"], d["silhouette"]["sd"],
                        d["k"]["mean"]])

    # ---------------- 2. k-sensitivity ----------------
    print("k-sensitivity: MiniLM + K-Means (n_init=%d), seeds 0..%d" % (NI, a.seeds - 1))
    print(f"{'k':>3s}{'ARI':>16s}{'NMI':>16s}{'purity':>14s}{'silhouette':>14s}")
    ksens = {}
    for k in range(2, 11):
        runs, parts = [], []
        for s in seeds:
            lab = KMeans(k, random_state=s, n_init=NI).fit_predict(emb)
            runs.append(metrics(lab)); parts.append(canon(lab))
        d = summarise(runs, parts); ksens[k] = d
        print(f"{k:3d}{fmt(d,'ari'):>16s}{fmt(d,'nmi'):>16s}{fmt(d,'purity'):>14s}{fmt(d,'silhouette'):>14s}")
    with open(os.path.join(a.out, "k_sensitivity.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["k", "ari_mean", "ari_sd", "nmi_mean", "nmi_sd", "purity_mean", "silhouette_mean"])
        for k, d in ksens.items():
            w.writerow([k, d["ari"]["mean"], d["ari"]["sd"], d["nmi"]["mean"], d["nmi"]["sd"], d["purity"]["mean"], d["silhouette"]["mean"]])
    print()

    # ---------------- 3. n_init ablation ----------------
    print("n_init ablation: MiniLM + K-Means (k=5), file order fixed, seeds 0..%d" % (a.seeds - 1))
    print(f"{'n_init':>7s}{'ARI mean ± SD':>20s}{'min':>8s}{'max':>8s}{'distinct':>10s}{'corr(inertia,ARI)':>20s}")
    ablation_rows = []
    for ni in (1, 10, 50, 100):
        aris, inert, parts = [], [], []
        for s in seeds:
            km = KMeans(5, random_state=s, n_init=ni).fit(emb)
            aris.append(adjusted_rand_score(truth, km.labels_)); inert.append(km.inertia_); parts.append(canon(km.labels_))
        aris = np.array(aris)
        corr = stats.spearmanr(inert, aris)[0] if np.std(inert) > 0 and aris.std() > 0 else float("nan")
        print(f"{ni:7d}{aris.mean():12.4f} ± {aris.std(ddof=1):.4f}{aris.min():8.3f}{aris.max():8.3f}{len(set(parts)):10d}{corr:20.3f}")
        ablation_rows.append(["n_init", ni, aris.mean(), aris.std(ddof=1), aris.min(), aris.max(), len(set(parts)), corr])
    print("(a negative correlation means lower inertia goes with higher ARI)\n")

    # ---------------- 4. row-order ablation ----------------
    print("Row-order ablation: seed fixed at 42, rows shuffled 0..%d times (this is the bug the fix removed)" % (a.seeds - 1))
    print(f"{'setting':>26s}{'ARI mean ± SD':>20s}{'min':>8s}{'max':>8s}{'distinct':>10s}")
    ni_list = [1, 10] + ([NI] if NI not in (1, 10) else [])
    for label, kw in ((f"shuffled, n_init={x}", dict(n_init=x)) for x in ni_list):
        aris, parts = [], []
        for s in seeds:
            perm = np.random.RandomState(s).permutation(n)
            km = KMeans(5, random_state=42, **kw).fit(emb[perm])
            lab = np.empty(n, dtype=int); lab[perm] = km.labels_
            aris.append(adjusted_rand_score(truth, lab)); parts.append(canon(lab))
        aris = np.array(aris)
        print(f"{label:>26s}{aris.mean():12.4f} ± {aris.std(ddof=1):.4f}{aris.min():8.3f}{aris.max():8.3f}{len(set(parts)):10d}")
        ablation_rows.append(["row_order", label, aris.mean(), aris.std(ddof=1), aris.min(), aris.max(), len(set(parts)), ""])
    km = KMeans(5, random_state=42, n_init=NI).fit(emb)
    print(f"{('sorted order, n_init=%d' % NI):>26s}{adjusted_rand_score(truth, km.labels_):12.4f}   (the fixed app: one deterministic result)\n")
    with open(os.path.join(a.out, "ablation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ablation", "setting", "ari_mean", "ari_sd", "ari_min", "ari_max", "distinct_partitions", "spearman_inertia_ari"])
        w.writerows(ablation_rows)

    with open(os.path.join(a.out, "baselines_results.json"), "w") as f:
        json.dump({"main": results, "k_sensitivity": ksens}, f, indent=2)
    print("Saved: baselines_table.csv, k_sensitivity.csv, ablation.csv, baselines_results.json in", a.out)


if __name__ == "__main__":
    main()
