"""
20 Newsgroups quality comparison and scaling (Tables 9 and 12, Figures 10 and 12).
The dataset is downloaded by scikit-learn on first use.

    python eval/benchmark_and_scaling.py --out results/bench
"""
import argparse, csv, os, shutil, sys, tempfile, time
from pathlib import Path
from collections import Counter

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=str(Path(__file__).resolve().parents[1] / "Core"))
    ap.add_argument("--out", default="results_bench")
    ap.add_argument("--n_init", type=int, default=100, help="use the SAME value as the app (KMEANS_N_INIT)")
    ap.add_argument("--n_quality", type=int, default=2000)
    ap.add_argument("--max_n", type=int, default=5000)
    ap.add_argument("--sizes", default="100,500,1000,2000,5000")
    ap.add_argument("--seeds", type=int, default=10, help="seeds for the k=20 methods")
    ap.add_argument("--seeds_big", type=int, default=3, help="seeds for the slow methods (k=n//5, silhouette-k)")
    ap.add_argument("--max_words", type=int, default=1000, help="cap on words per post (outliers only)")
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--grid", default="2,5,10,15,20,25,30,40,60",
                    help="candidate k values for the silhouette search (extend it if the best k is the largest value)")
    ap.add_argument("--folder", default=None, help="your own folder instead of 20 Newsgroups (quality part only)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, a.core)
    import FileGroup as FG
    from FileScanner import FileScanner
    from file_node import extract_text, SUPPORTED_TEXT_EXTENSIONS
    from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, silhouette_score)

    NI = a.n_init
    T = time.perf_counter

    # =====================================================================================
    # load documents
    # =====================================================================================
    labels_known = True
    if a.folder:
        sc = FileScanner(); sc.search_files(a.folder)
        nodes = [n for n in sorted(sc.file_data(), key=lambda n: n.path) if n.extension in SUPPORTED_TEXT_EXTENSIONS]
        texts, names, truth_raw, skipped = [], [], [], []
        root = os.path.abspath(a.folder)
        for nd in nodes:
            try:
                tx = extract_text(nd.path)
            except Exception as e:
                skipped.append((nd.filename, str(e))); continue
            if not tx.strip():
                skipped.append((nd.filename, "no extractable text")); continue
            texts.append(" ".join(tx.split()[:a.max_words])); names.append(nd.path)
            rel = os.path.relpath(os.path.dirname(nd.path), root)
            truth_raw.append(rel if rel != "." else "")
        labels_known = len(set(truth_raw)) >= 2 and "" not in set(truth_raw)
        print(f"Folder mode: {len(texts)} documents, {len(skipped)} skipped, labelled={labels_known}")
        for f, why in skipped[:20]:
            print(f"  skipped {f}: {why}")
        truth = np.array(truth_raw) if labels_known else None
    else:
        from sklearn.datasets import fetch_20newsgroups
        d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        lens = np.array([len(t.split()) for t in d.data])
        keep = np.where(lens >= 30)[0]
        rng = np.random.RandomState(a.data_seed); rng.shuffle(keep)
        keep = keep[:a.max_n]
        texts = [" ".join(d.data[i].split()[:a.max_words]) for i in keep]
        truth = np.array([d.target_names[d.target[i]] for i in keep])
        print(f"20 Newsgroups pool: {len(texts)} posts (>=30 words, first {a.max_words} words), {len(set(truth))} classes")

    grouper = FG.FileGrouper()
    unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

    # =====================================================================================
    # Part B (first half): scan / extract / model load / embed on the real app functions
    # =====================================================================================
    stage = {}
    if not a.folder:
        tmp = tempfile.mkdtemp(prefix="bench_files_")
        for i, tx in enumerate(texts):
            open(os.path.join(tmp, f"doc_{i:05d}.txt"), "w", encoding="utf-8").write(tx)
        t0 = T(); sc = FileScanner(); sc.search_files(tmp); nodes = sc.file_data(); stage["scan_cold_s"] = T() - t0
        t0 = T(); sc2 = FileScanner(); sc2.search_files(tmp); sc2.file_data(); stage["scan_warm_s"] = T() - t0
        t0 = T(); _ = [extract_text(n.path) for n in nodes]; stage["extract_cold_s"] = T() - t0
        t0 = T(); _ = [extract_text(n.path) for n in nodes]; stage["extract_warm_s"] = T() - t0
        shutil.rmtree(tmp, ignore_errors=True)
    t0 = T(); grouper._get_model(); stage["model_load_s"] = T() - t0
    t0 = T(); emb = np.asarray(grouper._encode_texts(texts), dtype=np.float64); stage["embed_s"] = T() - t0
    stage["embed_docs_per_s"] = len(texts) / stage["embed_s"]
    stage["n_files"] = len(texts)
    print("\nPART B  per-stage timing of the app's pipeline (this machine, CPU)")
    for k_, v_ in stage.items():
        print(f"   {k_:18s}{v_:12.3f}" if isinstance(v_, float) else f"   {k_:18s}{v_:12d}")

    # =====================================================================================
    # Part A: quality
    # =====================================================================================
    nq = min(a.n_quality, len(texts)) if not a.folder else len(texts)
    Eq, Tq = emb[:nq], texts[:nq]
    tq = truth[:nq] if labels_known else None
    k_true = len(set(tq)) if labels_known else None

    def fix_noise(lab):
        lab = np.asarray(lab).copy(); nxt = lab.max() + 1
        for i in np.where(lab == -1)[0]:
            lab[i] = nxt; nxt += 1
        return lab

    def metrics(lab):
        lab = fix_noise(lab); ks = len(set(lab.tolist()))
        sil = silhouette_score(Eq, lab, metric="cosine") if 2 <= ks <= nq - 1 else float("nan")
        out = {"k": ks, "silhouette": sil}
        if labels_known:
            pur = sum(Counter(tq[lab == c]).most_common(1)[0][1] for c in set(lab.tolist())) / nq
            out.update(ari=adjusted_rand_score(tq, lab), nmi=normalized_mutual_info_score(tq, lab), purity=pur)
        return out

    results = []

    def add(name, fn, seeds):
        runs, secs = [], []
        for s in seeds:
            t0 = T(); lab = fn(s); secs.append(T() - t0); runs.append(metrics(lab))
        row = {"method": name, "runs": len(runs), "seconds": float(np.mean(secs))}
        for key in ("ari", "nmi", "purity", "silhouette", "k"):
            if key in runs[0]:
                v = np.array([r[key] for r in runs], dtype=float)
                row[key] = float(np.nanmean(v)); row[key + "_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
        results.append(row)
        print("   " + name, {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in row.items() if k_ != "method"}, flush=True)

    seeds_small, seeds_big = list(range(a.seeds)), list(range(a.seeds_big))
    k_app = max(2, min(nq // 5, 500))
    grid = sorted({int(k) for k in a.grid.split(",") if 2 <= int(k) < nq})
    print(f"\nPART A  quality on {nq} documents" + (f", {k_true} classes" if labels_known else " (unlabelled)"))

    def sil_k(s):
        best, bl = -2, None
        for k in grid:
            lab = KMeans(k, random_state=s, n_init=NI).fit_predict(Eq)
            sc_ = silhouette_score(Eq, lab, metric="cosine")
            if sc_ > best: best, bl = sc_, lab
        return bl

    if labels_known:
        tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=10000, min_df=2).fit_transform(Tq)
        counts = CountVectorizer(stop_words="english", max_features=5000, min_df=3).fit_transform(Tq)
        add(f"TF-IDF + K-Means (k={k_true})", lambda s: KMeans(k_true, random_state=s, n_init=NI).fit_predict(tfidf), seeds_small)
        add(f"LDA ({k_true} topics, argmax)", lambda s: LatentDirichletAllocation(
            k_true, random_state=s, learning_method="online", batch_size=256, max_iter=20).fit_transform(counts).argmax(1), seeds_small)
        add(f"MiniLM + K-Means (k={k_true}, true k)", lambda s: KMeans(k_true, random_state=s, n_init=NI).fit_predict(Eq), seeds_small)
        add(f"MiniLM + Agglomerative ward (k={k_true})", lambda s: AgglomerativeClustering(k_true, linkage="ward").fit_predict(Eq), [0])
    add(f"MiniLM + K-Means (k=n//5={k_app}, the app's rule)", lambda s: KMeans(k_app, random_state=s, n_init=NI).fit_predict(Eq), seeds_big)
    add(f"MiniLM + K-Means, k by silhouette {grid}", sil_k, seeds_big)
    if results[-1].get("k") == max(grid):
        print(f"   NOTE: the silhouette search picked the LARGEST candidate k ({max(grid)}); extend --grid before trusting it.")
    add("MiniLM + HDBSCAN (min_cluster_size=5)", lambda s: HDBSCAN(min_cluster_size=5).fit_predict(Eq), [0])

    with open(os.path.join(a.out, "benchmark_quality.csv"), "w", newline="") as f:
        cols = ["method", "runs", "seconds", "ari", "ari_sd", "nmi", "nmi_sd", "purity", "purity_sd", "silhouette", "silhouette_sd", "k", "k_sd"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(results)

    if a.folder:
        lab = fix_noise(sil_k(0))
        with open(os.path.join(a.out, "folder_membership.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["file", "cluster"] + (["label"] if labels_known else []))
            for i, p in enumerate(names):
                w.writerow([p, int(lab[i])] + ([truth[i]] if labels_known else []))
        print("\nSaved folder_membership.csv (silhouette-chosen k, seed 0) and benchmark_quality.csv in", a.out)
        return

    # =====================================================================================
    # Part C: algorithm scaling
    # =====================================================================================
    sizes = [int(x) for x in a.sizes.split(",") if int(x) <= len(texts)]
    print("\nPART C  clustering time (seconds) vs number of documents; K-Means uses n_init=%d" % NI)
    print(f"{'n':>6s}{'KMeans k=20':>14s}{'KMeans k=n//5':>16s}{'ward k=20':>12s}{'HDBSCAN':>10s}")
    rows = []
    for n in sizes:
        X = emb[:n]; kk = max(2, min(n // 5, 500))
        def tm(fn, reps):
            ts = []
            for _ in range(reps):
                t0 = T(); fn(); ts.append(T() - t0)
            return float(np.median(ts))
        reps = 3   # median of 3 runs at every n
        t_km20 = tm(lambda: KMeans(20, random_state=0, n_init=NI).fit(X), reps)
        t_kmapp = tm(lambda: KMeans(kk, random_state=0, n_init=NI).fit(X), reps)
        t_ward = tm(lambda: AgglomerativeClustering(20, linkage="ward").fit(X), reps)
        t_hdb = tm(lambda: HDBSCAN(min_cluster_size=5).fit(X), reps)
        print(f"{n:6d}{t_km20:14.2f}{t_kmapp:16.2f}{t_ward:12.2f}{t_hdb:10.2f}", flush=True)
        rows.append([n, kk, t_km20, t_kmapp, t_ward, t_hdb])
    with open(os.path.join(a.out, "scaling.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n", "k_app_rule", "kmeans_k20_s", "kmeans_app_k_s", "ward_k20_s", "hdbscan_s"]); w.writerows(rows)
    with open(os.path.join(a.out, "stage_timing.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"]); w.writerows(stage.items())
    print("\nSaved benchmark_quality.csv, scaling.csv, stage_timing.csv in", a.out)


if __name__ == "__main__":
    main()
