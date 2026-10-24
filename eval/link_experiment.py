"""
Cluster-then-link mode on the paired set and on D-MIX for several z thresholds (Table 7).

    python eval/link_experiment.py --text_dir data/paired_notes --img_dir data/dimg --mix_dir data/dmix --truth_csv data/ground_truth/dmix_labels.csv --out results/link
"""
import argparse
from pathlib import Path
import csv
import os
import re
import sys
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=str(Path(__file__).resolve().parents[1] / "Core"))
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--mix_dir", required=True)
    ap.add_argument("--truth_csv", required=True)
    ap.add_argument("--out", default="results_link")
    ap.add_argument("--seeds", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, a.core)
    import FileGroup as FG
    from FileScanner import FileScanner
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    def scan(*folders):
        nodes = []
        for f in folders:
            sc = FileScanner()
            sc.search_files(f)
            nodes += sc.file_data()
        return sorted(nodes, key=lambda n: n.path)

    def name_class(fname):
        return re.match(r"[A-Za-z]+", os.path.splitext(fname)[0]).group(0).lower()

    truth_csv = {r["filename"]: r["ground_truth"] for r in csv.DictReader(open(a.truth_csv, encoding="utf-8-sig"))}
    datasets = [
        ("paired", scan(a.text_dir, a.img_dir), lambda n: name_class(n.filename), lambda c: c),
        ("D-MIX", scan(a.mix_dir), lambda n: truth_csv.get(n.filename, ""), lambda c: {"food": "recipe"}.get(c, c)),
    ]
    skip = {"", "unknown", "other (xlsx)", "_extra"}
    settings = [("joint K-Means (default)", None), ("separate, no linking", float("inf"))]
    settings += [(f"link, z = {z}", z) for z in (1.0, 1.5, 2.0, 2.5, 3.0)]

    rows = []
    for dname, nodes, get_class, to_topic in datasets:
        g = FG.FileGrouper()
        emb, ordered, unprocessed = g._build_embeddings(nodes)
        X = np.asarray(emb, dtype=np.float64)
        y = np.array([get_class(n) for n in ordered])
        keep = np.array([c not in skip for c in y])
        topic = np.array([to_topic(c) for c in y])
        is_img = np.array([g._is_image(n) for n in ordered])
        k = max(2, min(len(ordered) // 5, 500))
        print(f"\n=== {dname}: {len(ordered)} files ({int((~is_img).sum())} text, {int(is_img.sum())} images), "
              f"{len(set(y[keep]))} classes, joint k = {k}, n_init = {FG.KMEANS_N_INIT} ===")

        for sname, z in settings:
            res = []
            detail = None
            for s in range(a.seeds):
                if z is None:
                    lab = KMeans(k, random_state=s, n_init=FG.KMEANS_N_INIT).fit_predict(X)
                    links = []
                else:
                    lab, links = g._cluster_linked(X, ordered, seed=s, z=z)
                good = 0
                bad = 0
                for t_cl, _ in links:
                    t_major = Counter(topic[(lab == t_cl) & keep & ~is_img]).most_common(1)
                    i_major = Counter(topic[(lab == t_cl) & keep & is_img]).most_common(1)
                    if t_major and i_major and t_major[0][0] == i_major[0][0]:
                        good += 1
                    else:
                        bad += 1
                mixed = sum(1 for c in set(lab.tolist()) if is_img[lab == c].any() and (~is_img[lab == c]).any())
                res.append([adjusted_rand_score(y[keep], lab[keep]), normalized_mutual_info_score(y[keep], lab[keep]),
                            adjusted_rand_score(topic[keep], lab[keep]), len(links), good, bad, mixed])
                if s == 42:
                    detail = []
                    for t_cl, _ in links:
                        m = lab == t_cl
                        detail.append(dict(Counter(("img:" if is_img[i] else "txt:") + y[i] for i in np.where(m)[0])))
            r = np.array(res, dtype=float)
            mn = r.mean(axis=0)
            sd = r.std(axis=0, ddof=1) if len(r) > 1 else np.zeros(r.shape[1])
            any_bad = float((r[:, 5] > 0).mean())
            print(f"{sname:26s} ARI {mn[0]:.3f} ± {sd[0]:.3f}  NMI {mn[1]:.3f}  topic ARI {mn[2]:.3f} ± {sd[2]:.3f}  "
                  f"links {mn[3]:.2f} (correct {mn[4]:.2f}, wrong {mn[5]:.2f})  seeds with a wrong link {any_bad:.2f}")
            if detail:
                print(f"{'':26s} linked groups at seed 42: {detail}")
            rows.append([dname, sname, *mn, *sd, any_bad])

        lab, _ = g._cluster_linked(X, ordered, seed=42, z=float("inf"))
        print("mutual best text/image group pairs at seed 42 (score = z-value, link if above the chosen z):")
        for t_cl, i_cl, sc in sorted(g.last_link_scores, key=lambda x: -x[2]):
            txt = {str(c): v for c, v in Counter(y[lab == t_cl]).items()}
            img = {str(c): v for c, v in Counter(y[lab == i_cl]).items()}
            same = Counter(topic[lab == t_cl]).most_common(1)[0][0] == Counter(topic[lab == i_cl]).most_common(1)[0][0]
            print(f"   z = {sc:5.2f}  {'match ' if same else 'WRONG '} text {txt}  <->  images {img}")

    with open(os.path.join(a.out, "link_experiment.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["ari", "nmi", "topic_ari", "links", "correct_links", "wrong_links", "mixed_clusters"]
        w.writerow(["dataset", "setting"] + [f"mean_{c}" for c in cols] + [f"sd_{c}" for c in cols] + ["share_seeds_wrong_link"])
        w.writerows(rows)
    print("\nsaved", os.path.join(a.out, "link_experiment.csv"))


if __name__ == "__main__":
    main()
