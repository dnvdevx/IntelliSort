"""
D-IMG: CLIP pooling (last_hidden_state[:, 0] vs pooler_output), k = 5 vs k = 6, seed sweep and
row-order ablation (Table 6). The photos are listed in data/pexels_sources.csv.

    python eval/dimg_pooling_ablation.py --data data/dimg --out results/dimg
"""
import argparse, csv, os, re, sys
from pathlib import Path
from collections import Counter

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=str(Path(__file__).resolve().parents[1] / "Core"))
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "dimg"))
    ap.add_argument("--out", default="results_dimg")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n_init", type=int, default=100)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, a.core)
    import torch
    import FileGroup as FG
    from FileScanner import FileScanner
    from PIL import Image
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    sc = FileScanner(); sc.search_files(a.data)
    nodes = sorted(sc.file_data(), key=lambda n: n.path)
    truth = []
    for nd in nodes:
        m = re.match(r"[a-zA-Z]+", os.path.splitext(nd.filename)[0])
        truth.append(m.group(0).lower())
    truth = np.array(truth)
    n = len(nodes)
    print(f"{n} images, classes: { {str(k): v for k, v in Counter(truth).items()} }\n")

    g = FG.FileGrouper()
    model, processor = g._get_clip()
    old_vecs, new_vecs = [], []
    with torch.no_grad():
        for nd in nodes:
            img = Image.open(nd.path).convert("RGB")
            px = processor(images=img, return_tensors="pt")["pixel_values"].to(g.device)
            out = model.vision_model(pixel_values=px, return_dict=True)
            f_old = model.visual_projection(out.last_hidden_state[:, 0, :]).squeeze(0).cpu().numpy()
            f_new = model.visual_projection(out.pooler_output).squeeze(0).cpu().numpy()
            old_vecs.append(np.asarray(f_old, dtype=np.float64)); new_vecs.append(np.asarray(f_new, dtype=np.float64))
    norm = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
    variants = {"old pooling (no final LayerNorm)": norm(np.array(old_vecs)),
                "fixed pooling (pooler_output)": norm(np.array(new_vecs))}

    ref = np.array([g._embed_image(nd.path) for nd in nodes[:3]], dtype=np.float64)
    assert np.allclose(ref, variants["fixed pooling (pooler_output)"][:3], atol=1e-4), "app embedding != fixed variant"

    def canon(lab):
        d = {}
        for i, l in enumerate(lab):
            d.setdefault(int(l), []).append(i)
        return tuple(sorted(tuple(v) for v in d.values()))

    def purity(lab):
        return sum(Counter(truth[lab == c]).most_common(1)[0][1] for c in set(lab.tolist())) / n

    rows = []
    print(f"{'variant':36s}{'k':>3s}{'ARI mean ± SD':>18s}{'min':>7s}{'max':>7s}{'NMI':>16s}{'purity':>8s}{'distinct':>9s}{'ARI @seed42':>13s}")
    comp = []
    for vname, X in variants.items():
        for k in (5, 6):
            aris, nmis, purs, parts = [], [], [], []
            for s in range(a.seeds):
                lab = KMeans(k, random_state=s, n_init=a.n_init).fit_predict(X)
                aris.append(adjusted_rand_score(truth, lab)); nmis.append(normalized_mutual_info_score(truth, lab))
                purs.append(purity(lab)); parts.append(canon(lab))
            aris, nmis = np.array(aris), np.array(nmis)
            lab42 = KMeans(k, random_state=42, n_init=a.n_init).fit_predict(X)
            a42 = adjusted_rand_score(truth, lab42)
            sd = aris.std(ddof=1) if len(aris) > 1 else 0.0
            print(f"{vname:36s}{k:3d}{aris.mean():10.4f} ± {sd:.4f}{aris.min():7.3f}{aris.max():7.3f}"
                  f"{nmis.mean():9.3f} ± {(nmis.std(ddof=1) if len(nmis) > 1 else 0):.3f}{np.mean(purs):8.3f}{len(set(parts)):9d}{a42:13.4f}")
            rows.append([vname, k, aris.mean(), sd, aris.min(), aris.max(), nmis.mean(), np.mean(purs), len(set(parts)), a42])
            comp.append((vname, k, lab42))
        ward = AgglomerativeClustering(5, linkage="ward").fit_predict(X)
        print(f"{vname:36s}{'5':>3s}   ward (deterministic): ARI {adjusted_rand_score(truth, ward):.4f}  NMI {normalized_mutual_info_score(truth, ward):.4f}")
    print()
    for vname, k, lab in comp:
        print(f"Cluster composition, {vname}, k={k}, seed 42:")
        for c in sorted(set(lab.tolist())):
            print(f"   cluster {c}: { {str(k): v for k, v in Counter(truth[lab == c]).items()} }   misplaced: "
                  + ", ".join(nodes[i].filename for i in np.where(lab == c)[0] if Counter(truth[lab == c]).most_common(1)[0][0] != truth[i]) )
        print()
    print("(the 'members' list shows only the images that do NOT belong to their cluster's majority class)")

    # Row-order ablation: what the ORIGINAL pipeline effectively did (random row order, one k-means++ start).
    print("\nRow-order ablation: seed fixed at 42, rows shuffled 0..%d times" % (a.seeds - 1))
    print(f"{'variant':36s}{'k':>3s}{'n_init':>7s}{'ARI mean ± SD':>18s}{'min':>7s}{'max':>7s}{'distinct':>9s}{'share<=0.65':>13s}")
    order_rows = []
    for vname, X in variants.items():
        for k in (5, 6):
            for ni in sorted({1, a.n_init}):
                aris, parts = [], []
                for sh in range(a.seeds):
                    perm = np.random.RandomState(sh).permutation(n)
                    lab_p = KMeans(k, random_state=42, n_init=ni).fit_predict(X[perm])
                    lab = np.empty(n, dtype=int); lab[perm] = lab_p
                    aris.append(adjusted_rand_score(truth, lab)); parts.append(canon(lab))
                aris = np.array(aris); sd = aris.std(ddof=1) if len(aris) > 1 else 0.0
                share = float((aris <= 0.65).mean())
                print(f"{vname:36s}{k:3d}{ni:7d}{aris.mean():10.4f} ± {sd:.4f}{aris.min():7.3f}{aris.max():7.3f}{len(set(parts)):9d}{share:13.2f}")
                order_rows.append([vname, k, ni, aris.mean(), sd, aris.min(), aris.max(), len(set(parts)), share])
    print("(share<=0.65 = fraction of shuffles scoring at or below 0.65, i.e. as low as the manuscript's original 0.621)")
    with open(os.path.join(a.out, "dimg_row_order_ablation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "k", "n_init", "ari_mean", "ari_sd", "ari_min", "ari_max", "distinct_partitions", "share_ari_le_0.65"])
        w.writerows(order_rows)

    with open(os.path.join(a.out, "dimg_pooling_ablation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "k", "ari_mean", "ari_sd", "ari_min", "ari_max", "nmi_mean", "purity_mean", "distinct_partitions", "ari_seed42"])
        w.writerows(rows)
    print("Saved dimg_pooling_ablation.csv and dimg_row_order_ablation.csv in", a.out)


if __name__ == "__main__":
    main()
