"""
D-MIX with the experimental centring mode (cross_modal = True) against the default (Table 7).

    python eval/dmix_crossmodal.py --data data/dmix --truth_csv data/ground_truth/dmix_labels.csv --out results/dmix_crossmodal
"""
import argparse
from pathlib import Path
import csv
import os
import sys
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=str(Path(__file__).resolve().parents[1] / "Core"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--truth_csv", required=True)
    ap.add_argument("--out", default="results_dmix_crossmodal")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n_init", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, a.core)
    import FileGroup as FG
    from FileScanner import FileScanner
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    truth_map = {}
    for r in csv.DictReader(open(a.truth_csv, encoding="utf-8-sig")):
        truth_map[r["filename"]] = r["ground_truth"]
    skip = {"unknown", "other (xlsx)", "_extra", ""}
    topic = {"food": "recipe"}

    sc = FileScanner()
    sc.search_files(a.data)
    nodes = sorted(sc.file_data(), key=lambda n: n.path)

    rows = []
    for mode in (False, True):
        g = FG.FileGrouper()
        g.cross_modal = mode
        emb, ordered, unprocessed = g._build_embeddings(nodes)
        keep = [i for i, n in enumerate(ordered) if truth_map.get(n.filename, "") not in skip]
        X = np.asarray(emb, dtype=np.float64)[keep]
        nds = [ordered[i] for i in keep]
        y = np.array([truth_map[n.filename] for n in nds])
        y_topic = np.array([topic.get(c, c) for c in y])
        is_img = np.array([g._is_image(n) for n in nds])
        k = max(2, min(len(nds) // 5, 500))
        name = "CLIP text + centring" if mode else "MiniLM zero-padded (default)"

        recipe_txt = [i for i in range(len(nds)) if y[i] == "recipe" and not is_img[i]]
        food_img = [i for i in range(len(nds)) if y[i] == "food" and is_img[i]]

        def score(lab):
            mixed = 0
            for c in set(lab.tolist()):
                m = is_img[lab == c]
                if m.any() and (~m).any():
                    mixed += 1
            pairs = np.mean([lab[i] == lab[j] for i in recipe_txt for j in food_img]) if recipe_txt and food_img else float("nan")
            return [adjusted_rand_score(y, lab), normalized_mutual_info_score(y, lab),
                    adjusted_rand_score(y_topic, lab), mixed / len(set(lab.tolist())), pairs]

        lab42 = KMeans(k, random_state=42, n_init=a.n_init).fit_predict(X)
        app = score(lab42)
        sweep = np.array([score(KMeans(k, random_state=s, n_init=a.n_init).fit_predict(X)) for s in range(a.seeds)])

        print(f"\n=== {name}: {len(nds)} files, k={k}, n_init={a.n_init} ===")
        print(f"app result (seed 42): ARI {app[0]:.4f}  NMI {app[1]:.4f}  topic ARI {app[2]:.4f}  "
              f"mixed clusters {app[3]:.3f}  recipe-food pairs together {app[4]:.3f}")
        sd = sweep.std(axis=0, ddof=1) if len(sweep) > 1 else np.zeros(5)
        mn = sweep.mean(axis=0)
        print(f"{a.seeds} seeds: ARI {mn[0]:.4f} ± {sd[0]:.4f}  NMI {mn[1]:.4f} ± {sd[1]:.4f}  topic ARI {mn[2]:.4f} ± {sd[2]:.4f}  "
              f"mixed {mn[3]:.3f} ± {sd[3]:.3f}  pairs {mn[4]:.3f} ± {sd[4]:.3f}")
        print("clusters at seed 42:")
        for c in sorted(set(lab42.tolist())):
            idx = np.where(lab42 == c)[0]
            parts = Counter(("img:" if is_img[i] else "txt:") + y[i] for i in idx)
            print(f"  {c:2d}: {dict(parts)}")
        rows.append([name, len(nds), k, *app, *mn, *sd])

    with open(os.path.join(a.out, "dmix_crossmodal.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["ari", "nmi", "topic_ari", "mixed", "pairs"]
        w.writerow(["mode", "files", "k"] + [f"app_{c}" for c in cols] + [f"mean_{c}" for c in cols] + [f"sd_{c}" for c in cols])
        w.writerows(rows)
    print("\nsaved", os.path.join(a.out, "dmix_crossmodal.csv"))


if __name__ == "__main__":
    main()
