"""
Cross-modal positive control on the paired set: text-to-image retrieval and clustering for
V0 (MiniLM zero-padded + CLIP image), V1 (CLIP text + CLIP image) and V2 (V1, mean-centred) (Table 7, Figure 8).

    python eval/cross_modal_experiment.py --text_dir data/paired_notes --img_dir data/dimg --out results/crossmodal
"""
import argparse, csv, os, re, sys
from pathlib import Path
from collections import Counter

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=str(Path(__file__).resolve().parents[1] / "Core"))
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out", default="results_crossmodal")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n_init", type=int, default=100)
    ap.add_argument("--chunk_words", type=int, default=35, help="words per CLIP text chunk (CLIP reads 77 tokens)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, a.core)
    import torch
    import FileGroup as FG
    from FileScanner import FileScanner
    from file_node import extract_text
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    def load(folder):
        sc = FileScanner(); sc.search_files(folder)
        return sorted(sc.file_data(), key=lambda n: n.path)

    def cls_of(node):
        return re.match(r"[A-Za-z]+", os.path.splitext(node.filename)[0]).group(0).lower()

    t_nodes, i_nodes = load(a.text_dir), load(a.img_dir)
    t_cls = np.array([cls_of(n) for n in t_nodes]); i_cls = np.array([cls_of(n) for n in i_nodes])
    print(f"{len(t_nodes)} texts {dict(Counter(map(str, t_cls)))}\n{len(i_nodes)} images {dict(Counter(map(str, i_cls)))}\n")
    if sorted(set(t_cls)) != sorted(set(i_cls)):
        sys.exit("text and image classes differ")
    texts = [extract_text(n.path) for n in t_nodes]

    g = FG.FileGrouper()
    model, processor = g._get_clip()

    # ---------- representations ----------
    unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    img_clip = np.array([g._embed_image(n.path) for n in i_nodes], dtype=np.float64)       # app's CLIP image path
    img_clip = unit(img_clip)

    def clip_text(txts):
        tok = processor.tokenizer
        out = []
        with torch.no_grad():
            for t in txts:
                words = t.split() or [""]
                chunks = [" ".join(words[i:i + a.chunk_words]) for i in range(0, len(words), a.chunk_words)]
                enc = tok(chunks, padding=True, truncation=True, max_length=77, return_tensors="pt")
                o = model.text_model(input_ids=enc["input_ids"].to(g.device),
                                     attention_mask=enc["attention_mask"].to(g.device), return_dict=True)
                f = model.text_projection(o.pooler_output).cpu().numpy().astype(np.float64)
                f = unit(f)
                out.append(unit(f.mean(0, keepdims=True))[0])      # mean over chunks, re-normalised
        return np.array(out)

    txt_clip = clip_text(texts)
    mini = np.asarray(g._encode_texts(texts), dtype=np.float64)                          # 384-d MiniLM, unit norm
    D = img_clip.shape[1]                                                                # 512 for CLIP ViT-B/32
    txt_mini = np.zeros((len(texts), D)); txt_mini[:, :min(D, mini.shape[1])] = mini[:, :D]   # the app's zero-padding

    def centre(T, I):
        return unit(T - T.mean(0, keepdims=True)), unit(I - I.mean(0, keepdims=True))

    reps = {
        "V0  MiniLM zero-padded + CLIP image  [current app]": (txt_mini, img_clip),
        "V1  CLIP text + CLIP image": (txt_clip, img_clip),
        "V2  V1, each modality mean-centred": centre(txt_clip, img_clip),
    }

    # ---------- 1. retrieval ----------
    def retrieval(Q, D, q_cls, d_cls):
        S = Q @ D.T
        top1, p6 = [], []
        for qi in range(len(Q)):
            order = np.argsort(-S[qi])
            top1.append(d_cls[order[0]] == q_cls[qi])
            p6.append(np.mean(d_cls[order[:6]] == q_cls[qi]))
        return float(np.mean(top1)), float(np.mean(p6))

    print("1. RETRIEVAL (chance = 0.20 for both numbers)")
    print(f"{'representation':55s}{'text→img top-1':>16s}{'text→img P@6':>14s}{'img→text top-1':>16s}{'img→text P@6':>14s}")
    ret_rows = []
    for name, (T, I) in reps.items():
        a1, a2 = retrieval(T, I, t_cls, i_cls); b1, b2 = retrieval(I, T, i_cls, t_cls)
        print(f"{name:55s}{a1:16.3f}{a2:14.3f}{b1:16.3f}{b2:14.3f}")
        ret_rows.append([name, a1, a2, b1, b2])
    print()

    # ---------- 2. joint clustering ----------
    cls_all = np.concatenate([t_cls, i_cls]); mod_all = np.array([0] * len(t_cls) + [1] * len(i_cls))
    same_class_pairs = [(ti, len(t_cls) + ii) for ti in range(len(t_cls)) for ii in range(len(i_cls)) if t_cls[ti] == i_cls[ii]]

    def joint_metrics(lab):
        both = sum(1 for c in set(lab.tolist()) if len(set(mod_all[lab == c].tolist())) == 2) / len(set(lab.tolist()))
        co = np.mean([lab[x] == lab[y] for x, y in same_class_pairs])
        return (adjusted_rand_score(cls_all, lab), normalized_mutual_info_score(cls_all, lab),
                normalized_mutual_info_score(mod_all, lab), both, co)

    print("2. JOINT K-MEANS on all 60 files (mean ± SD over seeds 0..%d, n_init=%d)" % (a.seeds - 1, a.n_init))
    hdr = f"{'representation':55s}{'k':>3s}{'ARI':>14s}{'NMI':>14s}{'NMI(modality)':>15s}{'mixed clusters':>16s}{'pairs together':>16s}"
    print(hdr)
    clus_rows = []
    for name, (T, I) in reps.items():
        X = np.vstack([T, I])
        for k in (5, 12):
            vals = np.array([joint_metrics(KMeans(k, random_state=s, n_init=a.n_init).fit_predict(X)) for s in range(a.seeds)])
            m, sd = vals.mean(0), (vals.std(0, ddof=1) if len(vals) > 1 else np.zeros(5))
            cell = lambda j, w: f"{m[j]:.3f}±{sd[j]:.3f}".rjust(w)
            print(f"{name:55s}{k:3d}" + cell(0, 14) + cell(1, 14) + cell(2, 15) + cell(3, 16) + cell(4, 16))
            clus_rows.append([name, k] + [x for j in range(5) for x in (m[j], sd[j])])
    print("\nNMI(modality): 0 = clusters ignore text-vs-image, 1 = clusters are just 'texts' and 'images'.")
    print("mixed clusters: share of clusters holding both texts and images.  pairs together: share of the 180 same-class text/image pairs in one cluster.\n")

    # ---------- per-class retrieval for V1/V2 ----------
    print("Per-class text→image top-1 (V2):")
    T, I = reps["V2  V1, each modality mean-centred"]
    S = T @ I.T
    for c in sorted(set(t_cls)):
        idx = np.where(t_cls == c)[0]
        acc = np.mean([i_cls[np.argmax(S[q])] == c for q in idx])
        print(f"   {c:10s}{acc:6.2f}")

    with open(os.path.join(a.out, "crossmodal_retrieval.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["representation", "t2i_top1", "t2i_p6", "i2t_top1", "i2t_p6"]); w.writerows(ret_rows)
    with open(os.path.join(a.out, "crossmodal_clustering.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["representation", "k", "ari_mean", "ari_sd", "nmi_mean", "nmi_sd", "nmi_modality_mean", "nmi_modality_sd",
                    "mixed_clusters_mean", "mixed_clusters_sd", "pairs_together_mean", "pairs_together_sd"])
        w.writerows(clus_rows)
    print("\nSaved crossmodal_retrieval.csv and crossmodal_clustering.csv in", a.out)


if __name__ == "__main__":
    main()
