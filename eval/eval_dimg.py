"""
eval_dimg.py
============
D-IMG experiment:
  - Runs CLIP clustering on the 30 D-IMG photos (5 categories x 6); --k 5 by default, --k 6 for the app rule.
  - Reports ARI, per-file cluster table, car split, dog/food confusion.
  - CLIP zero-shot classification on all 30 photos — flags misplaced images.
  - Generates Figure 4 (PCA/UMAP scatter of CLIP embeddings).
"""

import sys, os, time, json
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

REPO_ROOT = Path(__file__).parent.parent
CORE_DIR  = REPO_ROOT / "Core"
EVAL_DIR  = REPO_ROOT / "eval"
FIG_DIR   = REPO_ROOT / "figures"
RES_DIR   = REPO_ROOT / "results"
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--data", default=str(REPO_ROOT / "data" / "dimg"))
_ap.add_argument("--k", type=int, default=5)
ARGS = _ap.parse_args()
DATA_DIR  = Path(ARGS.data)

sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(EVAL_DIR))

FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

ZERO_SHOT_LABELS = ["a photo of a car", "a photo of a desk",
                    "a photo of a dog", "a photo of food",
                    "a photo of a mountain"]
LABEL_NAMES      = ["car", "desk", "dog", "food", "mountain"]
N_CLUSTERS       = ARGS.k

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ground_truth import IMAGE_GROUND_TRUTH


def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading CLIP on {device} ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc, device


def embed_images(image_paths, model, proc, device):
    import torch
    vecs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            vision_out = model.vision_model(
                pixel_values=inputs["pixel_values"], return_dict=True)
            pooled   = vision_out.pooler_output
            features = model.visual_projection(pooled)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        vecs.append(features.squeeze(0).cpu().numpy())
    return np.array(vecs, dtype=np.float32)


def zero_shot_classify(image_paths, model, proc, device):
    import torch
    text_inputs = proc(text=ZERO_SHOT_LABELS, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_feats = model.get_text_features(**text_inputs)
        if hasattr(text_feats, "pooler_output") and text_feats.pooler_output is not None:
            text_feats = text_feats.pooler_output
        elif hasattr(text_feats, "last_hidden_state"):
            text_feats = text_feats.last_hidden_state[:, 0, :]
        text_feats = torch.nn.functional.normalize(text_feats, p=2, dim=-1)

    preds = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img_inputs = proc(images=img, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        with torch.no_grad():
            img_feats = model.get_image_features(**img_inputs)
            if hasattr(img_feats, "pooler_output") and img_feats.pooler_output is not None:
                img_feats = img_feats.pooler_output
            elif hasattr(img_feats, "last_hidden_state"):
                img_feats = img_feats.last_hidden_state[:, 0, :]
            img_feats = torch.nn.functional.normalize(img_feats, p=2, dim=-1)
            scores = (img_feats @ text_feats.T).squeeze(0)
        preds.append(LABEL_NAMES[scores.argmax().item()])
    return preds


def main():
    print("=" * 60)
    print("D-IMG CLIP Evaluation")
    print("=" * 60)

    # ── load images ──────────────────────────────────────────────────────────
    image_paths = sorted(DATA_DIR.glob("*.jpg"))
    filenames   = [p.name for p in image_paths]
    true_labels = [IMAGE_GROUND_TRUTH[f] for f in filenames]
    cats        = LABEL_NAMES
    cat_map     = {c: i for i, c in enumerate(cats)}
    y_true      = np.array([cat_map[l] for l in true_labels])

    print(f"\n  Found {len(filenames)} images in {DATA_DIR}")

    # ── load CLIP ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model, proc, device = load_clip()
    clip_load_time = time.perf_counter() - t0

    # ── embed ─────────────────────────────────────────────────────────────────
    print("  Embedding images ...")
    t0 = time.perf_counter()
    embeddings = embed_images(image_paths, model, proc, device)
    embed_time = time.perf_counter() - t0
    print(f"  Embeddings: {embeddings.shape}  ({embed_time:.2f}s)")

    # ── cluster ───────────────────────────────────────────────────────────────
    print("  Clustering (KMeans) ...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=100)
    y_pred = km.fit_predict(embeddings)

    ari = adjusted_rand_score(y_true, y_pred)
    print(f"\n  ARI = {ari:.4f}")

    # ── per-file table ────────────────────────────────────────────────────────
    # map predicted cluster → best label via Hungarian
    from scipy.optimize import linear_sum_assignment
    cm_raw = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm_raw)
    cluster_names = {col: cats[row] for row, col in zip(row_ind, col_ind)}
    y_pred_named  = [cluster_names.get(p, f"cluster_{p}") for p in y_pred]

    print(f"\n  {'File':<18} {'True':^10} {'Pred':^10} {'OK?':^5}")
    print("  " + "-" * 48)

    misplaced = []
    car_split_clusters = set()
    dog_food_confusions = []

    for fname, tl, pl in zip(filenames, true_labels, y_pred_named):
        ok = "OK" if tl == pl else "X"
        print(f"  {fname:<18} {tl:^10} {pl:^10} {ok:^5}")
        if tl != pl:
            misplaced.append((fname, tl, pl))
        # car split check
        if tl == "car":
            car_split_clusters.add(pl)
        # dog/food confusion
        if (tl == "dog" and pl == "food") or (tl == "food" and pl == "dog"):
            dog_food_confusions.append((fname, tl, pl))

    print(f"\n  Misplaced: {len(misplaced)} / {len(filenames)}")
    print(f"  Car-cluster split: {car_split_clusters}  "
          f"({'SPLIT X' if len(car_split_clusters) > 1 else 'no split OK'})")
    print(f"  Dog/food confusion: {dog_food_confusions or 'none OK'}")

    # ── zero-shot ─────────────────────────────────────────────────────────────
    print("\n  Running CLIP zero-shot classification ...")
    t0 = time.perf_counter()
    zs_preds = zero_shot_classify(image_paths, model, proc, device)
    zs_time  = time.perf_counter() - t0
    zs_ari   = adjusted_rand_score(y_true,
                                   [cat_map[p] for p in zs_preds])
    print(f"  Zero-shot ARI = {zs_ari:.4f}  ({zs_time:.2f}s)")

    print(f"\n  {'File':<18} {'True':^10} {'ZS Pred':^10} {'OK?':^5}")
    print("  " + "-" * 48)
    zs_misplaced = []
    for fname, tl, zp in zip(filenames, true_labels, zs_preds):
        ok = "OK" if tl == zp else "X"
        print(f"  {fname:<18} {tl:^10} {zp:^10} {ok:^5}")
        if tl != zp:
            zs_misplaced.append((fname, tl, zp))
    print(f"\n  Zero-shot misplaced: {zs_misplaced or 'none'}")

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "ari":               ari,
        "zs_ari":            zs_ari,
        "filenames":         filenames,
        "true_labels":       true_labels,
        "pred_labels":       y_pred_named,
        "zs_pred_labels":    zs_preds,
        "misplaced":         misplaced,
        "zs_misplaced":      zs_misplaced,
        "car_split":         len(car_split_clusters) > 1,
        "car_split_clusters":list(car_split_clusters),
        "dog_food_confusion":dog_food_confusions,
        "embed_time_s":      embed_time,
        "clip_load_time_s":  clip_load_time,
        "zs_time_s":         zs_time,
    }
    out = RES_DIR / "dimg.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out}")

    # ── Figure 4 — PCA scatter ────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    colours = {"car": "#e41a1c", "desk": "#377eb8", "dog": "#4daf4a",
               "food": "#ff7f00", "mountain": "#984ea3"}
    markers = {"car": "o", "desk": "s", "dog": "^", "food": "D", "mountain": "P"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (pred_lbl, title) in zip(axes,
            [(y_pred_named, "K-Means predicted"), (zs_preds, "CLIP zero-shot")]):
        for cat in cats:
            idx = [i for i, l in enumerate(true_labels) if l == cat]
            sc  = ax.scatter(coords[idx, 0], coords[idx, 1],
                             c=colours[cat], marker=markers[cat],
                             label=cat, s=80, alpha=0.85, edgecolors='k', linewidths=0.4)
        # circle misplaced with red ring
        for i, (fname, tl, pl) in enumerate(
                zip(filenames, true_labels, pred_lbl)):
            if tl != pl:
                fi = filenames.index(fname)
                ax.scatter(coords[fi, 0], coords[fi, 1],
                           s=180, facecolors='none', edgecolors='red', linewidths=1.5, zorder=5)

    # Inset actual food5.jpg image thumbnail into K-Means plot (axes[0])
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image as PILImage
    if "food5.jpg" in filenames:
        f5_idx = filenames.index("food5.jpg")
        f5_path = image_paths[f5_idx]
        pil_img = PILImage.open(f5_path).convert("RGB")
        pil_img.thumbnail((120, 150))
        img_arr = np.array(pil_img)
        # Create thumbnail image box
        imagebox = OffsetImage(img_arr, zoom=1.0)
        ab = AnnotationBbox(imagebox, (coords[f5_idx, 0], coords[f5_idx, 1]),
                            xybox=(-0.05, 0.35), xycoords='data',
                            boxcoords="data",
                            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                            bboxprops=dict(edgecolor="red", linewidth=1.5))
        axes[0].add_artist(ab)
        axes[0].text(-0.05, 0.52, "misplaced food5.jpg\n(ice cream cone sculpture)",
                     color="darkred", fontsize=8, fontweight="bold", ha="center")

    for ax, (pred_lbl, title) in zip(axes,
            [(y_pred_named, "K-Means predicted"), (zs_preds, "CLIP zero-shot")]):
        ax.set_title(f"{title} (k=5)\nARI={adjusted_rand_score(y_true, [cat_map.get(p, 0) for p in pred_lbl]):.3f}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle("D-IMG CLIP Embeddings (PCA, k=5)\n(Red circle marks misplaced food5.jpg, which clustered with mountain)", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig4_path = FIG_DIR / "fig4_dimg_pca.png"
    fig.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 4 saved -> {fig4_path}")

    # ── Figure 9 — confusion heatmap (reused for D-MIX, but also useful here) ─
    all_labels = sorted(set(true_labels) | set(y_pred_named))
    cm2 = confusion_matrix(true_labels, y_pred_named, labels=all_labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_labels, yticklabels=all_labels, ax=ax)
    ax.set_xlabel('Predicted cluster (mapped)')
    ax.set_ylabel('Ground truth')
    ax.set_title('D-IMG Cluster Confusion (k=5)')
    fig.tight_layout()
    dimg_cm_path = FIG_DIR / "dimg_confusion.png"
    fig.savefig(dimg_cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  D-IMG confusion matrix saved -> {dimg_cm_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
