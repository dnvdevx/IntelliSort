"""
eval_dmix.py
============
D-MIX experiment — mixed_test_files/ (66 files):
  - File inventory, including the 2 files that cannot be processed (a spreadsheet and a scanned PDF).
  - Evaluates digital vs scanned PDFs.
  - Evaluates processable files with unified 5-class text + image + maths GT labels.
  - Produces PDF placement table.
  - Figure 9 (cluster composition bar chart with complete legend & descriptive captions).
  - Exports supplement_membership.csv with corrected match/accuracy column.
"""

import sys, os, time, json, platform, subprocess
from pathlib import Path
from collections import defaultdict

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
_ap.add_argument("--data", default=str(REPO_ROOT / "data" / "dmix"))
ARGS = _ap.parse_args()
DATA_DIR  = Path(ARGS.data)

sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(EVAL_DIR))

FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

from ground_truth import (MIX_GROUND_TRUTH, PDF_METADATA,
                          UNPROCESSED_FILES)

PDF_TYPE = {fname: ("scanned" if meta[1] else "digital")
            for fname, meta in PDF_METADATA.items()}


def get_cpu_model():
    try:
        cmd = "powershell -NoProfile -Command (Get-CimInstance Win32_Processor).Name"
        out = subprocess.check_output(cmd, shell=True).decode().strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or platform.machine()


def check_pdf_has_text(pdf_path):
    """Return True if fitz extracts any text (-> digital); False = scanned."""
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return bool(text.strip())


def main():
    print("=" * 70)
    print("D-MIX Evaluation (Unified Ground Truth, n_init=100)")
    print("=" * 70)

    cpu_name = get_cpu_model()
    print(f"  CPU: {cpu_name}")

    # ── 1. File inventory ────────────────────────────────────────────────────
    all_files = sorted(DATA_DIR.iterdir())
    all_files = [f for f in all_files if f.is_file()]
    total_count = len(all_files)
    print(f"\n  Total files found: {total_count} (66 in the paper, 64 of them processable)")

    unprocessed_inventory = [f.name for f in all_files if f.name in UNPROCESSED_FILES]
    print(f"  Unprocessed / unsupported files ({len(unprocessed_inventory)}): {unprocessed_inventory}")

    # ── 2. PDF inspection & labeling ─────────────────────────────────────────
    pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]
    print(f"\n  PDF Inspection (manually read prior to clustering):")
    print(f"  {'Filename':<45} {'Type':^9} {'Has Text':^10} {'GT Label':^10}")
    print("  " + "-" * 78)
    pdf_report = []
    for pf in pdf_files:
        has_text = check_pdf_has_text(pf)
        actual_type = "digital" if has_text else "scanned"
        gt_label, is_scanned_meta = PDF_METADATA.get(pf.name, ("unknown", False))
        print(f"  {pf.name:<45} {actual_type:^9} {str(has_text):^10} {gt_label:^10}")
        pdf_report.append({
            "name": pf.name,
            "type": actual_type,
            "has_text": has_text,
            "ground_truth": gt_label
        })

    # ── 3. Scan timing ───────────────────────────────────────────────────────
    from FileScanner import FileScanner
    from file_node import FileNode, extract_text, SUPPORTED_TEXT_EXTENSIONS

    print("\n  [Timing] Scan stage...")
    t0 = time.perf_counter()
    scanner = FileScanner()
    scanner.search_files(DATA_DIR)
    file_nodes = scanner.file_data()
    scan_time_cold = time.perf_counter() - t0
    print(f"    Scan cold: {scan_time_cold:.3f}s ({len(file_nodes)} nodes)")

    t0 = time.perf_counter()
    scanner2 = FileScanner()
    scanner2.search_files(DATA_DIR)
    file_nodes2 = scanner2.file_data()
    scan_time_warm = time.perf_counter() - t0
    print(f"    Scan warm: {scan_time_warm:.3f}s")

    # ── 4. Extract timing ────────────────────────────────────────────────────
    print("\n  [Timing] Text extraction...")
    text_nodes = [n for n in file_nodes if n.extension in SUPPORTED_TEXT_EXTENSIONS]
    texts_cold, extract_errs = [], []
    t0 = time.perf_counter()
    for node in text_nodes:
        try:
            texts_cold.append((node.filename, extract_text(node.path)))
        except Exception as e:
            extract_errs.append((node.filename, str(e)))
    extract_time_cold = time.perf_counter() - t0
    print(f"    Extract cold: {extract_time_cold:.3f}s ({len(texts_cold)} ok, {len(extract_errs)} failed)")
    for fname, err in extract_errs:
        print(f"      ! {fname}: {err}")

    t0 = time.perf_counter()
    for node in text_nodes:
        try:
            extract_text(node.path)
        except Exception:
            pass
    extract_time_warm = time.perf_counter() - t0
    print(f"    Extract warm: {extract_time_warm:.3f}s")

    # ── 5. Embed timing ──────────────────────────────────────────────────────
    from FileGroup import FileGrouper
    grouper = FileGrouper()

    print("\n  [Timing] Embed stage...")
    t0 = time.perf_counter()
    embeddings_np, ordered_nodes, unprocessed = grouper._build_embeddings(file_nodes)
    embed_time_cold = time.perf_counter() - t0
    print(f"    Embed cold: {embed_time_cold:.3f}s ({len(ordered_nodes)} embedded, {len(unprocessed)} unprocessed)")
    for node, why in unprocessed:
        print(f"      ! {node.filename}: {why}")

    t0 = time.perf_counter()
    emb2, ord2, _ = grouper._build_embeddings(file_nodes)
    embed_time_warm = time.perf_counter() - t0
    print(f"    Embed warm: {embed_time_warm:.3f}s")

    # ── 6. Cluster stage with n_init=100 ────────────────────────────────────
    n_clusters = max(2, min(len(ordered_nodes) // 5, 500))
    print(f"\n  [Clustering] KMeans (k={n_clusters}, n_init=100, seed=42)...")
    t0 = time.perf_counter()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
    labels = km.fit_predict(embeddings_np)
    cluster_time = time.perf_counter() - t0
    print(f"    Cluster time: {cluster_time:.3f}s")

    # ── 7. ARI on all processable files ─────────────────────────────────────
    filenames_ordered = [n.filename for n in ordered_nodes]
    true_all, pred_all, fname_all = [], [], []
    for fname, pred in zip(filenames_ordered, labels):
        gt = MIX_GROUND_TRUTH.get(fname)
        if gt is not None and gt not in ("_extra", "unknown"):
            true_all.append(gt)
            pred_all.append(pred)
            fname_all.append(fname)

    cats = sorted(set(true_all))
    cat_map = {c: i for i, c in enumerate(cats)}
    y_true = np.array([cat_map[l] for l in true_all])
    ari_all = adjusted_rand_score(y_true, np.array(pred_all))
    print(f"\n  ARI (all processable files, 5-class unified GT): {ari_all:.4f}")

    # ── 8. Majority-vote cluster naming ──────────────────────────────────────
    cluster_to_gt = defaultdict(list)
    for gt_item, p_item in zip(true_all, pred_all):
        cluster_to_gt[p_item].append(gt_item)
    cluster_map = {}
    for cl_id, gts in cluster_to_gt.items():
        vals, counts = np.unique(gts, return_counts=True)
        majority = vals[np.argmax(counts)]
        cluster_map[cl_id] = f"{majority}_c{cl_id}"
    pred_named = [cluster_map.get(p, f"cluster_{p}") for p in labels]

    # ── 9. PDF Placement Table ───────────────────────────────────────────────
    print(f"\n  PDF Placement Table:")
    print(f"  {'Filename':<45} {'GT Label':^10} {'Cluster ID':^12} {'Cluster Name':^15} {'Majority Matches'}")
    print("  " + "-" * 95)
    pdf_placement = []
    for r in pdf_report:
        fname = r["name"]
        gt = r["ground_truth"]
        if fname in filenames_ordered:
            idx = filenames_ordered.index(fname)
            cid = int(labels[idx])
            cname = pred_named[idx]
            majority = cname.rsplit("_c", 1)[0]
            matched = (gt == majority)
        else:
            cid = -1
            cname = "_not_processed"
            matched = False
        pdf_placement.append({
            "filename": fname,
            "ground_truth": gt,
            "cluster_id": cid,
            "cluster_label": cname,
            "majority_matched": matched
        })
        print(f"  {fname:<45} {gt:^10} {cid:^12} {cname:^15} {matched}")

    # ── 10. Supplement Membership Table ──────────────────────────────────────
    rows = []
    for fname, pred_int, pred_nm in zip(filenames_ordered, labels, pred_named):
        gt = MIX_GROUND_TRUTH.get(fname, "unknown")
        majority = pred_nm.rsplit("_c", 1)[0]
        rows.append({
            "filename":      fname,
            "ground_truth":  gt,
            "cluster_id":    int(pred_int),
            "cluster_label": pred_nm,
            "correct":       (gt == majority),
        })

    not_proc_names = {n.filename for n, _ in unprocessed}
    for fname in sorted(DATA_DIR.iterdir(), key=lambda x: x.name):
        if fname.name in not_proc_names:
            gt = MIX_GROUND_TRUTH.get(fname.name, "unknown")
            rows.append({
                "filename":      fname.name,
                "ground_truth":  gt,
                "cluster_id":    -1,
                "cluster_label": "_not_processed",
                "correct":       False,
            })

    df = pd.DataFrame(rows)
    csv_path = RES_DIR / "dmix_membership.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Supplement CSV saved -> {csv_path}")

    # ── 11. Save raw results JSON ────────────────────────────────────────────
    results = {
        "cpu":                 cpu_name,
        "total_files":         total_count,
        "unprocessed_files":   unprocessed_inventory,
        "pdf_report":          pdf_report,
        "pdf_placement":       pdf_placement,
        "ari_all":             ari_all,
        "timing": {
            "scan_cold_s":    scan_time_cold,
            "scan_warm_s":    scan_time_warm,
            "extract_cold_s": extract_time_cold,
            "extract_warm_s": extract_time_warm,
            "embed_cold_s":   embed_time_cold,
            "embed_warm_s":   embed_time_warm,
            "cluster_s":      cluster_time,
        },
        "unprocessed": [(n.filename, why) for n, why in unprocessed],
    }
    out = RES_DIR / "dmix.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {out}")

    # ── 12. Figure 9: Cluster Summary Bar Chart ──────────────────────────────
    cluster_sizes = defaultdict(int)
    cluster_by_gt  = defaultdict(lambda: defaultdict(int))
    for row in rows:
        lbl = row["cluster_label"]
        gt  = row["ground_truth"]
        cluster_sizes[lbl] += 1
        cluster_by_gt[lbl][gt] += 1

    cluster_labels_sorted = sorted(cluster_sizes.keys())
    # Ensure all ground truth categories appear in legend
    gt_cats_sorted = sorted({r["ground_truth"] for r in rows})
    palette = plt.get_cmap("tab20", len(gt_cats_sorted))
    col_map = {c: palette(i) for i, c in enumerate(gt_cats_sorted)}

    fig, ax = plt.subplots(figsize=(max(11, len(cluster_labels_sorted) * 0.95), 6.5))
    bottoms = np.zeros(len(cluster_labels_sorted))
    for gt_cat in gt_cats_sorted:
        heights = [cluster_by_gt[cl].get(gt_cat, 0) for cl in cluster_labels_sorted]
        ax.bar(cluster_labels_sorted, heights, bottom=bottoms,
               color=col_map[gt_cat], label=gt_cat, edgecolor="black", linewidth=0.5)
        bottoms += np.array(heights, dtype=float)

    ax.set_xlabel("Partition / Cluster Name (format: majorityClass_cClusterID)", fontsize=10)
    ax.set_ylabel("Number of Files", fontsize=10)
    ax.set_title(f"D-MIX Cluster Composition (ARI = {ari_all:.4f}, n_init=100)\n"
                 f"Cluster labels indicate majority class and cluster ID (e.g., car_c11 is Cluster 11 dominated by cars)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig9_path = FIG_DIR / "fig9_dmix_clusters.png"
    fig.savefig(fig9_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 9 saved -> {fig9_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
