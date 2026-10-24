"""
Per-stage timing of the pipeline on D-MIX, repeated five times (Table 9).

    python eval/measure_dmix_timing.py --data data/dmix --out results/dmix
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--core", default=str(Path(__file__).resolve().parents[1] / "Core"))
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "dmix"))
ap.add_argument("--out", default="results/dmix")
ap.add_argument("--runs", type=int, default=5)
args = ap.parse_args()

sys.path.insert(0, args.core)

from FileScanner import FileScanner
from file_node import SUPPORTED_TEXT_EXTENSIONS, extract_text
import FileGroup as FG
from sklearn.cluster import KMeans

DATA_DIR = Path(args.data)
results = []

print(f"Running D-MIX timing {args.runs} times on {DATA_DIR}...")

for r in range(args.runs):
    t0 = time.perf_counter()
    sc = FileScanner()
    sc.search_files(DATA_DIR)
    nodes = sc.file_data()
    t_scan_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    sc2 = FileScanner()
    sc2.search_files(DATA_DIR)
    sc2.file_data()
    t_scan_warm = time.perf_counter() - t0

    text_nodes = [n for n in nodes if n.extension in SUPPORTED_TEXT_EXTENSIONS]
    t0 = time.perf_counter()
    for n in text_nodes:
        try:
            extract_text(n.path)
        except Exception:
            pass
    t_ext_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    for n in text_nodes:
        try:
            extract_text(n.path)
        except Exception:
            pass
    t_ext_warm = time.perf_counter() - t0

    grouper = FG.FileGrouper()
    t0 = time.perf_counter()
    emb_cold, ord_nodes, unproc = grouper._build_embeddings(nodes)
    t_emb_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    grouper._build_embeddings(nodes)
    t_emb_warm = time.perf_counter() - t0

    k = max(2, min(len(ord_nodes) // 5, 500))
    t0 = time.perf_counter()
    KMeans(n_clusters=k, random_state=42, n_init=FG.KMEANS_N_INIT).fit_predict(emb_cold)
    t_clus = time.perf_counter() - t0

    results.append({
        "run": r + 1,
        "scan_cold_s": t_scan_cold,
        "scan_warm_s": t_scan_warm,
        "extract_cold_s": t_ext_cold,
        "extract_warm_s": t_ext_warm,
        "embed_cold_s": t_emb_cold,
        "embed_warm_s": t_emb_warm,
        "cluster_s": t_clus,
    })
    print(f"Run {r + 1}: cold_embed={t_emb_cold:.2f}s, warm_embed={t_emb_warm:.2f}s, cluster={t_clus:.3f}s")

stages = ["scan_cold_s", "scan_warm_s", "extract_cold_s", "extract_warm_s", "embed_cold_s", "embed_warm_s", "cluster_s"]
print("\n" + "=" * 65)
print(f"{'Stage':20s} {'Median (s)':>12s} {'Min (s)':>12s} {'Max (s)':>12s}")
print("=" * 65)
summary = {}
for s in stages:
    vals = [res[s] for res in results]
    summary[s] = {"median": float(np.median(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "all": vals}
    print(f"{s:20s} {summary[s]['median']:12.3f} {summary[s]['min']:12.3f} {summary[s]['max']:12.3f}")

os.makedirs(args.out, exist_ok=True)
out_file = os.path.join(args.out, "dmix_timing_repeats.json")
with open(out_file, "w") as f:
    json.dump({"runs": results, "summary": summary}, f, indent=2)
print(f"\nSaved {out_file}")
