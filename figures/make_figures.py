import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
C_IS, C_BASE, C_DET = "#1b9e77", "#8da0cb", "#bdbdbd"

def hbar(ax, labels, means, sds, colors, xlabel):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, xerr=[s if s else 0 for s in sds], color=colors, edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="black", capsize=3, lw=1))
    for yi, m, s in zip(y, means, sds):
        ax.text(max(m, 0) + (s or 0) + 0.015, yi, f"{m:.3f}" + (f" ± {s:.3f}" if s else ""), va="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlabel(xlabel); ax.set_xlim(-0.05, 1.12); ax.axvline(0, color="k", lw=0.6)

# Figure 11: D-TEXT baselines (values from results/dtext/baselines_table.csv)
rows = [("Extension grouping", 0.000, 0, C_DET), ("Filename similarity, thr 0.80", 0.023, 0, C_DET),
        ("Filename similarity, thr 0.50", 0.190, 0, C_DET), ("Filename similarity, thr 0.30", 0.014, 0, C_DET),
        ("LDA (default priors)", 0.038, 0.055, C_BASE), ("LDA (sparse priors)", 0.192, 0.121, C_BASE),
        ("TF-IDF + K-Means", 0.560, 0.088, C_BASE), ("MiniLM + DBSCAN, eps 0.9", -0.001, 0, C_DET),
        ("MiniLM + HDBSCAN", 0.655, 0, C_DET), ("MiniLM + agglomerative (average, cosine)", 0.693, 0, C_DET),
        ("MiniLM + DBSCAN, eps 0.8", 0.827, 0, C_DET), ("MiniLM + K-Means, k by silhouette", 0.816, 0.037, C_BASE),
        ("MiniLM + K-Means, k=5 (IntelliSort)", 0.828, 0.023, C_IS), ("MiniLM + agglomerative (ward)", 0.845, 0, C_DET)]
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9, 5.4))
hbar(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], "Adjusted Rand Index (D-TEXT, 29 files, 5 classes)")
fig.tight_layout(); fig.savefig("figures/fig11_baselines_dtext.png", dpi=200); plt.close(fig)

# Figure 5: n_init and row order (results/dtext/ablation.csv)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
ni = [1, 10, 50, 100]; m = [0.6022, 0.7564, 0.8158, 0.8280]; s = [0.1166, 0.0755, 0.0363, 0.0231]; d = [95, 55, 17, 11]
axs[0].bar([str(x) for x in ni], m, yerr=s, color=C_BASE, edgecolor="black", lw=0.5, capsize=4)
for i, (mm, ss, dd) in enumerate(zip(m, s, d)): axs[0].text(i, mm + ss + 0.02, f"{dd} distinct", ha="center", fontsize=8.5)
axs[0].set_xlabel("n_init (file order fixed, seeds 0 to 99)"); axs[0].set_ylabel("ARI"); axs[0].set_ylim(0, 1.05)
axs[0].set_title("(a) Effect of K-Means restarts", fontsize=10, loc="left")
lab = ["shuffled\nn_init=1", "shuffled\nn_init=10", "shuffled\nn_init=100", "sorted\nn_init=100"]
m2 = [0.5882, 0.7622, 0.8312, 0.8293]; s2 = [0.1163, 0.0716, 0.0287, 0]; d2 = [97, 56, 12, 1]
axs[1].bar(lab, m2, yerr=s2, color=[C_BASE, C_BASE, C_BASE, C_IS], edgecolor="black", lw=0.5, capsize=4)
for i, (mm, ss, dd) in enumerate(zip(m2, s2, d2)): axs[1].text(i, mm + ss + 0.02, f"{dd} distinct", ha="center", fontsize=8.5)
axs[1].set_xlabel("row order of the embedding matrix (seed fixed at 42, 100 shuffles)")
axs[1].set_title("(b) Effect of file order", fontsize=10, loc="left")
fig.suptitle("Why the output varied: row order and a single initialisation (D-TEXT, ARI mean ± SD)", fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("figures/fig05_ninit_roworder.png", dpi=200); plt.close(fig)

# Figure 10: scaling (results/bench/scaling.csv)
n = [100, 500, 1000, 2000, 5000]
series = {"K-Means, k=20": [0.71, 1.09, 1.51, 2.30, 18.22], "K-Means, k=n/5 (app rule, capped at 500)": [0.68, 2.12, 4.80, 31.62, 107.07],
          "Ward, k=20": [0.0, 0.02, 0.09, 0.30, 2.89], "HDBSCAN": [0.0, 0.08, 0.28, 1.10, 10.44]}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for (k, v), mk in zip(series.items(), "osd^"):
    vv = [x if x > 0 else np.nan for x in v]; ax.plot(n, vv, marker=mk, label=k)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("number of documents (20 Newsgroups posts)"); ax.set_ylabel("clustering time (s), median of 3")
ax.set_xticks(n); ax.set_xticklabels([str(x) for x in n]); ax.legend(fontsize=8.5, frameon=False)
ax.set_title("Clustering time vs collection size (K-Means n_init=100; CPU)", fontsize=10, loc="left")
ax.text(0.01, 0.02, "Ward and HDBSCAN values below 10 ms are not plotted.", transform=ax.transAxes, fontsize=7.5, color="gray")
fig.tight_layout(); fig.savefig("figures/fig10_scaling.png", dpi=200); plt.close(fig)

# Figure 8: cross-modal (results/crossmodal/)
reps = ["V0 MiniLM zero-padded\n+ CLIP image (current)", "V1 CLIP text\n+ CLIP image", "V2 V1 with each\nmodality centred"]
fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2))
x = np.arange(3); w = 0.38
axs[0].bar(x - w/2, [0.267, 1.0, 1.0], w, label="text → image", color=C_BASE, edgecolor="black", lw=0.5)
axs[0].bar(x + w/2, [0.233, 0.967, 1.0], w, label="image → text", color=C_IS, edgecolor="black", lw=0.5)
axs[0].axhline(0.2, color="r", ls="--", lw=0.8); axs[0].text(2.45, 0.215, "chance", color="r", fontsize=8, ha="right")
axs[0].set_title("(a) Retrieval, top-1", fontsize=10, loc="left"); axs[0].set_ylim(0, 1.1); axs[0].legend(fontsize=8, frameon=False, loc="center left")
axs[1].bar(x - w/2, [0.0, 0.0, 1.0], w, label="k = 5", color=C_BASE, edgecolor="black", lw=0.5)
axs[1].bar(x + w/2, [0.0, 0.0, 0.329], w, label="k = 12 (app rule)", color=C_IS, edgecolor="black", lw=0.5)
axs[1].set_title("(b) Clusters containing both texts and images", fontsize=10, loc="left"); axs[1].set_ylim(0, 1.1); axs[1].legend(fontsize=8, frameon=False)
axs[2].bar(x - w/2, [0.0, 0.0, 1.0], w, label="k = 5", color=C_BASE, edgecolor="black", lw=0.5)
axs[2].bar(x + w/2, [0.0, 0.0, 0.149], w, label="k = 12 (app rule)", color=C_IS, edgecolor="black", lw=0.5)
axs[2].set_title("(c) Same-class text/image pairs in one cluster", fontsize=10, loc="left"); axs[2].set_ylim(0, 1.1); axs[2].legend(fontsize=8, frameon=False)
for a in axs: a.set_xticks(x); a.set_xticklabels(reps, fontsize=8)
fig.suptitle("Cross-modal grouping on the paired set (30 texts + 30 photos; K-Means n_init=100, mean over 100 seeds)", fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig("figures/fig08_crossmodal.png", dpi=200); plt.close(fig)

# Figure 12: 20 Newsgroups (results/bench/benchmark_quality.csv)
rows = [("LDA (20 topics)", 0.094, 0.022, C_BASE), ("TF-IDF + K-Means (k=20)", 0.156, 0.025, C_BASE),
        ("MiniLM + HDBSCAN", 0.040, 0, C_DET), ("MiniLM + K-Means, k=n/5=400 (app rule)", 0.063, 0.002, C_BASE),
        ("MiniLM + K-Means, k by silhouette (≈12)", 0.380, 0.041, C_BASE), ("MiniLM + ward (k=20)", 0.350, 0, C_DET),
        ("MiniLM + K-Means (k=20, true k)", 0.428, 0.014, C_IS)]
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9, 3.9))
hbar(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], "Adjusted Rand Index (2,000 posts, 20 classes)")
ax.set_xlim(-0.02, 0.62)
fig.tight_layout(); fig.savefig("figures/fig12_20ng.png", dpi=200); plt.close(fig)
print("ok")
