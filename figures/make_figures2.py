import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 3: D-TEXT confusion matrix of the deterministic app result (ARI 0.829)
import seaborn as sns
L = ["dev", "finance", "ml", "personal", "recipe"]
cm = np.array([[4, 1, 0, 1, 0], [0, 6, 0, 0, 0], [0, 0, 6, 0, 0], [0, 0, 0, 5, 0], [0, 0, 0, 0, 6]])
fig, ax = plt.subplots(figsize=(6.4, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=L, yticklabels=L, cbar=True, ax=ax, vmin=0, vmax=6)
ax.set_xlabel("Predicted cluster (mapped to majority class)"); ax.set_ylabel("Ground truth")
ax.set_title("D-TEXT confusion matrix, deterministic app result (k=5, n_init=100, ARI 0.829)", fontsize=9.5)
fig.tight_layout(); fig.savefig("figures/fig03_dtext_confusion.png", dpi=200); plt.close(fig)

# Figure 4: launch stability and seed sweep
d = json.load(open("results/dtext/stability_launches.json"))
before = np.array(d["before_per_launch_aris"]); sweep = np.array(d["seed_sweep_aris"])
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
a = axs[0]; vp = a.violinplot(before, positions=[0], showmedians=True, widths=0.7)
for b in vp["bodies"]: b.set_facecolor("#e8a06a"); b.set_alpha(0.9)
a.scatter(np.random.RandomState(0).normal(0, 0.03, len(before)), before, s=10, color="#6a5acd", alpha=0.6, zorder=3, label="original launches (N=100)")
a.axhline(d["after_deterministic_ari"], color="#1b9e77", lw=2.5, label=f"fixed app, all 100 launches: ARI {d['after_deterministic_ari']:.3f} (1 grouping)")
a.set_xticks([0]); a.set_xticklabels([f"original pipeline\n(100 launches, {d['before_distinct_partitions']} distinct groupings)"])
a.set_ylim(0, 1.05); a.set_ylabel("Adjusted Rand Index"); a.legend(fontsize=8, loc="lower right", frameon=False)
a.set_title(f"Launch-to-launch stability\noriginal: {d['before_mean_ari']:.3f} ± {d['before_sd_ari']:.3f} (range {d['before_min_ari']:.2f}–{d['before_max_ari']:.2f})", fontsize=10)
b = axs[1]; b.plot(range(len(sweep)), sweep, marker="o", ms=3.5, lw=0.8, color="#3aa17e")
b.axhline(sweep.mean(), color="#d95f02", ls="--", lw=1.4, label=f"mean {sweep.mean():.3f}, SD {sweep.std(ddof=1):.3f}")
b.set_ylim(0, 1.05); b.set_xlabel("K-Means seed (random_state 0 to 99; n_init=100, file order fixed)"); b.set_ylabel("ARI")
b.set_title("Seed sweep on identical embeddings", fontsize=10); b.legend(fontsize=8, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig("figures/fig04_stability.png", dpi=200); plt.close(fig)
print("ok", before.mean(), before.std(ddof=1), sweep.mean(), sweep.std(ddof=1))
