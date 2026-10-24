import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

INK = "#1f2933"
MUTED = "#52606d"
RED = "#b42318"
STAGES = [
    ("1  Input", 1, 17, "#e8f0fb", "#3b6fb6"),
    ("2  Content extraction", 19, 42, "#e9f5ec", "#2f855a"),
    ("3  Embedding", 44, 64, "#fdf3e4", "#b7791f"),
    ("4  Clustering", 66, 84, "#efeafb", "#6b46c1"),
    ("5  Output", 86, 99, "#e6f6f7", "#2c7a7b"),
]
TOP, BOTTOM = 47, 12

fig, ax = plt.subplots(figsize=(16, 7.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.axis("off")

for title, x0, x1, fill, edge in STAGES:
    ax.add_patch(FancyBboxPatch((x0, BOTTOM), x1 - x0, TOP - BOTTOM, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fill, ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((x0, TOP - 3.2), x1 - x0, 3.2, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=edge, ec="none", zorder=1))
    ax.text((x0 + x1) / 2, TOP - 1.6, title, ha="center", va="center", fontsize=12.5, weight="bold",
            color="white", zorder=2)


def box(x0, y0, w, h, head, body="", edge=INK, fill="white", dashed=False):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.6", fc=fill, ec=edge,
                                lw=1.3, ls="--" if dashed else "-", zorder=3))
    cx = x0 + w / 2
    if body:
        ax.text(cx, y0 + h - 1.1, head, ha="center", va="top", fontsize=10.5, weight="bold", color=edge, zorder=4)
        ax.text(cx, y0 + h - 3.4, body, ha="center", va="top", fontsize=9, color=INK, linespacing=1.45, zorder=4)
    else:
        ax.text(cx, y0 + h / 2, head, ha="center", va="center", fontsize=10.5, weight="bold", color=edge, zorder=4)
    return dict(l=x0, r=x0 + w, b=y0, t=y0 + h, cx=cx, cy=y0 + h / 2)


def arrow(p, q, color=INK, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color,
                                 ls=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0, zorder=5))


folder = box(2.5, 37, 13, 5, "Selected folder", "any depth")
scan = box(2.5, 24.5, 13, 9.5, "FileScanner", "recursive walk\nfiles sorted by path\n(same order every run)")
meta = box(2.5, 13.5, 13, 9, "File metadata", "name, extension,\nsize, created and\nmodified time")

text = box(20.5, 31, 20, 11, "Text files", ".txt  read as UTF-8\n.docx  python-docx\n.pdf  PyMuPDF text layer")
ocr = box(20.5, 23.2, 20, 5.8, "Pages without a text layer", "Tesseract OCR, if installed")
img = box(20.5, 14, 20, 7.2, "Image files", ".png .jpg .jpeg .gif .bmp .webp\nopened with PIL, converted to RGB")

temb = box(45.5, 31, 17, 11, "Text embedding", "all-MiniLM-L6-v2\n250-token windows (max 16),\nlength-weighted mean, L2\n384-d")
pad = box(45.5, 24.5, 17, 4.5, "Zero-pad 384 → 512", "")
iemb = box(45.5, 13.5, 17, 9.2, "Image embedding", "CLIP ViT-B/32 image encoder\npooled output → projection, L2\n512-d")

mat = box(67.5, 36.5, 15, 5.5, "Embedding matrix", "N files × 512")
km = box(67.5, 21.5, 15, 13, "K-Means", "Euclidean on L2-normalised\nvectors\nk = max(2, min(⌊N/5⌋, 500))\nk-means++, n_init = 100\nseed 42")
lab = box(67.5, 14, 15, 5.5, "One cluster per file", "")

fold = box(87.2, 13.5, 10.6, 11, "Folders", "one per cluster,\nnamed after the\nfile nearest the\ncentroid")
copy = box(87.2, 28, 10.6, 9, "Files copied\nor moved into\ntheir folders", "")
outnp = box(86.3, 3.2, 12.4, 5.6, "_not_processed", "", edge=RED, fill="#fdecea")

excl = box(20.5, 3.2, 61, 5.6, "Unsupported, empty or unreadable files (and scanned PDFs when OCR is unavailable)",
           "", edge=RED, fill="#fdecea", dashed=True)
ax.text(excl["cx"], 2.0, "not embedded and not clustered; listed with the reason", ha="center", va="center",
        fontsize=9, color=RED, style="italic")

arrow((folder["cx"], folder["b"]), (scan["cx"], scan["t"]))
arrow((scan["cx"], scan["b"]), (meta["cx"], meta["t"]))
arrow((scan["r"], scan["cy"] + 2), (text["l"], text["cy"]))
arrow((scan["r"], scan["cy"] - 2), (img["l"], img["cy"]))
arrow((text["cx"], text["b"]), (ocr["cx"], ocr["t"]), dashed=True)
arrow((text["r"], text["cy"]), (temb["l"], temb["cy"]))
arrow((ocr["r"], ocr["cy"]), (temb["l"], temb["b"] + 1.5))
arrow((img["r"], img["cy"]), (iemb["l"], iemb["cy"]))
arrow((temb["cx"], temb["b"]), (pad["cx"], pad["t"]))
arrow((pad["r"], pad["cy"]), (mat["l"], mat["cy"] - 1.5))
arrow((iemb["r"], iemb["cy"]), (mat["l"], mat["b"] + 0.6))
arrow((mat["cx"], mat["b"]), (km["cx"], km["t"]))
arrow((km["cx"], km["b"]), (lab["cx"], lab["t"]))
arrow((lab["r"], lab["cy"]), (fold["l"], lab["cy"]))
arrow((fold["cx"], fold["t"]), (copy["cx"], copy["b"]))
arrow((meta["cx"], meta["b"]), (excl["l"], excl["cy"]), color=RED, dashed=True, rad=0.3)
arrow((excl["r"], excl["cy"]), (outnp["l"], outnp["cy"]), color=RED, dashed=True)

ax.text(1, 49.8, "Solid arrows: data flow.   Dashed grey arrow: only for PDF pages with no text layer.   "
        "Red dashed arrows: files that cannot be processed.   All processing runs locally.",
        fontsize=9, color=MUTED, va="center")

fig.savefig("figures/fig01_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("figures/fig01_architecture.pdf", bbox_inches="tight", facecolor="white")
