# IntelliSort

Code, data and results for the paper

> D. N. Varkey and J. P. N., "IntelliSort: Design and Empirical Evaluation of a Multi-Modal Semantic File Organization System Using Sentence Transformers and CLIP," *Frontiers in Artificial Intelligence* (under review).

IntelliSort is a local desktop tool that groups files by their content. Text files (.txt, .docx, .pdf) are embedded with all-MiniLM-L6-v2, images (.png, .jpg, .jpeg, .gif, .bmp, .webp) with the CLIP ViT-B/32 image encoder, and the combined vectors are clustered with K-Means. Everything runs on the user's computer.

## Versions

| Tag | What it is |
|---|---|
| `v0-original` | The application code used for the first submission of the paper. |
| `v1-fixed` | The revised code used for every result in the revised paper: files sorted by path before embedding, `KMEANS_N_INIT = 100`, chunked text embedding, CLIP pooled output, unprocessable files excluded, and the two experimental cross-modal modes (off by default). |

## Repository layout

```
Core/        application code (GUI: tk_ui_app.py, command line: main.py)
eval/        one script per experiment
data/        D-TEXT files, paired notes, D-MIX labels, list of Pexels photos
results/     the CSV/JSON outputs behind the tables in the paper
figures/     the figures in the paper and the scripts that draw them
```

## Installation

Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Linux / macOS)
pip install -r requirements-lock.txt     # exact versions used for the paper
```

`requirements.txt` lists the same packages without versions.

The models (all-MiniLM-L6-v2 and openai/clip-vit-base-patch32) are downloaded from Hugging Face the first time they are used. Tesseract OCR is optional; it is only used for PDF pages without a text layer. Set `TESSERACT_CMD` if it is not on your PATH.

## Running the application

```
cd Core
python tk_ui_app.py      # graphical interface
python main.py           # command line
```

## Data

| Folder | Contents |
|---|---|
| `data/dtext/` | D-TEXT: the 29 text files (5 classes), written for this study. |
| `data/paired_notes/` | The 30 short notes of the paired set (6 per D-IMG class). They were drafted with AI assistance and checked by the authors. |
| `data/pexels_sources.csv` | The 30 D-IMG photographs with their Pexels links. The photos are not redistributed here; download them into `data/dimg/` under the names in the file (e.g. `dog1.jpg`). |
| `data/ground_truth/dmix_labels.csv` | Labels for all 66 D-MIX files. |
| `eval/ground_truth.py` | The same labels as Python dictionaries. |

D-MIX consists of the 29 D-TEXT files, the 30 D-IMG photos, six PDF documents and one spreadsheet. The PDFs are university course material and the spreadsheet is a personal file, so they are not included; the D-MIX numbers can therefore not be reproduced exactly without them. The spreadsheet is listed under the neutral name `spreadsheet.xlsx`. To run the D-MIX scripts on the public part, put the D-TEXT files and the photos together in `data/dmix/`.

20 Newsgroups is downloaded automatically by scikit-learn.

## Reproducing the results

Run every command from the repository root. Each script writes its CSV/JSON output to the folder given by `--out`; the outputs used in the paper are already in `results/`.

| Paper | Command | Output in `results/` |
|---|---|---|
| Table 3, Table 4, Table 10 (D-TEXT), Table 11, Figures 5 and 11 | `python eval/baselines_and_ablation.py --data data/dtext --out results/dtext` | `dtext/baselines_table.csv`, `dtext/ablation.csv`, `dtext/k_sensitivity.csv` |
| Table 5, Figure 4 (100 separate launches) | `git worktree add ../intellisort-v0 v0-original` then `python eval/stability_launches.py --data data/dtext --before ../intellisort-v0/Core --after Core --n 100` | `dtext/stability_launches.json` |
| Table 6 (pooling, k = 5 / 6) | `python eval/dimg_pooling_ablation.py --data data/dimg --out results/dimg` | `dimg/dimg_pooling_ablation.csv` |
| Table 6, Table 10 (D-IMG) | `python eval/mixed_sets_extra.py --data data/dimg --name D-IMG --out results/dimg` | `dimg/D-IMG_extra.csv` |
| Figures 6 and 7 | `python eval/eval_dimg.py --data data/dimg` | `dimg/dimg.json` |
| Table 7, Figure 8 (paired set) | `python eval/cross_modal_experiment.py --text_dir data/paired_notes --img_dir data/dimg --out results/crossmodal` | `crossmodal/` |
| Table 7 (centring mode on D-MIX) | `python eval/dmix_crossmodal.py --data data/dmix --truth_csv data/ground_truth/dmix_labels.csv --out results/dmix_crossmodal` | `dmix_crossmodal/dmix_crossmodal.csv` |
| Table 7 (cluster-then-link mode) | `python eval/link_experiment.py --text_dir data/paired_notes --img_dir data/dimg --mix_dir data/dmix --truth_csv data/ground_truth/dmix_labels.csv --out results/link` | `link/link_experiment.csv` |
| Table 8, Table 10 (D-MIX) | `python eval/mixed_sets_extra.py --data data/dmix --truth_csv data/ground_truth/dmix_labels.csv --name D-MIX --out results/dmix` | `dmix/D-MIX_extra.csv` |
| Table 8, Figure 9, Supplementary Table S1 | `python eval/eval_dmix.py --data data/dmix` | `dmix/dmix.json`, `dmix/dmix_membership_app.csv` |
| Table 9 (D-MIX timing, 5 runs) | `python eval/measure_dmix_timing.py --data data/dmix --out results/dmix` | `dmix/dmix_timing_repeats.json` |
| Table 9, Table 12, Figures 10 and 12 (20 Newsgroups, scaling) | `python eval/benchmark_and_scaling.py --out results/bench` | `bench/` |
| Figures 1, 3, 4, 5, 8, 10, 11, 12 | `python figures/make_architecture.py`, `python figures/make_figures.py`, `python figures/make_figures2.py` | `figures/` |

All stochastic results use seeds 0 to 99 and `n_init = 100` unless stated otherwise; the application itself uses seed 42. Timings were measured on an Intel Core Ultra 9 275HX with 32 GB RAM under Windows 11, CPU only.

## Citation

See `CITATION.cff`.

## License

See `LICENSE`.
