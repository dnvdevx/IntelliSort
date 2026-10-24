"""
Launch-to-launch stability of the full application pipeline (Table 5, Figure 4).
Each launch is a separate Python process using the application's own settings.
--before is the Core folder of the original code (tag v0-original), --after the revised one.

    git worktree add ../intellisort-v0 v0-original
    python eval/stability_launches.py --data data/dtext --before ../intellisort-v0/Core --after Core --n 100
"""
import argparse, json, os, re, subprocess, sys
from collections import Counter

# D-TEXT 5-class ground truth (ML and software development are separate classes); images are handled in truth_of()
GT = {}
for cls, stems in {
    "ml": ["cnn_architecture_notes", "gradient_descent_explained", "ml_supervised_learning",
           "ml_unsupervised_learning", "neural_networks_intro", "transformer_attention"],
    "dev": ["python_tips", "debugging_checklist", "docker_basics", "git_commands",
            "rest_api_design", "sql_join_notes"],
    "recipe": ["recipe_banana_bread", "recipe_curry", "recipe_fried_rice", "recipe_lentil_soup",
               "recipe_omelette", "recipe_pasta"],
    "finance": ["budget_february", "budget_january", "expense_report_q1", "loan_repayment_plan",
                "tax_filing_notes", "investment_summary"],
    "personal": ["birthday_party_plan", "gym_routine", "travel_journal_goa", "weekend_plans",
                 "book_notes_atomic_habits"],
}.items():
    for s in stems:
        GT[s + ".txt"] = cls


def truth_of(fname):
    """D-TEXT: from the table above. D-IMG: class = leading letters of the file name (dog1.jpg -> dog)."""
    if fname in GT:
        return GT[fname]
    m = re.match(r"([A-Za-z]+)\d*\.(jpg|jpeg|png)$", fname, re.I)
    return m.group(1).lower() if m else None


def child(core, data):
    sys.path.insert(0, core)
    from FileScanner import FileScanner
    from FileGroup import FileGrouper
    sc = FileScanner()
    sc.search_files(data)
    nodes = sc.file_data()
    g = FileGrouper()
    g.group_by_combined(nodes)
    mapping = {n.filename: name for name, s in g.group["combined"].items() for n in s}
    print("RESULT" + json.dumps(mapping))


def canonical(mapping):
    """Partition independent of folder/cluster names."""
    groups = {}
    for f, c in mapping.items():
        groups.setdefault(c, []).append(f)
    return tuple(sorted(tuple(sorted(v)) for v in groups.values()))


def run(core, data, n, label):
    from sklearn.metrics import adjusted_rand_score as ari
    import numpy as np
    from scipy import stats
    partitions, scores = [], []
    for i in range(n):
        out = subprocess.check_output(
            [sys.executable, os.path.abspath(__file__), "--child", "--core", core, "--data", data],
            stderr=subprocess.DEVNULL).decode(errors="replace")
        line = [l for l in out.splitlines() if l.startswith("RESULT")][-1]
        mapping = json.loads(line[len("RESULT"):])
        names = sorted(mapping)
        missing = [x for x in names if truth_of(x) is None]
        if missing:
            sys.exit(f"No ground truth for: {missing}")
        scores.append(ari([truth_of(x) for x in names], [mapping[x] for x in names]))
        partitions.append(canonical(mapping))
        print(f"  [{label}] launch {i + 1}/{n}  ARI={scores[-1]:.4f}", flush=True)
    s = np.array(scores)
    ci = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s)) if len(s) > 1 and s.std() > 0 else (s.mean(), s.mean())
    ci = (max(ci[0], -1.0), min(ci[1], 1.0))
    counts = Counter(partitions)
    print(f"\n=== {label}: {n} separate launches, seed fixed by the app (42) ===")
    print(f"distinct partitions : {len(counts)}   (sizes of each: {sorted(counts.values(), reverse=True)})")
    print(f"ARI (5-class)       : mean={s.mean():.4f}  SD={s.std(ddof=1) if len(s) > 1 else 0:.5f}  "
          f"95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]  min={s.min():.4f}  max={s.max():.4f}\n")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--core"); ap.add_argument("--data")
    ap.add_argument("--before"); ap.add_argument("--after"); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    if a.child:
        child(a.core, a.data)
    else:
        if not (a.data and a.before and a.after):
            ap.error("--data, --before and --after are required")
        run(a.before, a.data, a.n, "BEFORE (original code)")
        run(a.after, a.data, a.n, "AFTER (fixed code)")
