"""
Ground-truth labels for D-TEXT, D-IMG and D-MIX, fixed before any clustering was run.
"""

# ── D-TEXT: 29 text files → 5 semantic categories ───────────────────────────
TEXT_GROUND_TRUTH = {
    # Finance / Budget (6 files)
    "budget_february.txt":      "finance",
    "budget_january.txt":       "finance",
    "expense_report_q1.txt":    "finance",
    "investment_summary.txt":   "finance",
    "loan_repayment_plan.txt":  "finance",
    "tax_filing_notes.txt":     "finance",

    # Machine Learning (6 files)
    "cnn_architecture_notes.txt":       "ml",
    "gradient_descent_explained.txt":   "ml",
    "ml_supervised_learning.txt":       "ml",
    "ml_unsupervised_learning.txt":     "ml",
    "neural_networks_intro.txt":        "ml",
    "transformer_attention.txt":        "ml",

    # Software Development (6 files)
    "debugging_checklist.txt":          "dev",
    "docker_basics.txt":                "dev",
    "git_commands.txt":                 "dev",
    "python_tips.txt":                  "dev",
    "rest_api_design.txt":              "dev",
    "sql_join_notes.txt":               "dev",

    # Recipes / Food (6 files)
    "recipe_banana_bread.txt":  "recipe",
    "recipe_curry.txt":         "recipe",
    "recipe_fried_rice.txt":    "recipe",
    "recipe_lentil_soup.txt":   "recipe",
    "recipe_omelette.txt":      "recipe",
    "recipe_pasta.txt":         "recipe",

    # Personal / Lifestyle (5 files)
    "birthday_party_plan.txt":      "personal",
    "book_notes_atomic_habits.txt": "personal",
    "gym_routine.txt":              "personal",
    "travel_journal_goa.txt":       "personal",
    "weekend_plans.txt":            "personal",
}

# ── D-IMG: 30 image files → 5 visual categories ─────────────────────────────
IMAGE_GROUND_TRUTH = {
    **{f"car{i}.jpg":      "car"      for i in range(1, 7)},
    **{f"desk{i}.jpg":     "desk"     for i in range(1, 7)},
    **{f"dog{i}.jpg":      "dog"      for i in range(1, 7)},
    **{f"food{i}.jpg":     "food"     for i in range(1, 7)},
    **{f"mountain{i}.jpg": "mountain" for i in range(1, 7)},
}

# ── D-MIX: mixed_test_files — 66 files ───────────────────────────────────────
# PDF metadata: (ground_truth_topic, is_scanned)
# Labels were assigned after reading each document and before clustering.
PDF_METADATA = {
    "16-Sets.pdf":                               ("dev",   False),   # digital Python lecture on Sets
    "BACSE105_DATA-STRUCTURES-AND-ALGORITHMS_ETH_1.0_80_BACSE105_DATA-STRUCTURES-AND-ALGORITHMS_ELA_1.0_84_BACSE105.pdf":
                                                 ("dev",   False),   # digital Data Structures & Algorithms syllabus
    "Exp6_Triple Integrals matlab -new (1) (3).pdf": ("maths", False), # digital calculus/matlab lab sheet
    "Module 2-Scope.pdf":                        ("maths", False),   # digital maths module doc
    "Module_2_Tutorial_Sheet_02.pdf":            ("maths", False),   # digital maths tutorial sheet
    "random.pdf":                                ("unknown", True),  # scanned page, no text layer
}

# Files that fail processing / extraction (2 files):
UNPROCESSED_FILES = {
    "spreadsheet.xlsx",   # unsupported file type
    "random.pdf",         # scanned PDF, not processed when OCR is unavailable
}

MIX_GROUND_TRUTH = {
    # ── 29 text files (5 classes: ml, dev, recipe, finance, personal) ──
    **{k: v for k, v in TEXT_GROUND_TRUTH.items()},
    # ── 30 images (5 visual classes: car, desk, dog, food, mountain) ──
    **{k: v for k, v in IMAGE_GROUND_TRUTH.items()},
    # ── 6 PDFs (dev, maths, unknown) ──
    **{fname: meta[0] for fname, meta in PDF_METADATA.items()},
    # ── 1 xlsx (unsupported) ──
    "spreadsheet.xlsx": "other (xlsx)",
}
