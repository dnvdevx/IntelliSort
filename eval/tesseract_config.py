"""
tesseract_config.py
===================
Finds and configures the Tesseract binary for pytesseract.
Import this BEFORE importing pytesseract anywhere in the eval suite.

Priority order:
  1. TESSERACT_CMD environment variable
  2. Common Windows install paths
  3. PATH lookup
  4. Warn and disable OCR if not found
"""
import os
import shutil
from pathlib import Path

TESSERACT_FOUND = False
TESSERACT_PATH  = None

CANDIDATE_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\tools\Tesseract-OCR\tesseract.exe",
]

def _find_tesseract():
    global TESSERACT_FOUND, TESSERACT_PATH

    # 1. env override
    env = os.environ.get("TESSERACT_CMD")
    if env and Path(env).exists():
        TESSERACT_PATH = env
        TESSERACT_FOUND = True
        return

    # 2. known paths
    for p in CANDIDATE_PATHS:
        if Path(p).exists():
            TESSERACT_PATH = p
            TESSERACT_FOUND = True
            return

    # 3. PATH
    found = shutil.which("tesseract")
    if found:
        TESSERACT_PATH = found
        TESSERACT_FOUND = True
        return

    # 4. not found — OCR will be disabled
    print("  [WARNING] Tesseract not found — scanned-PDF OCR will be skipped.")
    print("  Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    TESSERACT_FOUND = False


_find_tesseract()

if TESSERACT_FOUND:
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        print(f"  [OK] Tesseract configured: {TESSERACT_PATH}")
    except ImportError:
        print("  [WARNING] pytesseract package not installed")
        TESSERACT_FOUND = False
