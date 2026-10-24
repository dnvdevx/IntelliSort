import os
import hashlib
import datetime
from pathlib import Path

import fitz
import docx
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

SIZE_TINY = 1_000_000
SIZE_SMALL = 10_000_000
SIZE_MEDIUM = 100_000_000
SUPPORTED_TEXT_EXTENSIONS = frozenset({".txt", ".docx", ".pdf"})


def extract_text(filepath):
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext == ".txt":
        return filepath.read_text(encoding="utf-8", errors="ignore")

    elif ext == ".docx":
        doc = docx.Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".pdf":
        doc = fitz.open(filepath)
        parts = []
        ocr_error = None
        try:
            for page in doc:
                extracted = page.get_text()
                if extracted.strip():
                    parts.append(extracted)
                else:
                    try:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        parts.append(pytesseract.image_to_string(img))
                    except Exception as e:
                        ocr_error = e
        finally:
            doc.close()
        text = "".join(parts)
        if not text.strip() and ocr_error is not None:
            raise RuntimeError(f"OCR failed ({type(ocr_error).__name__}: {ocr_error})")
        return text

    elif ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(filepath)
        return pytesseract.image_to_string(img)

    return ""


class FileNode:

    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")
        if not os.path.isfile(self.path):
            raise ValueError(f"Path is not a file: {self.path}")

        p = Path(self.path)
        self.name = p.stem
        self.filename = p.name
        self.extension = p.suffix.lower()

        try:
            stat = os.stat(self.path)
            self.size = stat.st_size
            self.time_created = datetime.datetime.fromtimestamp(stat.st_ctime)
            self.last_modified = datetime.datetime.fromtimestamp(stat.st_mtime)
        except PermissionError:
            import warnings
            warnings.warn(f"Permission denied reading stat for: {self.path}", RuntimeWarning, stacklevel=2)
            self.size = 0
            self.time_created = None
            self.last_modified = None

        self.hash = None

    def hash_value(self):
        if self.hash is None:
            try:
                hasher = hashlib.sha256()
                with open(self.path, 'rb') as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
                self.hash = hasher.hexdigest()
            except Exception as error:
                print(f"Warning: could not hash {self.filename}: {error}")
                self.hash = None
        return self.hash

    def size_format(self):
        size = self.size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"  # fallback for extremely large files

    def size_category(self):
        if self.size < SIZE_TINY:
            return 'tiny'
        elif self.size < SIZE_SMALL:
            return 'small'
        elif self.size < SIZE_MEDIUM:
            return 'medium'
        else:
            return 'large'

    def data(self, include_hash=False):
        d = {
            'name': self.name,
            'filename': self.filename,
            'extension': self.extension,
            'size': self.size,
            'size_formatted': self.size_format(),
            'size_category': self.size_category(),
            'time_created': self.time_created.isoformat() if self.time_created else None,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
        }
        if include_hash:
            d['hash'] = self.hash_value()
        return d

    def __str__(self):
        return f"{self.filename} ({self.size_format()})"

    def __repr__(self):
        return f"FileNode('{self.filename}', {self.size_format()}, {self.extension})"