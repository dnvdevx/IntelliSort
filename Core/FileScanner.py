import re
from pathlib import Path
from datetime import datetime, timedelta

from file_node import FileNode


class FileScanner:

    def __init__(self):
        self.files = set()
        self.errors = []
        self.scanned_count = 0
        self.ignored_count = 0

    def search_files(self, path):
        path = Path(path)
        try:
            if path.is_file():
                print(f"Scanning: {path}")
                self.files.add(path)
                self.scanned_count += 1
                return

            print(f"Scanning: {path}")
            for item in path.iterdir():
                if item.is_file():
                    self.files.add(item)
                    self.scanned_count += 1
                elif item.is_dir():
                    self.search_files(item)
        except (PermissionError, OSError):
            self.errors.append(str(path))

    def file_data(self):
        file_data_list = set()
        skipped = 0
        for file in self.files:
            try:
                node = FileNode(file)
                file_data_list.add(node)
            except Exception as e:
                print(f"Skipping {file.name}: {e}")
                skipped += 1
        if skipped:
            print(f"Skipped {skipped} file(s) due to errors.")
        return file_data_list

    def find_files(self, search_root, keyword=None, extensions=None, time_filter=None):
        search_root = Path(search_root)
        results = []

        for item in search_root.rglob("*"):
            if not item.is_file():
                continue

            try:
                node = FileNode(item)
            except Exception:
                continue

            if keyword:
                clean_name = re.sub(r"[-_.\s]+", "", node.name.lower())
                keyword_words = keyword.lower().split()
                if not all(word in clean_name for word in keyword_words):
                    continue

            if extensions:
                if node.extension.lower() not in [e.lower() for e in extensions]:
                    continue

            if time_filter and node.last_modified:
                if not self._matches_time_filter(node.last_modified, time_filter):
                    continue

            results.append(node)

        results.sort(key=lambda n: n.last_modified or datetime.min, reverse=True)
        return results

    def _matches_time_filter(self, last_modified, time_filter):
        now = datetime.now()
        filter_type = time_filter.get('type')

        if filter_type == 'recent_week':
            return last_modified >= now - timedelta(days=7)

        elif filter_type == 'recent_month':
            return last_modified >= now - timedelta(days=30)

        elif filter_type == 'year':
            return last_modified.year == time_filter['year']

        elif filter_type == 'month':
            return (last_modified.year == time_filter['year'] and
                    last_modified.month == time_filter['month'])

        elif filter_type == 'week':
            if (last_modified.year != time_filter['year'] or
                    last_modified.month != time_filter['month']):
                return False
            week_num = time_filter['week']
            day = last_modified.day
            file_week = (day - 1) // 7 + 1
            return file_week == week_num

        else:
            raise ValueError(f"Unknown time filter type: {filter_type!r}")