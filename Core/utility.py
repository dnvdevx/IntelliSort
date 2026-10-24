import shutil
from pathlib import Path

from FileGroup import FileGrouper


def copy_files_to_folder(file_set, dest_dir, log_fn=None, move=False):
    _log = log_fn or print
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    action = shutil.move if move else shutil.copy2
    verb = "Moving" if move else "Copying"

    stats = {"copied": 0, "errors": 0}
    for file_node in file_set:
        try:
            dest_path = dest_dir / file_node.filename
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{file_node.name}_{counter}{file_node.extension}"
                counter += 1
            action(file_node.path, dest_path)
            stats["copied"] += 1
        except Exception as exc:
            _log(f"Error {verb.lower()} {file_node.filename}: {exc}")
            stats["errors"] += 1
    return stats


def organize_files(file_nodes, dest_dir, group_type, log_fn=None, move=False):
    verb = "Moving" if move else "Copying"
    _log = log_fn or print
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    grouper = FileGrouper()
    dispatch = {
        "extensions": grouper.group_extensions,
        "size":       grouper.group_by_size,
        "date":       grouper.group_by_date,
        "name":       grouper.group_by_name,
    }
    if group_type not in dispatch:
        raise ValueError(f"Unknown group type: {group_type!r}")

    dispatch[group_type](file_nodes)
    groups = grouper.group[group_type]

    total_copied = 0
    total_errors = 0
    groups_created = 0

    for group_name, file_set in groups.items():
        group_folder = dest_dir / str(group_name)
        group_folder.mkdir(parents=True, exist_ok=True)
        groups_created += 1
        _log(f"  - {verb} group: {group_name} ({len(file_set)} files)")
        stats = copy_files_to_folder(file_set, group_folder, log_fn=log_fn, move=move)
        total_copied += stats["copied"]
        total_errors += stats["errors"]

    return {"total_files": total_copied, "groups_created": groups_created, "errors": total_errors}


def group_similar_by_combined(file_nodes, dest_dir, log_fn=None, cancel_event=None, move=False):
    _log = log_fn or print
    verb = "Moving" if move else "Copying"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    grouper = FileGrouper()
    grouper.group_by_combined(file_nodes)
    combined = grouper.group.get("combined", {})

    stats_map = {}
    for group_name, file_set in combined.items():
        if cancel_event is not None and cancel_event.is_set():
            _log("Cancelled — stopping before copying remaining groups.")
            raise RuntimeError("Cancelled by user.")
        group_folder = dest_dir / str(group_name)
        group_folder.mkdir(parents=True, exist_ok=True)
        _log(f"  - {verb} similarity group: {group_name} ({len(file_set)} files)")
        stats = copy_files_to_folder(file_set, group_folder, log_fn=log_fn, move=move)
        stats_map[str(group_name)] = stats

    return combined, stats_map