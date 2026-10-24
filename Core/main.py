import os
from pathlib import Path
from FileScanner import FileScanner
from FileGroup import FileGrouper
from utility import copy_files_to_folder, organize_files, group_similar_by_combined

MONTH_NAMES = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}


def get_directory(prompt):
    while True:
        path = input(prompt).strip()
        if path.lower() == 'q':
            return None
        if os.path.isdir(path):
            return Path(path)
        print("Invalid directory. Try again or type 'q' to cancel.")


def parse_month(value):
    value = value.strip().lower()
    if value.isdigit():
        m = int(value)
        return m if 1 <= m <= 12 else None
    return MONTH_NAMES.get(value)


def build_time_filter():
    print("\nHow do you want to filter by date?")
    print("1. This week (last 7 days)")
    print("2. This month (last 30 days)")
    print("3. A specific year (e.g. 2024)")
    print("4. A specific month (e.g. March 2024)")
    print("5. A specific week (e.g. Week 2 of March 2024)")
    print("6. Skip")

    choice = input("Choice (1-6): ").strip()

    if choice == '1':
        return {'type': 'recent_week'}

    elif choice == '2':
        return {'type': 'recent_month'}

    elif choice == '3':
        year_input = input("Year: ").strip()
        if year_input.isdigit():
            return {'type': 'year', 'year': int(year_input)}
        print("Invalid year, skipping.")
        return None

    elif choice == '4':
        year_input = input("Year: ").strip()
        month_input = input("Month (e.g. 3 or march): ").strip()
        month = parse_month(month_input)
        if year_input.isdigit() and month:
            return {'type': 'month', 'year': int(year_input), 'month': month}
        print("Invalid input, skipping.")
        return None

    elif choice == '5':
        year_input = input("Year: ").strip()
        month_input = input("Month: ").strip()
        week_input = input("Week number (1=days 1-7, 2=days 8-14, etc.): ").strip()
        month = parse_month(month_input)
        if year_input.isdigit() and month and week_input.isdigit():
            return {'type': 'week', 'year': int(year_input), 'month': month, 'week': int(week_input)}
        print("Invalid input, skipping.")
        return None

    return None


def run_file_finder(scanner):
    print("\nFile finder — press Enter to skip any filter.")

    search_root = get_directory("Folder to search: ")
    if search_root is None:
        return

    keyword = input("Keyword in filename: ").strip() or None

    ext_input = input("File types (e.g. jpg png pdf), or Enter for all: ").strip()
    if ext_input:
        extensions = [e if e.startswith('.') else f'.{e}' for e in ext_input.lower().split()]
    else:
        extensions = None

    if input("Filter by date? (y/n): ").strip().lower() == 'y':
        time_filter = build_time_filter()
    else:
        time_filter = None

    if not keyword and not extensions and not time_filter:
        print("No filters — showing all files.")

    print(f"\nSearching {search_root} ...")
    results = scanner.find_files(
        search_root=search_root,
        keyword=keyword,
        extensions=extensions,
        time_filter=time_filter
    )

    print(f"\nFound {len(results)} file(s).")

    if not results:
        print("No files matched.")
        return

    for i, node in enumerate(results, 1):
        modified = node.last_modified.strftime("%d %b %Y  %H:%M") if node.last_modified else "unknown"
        print(f"  {i}. {node.filename}")
        print(f"     {node.path}")
        print(f"     {node.size_format()}  |  {modified}")


def main():
    print("File Organizer")

    while True:
        print("\nWhat do you want to do?")
        print("1. Organize a single folder")
        print("2. Organize multiple folders")
        print("3. Group similar files by content + name")
        print("4. Find / search files")
        print("5. Exit")

        choice = input("\nChoice (1-5): ").strip()
        scanner = FileScanner()

        if choice == '5':
            print("Bye.")
            break

        elif choice == '1':
            folder_path = get_directory("Folder path: ")
            if folder_path is None:
                continue
            scanner.search_files(folder_path)

        elif choice == '2':
            print("Enter folder paths (type 'done' when finished):")
            folder_count = 0
            while True:
                path_input = input(f"Folder {folder_count + 1}: ").strip()
                if path_input.lower() == 'done':
                    if folder_count == 0:
                        print("Enter at least one folder.")
                        continue
                    break
                if os.path.isdir(path_input):
                    scanner.search_files(Path(path_input))
                    folder_count += 1
                else:
                    print(f"Invalid path: {path_input}")

        elif choice == '3':
            print("\n1. Group all files in a folder by similarity")
            print("2. Find files similar to a specific reference file")
            sub = input("Choice (1 or 2): ").strip()

            if sub == '2':
                print("Reference file path:")
                while True:
                    ref_path = input("  Path: ").strip()
                    if os.path.isfile(ref_path):
                        break
                    print("  File not found.")

                folder_path = get_directory("Folder to search for similar files: ")
                if folder_path is None:
                    continue
                scanner.search_files(folder_path)

                file_nodes = scanner.file_data()
                if not file_nodes:
                    print("No files found.")
                    continue

                print(f"\nFound {len(file_nodes)} files. Comparing to: {Path(ref_path).name}")
                print("Similarity threshold:")
                print("  1. Loose (0.3)")
                print("  2. Medium (0.5) — recommended")
                print("  3. Strict (0.7)")
                thresh_choice = input("Choice (1-3, default 2): ").strip()
                threshold = 0.3 if thresh_choice == '1' else 0.7 if thresh_choice == '3' else 0.5

                grouper = FileGrouper()
                similar_nodes = grouper.group_by_reference_file(ref_path, file_nodes, threshold=threshold)

                if not similar_nodes:
                    print(f"No files similar enough to '{Path(ref_path).name}'. Try a looser threshold.")
                    if input("Do something else? (y/n): ").lower() != 'y':
                        print("Bye.")
                        break
                    continue

                print(f"\nFound {len(similar_nodes)} similar file(s):")
                for node in similar_nodes:
                    print(f"  {node.filename}")

                dest_parent = get_directory("Copy them to: ")
                if dest_parent is None:
                    continue
                folder_name = input("Folder name (Enter for 'Similar_Files'): ").strip() or "Similar_Files"

                dest_dir = dest_parent / folder_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                stats = copy_files_to_folder(set(similar_nodes), dest_dir)
                print(f"\nDone. {stats['copied']} file(s) copied to {dest_dir}")
                if stats['errors']:
                    print(f"Errors: {stats['errors']}")

            else:
                folder_path = get_directory("Folder to group: ")
                if folder_path is None:
                    continue
                scanner.search_files(folder_path)

                file_nodes = scanner.file_data()
                if not file_nodes:
                    print("No files found.")
                    continue

                print(f"\nFound {len(file_nodes)} files. Grouping by content...")

                dest_parent = get_directory("Destination path: ")
                if dest_parent is None:
                    continue
                folder_name = input("Folder name (Enter for 'Grouped_Files'): ").strip() or "Grouped_Files"

                dest_dir = dest_parent / folder_name
                combined, stats_map = group_similar_by_combined(file_nodes, dest_dir)
                groups_created = len(combined)
                total_copied = sum(s["copied"] for s in stats_map.values())
                total_errors = sum(s["errors"] for s in stats_map.values())

                print(f"\nDone. {total_copied} file(s) grouped into {groups_created} folder(s).")
                if total_errors:
                    print(f"Errors: {total_errors}")
                print(f"Saved to: {dest_dir}")

            if input("\nDo something else? (y/n): ").lower() != 'y':
                print("Bye.")
                break
            continue

        elif choice == '4':
            run_file_finder(scanner)
            if input("\nSearch again? (y/n): ").lower() != 'y':
                print("Bye.")
                break
            continue

        else:
            print("Invalid choice.")
            continue

        print(f"\nScan done. {scanner.scanned_count} file(s) found, {len(scanner.errors)} error(s).")

        if scanner.scanned_count == 0:
            print("No files found.")
            if input("Try again? (y/n): ").lower() != 'y':
                break
            continue

        file_nodes = scanner.file_data()
        if not file_nodes:
            print("No valid files to organize.")
            if input("Try again? (y/n): ").lower() != 'y':
                break
            continue

        print("\nGroup by:")
        print("1. Extension")
        print("2. Size")
        print("3. Date")
        print("4. Name similarity")

        group_choice = input("Choice (1-4): ").strip()
        group_map = {'1': 'extensions', '2': 'size', '3': 'date', '4': 'name'}
        group_type = group_map.get(group_choice)

        if not group_type:
            print("Invalid choice.")
            continue

        dest_parent = get_directory("Destination path: ")
        if dest_parent is None:
            continue

        folder_name = input("Folder name (Enter for 'Organized_Files'): ").strip() or "Organized_Files"
        dest_dir = dest_parent / folder_name

        print(f"\n{len(file_nodes)} file(s) -> {dest_dir} (grouped by {group_type})")
        if input("Proceed? (y/n): ").lower() != 'y':
            print("Cancelled.")
            if input("Start over? (y/n): ").lower() != 'y':
                break
            continue

        stats = organize_files(file_nodes, dest_dir, group_type)

        print(f"\nDone. {stats['total_files']} file(s) in {stats['groups_created']} group(s).")
        if stats['errors']:
            print(f"Errors: {stats['errors']}")
        print(f"Saved to: {dest_dir}")

        if input("Organize more? (y/n): ").lower() != 'y':
            print("Bye.")
            break


if __name__ == "__main__":
    main()