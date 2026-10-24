import os
import sys
import threading
import queue
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = WORKSPACE_ROOT / "Core"

sys.path.insert(0, str(CORE_DIR))

from utility import copy_files_to_folder, organize_files, group_similar_by_combined


class FileOrganizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IntelliSort")
        self.geometry("860x600")

        self._work_queue = queue.Queue()
        self._worker_thread = None
        self._cancel_event = threading.Event()
        self.move_mode = tk.IntVar(value=0)

        self.sources_1 = []
        self.sources_2 = []
        self.sources_3 = []

        self._build_ui()
        self.after(100, self._drain_queue)

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="both", expand=False)

        ttk.Label(top, text="Choose operation:").grid(row=0, column=0, sticky="w")

        self.btn_op1 = ttk.Button(top, text="1) Organize Files (Single Choice)", command=self._show_op1)
        self.btn_op2 = ttk.Button(top, text="2) Organize Multiple", command=self._show_op2)
        self.btn_op3 = ttk.Button(top, text="3) Group Similar (Content + Name)", command=self._show_op3)
        self.btn_op4 = ttk.Button(top, text="4) Find / Search Files", command=self._show_op4)
        self.btn_op1.grid(row=1, column=0, sticky="w", pady=5)
        self.btn_op2.grid(row=2, column=0, sticky="w", pady=5)
        self.btn_op3.grid(row=3, column=0, sticky="w", pady=5)
        self.btn_op4.grid(row=4, column=0, sticky="w", pady=5)

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(top, textvariable=self.status_var).grid(row=5, column=0, sticky="w", pady=(8, 2))

        progress_frame = ttk.Frame(top)
        progress_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        top.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate", length=400)
        self.progress_bar.pack(side="left", fill="x", expand=True)

        self.step_var = tk.StringVar(value="")
        self.step_label = ttk.Label(progress_frame, textvariable=self.step_var,
                                    width=28, anchor="w", foreground="#555555")
        self.step_label.pack(side="left", padx=(10, 0))

        self.container = ttk.Frame(self, padding=10)
        self.container.pack(fill="both", expand=True)

        self.frame_op1 = ttk.LabelFrame(self.container, text="Option 1: Organize")
        self._build_op1(self.frame_op1)

        self.frame_op2 = ttk.LabelFrame(self.container, text="Option 2: Organize Multiple")
        self._build_op2(self.frame_op2)

        self.frame_op3 = ttk.LabelFrame(self.container, text="Option 3: Group Similar (ML)")
        self._build_op3(self.frame_op3)

        self.frame_op4 = ttk.LabelFrame(self.container, text="Option 4: Find / Search Files")
        self._build_op4(self.frame_op4)

        self._show_op1()

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=False, padx=10, pady=(0, 10))

        self.log_text = ScrolledText(log_frame, height=12, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _build_op1(self, parent):
        src_frame = ttk.Frame(parent)
        src_frame.pack(fill="x", pady=8)

        ttk.Label(src_frame, text="Source folders/files:").grid(row=0, column=0, sticky="w")
        self.list_op1 = tk.Listbox(src_frame, height=6)
        self.list_op1.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        btns = ttk.Frame(src_frame)
        btns.grid(row=1, column=1, sticky="ns")

        ttk.Button(btns, text="Add Folder", command=self._add_op1_folder).pack(fill="x", pady=2)
        ttk.Button(btns, text="Add Files", command=self._add_op1_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove Selected", command=self._remove_selected_op1).pack(fill="x", pady=2)

        src_frame.columnconfigure(0, weight=1)
        src_frame.rowconfigure(1, weight=1)

        grp_frame = ttk.Frame(parent)
        grp_frame.pack(fill="x", pady=8)
        ttk.Label(grp_frame, text="Group by:").grid(row=0, column=0, sticky="w")
        self.cmb_op1_group = ttk.Combobox(grp_frame, state="readonly", values=["extensions", "size", "date", "name"])
        self.cmb_op1_group.current(0)
        self.cmb_op1_group.grid(row=0, column=1, sticky="w", padx=10)

        dst_frame = ttk.Frame(parent)
        dst_frame.pack(fill="x", pady=8)
        ttk.Label(dst_frame, text="Destination parent:").grid(row=0, column=0, sticky="w")
        self.entry_op1_dest = ttk.Entry(dst_frame, width=55)
        self.entry_op1_dest.grid(row=0, column=1, sticky="we", padx=10)
        ttk.Button(dst_frame, text="Browse", command=self._browse_op1_dest).grid(row=0, column=2, padx=(0, 10))

        ttk.Label(dst_frame, text="Folder name:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.entry_op1_name = ttk.Entry(dst_frame, width=25)
        self.entry_op1_name.insert(0, "Organized_Files")
        self.entry_op1_name.grid(row=1, column=1, sticky="w", padx=10, pady=(8, 0))

        run_frame = ttk.Frame(parent)
        run_frame.pack(fill="x", pady=12)
        ttk.Checkbutton(run_frame, text="Move files instead of copying",
                        variable=self.move_mode).pack(side="left", padx=(0, 16))
        self.btn_op1_run = ttk.Button(run_frame, text="Start Option 1", command=self._start_op1)
        self.btn_op1_run.pack(side="left")

    def _build_op2(self, parent):
        src_frame = ttk.Frame(parent)
        src_frame.pack(fill="x", pady=8)

        ttk.Label(src_frame, text="Source folders/files:").grid(row=0, column=0, sticky="w")
        self.list_op2 = tk.Listbox(src_frame, height=6)
        self.list_op2.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        btns = ttk.Frame(src_frame)
        btns.grid(row=1, column=1, sticky="ns")

        ttk.Button(btns, text="Add Folder", command=self._add_op2_folder).pack(fill="x", pady=2)
        ttk.Button(btns, text="Add Files", command=self._add_op2_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove Selected", command=self._remove_selected_op2).pack(fill="x", pady=2)
        ttk.Button(btns, text="Clear", command=self._clear_op2_sources).pack(fill="x", pady=2)

        src_frame.columnconfigure(0, weight=1)
        src_frame.rowconfigure(1, weight=1)

        grp_frame = ttk.Frame(parent)
        grp_frame.pack(fill="x", pady=8)
        ttk.Label(grp_frame, text="Group by:").grid(row=0, column=0, sticky="w")
        self.cmb_op2_group = ttk.Combobox(grp_frame, state="readonly", values=["extensions", "size", "date", "name"])
        self.cmb_op2_group.current(0)
        self.cmb_op2_group.grid(row=0, column=1, sticky="w", padx=10)

        dst_frame = ttk.Frame(parent)
        dst_frame.pack(fill="x", pady=8)
        ttk.Label(dst_frame, text="Destination parent:").grid(row=0, column=0, sticky="w")
        self.entry_op2_dest = ttk.Entry(dst_frame, width=55)
        self.entry_op2_dest.grid(row=0, column=1, sticky="we", padx=10)
        ttk.Button(dst_frame, text="Browse", command=self._browse_op2_dest).grid(row=0, column=2, padx=(0, 10))

        ttk.Label(dst_frame, text="Folder name:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.entry_op2_name = ttk.Entry(dst_frame, width=25)
        self.entry_op2_name.insert(0, "Organized_Files")
        self.entry_op2_name.grid(row=1, column=1, sticky="w", padx=10, pady=(8, 0))

        run_frame = ttk.Frame(parent)
        run_frame.pack(fill="x", pady=12)
        ttk.Checkbutton(run_frame, text="Move files instead of copying",
                        variable=self.move_mode).pack(side="left", padx=(0, 16))
        self.btn_op2_run = ttk.Button(run_frame, text="Start Option 2", command=self._start_op2)
        self.btn_op2_run.pack(side="left")

    def _build_op3(self, parent):
        src_frame = ttk.Frame(parent)
        src_frame.pack(fill="x", pady=8)

        ttk.Label(src_frame, text="Source folders/files:").grid(row=0, column=0, sticky="w")
        self.list_op3 = tk.Listbox(src_frame, height=6)
        self.list_op3.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        btns = ttk.Frame(src_frame)
        btns.grid(row=1, column=1, sticky="ns")
        ttk.Button(btns, text="Add Folder", command=self._add_op3_folder).pack(fill="x", pady=2)
        ttk.Button(btns, text="Add Files", command=self._add_op3_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove Selected", command=self._remove_selected_op3).pack(fill="x", pady=2)
        ttk.Button(btns, text="Clear", command=self._clear_op3_sources).pack(fill="x", pady=2)

        src_frame.columnconfigure(0, weight=1)
        src_frame.rowconfigure(1, weight=1)

        dst_frame = ttk.Frame(parent)
        dst_frame.pack(fill="x", pady=8)
        ttk.Label(dst_frame, text="Destination parent:").grid(row=0, column=0, sticky="w")
        self.entry_op3_dest = ttk.Entry(dst_frame, width=55)
        self.entry_op3_dest.grid(row=0, column=1, sticky="we", padx=10)
        ttk.Button(dst_frame, text="Browse", command=self._browse_op3_dest).grid(row=0, column=2, padx=(0, 10))

        ttk.Label(dst_frame, text="Folder name:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.entry_op3_name = ttk.Entry(dst_frame, width=25)
        self.entry_op3_name.insert(0, "Similar_Files")
        self.entry_op3_name.grid(row=1, column=1, sticky="w", padx=10, pady=(8, 0))

        run_frame = ttk.Frame(parent)
        run_frame.pack(fill="x", pady=12)
        ttk.Checkbutton(run_frame, text="Move files instead of copying",
                        variable=self.move_mode).pack(side="left", padx=(0, 16))
        self.btn_op3_run = ttk.Button(run_frame, text="Start Option 3 (ML)", command=self._start_op3)
        self.btn_op3_run.pack(side="left", padx=(0, 8))
        self.btn_op3_cancel = ttk.Button(run_frame, text="Cancel", command=self._cancel_op3,
                                          state="disabled")
        self.btn_op3_cancel.pack(side="left")

    def _build_op4(self, parent):
        root_frame = ttk.Frame(parent)
        root_frame.pack(fill="x", pady=8)
        ttk.Label(root_frame, text="Search in folder:").grid(row=0, column=0, sticky="w")
        self.entry_op4_root = ttk.Entry(root_frame, width=55)
        self.entry_op4_root.grid(row=0, column=1, sticky="we", padx=10)
        ttk.Button(root_frame, text="Browse", command=self._browse_op4_root).grid(row=0, column=2)
        root_frame.columnconfigure(1, weight=1)

        filter_frame = ttk.LabelFrame(parent, text="Filters  (all optional)")
        filter_frame.pack(fill="x", pady=8, padx=4)

        ttk.Label(filter_frame, text="Keyword in filename:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.entry_op4_keyword = ttk.Entry(filter_frame, width=28)
        self.entry_op4_keyword.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(filter_frame, text="File types (e.g. jpg pdf):").grid(row=0, column=2, sticky="w", padx=(16, 4), pady=4)
        self.entry_op4_ext = ttk.Entry(filter_frame, width=22)
        self.entry_op4_ext.grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(filter_frame, text="Date filter:").grid(row=1, column=0, sticky="w", padx=6, pady=(4, 2))
        self.cmb_op4_date = ttk.Combobox(
            filter_frame, state="readonly", width=24,
            values=["None", "This week (last 7 days)", "This month (last 30 days)",
                    "Specific year", "Specific month", "Specific week"]
        )
        self.cmb_op4_date.current(0)
        self.cmb_op4_date.grid(row=1, column=1, sticky="w", padx=6, pady=(4, 2))
        self.cmb_op4_date.bind("<<ComboboxSelected>>", self._on_date_filter_change)

        self.frame_op4_date_sub = ttk.Frame(filter_frame)
        self.frame_op4_date_sub.grid(row=2, column=0, columnspan=4, sticky="w", padx=6, pady=4)

        ttk.Label(self.frame_op4_date_sub, text="Year:").grid(row=0, column=0, sticky="w")
        self.entry_op4_year = ttk.Entry(self.frame_op4_date_sub, width=8)
        self.entry_op4_year.grid(row=0, column=1, padx=6, sticky="w")

        self.lbl_op4_month = ttk.Label(self.frame_op4_date_sub, text="Month (name or 1-12):")
        self.lbl_op4_month.grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.entry_op4_month = ttk.Entry(self.frame_op4_date_sub, width=10)
        self.entry_op4_month.grid(row=0, column=3, padx=6, sticky="w")

        self.lbl_op4_week = ttk.Label(self.frame_op4_date_sub, text="Week no. (1-5):")
        self.lbl_op4_week.grid(row=0, column=4, sticky="w", padx=(12, 0))
        self.entry_op4_week = ttk.Entry(self.frame_op4_date_sub, width=5)
        self.entry_op4_week.grid(row=0, column=5, padx=6, sticky="w")

        self._on_date_filter_change()

        run_frame = ttk.Frame(parent)
        run_frame.pack(fill="x", pady=8)
        self.btn_op4_run = ttk.Button(run_frame, text="Search", command=self._start_op4)
        self.btn_op4_run.pack(side="left", padx=4)
        ttk.Button(run_frame, text="Clear Results", command=self._clear_op4_results).pack(side="left", padx=4)

        res_frame = ttk.LabelFrame(parent, text="Results")
        res_frame.pack(fill="both", expand=True, pady=6, padx=4)

        cols = ("Filename", "Path", "Size", "Last Modified")
        self.tree_op4 = ttk.Treeview(res_frame, columns=cols, show="headings", height=10)
        for col in cols:
            self.tree_op4.heading(col, text=col)
        self.tree_op4.column("Filename", width=180, anchor="w")
        self.tree_op4.column("Path", width=300, anchor="w")
        self.tree_op4.column("Size", width=90, anchor="e")
        self.tree_op4.column("Last Modified", width=140, anchor="w")

        vsb = ttk.Scrollbar(res_frame, orient="vertical", command=self.tree_op4.yview)
        hsb = ttk.Scrollbar(res_frame, orient="horizontal", command=self.tree_op4.xview)
        self.tree_op4.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree_op4.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        res_frame.rowconfigure(0, weight=1)
        res_frame.columnconfigure(0, weight=1)

        self.lbl_op4_count = ttk.Label(parent, text="")
        self.lbl_op4_count.pack(anchor="w", padx=4)


    def _on_date_filter_change(self, _event=None):
        sel = self.cmb_op4_date.get()
        needs_year = sel in ("Specific year", "Specific month", "Specific week")
        needs_month = sel in ("Specific month", "Specific week")
        needs_week = sel == "Specific week"

        for widget in [self.entry_op4_year]:
            widget.grid() if needs_year else widget.grid_remove()
        for widget in [self.lbl_op4_month, self.entry_op4_month]:
            widget.grid() if needs_month else widget.grid_remove()
        for widget in [self.lbl_op4_week, self.entry_op4_week]:
            widget.grid() if needs_week else widget.grid_remove()

        children = self.frame_op4_date_sub.winfo_children()
        year_label = children[0] if children else None
        if year_label:
            year_label.grid() if needs_year else year_label.grid_remove()

    def _build_time_filter(self):
        sel = self.cmb_op4_date.get()
        if sel == "None" or not sel:
            return None
        if sel == "This week (last 7 days)":
            return {"type": "recent_week"}
        if sel == "This month (last 30 days)":
            return {"type": "recent_month"}

        year_raw = self.entry_op4_year.get().strip()
        if not year_raw.isdigit():
            messagebox.showwarning("Date filter", "Please enter a valid year (e.g. 2024).")
            return "invalid"
        year = int(year_raw)

        if sel == "Specific year":
            return {"type": "year", "year": year}

        month_raw = self.entry_op4_month.get().strip()
        month = self._parse_month(month_raw)
        if month is None:
            messagebox.showwarning("Date filter", "Please enter a valid month (e.g. 3 or March).")
            return "invalid"

        if sel == "Specific month":
            return {"type": "month", "year": year, "month": month}

        week_raw = self.entry_op4_week.get().strip()
        if not week_raw.isdigit() or not (1 <= int(week_raw) <= 5):
            messagebox.showwarning("Date filter", "Please enter a week number between 1 and 5.")
            return "invalid"
        return {"type": "week", "year": year, "month": month, "week": int(week_raw)}

    @staticmethod
    def _parse_month(value):
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        value = value.strip().lower()
        if value.isdigit():
            m = int(value)
            return m if 1 <= m <= 12 else None
        return month_names.get(value)

    def _browse_op4_root(self):
        path = filedialog.askdirectory(title="Select folder to search")
        if path:
            self.entry_op4_root.delete(0, tk.END)
            self.entry_op4_root.insert(0, path)

    def _clear_op4_results(self):
        self.tree_op4.delete(*self.tree_op4.get_children())
        self.lbl_op4_count.configure(text="")

    def _show_op1(self):
        self.frame_op2.pack_forget()
        self.frame_op3.pack_forget()
        self.frame_op4.pack_forget()
        self.frame_op1.pack(fill="both", expand=False)

    def _show_op2(self):
        self.frame_op1.pack_forget()
        self.frame_op3.pack_forget()
        self.frame_op4.pack_forget()
        self.frame_op2.pack(fill="both", expand=False)

    def _show_op3(self):
        self.frame_op1.pack_forget()
        self.frame_op2.pack_forget()
        self.frame_op4.pack_forget()
        self.frame_op3.pack(fill="both", expand=False)

    def _show_op4(self):
        self.frame_op1.pack_forget()
        self.frame_op2.pack_forget()
        self.frame_op3.pack_forget()
        self.frame_op4.pack(fill="both", expand=True)

    def _add_source_paths(self, mode, paths):
        dest = getattr(self, f"sources_{mode}")
        for p in paths:
            p = str(Path(p))
            if p not in dest:
                dest.append(p)
        self._refresh_listbox(mode)

    def _refresh_listbox(self, mode):
        lb = getattr(self, f"list_op{mode}")
        sources = getattr(self, f"sources_{mode}")
        lb.delete(0, tk.END)
        for s in sources:
            lb.insert(tk.END, s)

    def _remove_selected(self, mode):
        lb = getattr(self, f"list_op{mode}")
        sources = getattr(self, f"sources_{mode}")
        indices = list(lb.curselection())
        if not indices:
            return
        for idx in sorted(indices, reverse=True):
            del sources[idx]
        self._refresh_listbox(mode)

    def _clear_sources(self, mode):
        setattr(self, f"sources_{mode}", [])
        self._refresh_listbox(mode)

    def _browse_dest(self, entry):
        path = filedialog.askdirectory(title="Select destination parent folder")
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def _browse_op1_dest(self): self._browse_dest(self.entry_op1_dest)
    def _browse_op2_dest(self): self._browse_dest(self.entry_op2_dest)
    def _browse_op3_dest(self): self._browse_dest(self.entry_op3_dest)

    def _add_op1_folder(self):
        path = filedialog.askdirectory(title="Select a folder to organize")
        if path:
            self._add_source_paths(1, [path])

    def _add_op1_files(self):
        paths = filedialog.askopenfilenames(title="Select files to organize", filetypes=[("All files", "*.*")])
        if paths:
            self._add_source_paths(1, list(paths))

    def _remove_selected_op1(self):
        self._remove_selected(1)

    def _add_op2_folder(self):
        path = filedialog.askdirectory(title="Select a folder to add")
        if path:
            self._add_source_paths(2, [path])

    def _add_op2_files(self):
        paths = filedialog.askopenfilenames(title="Select files to add", filetypes=[("All files", "*.*")])
        if paths:
            self._add_source_paths(2, list(paths))

    def _remove_selected_op2(self):
        self._remove_selected(2)

    def _clear_op2_sources(self):
        self._clear_sources(2)

    def _add_op3_folder(self):
        path = filedialog.askdirectory(title="Select a folder to group by similarity")
        if path:
            self._add_source_paths(3, [path])

    def _add_op3_files(self):
        paths = filedialog.askopenfilenames(title="Select files to group by similarity", filetypes=[("All files", "*.*")])
        if paths:
            self._add_source_paths(3, list(paths))

    def _remove_selected_op3(self):
        self._remove_selected(3)

    def _clear_op3_sources(self):
        self._clear_sources(3)

    def _log(self, msg):
        self._work_queue.put(("log", msg))

    def _progress(self, msg):
        self._work_queue.put(("progress", msg))

    def _drain_queue(self):
        did_work = False
        while True:
            try:
                item = self._work_queue.get_nowait()
            except queue.Empty:
                break
            did_work = True

            kind = item[0]
            if kind == "log":
                msg = item[1]
                self._append_log(msg)
            elif kind == "status":
                self.status_var.set(item[1])
            elif kind == "progress":
                self.step_var.set(item[1])
            elif kind == "search_results":
                nodes = item[1]
                self.tree_op4.delete(*self.tree_op4.get_children())
                for node in nodes:
                    modified = node.last_modified.strftime("%d %b %Y  %H:%M") if node.last_modified else "unknown"
                    self.tree_op4.insert("", tk.END, values=(
                        node.filename,
                        node.path,
                        node.size_format(),
                        modified,
                    ))
                count = len(nodes)
                self.lbl_op4_count.configure(
                    text=f"{count} file(s) found." if count else "No files matched the filters."
                )
            elif kind == "cancel_done":
                self.btn_op3_cancel.configure(state="disabled")
            elif kind == "done":
                self._set_busy(False)
                ok, payload = item[1], item[2]
                if ok:
                    self.status_var.set(payload)
                    messagebox.showinfo("Done", payload)
                else:
                    self.status_var.set("Error.")
                    messagebox.showerror("Error", str(payload))

        if did_work:
            self.update_idletasks()
        self.after(100, self._drain_queue)

    def _append_log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _set_busy(self, is_busy):
        state = "disabled" if is_busy else "normal"
        for w in [self.btn_op1_run, self.btn_op2_run, self.btn_op3_run, self.btn_op4_run,
                  self.btn_op1, self.btn_op2, self.btn_op3, self.btn_op4]:
            w.configure(state=state)
        if not is_busy:
            self.btn_op3_cancel.configure(state="disabled")
        if is_busy:
            self.status_var.set("Working...")
            self.progress_bar.start(12)
        else:
            self.status_var.set("Idle.")
            self.progress_bar.stop()
            self.step_var.set("")

    def _run_organize_worker(self, label, sources, dest_dir, group_type, move=False):
        """Shared worker for ops 1 & 2 — scans, groups, and copies/moves files."""
        self._set_busy(True)

        def worker():
            try:
                from FileScanner import FileScanner

                self._progress("Scanning files...")
                self._log(f"{label}: Scanning sources...")
                scanner = FileScanner()
                for s in sources:
                    scanner.search_files(s)
                self._log(f"Scanned {scanner.scanned_count} file(s).")

                self._progress("Loading file info...")
                file_nodes = scanner.file_data()
                self._log(f"Loaded {len(file_nodes)} valid file node(s).")
                if not file_nodes:
                    raise RuntimeError("No valid files to organize.")

                self._progress(f"Grouping by {group_type}...")
                self._log(f"Organizing with group type: {group_type}")
                result = organize_files(
                    file_nodes=file_nodes,
                    dest_dir=dest_dir,
                    group_type=group_type,
                    log_fn=lambda m: self._log(m),
                    move=move,
                )
                verb = "Moved" if move else "Copied"
                self._progress(f"{verb} files...")
                msg = (
                    f"{label} done. {verb} {result['total_files']} files into "
                    f"{result['groups_created']} group folder(s) at: {dest_dir}"
                )
                self._work_queue.put(("done", True, msg))
            except Exception as e:
                self._work_queue.put(("done", False, e))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _start_op1(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        sources = list(self.sources_1)
        dest_parent = self.entry_op1_dest.get().strip()
        folder_name = self.entry_op1_name.get().strip() or "Organized_Files"
        group_type = self.cmb_op1_group.get().strip()
        if not sources:
            messagebox.showwarning("Missing input", "Add at least one source folder/file.")
            return
        if not dest_parent:
            messagebox.showwarning("Missing input", "Choose a destination parent folder.")
            return
        self._run_organize_worker("Option 1", sources, Path(dest_parent) / folder_name, group_type, move=bool(self.move_mode.get()))

    def _start_op2(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        sources = list(self.sources_2)
        dest_parent = self.entry_op2_dest.get().strip()
        folder_name = self.entry_op2_name.get().strip() or "Organized_Files"
        group_type = self.cmb_op2_group.get().strip()
        if not sources:
            messagebox.showwarning("Missing input", "Add at least one source folder/file.")
            return
        if not dest_parent:
            messagebox.showwarning("Missing input", "Choose a destination parent folder.")
            return
        self._run_organize_worker("Option 2", sources, Path(dest_parent) / folder_name, group_type, move=bool(self.move_mode.get()))

    def _cancel_op3(self):
        """Signal the running ML worker to stop after its current file."""
        self._cancel_event.set()
        self._log("Cancellation requested — stopping after current file...")
        self.btn_op3_cancel.configure(state="disabled")

    def _start_op3(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        sources = list(self.sources_3)
        dest_parent = self.entry_op3_dest.get().strip()
        folder_name = self.entry_op3_name.get().strip() or "Similar_Files"

        if not sources:
            messagebox.showwarning("Missing input", "Add at least one source folder/file.")
            return
        if not dest_parent:
            messagebox.showwarning("Missing input", "Choose a destination parent folder.")
            return

        dest_dir = Path(dest_parent) / folder_name
        self._cancel_event.clear()
        self._set_busy(True)
        self.btn_op3_cancel.configure(state="normal")

        cancel_event = self._cancel_event
        move = bool(self.move_mode.get())

        def worker():
            try:
                from FileScanner import FileScanner

                self._progress("Scanning files...")
                self._log("Option 3: Scanning sources for ML grouping...")
                scanner = FileScanner()
                for s in sources:
                    if cancel_event.is_set():
                        raise RuntimeError("Cancelled by user.")
                    scanner.search_files(s)
                self._log(f"Scanned {scanner.scanned_count} file(s).")

                self._progress("Loading file info...")
                file_nodes = scanner.file_data()
                self._log(f"Loaded {len(file_nodes)} valid file node(s).")
                if not file_nodes:
                    raise RuntimeError("No valid files to group.")

                if cancel_event.is_set():
                    raise RuntimeError("Cancelled by user.")

                self._progress("Extracting text & building embeddings...")
                self._log("Running similarity grouping (content + name)...")
                combined, _stats_map = group_similar_by_combined(
                    file_nodes=file_nodes,
                    dest_dir=dest_dir,
                    log_fn=lambda m: self._log(m),
                    cancel_event=cancel_event,
                    move=move,
                )
                verb = "Moved" if move else "Copied"
                self._progress(f"{verb} grouped files...")
                group_names = list(combined.keys())
                msg = f"Option 3 done. {verb} files into {len(group_names)} similarity group folder(s) at: {dest_dir}"
                self._work_queue.put(("done", True, msg))
            except Exception as e:
                self._work_queue.put(("done", False, e))
            finally:
                self._work_queue.put(("cancel_done", None))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()


    def _start_op4(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        search_root = self.entry_op4_root.get().strip()
        if not search_root or not Path(search_root).is_dir():
            messagebox.showwarning("Missing input", "Please select a valid folder to search.")
            return

        keyword = self.entry_op4_keyword.get().strip() or None

        ext_raw = self.entry_op4_ext.get().strip()
        extensions = None
        if ext_raw:
            extensions = [e if e.startswith(".") else f".{e}" for e in ext_raw.lower().split()]

        time_filter = self._build_time_filter()
        if time_filter == "invalid":
            return

        self._set_busy(True)
        self._clear_op4_results()
        self._log(f"Option 4: Searching in {search_root} ...")

        def worker():
            try:
                from FileScanner import FileScanner
                self._progress("Searching files...")
                scanner = FileScanner()
                results = scanner.find_files(
                    search_root=search_root,
                    keyword=keyword,
                    extensions=extensions,
                    time_filter=time_filter,
                )
                self._progress("Populating results...")
                self._work_queue.put(("search_results", results))
                msg = f"Search complete. {len(results)} file(s) found."
                self._work_queue.put(("done", True, msg))
            except Exception as e:
                self._work_queue.put(("done", False, e))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()


def main():
    app = FileOrganizerApp()
    app.mainloop()


if __name__ == "__main__":
    main()