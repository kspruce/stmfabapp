"""
GUI Device Management Module - Adds device/sample management capabilities

This module provides:
1. Device selection and management interface
2. Rename and delete operations
3. Thermal budget viewer with device selection
4. Enhanced database browser

Add to your stm_fab_gui_enhanced.py
"""

import customtkinter as ctk
from tkinter import messagebox, simpledialog
import tkinter as tk
from tkinter import ttk
from datetime import datetime

class DeviceManagementMixin:
    """
    Mixin class to add device management features (Treeview table renderer)
    """

    # ==================== STATE ====================

    def _ensure_db_tab_vars(self):
        """Initialize variables/state for the Database tab (lazy init)."""
        if getattr(self, "_db_ui_inited", False):
            return

        self._db_state = {
            "query": "",
            "status": "All",  # All, in_progress, complete, failed
        }

        # Tk variables for bindings
        self.db_search_var = tk.StringVar(value="")
        self.db_status_filter = tk.StringVar(value="All")
        self.db_multi_sort_var = tk.BooleanVar(value=False)
        self._db_search_after = None     # debounce handle
        self._rebuild_after = None       # rebuild coalescing
        self.db_fast_mode_var = tk.BooleanVar(value=True)  # fast mode: skip heavy metrics initially


        # Multi-column sort keys: list of (column_key, descending_bool)
        self._tree_sort_keys = []  # e.g., [("completion", True), ("scans", False)]

        # Row index: tree iid -> device object (or ids)
        self._tree_index = {}

        self._db_ui_inited = True

    # ==================== MAIN ENTRY ====================

    def create_enhanced_database_tab(self):
        """Enhanced database tab using ttk.Treeview with multi-column sorting."""
        self._ensure_db_tab_vars()

        # Clear existing container
        for w in self.db_container.winfo_children():
            w.destroy()

        # Header
        header = ctk.CTkFrame(self.db_container, fg_color="transparent")
        header.pack(fill="x", pady=5, padx=10)

        ctk.CTkLabel(
            header, text="Database Manager (Table View)", font=ctk.CTkFont(size=20, weight="bold")
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            header, text="🔄 Refresh", command=self.refresh_database_view, width=100, height=30
        ).pack(side="right", padx=5)

        # Toolbar
        toolbar = ctk.CTkFrame(self.db_container, corner_radius=10)
        toolbar.pack(fill="x", padx=10, pady=(0,10))

        row = ctk.CTkFrame(toolbar, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=8)

        # Search
        search_frame = ctk.CTkFrame(row, fg_color="transparent")
        search_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(search_frame, text="Search", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        self.db_search_entry = ctk.CTkEntry(
            search_frame, textvariable=self.db_search_var, placeholder_text="Search samples or devices..."
        )
        self.db_search_entry.pack(fill="x", padx=(0, 10))

        def on_search_change(*_):
            self._db_state["query"] = (self.db_search_var.get() or "").strip()
            if self._db_search_after:
                self.root.after_cancel(self._db_search_after)
            # Debounce for 200 ms
            self._db_search_after = self.root.after(200, self._rebuild_tree)


        self.db_search_var.trace_add("write", on_search_change)

        # Status filter
        status_frame = ctk.CTkFrame(row, fg_color="transparent")
        status_frame.pack(side="left", padx=10)
        ctk.CTkLabel(status_frame, text="Device Status", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        ctk.CTkOptionMenu(
            status_frame,
            values=["All", "in_progress", "complete", "failed"],
            variable=self.db_status_filter,
            command=lambda *_: self._on_status_changed()
        ).pack()

        # Multi-sort + Clear sort
        sort_frame = ctk.CTkFrame(row, fg_color="transparent")
        sort_frame.pack(side="left", padx=10)
        ctk.CTkLabel(sort_frame, text="Sorting", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        ctk.CTkCheckBox(
            sort_frame, text="Multi-sort", variable=self.db_multi_sort_var
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            sort_frame, text="Clear Sort", width=100, command=self._clear_sort
        ).pack(side="left", padx=6)

        # After the sort_frame in toolbar row
        actions_frame = ctk.CTkFrame(row, fg_color="transparent")
        actions_frame.pack(side="right")

        ctk.CTkCheckBox(
            actions_frame, text="Smooth Mode (fast)", variable=self.db_fast_mode_var,
            command=lambda: self._rebuild_tree()
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            actions_frame, text="Edit Selected…", width=160, command=self._toolbar_edit_selected
        ).pack(side="left", padx=6)



        # Table container
        table_wrap = ctk.CTkFrame(self.db_container, corner_radius=10)
        table_wrap.pack(fill="both", expand=True, padx=10, pady=10)

        # Create style for dark mode
        self._init_treeview_style()

        # Treeview columns
        self._tree_columns = [
            ("sample", "Sample"),
            ("device", "Device"),
            ("status", "Status"),
            ("completion", "Completion"),
            ("steps", "Steps"),
            ("scans", "Scans"),
            ("start", "Start"),
            ("end", "End"),
            ("operator", "Operator"),
        ]

        cols = [c[0] for c in self._tree_columns]
        self._tree = ttk.Treeview(
            table_wrap,
            columns=cols,
            show="headings",
            style="Dark.Treeview",
            selectmode="extended"
        )

        # Headings with sort commands
        for key, text in self._tree_columns:
            self._tree.heading(key, text=text, command=lambda k=key: self._on_tree_heading_click(k))

        # Column widths / anchors
        self._tree.column("sample", width=260, anchor="w")
        self._tree.column("device", width=260, anchor="w")
        self._tree.column("status", width=130, anchor="center")
        self._tree.column("completion", width=130, anchor="center")
        self._tree.column("steps", width=120, anchor="center")
        self._tree.column("scans", width=80, anchor="center")
        self._tree.column("start", width=120, anchor="center")
        self._tree.column("end", width=120, anchor="center")
        self._tree.column("operator", width=120, anchor="w")

        # Scrollbars
        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(table_wrap, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscroll=vsb.set, xscroll=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_wrap.grid_rowconfigure(0, weight=1)
        table_wrap.grid_columnconfigure(0, weight=1)

        # Bindings
        self._tree.bind("<Double-1>", self._on_tree_double_click)
        self._tree.bind("<Button-3>", self._on_tree_right_click)  # context menu
        self._tree.bind("<Delete>", self._on_tree_delete_key)

        # Context menu
        self._tree_menu = tk.Menu(self._tree, tearoff=0)
        self._tree_menu.add_command(label="Select", command=self._ctx_select)
        self._tree_menu.add_separator()
        self._tree_menu.add_command(label="View Report", command=self._ctx_view_report)  
        self._tree_menu.add_command(label="View BMR…", command=self._ctx_view_bmr) 
        self._tree_menu.add_command(label="Thermal Budget…", command=self._ctx_view_thermal_budget)
        self._tree_menu.add_command(label="Edit…", command=self._ctx_edit)
        self._tree_menu.add_command(label="Rename Device", command=self._ctx_rename)
        self._tree_menu.add_command(label="Delete Device", command=self._ctx_delete)


        # Initial fill
        self._rebuild_tree()

    # ==================== STYLE ====================
    def _toolbar_edit_selected(self):
        iids = self._tree.selection()
        if not iids:
            messagebox.showinfo("Edit Device", "Select a device row first.")
            return
        dev = self._tree_index.get(iids[0])
        if dev:
            self._open_device_edit_dialog(dev)

    def _ctx_edit(self):
        iids = self._tree.selection()
        if not iids:
            return
        dev = self._tree_index.get(iids[0])
        if dev:
            self._open_device_edit_dialog(dev)
    
    def _ctx_view_thermal_budget(self):
        """Open the Thermal Budget viewer for the selected device's sample."""
        iids = self._tree.selection()
        if not iids:
            return
        dev = self._tree_index.get(iids[0])
        if not dev:
            return
        try:
            # Reuse the existing viewer that works at the sample level
            self.show_thermal_budget_for_sample(dev.sample)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open thermal budget:\n{e}")
        
    
    def _ctx_view_report(self):
        """Open the HTML fabrication report for the selected device"""
        iids = self._tree.selection()
        if not iids:
            return
        dev = self._tree_index.get(iids[0])
        if not dev:
            return
        
        if dev.report_path:
            import os
            import webbrowser
            report_path = os.path.abspath(dev.report_path)
            if os.path.exists(report_path):
                webbrowser.open(f'file://{report_path}')
            else:
                messagebox.showwarning("Not Found", f"Report file not found:\n{dev.report_path}")
        else:
            messagebox.showinfo("No Report", "No fabrication report has been generated for this device yet.")

    def _open_device_edit_dialog(self, device):
        """Modal dialog to edit device fields."""
        dlg = ctk.CTkToplevel(self.root)
        dlg.title(f"Edit Device — {device.device_name}")
        dlg.geometry("520x380")
        dlg.grab_set()

        frame = ctk.CTkFrame(dlg, corner_radius=10)
        frame.pack(fill="both", expand=True, padx=16, pady=16)

        # Title
        ctk.CTkLabel(frame, text="Edit Device", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(8, 12))

        # Grid container
        grid = ctk.CTkFrame(frame)
        grid.pack(fill="x", padx=6)

        # Fields
        name_var = tk.StringVar(value=device.device_name or "")
        oper_var = tk.StringVar(value=device.operator or "")
        status_var = tk.StringVar(value=(device.overall_status or "in_progress"))
        start_var = tk.StringVar(value=self._fmt_dt(device.fabrication_start) if device.fabrication_start else "")
        end_var   = tk.StringVar(value=self._fmt_dt(device.fabrication_end) if device.fabrication_end else "")

        def labeled(row, label, widget):
            ctk.CTkLabel(grid, text=label, width=140, anchor="w").grid(row=row, column=0, sticky="w", padx=4, pady=6)
            widget.grid(row=row, column=1, sticky="ew", padx=4, pady=6)

        # Device Name
        name_entry = ctk.CTkEntry(grid, textvariable=name_var, height=34, placeholder_text="Device name")
        labeled(0, "Device Name", name_entry)

        # Operator
        oper_entry = ctk.CTkEntry(grid, textvariable=oper_var, height=34, placeholder_text="Operator")
        labeled(1, "Operator", oper_entry)

        # Status
        status_menu = ctk.CTkOptionMenu(grid, values=["in_progress","complete","failed"], variable=status_var)
        labeled(2, "Status", status_menu)

        # Start Date
        start_row = ctk.CTkFrame(grid, fg_color="transparent")
        start_entry = ctk.CTkEntry(start_row, textvariable=start_var, height=34, placeholder_text="YYYY-MM-DD (blank to clear)")
        start_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(start_row, text="Today", width=60, command=lambda: start_var.set(datetime.now().strftime("%Y-%m-%d"))).pack(side="left", padx=6)
        labeled(3, "Start Date", start_row)

        # End Date
        end_row = ctk.CTkFrame(grid, fg_color="transparent")
        end_entry = ctk.CTkEntry(end_row, textvariable=end_var, height=34, placeholder_text="YYYY-MM-DD (blank to clear)")
        end_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(end_row, text="Today", width=60, command=lambda: end_var.set(datetime.now().strftime("%Y-%m-%d"))).pack(side="left", padx=6)
        labeled(4, "End Date", end_row)

        grid.grid_columnconfigure(1, weight=1)

        # Hints
        ctk.CTkLabel(
            frame,
            text="Tip: Use YYYY-MM-DD for dates. Leave blank to clear.",
            text_color="gray"
        ).pack(anchor="w", padx=6, pady=(6, 4))

        # Buttons
        btns = ctk.CTkFrame(frame, fg_color="transparent")
        btns.pack(fill="x", pady=(12, 6))
        ctk.CTkButton(btns, text="Cancel", command=dlg.destroy, width=100).pack(side="right", padx=6)
        def save():
            try:
                new_name = (name_var.get() or "").strip()
                operator = (oper_var.get() or "").strip()
                status   = (status_var.get() or "in_progress").strip()
                start_dt = self._parse_date_or_none(start_var.get())
                end_dt   = self._parse_date_or_none(end_var.get())

                # 1) Rename if changed
                if new_name and new_name != device.device_name:
                    self.db_ops.rename_device(device.device_id, new_name)

                # 2) Update other fields
                self.db_ops.update_device(
                    device.device_id,
                    operator=operator if operator else None,
                    overall_status=status,
                    fabrication_start=start_dt,
                    fabrication_end=end_dt,
                )

                messagebox.showinfo("Saved", "Device updated successfully.")
                dlg.destroy()
                self._rebuild_tree()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save changes:\n{e}")

        ctk.CTkButton(btns, text="Save", command=save, width=120, fg_color="green", hover_color="darkgreen").pack(side="right", padx=6)

    def _ctx_view_bmr(self):
        """Open the BMR for the selected device.
    
        Preference order:
          1) Open linked PDF if present and exists
          2) Load in BMR tab UI (most recent in-progress else most recent)
        """
        iids = self._tree.selection()
        if not iids:
            return
        dev = self._tree_index.get(iids[0])
        if not dev:
            return
    
        try:
            from stm_fab.db.models import BatchManufacturingRecord
            import os
            import webbrowser
    
            # Get BMRs for the device (most recent first)
            bmrs = (self.db_session.query(BatchManufacturingRecord)
                    .filter_by(device_id=dev.device_id)
                    .order_by(BatchManufacturingRecord.created_at.desc())
                    .all())
    
            if not bmrs:
                messagebox.showinfo("No BMR", f"No BMRs found for device '{dev.device_name}'.")
                return
    
            # Prefer in-progress, else most recent
            bmr = next((b for b in bmrs if (b.status or "").lower() == "in_progress"), bmrs[0])
    
            # 1) If a PDF is linked and exists, open it
            pdf_path = getattr(bmr, "pdf_file_path", None)
            if pdf_path and os.path.exists(pdf_path):
                try:
                    # Use webbrowser here (works on most platforms for PDFs).
                    webbrowser.open(f"file://{os.path.abspath(pdf_path)}")
                    return
                except Exception:
                    # Fall back to BMR tab UI
                    pass
    
            # 2) Otherwise load into the GUI BMR tab (if available)
            if hasattr(self, "bmr_tab"):
                try:
                    # Bring BMR tab forward
                    # Try typical tab labels used in your app
                    for tab_name in ["📋  Batch Record", "Batch Record", "BMR"]:
                        try:
                            self.tabview.set(tab_name)
                            break
                        except Exception:
                            continue
    
                    # Use the BMR tab's loader (already implemented)
                    self.bmr_tab._load_bmr_from_database(bmr.bmr_id)
    
                    # Optional: notify which BMR was opened
                    try:
                        progress = bmr.calculate_completion()
                    except Exception:
                        progress = 0.0
                    messagebox.showinfo(
                        "BMR Loaded",
                        f"Device: {dev.device_name}\n"
                        f"Batch:  {bmr.batch_number}\n"
                        f"Status: {bmr.status}\n"
                        f"Progress: {progress:.0f}%"
                    )
                    return
                except Exception as e:
                    messagebox.showerror("BMR Error", f"Failed to open BMR in tab:\n{e}")
                    return
    
            # If we get here, we have no PDF and no BMR tab to show it
            # As a fallback, try opening the JSON if linked
            json_path = getattr(bmr, "json_file_path", None)
            if json_path and os.path.exists(json_path):
                try:
                    webbrowser.open(f"file://{os.path.abspath(json_path)}")
                    return
                except Exception:
                    pass
    
            messagebox.showinfo(
                "BMR",
                "BMR found but no PDF is linked and BMR tab is not available."
            )
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open BMR:\n{e}")
    

    def _parse_date_or_none(self, s: str):
        s = (s or "").strip()
        if not s:
            return None
        try:
            # Accept YYYY-MM-DD
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            # Try full ISO
            try:
                return datetime.fromisoformat(s)
            except Exception:
                raise ValueError(f"Invalid date format: {s} (expected YYYY-MM-DD)")


    def _init_treeview_style(self):
        """Dark theme Treeview style to blend with CustomTkinter dark mode, larger fonts."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Larger fonts for better legibility
        body_font = ("TkDefaultFont", 14)
        head_font = ("TkDefaultFont", 14, "bold")

        style.configure(
            "Dark.Treeview",
            background="#1f1f1f",
            fieldbackground="#1f1f1f",
            foreground="white",
            rowheight=38,            # taller rows
            bordercolor="#2a2a2a",
            borderwidth=0,
            font=body_font,
        )
        style.map(
            "Dark.Treeview",
            background=[("selected", "#3a86ff")],
            foreground=[("selected", "white")],
        )
        style.configure(
            "Dark.Treeview.Heading",
            background="#252525",
            foreground="white",
            relief="flat",
            font=head_font,
            padding=8
        )
        style.map(
            "Dark.Treeview.Heading",
            background=[("active", "#333333")]
        )
        style.layout("Dark.Treeview", style.layout("Treeview"))



    # ==================== EVENTS ====================

    def _on_status_changed(self):
        self._db_state["status"] = self.db_status_filter.get()
        self._rebuild_tree()

    def _clear_sort(self):
        self._tree_sort_keys = []
        self._rebuild_tree()

    def _refresh_heading_labels(self):
        # Optional: show ▲/▼ on sorted columns
        arrows = {True: "▼", False: "▲"}
        current = {k: None for k, _ in self._tree_columns}
        for k, desc in self._tree_sort_keys:
            current[k] = arrows[desc]
        for k, text in self._tree_columns:
            arrow = f" {current[k]}" if current[k] else ""
            self._tree.heading(k, text=text + arrow)

    def _on_tree_heading_click(self, col_key: str):
        """
        Add/Toggle sort on a column.
        - If Multi-sort checkbox is off: replace the sort list with this column.
        - If on: append or flip this column in the sort list.
        """
        # Existing entry?
        existing_idx = next((i for i, (k, _) in enumerate(self._tree_sort_keys) if k == col_key), None)
        if existing_idx is not None:
            # Flip direction
            k, desc = self._tree_sort_keys[existing_idx]
            self._tree_sort_keys[existing_idx] = (k, not desc)
        else:
            if self.db_multi_sort_var.get():
                self._tree_sort_keys.append((col_key, False))  # default ascending
            else:
                self._tree_sort_keys = [(col_key, False)]

        self._refresh_heading_labels()
        self._rebuild_tree()

    def _on_tree_double_click(self, event):
        """Double-click selects the device."""
        item = self._tree.identify_row(event.y)
        if not item:
            return
        device = self._tree_index.get(item)
        if device:
            self.select_device(device)

    def _on_tree_right_click(self, event):
        iid = self._tree.identify_row(event.y)
        if iid:
            self._tree.selection_set(iid)
            try:
                self._tree_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self._tree_menu.grab_release()


    def _on_tree_delete_key(self, event):
        """Delete selected device(s) with Delete key."""
        iids = self._tree.selection()
        if not iids:
            return
        if not messagebox.askyesno("Confirm Deletion", f"Delete {len(iids)} selected device(s)?", icon="warning"):
            return
        for iid in iids:
            dev = self._tree_index.get(iid)
            if dev:
                try:
                    self.db_ops.delete_device(dev.device_id)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete {dev.device_name}:\n{e}")
        self._rebuild_tree()

    # Context menu actions
    def _ctx_select(self):
        iids = self._tree.selection()
        if not iids:
            return
        dev = self._tree_index.get(iids[0])
        if dev:
            self.select_device(dev)

    def _ctx_rename(self):
        iids = self._tree.selection()
        if not iids:
            return
        dev = self._tree_index.get(iids[0])
        if dev:
            self.rename_device_dialog(dev)
            self._rebuild_tree()

    def _ctx_delete(self):
        iids = self._tree.selection()
        if not iids:
            return
        dev = self._tree_index.get(iids[0])
        if dev:
            self.delete_device_dialog(dev)
            self._rebuild_tree()

    # ==================== DATA BUILD ====================

    def _insert_rows_chunked(self, rows, selected_ids, y0, start=0, chunk=400):
        end = min(start + chunk, len(rows))
        col_index = {k: i for i, (k, _) in enumerate(self._tree_columns)}

        for idx in range(start, end):
            r = rows[idx]
            # Display placeholders if None (fast mode)
            steps_text = (
                f"{r['steps'][0]}/{r['steps'][1]}"
                if isinstance(r['steps'][0], int) and isinstance(r['steps'][1], int)
                else "—/—"
            )
            scans_text = str(r["scans"]) if isinstance(r["scans"], int) else "—"

            values = [
                r["sample"],
                r["device"],
                self._status_label(r["status"]),
                f"{r['completion']:.0f}%",
                steps_text,
                scans_text,
                self._fmt_dt(r["start"]),
                self._fmt_dt(r["end"]),
                r["operator"],
            ]
            tag = "odd" if (idx % 2) else "even"
            iid = self._tree.insert("", "end", values=values, tags=(tag,))
            self._tree_index[iid] = r["_device_obj"]

            # Restore selection for this row if needed
            if r["_device_obj"].device_id in selected_ids:
                self._tree.selection_add(iid)

        # If more rows to insert, schedule next chunk
        if end < len(rows):
            self.root.after(1, lambda: self._insert_rows_chunked(rows, selected_ids, y0, end, chunk))
        else:
            # Restore scroll
            try:
                self._tree.yview_moveto(y0)
            except Exception:
                pass

    def _backfill_metrics(self, rows, start=0, batch=40):
        """
        Gradually compute heavy metrics (steps/scans/completion) and update rows.
        Runs in the main thread in small batches to keep UI responsive.
        """
        end = min(start + batch, len(rows))
        # Precompute column indices
        col_index = {k: i for i, (k, _) in enumerate(self._tree_columns)}

        # Build a map from device_id to iid for quick updates
        dev_to_iid = {}
        for iid, dev in self._tree_index.items():
            dev_to_iid[dev.device_id] = iid

        for r in rows[start:end]:
            d = r["_device_obj"]
            try:
                summary = self.db_ops.get_device_summary(d.device_id)
                total_steps = summary.get("total_steps") or 0
                completed_steps = summary.get("completed_steps") or 0
                total_scans = summary.get("total_scans") or 0
                completion = summary.get("completion", d.completion_percentage or 0.0) or 0.0

                # Update table row values if row is still present
                iid = dev_to_iid.get(d.device_id)
                if iid:
                    current = list(self._tree.item(iid, "values"))
                    current[col_index["completion"]] = f"{completion:.0f}%"
                    current[col_index["steps"]] = f"{completed_steps}/{total_steps}"
                    current[col_index["scans"]] = str(total_scans)
                    self._tree.item(iid, values=current)

            except Exception:
                pass

        # Schedule next batch if needed
        if end < len(rows):
            self.root.after(10, lambda: self._backfill_metrics(rows, end, batch))


    def _rebuild_tree(self):
        # Coalesce rapid calls
        if self._rebuild_after:
            self.root.after_cancel(self._rebuild_after)
        self._rebuild_after = self.root.after(1, self._rebuild_tree_now)

    def _rebuild_tree_now(self):
        self._rebuild_after = None

        # Save selection (by device_id) and scroll
        selected_ids = []
        for iid in self._tree.selection():
            dev = self._tree_index.get(iid)
            if dev:
                selected_ids.append(dev.device_id)
        try:
            y0, y1 = self._tree.yview()
        except Exception:
            y0, y1 = (0, 1)

        # Clear
        for iid in self._tree.get_children():
            self._tree.delete(iid)
        self._tree_index.clear()

        query = (self._db_state["query"] or "").lower()
        status_filter = (self._db_state["status"] or "All").lower()
        fast_mode = bool(self.db_fast_mode_var.get())

        # Build rows quickly
        rows = []
        samples = self.db_ops.list_samples()
        for sample in samples:
            devices = self.db_ops.list_devices(sample_id=sample.sample_id)
            if status_filter != "all":
                devices = [d for d in devices if (d.overall_status or "").lower() == status_filter]

            for d in devices:
                s_match = (query in (sample.sample_name or "").lower()) if query else True
                d_match = (query in (d.device_name or "").lower()) if query else True
                if query and not (s_match or d_match):
                    continue

                # FAST path: skip heavy metrics initially
                if fast_mode:
                    total_steps = completed_steps = total_scans = None
                    completion = d.completion_percentage or 0.0
                else:
                    # Careful: can be heavy for large datasets
                    try:
                        summary = self.db_ops.get_device_summary(d.device_id)
                        total_steps = summary.get("total_steps") or 0
                        completed_steps = summary.get("completed_steps") or 0
                        total_scans = summary.get("total_scans") or 0
                        completion = summary.get("completion", d.completion_percentage or 0.0) or 0.0
                    except Exception:
                        total_steps = completed_steps = total_scans = 0
                        completion = d.completion_percentage or 0.0

                rows.append({
                    "sample": sample.sample_name or "",
                    "device": (d.device_name or "") + (" 📄" if d.report_path else ""),
                    "status": (d.overall_status or "in_progress").lower(),
                    "completion": float(completion),
                    "steps": (completed_steps, total_steps),  # can be None in fast mode
                    "scans": total_scans,                     # can be None in fast mode
                    "start": d.fabrication_start,
                    "end": d.fabrication_end,
                    "operator": d.operator or "",
                    "_device_obj": d,
                })

        # Sort
        rows = self._sort_rows(rows)

        # Insert in chunks (prevents UI freeze)
        self._tree.tag_configure("odd", background="#222222")
        self._tree.tag_configure("even", background="#1c1c1c")
        self._insert_rows_chunked(rows, selected_ids, y0, chunk=400)

        # If in fast mode, backfill metrics progressively
        if fast_mode:
            self._backfill_metrics(rows, batch=40)


    def _sort_rows(self, rows):
        """Apply current multi-column sort keys to rows."""
        if not self._tree_sort_keys:
            # Default: Start desc, then Sample asc
            sort_keys = [("start", True), ("sample", False)]
        else:
            sort_keys = list(self._tree_sort_keys)

        def status_rank(s: str) -> int:
            order = {"in_progress": 0, "complete": 1, "failed": 2}
            return order.get((s or "").lower(), 99)

        # Build a tuple key for Python sort
        def make_key(r):
            key_parts = []
            for col, desc in sort_keys:
                if col == "sample":
                    v = (r["sample"] or "").lower()
                elif col == "device":
                    v = (r["device"] or "").lower()
                elif col == "status":
                    v = status_rank(r["status"])
                elif col == "completion":
                    v = float(r["completion"])
                elif col == "steps":
                    # Sort by completion ratio, fallback to totals
                    comp, tot = r["steps"]
                    v = (comp / tot) if tot > 0 else 0.0
                elif col == "scans":
                    v = int(r["scans"])
                elif col == "start":
                    v = r["start"] or datetime.min
                elif col == "end":
                    v = r["end"] or datetime.min
                elif col == "operator":
                    v = (r["operator"] or "").lower()
                else:
                    v = r.get(col)
                # Apply descending by negating or reordering
                if isinstance(v, (int, float)):
                    key_parts.append(-v if desc else v)
                else:
                    # For non-numeric, we can't negate; store (desc_flag, value) to reverse order cleanly
                    key_parts.append((1 if desc else 0, v))
            return tuple(key_parts)

        rows.sort(key=make_key)
        return rows

    # ==================== UTILS ====================

    def _fmt_dt(self, dt):
        if not dt:
            return "—"
        try:
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return str(dt)

    def _status_label(self, status: str) -> str:
        emoji = {
            "in_progress": "🔄",
            "complete": "✅",
            "failed": "❌",
        }.get((status or "").lower(), "🔬")
        return f"{emoji} {status}"
    
    # ==================== DIALOG FUNCTIONS ====================
    
    def rename_sample_dialog(self, sample):
        """Show dialog to rename a sample"""
        new_name = simpledialog.askstring(
            "Rename Sample",
            f"Enter new name for '{sample.sample_name}':",
            initialvalue=sample.sample_name
        )
        
        if not new_name or new_name == sample.sample_name:
            return
        
        try:
            self.db_ops.rename_sample(sample.sample_id, new_name)
            self.update_status(f"Renamed sample to '{new_name}'")
            self.refresh_database_view()
            messagebox.showinfo("Success", f"Sample renamed to '{new_name}'")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rename sample:\n{str(e)}")
    
    def delete_sample_dialog(self, sample):
        """Show confirmation dialog to delete a sample"""
        devices = self.db_ops.list_devices(sample_id=sample.sample_id)
        
        if devices:
            msg = (
                f"Sample '{sample.sample_name}' has {len(devices)} device(s).\n\n"
                "Deleting the sample will also delete all devices and their data.\n\n"
                "Are you sure you want to continue?"
            )
        else:
            msg = f"Are you sure you want to delete sample '{sample.sample_name}'?"
        
        if not messagebox.askyesno("Confirm Deletion", msg, icon='warning'):
            return
        
        try:
            self.db_ops.delete_sample(sample.sample_id, force=True)
            self.update_status(f"Deleted sample '{sample.sample_name}'")
            self.refresh_database_view()
            messagebox.showinfo("Success", "Sample deleted successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete sample:\n{str(e)}")
    
    def rename_device_dialog(self, device):
        """Show dialog to rename a device"""
        new_name = simpledialog.askstring(
            "Rename Device",
            f"Enter new name for '{device.device_name}':",
            initialvalue=device.device_name
        )
        
        if not new_name or new_name == device.device_name:
            return
        
        try:
            self.db_ops.rename_device(device.device_id, new_name)
            self.update_status(f"Renamed device to '{new_name}'")
            self.refresh_database_view()
            messagebox.showinfo("Success", f"Device renamed to '{new_name}'")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rename device:\n{str(e)}")
    
    def delete_device_dialog(self, device):
        """Show confirmation dialog to delete a device"""
        msg = (
            f"Are you sure you want to delete device '{device.device_name}'?\n\n"
            "This will delete all fabrication steps and STM scans for this device."
        )
        
        if not messagebox.askyesno("Confirm Deletion", msg, icon='warning'):
            return
        
        try:
            self.db_ops.delete_device(device.device_id)
            self.update_status(f"Deleted device '{device.device_name}'")
            self.refresh_database_view()
            messagebox.showinfo("Success", "Device deleted successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete device:\n{str(e)}")
    
    def select_device(self, device):
        self.current_device = device
        self.current_sample = device.sample
        self.device_name.set(device.device_name)
        self.sample_name.set(device.sample.sample_name)
    
        # NEW: auto-fill Setup and LabVIEW Analysis tab with stored paths
        try:
            self.apply_sample_paths_to_ui(self.current_sample)
        except Exception:
            pass
    
        self.update_status(f"Selected device: {device.device_name}")
        self.update_thermal_budget_display()
        
        # Switch to setup tab - try to find the correct tab name
        try:
            # Try common tab name variations
            for tab_name in ["⚙️  Setup", "Setup", "⚙ Setup", "setup", "📁  LabVIEW Files"]:
                try:
                    self.tabview.set(tab_name)
                    break
                except ValueError:
                    continue
        except Exception as e:
            # If tab switching fails, just skip it
            pass
        
        messagebox.showinfo(
            "Device Selected",
            f"Selected device '{device.device_name}'\n"
            f"Sample: {device.sample.sample_name}"
        )
    
    def refresh_database_view(self):
        """Refresh the database view"""
        self.create_enhanced_database_tab()
    
    # ==================== THERMAL BUDGET VIEWER ====================
    
    def show_thermal_budget_selector(self):
        """Show dialog to select a device and view its thermal budget"""
        # Create selection dialog
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("View Thermal Budget")
        dialog.geometry("800x600")
        
        # Main frame
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            main_frame,
            text="Select Sample/Device to View Thermal Budget",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Scrollable frame for samples
        scroll_frame = ctk.CTkScrollableFrame(main_frame, height=400)
        scroll_frame.pack(fill="both", expand=True, pady=10)
        
        # Get all samples
        samples = self.db_ops.list_samples()
        
        for sample in samples:
            sample_frame = ctk.CTkFrame(scroll_frame, corner_radius=10)
            sample_frame.pack(fill="x", pady=5, padx=5)
            
            # Sample button
            ctk.CTkButton(
                sample_frame,
                text=f"🧪 {sample.sample_name}",
                command=lambda s=sample: self.show_thermal_budget_for_sample(s, dialog),
                anchor="w",
                font=ctk.CTkFont(size=14, weight="bold"),
                fg_color="blue",
                hover_color="darkblue",
                height=40
            ).pack(fill="x", padx=10, pady=5)
            
            # Show devices under this sample
            devices = self.db_ops.list_devices(sample_id=sample.sample_id)
            if devices:
                device_container = ctk.CTkFrame(sample_frame)
                device_container.pack(fill="x", padx=30, pady=(0, 5))
                
                for device in devices:
                    ctk.CTkLabel(
                        device_container,
                        text=f"  🔬 {device.device_name}",
                        font=ctk.CTkFont(size=12),
                        anchor="w"
                    ).pack(anchor="w", padx=5, pady=2)
        
        # Close button
        ctk.CTkButton(
            main_frame,
            text="Close",
            command=dialog.destroy,
            width=100,
            height=35
        ).pack(pady=10)
    
    def show_thermal_budget_for_sample(self, sample, dialog=None):
        """Show thermal budget details for a selected sample"""
        if dialog:
            dialog.destroy()
        
        # Get thermal budget
        thermal_budget = self.db_ops.get_thermal_budget(sample.sample_id)
        
        if not thermal_budget:
            messagebox.showwarning(
                "No Data",
                f"No thermal budget data found for sample '{sample.sample_name}'"
            )
            return
        
        # Create display window
        budget_window = ctk.CTkToplevel(self.root)
        budget_window.title(f"Thermal Budget - {sample.sample_name}")
        budget_window.geometry("700x600")
        
        # Main frame
        main_frame = ctk.CTkFrame(budget_window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        ctk.CTkLabel(
            main_frame,
            text=f"Thermal Budget for {sample.sample_name}",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=10)
        
        # Status indicator
        status = thermal_budget.status()
        status_colors = {
            'normal': 'green',
            'warning': 'orange',
            'critical': 'red'
        }
        status_text = {
            'normal': '✓ NORMAL',
            'warning': '⚠ WARNING',
            'critical': '⚠️ CRITICAL'
        }
        
        status_frame = ctk.CTkFrame(
            main_frame,
            fg_color=status_colors.get(status, 'gray'),
            corner_radius=10
        )
        status_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            status_frame,
            text=status_text.get(status, 'UNKNOWN'),
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        ).pack(pady=10)
        
        # Total budget
        total_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        total_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            total_frame,
            text="Total Thermal Budget:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        ctk.CTkLabel(
            total_frame,
            text=f"{thermal_budget.total_budget:.2e} °C·s",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=5)
        
        # Breakdown
        breakdown_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        breakdown_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            breakdown_frame,
            text="Breakdown by Process:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Process contributions
        processes = [
            ('Degas', thermal_budget.degas_contribution),
            ('Flash', thermal_budget.flash_contribution),
            ('H-Termination', thermal_budget.hterm_contribution),
            ('Incorporation', thermal_budget.incorporation_contribution),
            ('Overgrowth', thermal_budget.overgrowth_contribution),
            ('Other', thermal_budget.other_contribution)
        ]
        
        for process_name, contribution in processes:
            if contribution > 0:
                percentage = (contribution / thermal_budget.total_budget * 100) if thermal_budget.total_budget > 0 else 0
                
                item_frame = ctk.CTkFrame(breakdown_frame)
                item_frame.pack(fill="x", padx=10, pady=2)
                
                ctk.CTkLabel(
                    item_frame,
                    text=f"{process_name}:",
                    font=ctk.CTkFont(size=12),
                    width=150,
                    anchor="w"
                ).pack(side="left", padx=5)
                
                ctk.CTkLabel(
                    item_frame,
                    text=f"{contribution:.2e} °C·s ({percentage:.1f}%)",
                    font=ctk.CTkFont(size=12)
                ).pack(side="left", padx=5)
        
        # Close button
        ctk.CTkButton(
            main_frame,
            text="Close",
            command=budget_window.destroy,
            width=100,
            height=35
        ).pack(pady=10)