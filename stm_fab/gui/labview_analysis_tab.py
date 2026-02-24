# gui_labview_analysis_tab.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import os
import webbrowser
from datetime import datetime
import threading
import warnings

# Suppress tkinter threading warnings (harmless cleanup warnings)
warnings.filterwarnings('ignore', message='.*main thread is not in main loop.*')

from stm_fab.analysis.labview_analysis import (
    load_labview_df, plot_flash, plot_termination, plot_susi, plot_dose, plot_outgas,
    save_figures, export_plots_for_folder
)
from stm_fab.labview.labview_parser import LabVIEWParser
from stm_fab.analysis.cooldown_analysis import CooldownAnalyzer
from stm_fab.analysis.batch_rampdown import (
    process_folder, process_file as bp_process_file, figure_per_file, generate_comparison_figures, export_summary_excel
)
from stm_fab.analysis.process_metrics import analyze_process_file
from stm_fab.analysis.batch_metrics import analyze_folder_metrics, format_batch_results, export_batch_results_excel
from stm_fab.analysis.susi_calibration import SUSICalibrationManager
from stm_fab.analysis.html_report import build_device_report
from stm_fab.scripts.convert_autoheater_to_flash_format import convert_file as convert_autoflash

PROCESS_DETECT_RULES = [
    ('Flash', lambda s: any(k in s for k in ['flash','cooldown'])),
    ('Termination', lambda s: 'term' in s),
    ('SUSI', lambda s: any(k in s for k in ['susi','ovg','overgrowth'])),
    ('Dose', lambda s: 'dose' in s),
    ('Outgas', lambda s: 'outgas' in s),
]

def detect_analysis_type(filename: str) -> str:
    s = filename.lower()
    for name, rule in PROCESS_DETECT_RULES:
        if rule(s):
            return name
    return 'Flash'  # fallback

class LabVIEWAnalysisTab:
    def __init__(self, parent_gui, tabview):
        self.gui = parent_gui
        self.tab = tabview.add("📈  LabVIEW Analysis")
        self._current_file = None
        self._plot_canvases = []
        self._batch_results = []
        self._batch_results = None
        self._build_ui()


    def _ui(self, key, default=None):
        """Pull UI constants from parent GUI if available, with safe fallbacks."""
        return getattr(self, "UI", {}).get(key, default)

    def _make_card(self, parent, title=None, subtitle=None):
        """Standard card container used across this tab."""
        card = ctk.CTkFrame(
            parent,
            corner_radius=self._ui("RADIUS", 14),
            fg_color=self._ui("CARD_BG", None),
            border_width=1,
            border_color=self._ui("CARD_BORDER", None),
        )

        if title:
            head = ctk.CTkFrame(card, fg_color="transparent")
            head.pack(fill="x", padx=self._ui("CARD_PAD", 14), pady=(self._ui("CARD_PAD", 14), 8))

            ctk.CTkLabel(head, text=title, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
            if subtitle:
                ctk.CTkLabel(
                    head,
                    text=subtitle,
                    font=ctk.CTkFont(size=11),
                    text_color=self._ui("MUTED", "gray"),
                ).pack(anchor="w", pady=(2, 0))

            body = ctk.CTkFrame(card, fg_color="transparent")
            body.pack(fill="both", expand=True, padx=self._ui("CARD_PAD", 14), pady=(0, self._ui("CARD_PAD", 14)))
            return card, body

        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=self._ui("CARD_PAD", 14), pady=self._ui("CARD_PAD", 14))
        return card, body

    def _mkbtn(self, parent, text, command, kind="secondary", width=None, height=34):
        """Semantic buttons: primary / secondary / success / danger."""
        cfg = {"height": height, "command": command, "text": text}
        if width is not None:
            cfg["width"] = width

        if kind == "primary":
            cfg.update(fg_color=self._ui("PRIMARY", None), hover_color=self._ui("PRIMARY_HOVER", None))
        elif kind == "success":
            cfg.update(fg_color=self._ui("SUCCESS", None), hover_color=self._ui("SUCCESS_HOVER", None))
        elif kind == "danger":
            cfg.update(fg_color=self._ui("DANGER", None), hover_color=self._ui("DANGER_HOVER", None))
        else:
            # secondary: let CTk theme decide
            pass

        return ctk.CTkButton(parent, **cfg)

    def _set_busy(self, busy: bool, message: str = "Working…"):
        """Enable / disable UI interaction and update global status."""
        self._is_busy = busy

        # Update header status chip (if available)
        if hasattr(self.gui, "update_status"):
            self.gui.update_status(message if busy else "Ready")

        # Cursor feedback
        try:
            self.tab.configure(cursor="watch" if busy else "")
        except Exception:
            pass

        # Disable / enable all buttons in this tab
        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child, ctk.CTkButton):
                    child.configure(state="disabled" if busy else "normal")
                walk(child)

        walk(self.tab)

        # Keep UI responsive
        try:
            self.tab.update_idletasks()
        except Exception:
            pass

    def _add_file_row(self, path: Path, file_type: str):
        """Add one file row to the scroll list with a subtle type indicator."""
        colors = {
            "Flash": "#DC2626",
            "Dose": "#2563EB",
            "SUSI": "#16A34A",
            "Termination": "#F59E0B",
            "Outgas": "#7C3AED",
        }
        accent = colors.get(file_type, "#6B7280")

        row = ctk.CTkFrame(
            self.file_list,
            corner_radius=12,
            fg_color=("white", "#0F1525"),
            border_width=1,
            border_color=self._ui("CARD_BORDER", None),
        )
        row.pack(fill="x", pady=4, padx=4)

        # Accent bar
        ctk.CTkFrame(row, width=6, fg_color=accent, corner_radius=12).pack(
            side="left", fill="y", padx=(0, 10), pady=8
        )

        mid = ctk.CTkFrame(row, fg_color="transparent")
        mid.pack(side="left", fill="x", expand=True, pady=6)

        ctk.CTkLabel(mid, text=path.name, anchor="w").pack(anchor="w")

        ctk.CTkLabel(
            mid,
            text=file_type,
            font=ctk.CTkFont(size=10),
            text_color=self._ui("MUTED", "gray"),
        ).pack(anchor="w", pady=(2, 0))

        self._mkbtn(row, "Open", lambda p=path: self._plot_file(p), kind="primary", width=70, height=30).pack(
            side="right", padx=10, pady=10
        )

    def _add_plot_tile(self, fig, title: str, subtitle: str):
        """Render a matplotlib figure inside a clean 'plot tile' card."""
        tile = ctk.CTkFrame(
            self.plot_container,
            corner_radius=self._ui("RADIUS", 14),
            fg_color=self._ui("CARD_BG", None),
            border_width=1,
            border_color=self._ui("CARD_BORDER", None),
        )
        tile.pack(fill="both", expand=True, pady=8, padx=4)

        # ---- Header ----
        header = ctk.CTkFrame(tile, fg_color="transparent")
        header.pack(fill="x", padx=self._ui("CARD_PAD", 14), pady=(self._ui("CARD_PAD", 14), 6))

        text_col = ctk.CTkFrame(header, fg_color="transparent")
        text_col.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            text_col,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w")

        ctk.CTkLabel(
            text_col,
            text=subtitle,
            font=ctk.CTkFont(size=10),
            text_color=self._ui("MUTED", "gray")
        ).pack(anchor="w", pady=(2, 0))

        # ---- Actions ----
        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.pack(side="right")

        def save_png():
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG image", "*.png")],
                initialfile=title.replace(" ", "_").lower()
            )
            if path:
                fig.savefig(path, dpi=300, bbox_inches="tight")


        def pop_out():
            win = ctk.CTkToplevel(self.tab)
            win.title(title)
            win.geometry("900x700")

            frame = ctk.CTkFrame(win)
            frame.pack(fill="both", expand=True, padx=10, pady=10)

            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        self._mkbtn(actions, "Save PNG", save_png, kind="secondary", width=90, height=28).pack(
            side="left", padx=(0, 6)
        )
        self._mkbtn(actions, "Pop-out", pop_out, kind="secondary", width=80, height=28).pack(
            side="left"
        )

        # ---- Figure ----
        canvas = FigureCanvasTkAgg(fig, master=tile)
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=self._ui("CARD_PAD", 14),
            pady=(0, self._ui("CARD_PAD", 14))
        )

        self._plot_canvases.append(canvas)

    def _build_ui(self):
        """Build a calmer, card-based UI while keeping existing workflows intact."""
        # Pull UI constants from parent GUI (keeps this tab consistent with the rest of the app)
        self.UI = getattr(self.gui, "UI", {
            "PAD": 12,
            "CARD_PAD": 14,
            "RADIUS": 14,
            "MUTED": ("#6B7280", "#9CA3AF"),
            "CARD_BG": ("#F5F6F8", "#141922"),
            "CARD_BORDER": ("#E5E7EB", "#273043"),
            "PRIMARY": ("#2563EB", "#2563EB"),
            "PRIMARY_HOVER": ("#1D4ED8", "#1D4ED8"),
            "SUCCESS": ("#16A34A", "#16A34A"),
            "SUCCESS_HOVER": ("#15803D", "#15803D"),
            "DANGER": ("#DC2626", "#DC2626"),
            "DANGER_HOVER": ("#B91C1C", "#B91C1C"),
        })

        main = ctk.CTkFrame(self.tab, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=self._ui("PAD", 12), pady=self._ui("PAD", 12))

        # ========== TOP: DATA + ACTIONS ==========
        top = ctk.CTkFrame(main, fg_color="transparent")
        top.pack(fill="x")

        top.grid_columnconfigure(0, weight=3)
        top.grid_columnconfigure(1, weight=2)

        # Data source card
        data_card, data_body = self._make_card(
            top,
            title="Data source",
        )
        data_card.grid(row=0, column=0, sticky="ew", padx=(0, self._ui("PAD", 12)), pady=(0, self._ui("PAD", 12)))

        folder_row = ctk.CTkFrame(data_body, fg_color="transparent")
        folder_row.pack(fill="x", pady=2)

        ctk.CTkLabel(folder_row, text="Folder", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")

        self.folder_var = ctk.StringVar()
        self.folder_entry = ctk.CTkEntry(
            folder_row,
            textvariable=self.folder_var,
            placeholder_text="Select folder with .txt files",
        )
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=10)

        self._mkbtn(folder_row, "Browse", self._browse, kind="secondary", width=90).pack(side="left", padx=(0, 6))
        self._mkbtn(folder_row, "Scan", self._scan, kind="primary", width=90).pack(side="left")


        # ==========================
        # Actions dropdown (menu-style)
        # ==========================

        actions_frame = ctk.CTkFrame(top, fg_color="transparent")
        actions_frame.grid(row=0, column=1, sticky="e", pady=(0, self._ui("PAD", 12)))

        def open_actions_menu():
            # --- Floating menu window ---
            menu = ctk.CTkToplevel(self.tab)
            menu.overrideredirect(True)
            menu.attributes("-topmost", True)

            body = ctk.CTkFrame(
                menu,
                corner_radius=14,
                fg_color=("white", "#1B1F2A"),
                border_width=1,
                border_color=self._ui("CARD_BORDER", None),
            )
            body.pack(fill="both", expand=True, padx=6, pady=6)

            # --- Close behaviour ---
            def close_menu(event=None):
                if menu.winfo_exists():
                    menu.destroy()

            menu.bind("<FocusOut>", close_menu)

            # --- Menu item helper ---
            def menu_item(text, command, primary=False):
                bg = ("#F4F4F5", "#1B1F2A")
                hover = ("#EDEEF0", "#27304A")
                accent = "#2563EB" if primary else None

                row = ctk.CTkFrame(
                    body,
                    fg_color=bg,
                    corner_radius=8,
                    border_width=1 if primary else 0,
                    border_color=accent,
                )
                row.pack(fill="x", pady=2)

                label = ctk.CTkLabel(
                    row,
                    text=text,
                    anchor="w",
                    padx=12,
                    height=30,
                )
                label.pack(fill="x")

                def on_enter(_):
                    row.configure(fg_color=hover)

                def on_leave(_):
                    row.configure(fg_color=bg)

                def on_click(_):
                    close_menu()
                    command()

                row.bind("<Enter>", on_enter)
                row.bind("<Leave>", on_leave)
                row.bind("<Button-1>", on_click)
                label.bind("<Button-1>", on_click)

            # -------- Menu content --------

            ctk.CTkLabel(body, text="Analysis", font=ctk.CTkFont(size=12, weight="bold")).pack(
                anchor="w", pady=(6, 4), padx=6
            )
            menu_item("Analyze current file", self._analyze_current_file_metrics)
            menu_item("Analyze folder", self._analyze_folder_metrics)

            ctk.CTkLabel(body, text="Export", font=ctk.CTkFont(size=12, weight="bold")).pack(
                anchor="w", pady=(12, 4), padx=6
            )
            menu_item("Save current plots", self._save_current_plots)
            menu_item("Export all plots", self._export_plots_folder)
            menu_item("Export HTML summary", self._analyze_and_export_html, primary=True)

            ctk.CTkLabel(body, text="Calibration", font=ctk.CTkFont(size=12, weight="bold")).pack(
                anchor="w", pady=(12, 4), padx=6
            )
            menu_item("Quick add SUSI calibration", self._quick_add_susi_cal)
            menu_item("SUSI calibrations…", self._open_susi_calibration_manager)

            # --- Positioning (clamped + auto-flip) ---
            menu.update_idletasks()

            menu_w = 300
            menu_h = body.winfo_reqheight()

            btn_x = actions_button.winfo_rootx()
            btn_y = actions_button.winfo_rooty()
            btn_h = actions_button.winfo_height()
            screen_h = menu.winfo_screenheight()

            x = btn_x
            y = btn_y + btn_h + 6

            if y + menu_h > screen_h:
                y = btn_y - menu_h - 6

            menu.geometry(f"{menu_w}x{menu_h}+{x}+{y}")
            menu.focus_force()

        # --- Actions trigger button ---
        actions_button = self._mkbtn(
            actions_frame,
            "Actions",
            open_actions_menu,
            kind="secondary",
            width=110,
            height=36,
        )
        actions_button.pack(anchor="e")


        # ========== MAIN CONTENT AREA ==========
        content = ctk.CTkFrame(main, fg_color="transparent")
        content.pack(fill="both", expand=True)

        content.grid_columnconfigure(0, weight=1)   # Files
        content.grid_columnconfigure(1, weight=3)   # Plots (hero)
        content.grid_rowconfigure(0, weight=1)

        # Files card (left)
        files_card, files_body = self._make_card(
            content,
            title="Files",
            subtitle="Scan a folder, then open a file to render plots.",
        )
        files_card.grid(row=0, column=0, sticky="nsw", padx=(0, self._ui("PAD", 12)))

        head_row = ctk.CTkFrame(files_body, fg_color="transparent")
        head_row.pack(fill="x", pady=(0, 8))

        self.file_count_label = ctk.CTkLabel(
            head_row,
            text="0 files",
            font=ctk.CTkFont(size=11),
            text_color=self._ui("MUTED", "gray"),
        )
        self.file_count_label.pack(side="left")

        self.file_list = ctk.CTkScrollableFrame(files_body, width=260)
        self.file_list.pack(fill="both", expand=True)

        # Plots card (right)
        plots_card, plots_body = self._make_card(
            content,
            title="Plots",
            subtitle="Selected file plots appear here. Use export actions above to save.",
        )
        plots_card.grid(row=0, column=1, sticky="nsew")

        plot_head = ctk.CTkFrame(plots_body, fg_color="transparent")
        plot_head.pack(fill="x", pady=(0, 8))

        self.current_file_label = ctk.CTkLabel(
            plot_head,
            text="No file selected",
            font=ctk.CTkFont(size=11),
            text_color=self._ui("MUTED", "gray"),
        )
        self.current_file_label.pack(side="left")

        self.plot_container = ctk.CTkScrollableFrame(plots_body)
        self.plot_container.pack(fill="both", expand=True)

        self._plot_canvases = []

        # ========== BATCH RAMP-DOWN (optional but preserved) ==========
        batch_card, batch_body = self._make_card(
            main,
            title="Batch ramp-down",
            subtitle="Process a folder of ramp-down files, view per-file plots, generate comparisons, and export a summary Excel.",
        )
        batch_card.pack(fill="x", pady=(self._ui("PAD", 12), 0))

        batch_top = ctk.CTkFrame(batch_body, fg_color="transparent")
        batch_top.pack(fill="x", pady=(0, 10))

        self.batch_folder_var = ctk.StringVar()
        batch_entry = ctk.CTkEntry(batch_top, textvariable=self.batch_folder_var, placeholder_text="Select ramp-down folder")
        batch_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self._mkbtn(batch_top, "Browse", self._batch_browse_folder, kind="secondary", width=90).pack(side="left", padx=(0, 6))
        self._mkbtn(batch_top, "Run", self._batch_run, kind="primary", width=70).pack(side="left", padx=(0, 6))
        self._mkbtn(batch_top, "Compare", self._batch_show_comparisons, kind="secondary", width=90).pack(side="left", padx=(0, 6))
        self._mkbtn(batch_top, "Export Excel", self._batch_export_excel, kind="secondary", width=110).pack(side="left")

        self.batch_files_frame = ctk.CTkScrollableFrame(batch_body, height=140)
        self.batch_files_frame.pack(fill="x")

    def _browse(self):
        folder = filedialog.askdirectory(title="Select LabVIEW Data Folder")
        if folder:
            self.folder_var.set(folder)

    # Add this method inside the LabVIEWAnalysisTab class

    def _quick_add_susi_cal(self):
        """Quick dialog to create a minimal SUSI calibration (Lock & LTE rates) and set it active."""
        try:
            mgr = SUSICalibrationManager()  # default DB path
        except Exception as e:
            messagebox.showerror("Calibration DB", f"Failed to open calibration DB:\n{e}")
            return

        win = ctk.CTkToplevel(self.tab)
        win.title("Quick Add SUSI Calibration")
        win.geometry("420x360")

        container = ctk.CTkFrame(win)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        fields = {}

        def add_row(label, key, placeholder="", default_val=None):
            row = ctk.CTkFrame(container)
            row.pack(fill="x", pady=6)
            ctk.CTkLabel(row, text=label, width=160, anchor="w").pack(side="left", padx=(0, 10))
            var = ctk.StringVar(value="" if default_val is None else str(default_val))
            ent = ctk.CTkEntry(row, textvariable=var, placeholder_text=placeholder)
            ent.pack(side="left", fill="x", expand=True)
            fields[key] = var

        add_row("Name", "name", "e.g. Feb 2026 Growth")
        add_row("Lock rate (ML/min)", "lock_rate", "e.g. 0.85")
        add_row("LTE rate (ML/min)", "lte_rate", "e.g. 0.55")
        add_row("Notes (optional)", "notes", "Anything helpful…")

        btns = ctk.CTkFrame(container)
        btns.pack(fill="x", pady=(18, 0))

        def save():
            name = fields["name"].get().strip()
            if not name:
                messagebox.showwarning("Missing name", "Please enter a calibration name.")
                return
            try:
                lock_rate = float(fields["lock_rate"].get().strip())
                lte_rate = float(fields["lte_rate"].get().strip())
            except Exception:
                messagebox.showwarning("Invalid rates", "Enter numeric values for Lock and LTE rates.")
                return

            notes = fields["notes"].get().strip()

            try:
                cal_id = mgr.add_calibration(
                    name=name,
                    created_at=datetime.utcnow().isoformat(),
                    lock_growth_rate_ml_per_min=lock_rate,
                    lte_growth_rate_ml_per_min=lte_rate,
                    notes=notes
                )
                mgr.set_active(cal_id)
            except Exception as e:
                messagebox.showerror("Save error", f"Failed to save calibration:\n{e}")
                return

            try:
                self._refresh_susi_cal_list()
            except Exception:
                pass

            messagebox.showinfo("Saved", f"Calibration saved and set active:\n{name}")
            win.destroy()

        ctk.CTkButton(btns, text="Cancel", command=win.destroy).pack(side="right")
        ctk.CTkButton(btns, text="Save", command=save).pack(side="right", padx=8)


    def _scan(self):
        """Scan folder for .txt files, auto-convert autoflash files, and display all files."""
        self._set_busy(True, "Scanning folder…")
        try:
            # Clear existing rows
            for w in self.file_list.winfo_children():
                w.destroy()

            folder = self.folder_var.get().strip()
            if not folder:
                messagebox.showwarning("No Folder", "Please select a folder first")
                return

            folder_path = Path(folder)

            all_txt_files = list(folder_path.glob("*.txt"))
            if not all_txt_files:
                messagebox.showinfo("No Files", "No .txt files found")
                return

            # Auto-flash conversion
            autoflash_files = [f for f in all_txt_files if "auto" in f.name.lower()]
            if autoflash_files:
                conversion_results = self._convert_autoflash_files(autoflash_files)
                if conversion_results["success"]:
                    msg = f"✓ Converted {conversion_results['success']} autoflash file(s)\n\n"
                    for old, new in conversion_results["conversions"]:
                        msg += f"  {old.name} → {new.name}\n"
                    messagebox.showinfo("Auto-Flash Conversion", msg)

            files = sorted(folder_path.glob("*.txt"), key=lambda p: p.name.lower())
            self.file_count_label.configure(text=f"{len(files)} files")

            for p in files:
                file_type = detect_analysis_type(p.name)
                self._add_file_row(p, file_type)

        except Exception as e:
            messagebox.showerror("Scan Error", f"Failed to scan folder:\n{e}")

        finally:
            self._set_busy(False)
            
    def _convert_autoflash_files(self, autoflash_files):
        """Convert autoheater/autoflash files to standard flash format."""
        results = {'success': 0, 'conversions': [], 'error': None}
        
        try:
            for file_path in autoflash_files:
                # Create new filename with 'flash' instead of 'auto'
                new_name = file_path.name.lower().replace('auto', 'flash')
                new_path = file_path.parent / new_name
                
                # Skip if already converted
                if new_path.exists():
                    continue
                
                # Convert file
                convert_autoflash(str(file_path), str(new_path))
                results['success'] += 1
                results['conversions'].append((file_path, new_path))
                
        except Exception as e:
            results['error'] = str(e)
            
        return results

    def _open_susi_calibration_manager(self):
        """Open a dialog window to manage SUSI calibrations (create, set active, delete, import/export)."""
        win = ctk.CTkToplevel(self.tab)
        win.title("SUSI Calibration Manager")
        win.geometry("820x520")

        try:
            self._susi_mgr = SUSICalibrationManager()
        except Exception as e:
            messagebox.showerror("Calibration DB", f"Failed to open calibration DB:\n{e}")
            win.destroy()
            return

        layout = ctk.CTkFrame(win)
        layout.pack(fill="both", expand=True, padx=12, pady=12)
        layout.grid_columnconfigure(0, weight=1)
        layout.grid_columnconfigure(1, weight=1)
        layout.grid_rowconfigure(1, weight=1)

        # Header
        ctk.CTkLabel(layout, text="SUSI Calibrations", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        # Left: list
        left = ctk.CTkFrame(layout)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(left, text="Saved calibrations", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 6)
        )
        self._susi_list = ctk.CTkScrollableFrame(left)
        self._susi_list.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Right: editor
        right = ctk.CTkFrame(layout)
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)

        form = ctk.CTkFrame(right)
        form.pack(fill="both", expand=True, padx=10, pady=10)

        fields = {}

        def add_row(label, key, placeholder="", default_val=""):
            row = ctk.CTkFrame(form)
            row.pack(fill="x", pady=6)
            ctk.CTkLabel(row, text=label, width=220, anchor="w").pack(side="left", padx=(0, 10))
            var = ctk.StringVar(value=default_val)
            ent = ctk.CTkEntry(row, textvariable=var, placeholder_text=placeholder)
            ent.pack(side="left", fill="x", expand=True)
            fields[key] = var

        add_row("Name", "name", "e.g. Feb 2026 Growth")
        add_row("Lock growth rate (ML/min)", "lock_rate", "e.g. 0.85")
        add_row("LTE growth rate (ML/min)", "lte_rate", "e.g. 0.55")
        add_row("Notes", "notes", "optional", "")

        buttons = ctk.CTkFrame(form)
        buttons.pack(fill="x", pady=(18, 0))

        self._active_id_var = ctk.StringVar(value="")

        def refresh():
            self._refresh_susi_cal_list()

        def create():
            self._add_susi_calibration(fields)

        def export():
            self._export_susi_calibrations()

        def imp():
            self._import_susi_calibrations()

        ctk.CTkButton(buttons, text="Create / Save", command=create).pack(side="left", padx=(0, 8))
        ctk.CTkButton(buttons, text="Export…", command=export).pack(side="left", padx=(0, 8))
        ctk.CTkButton(buttons, text="Import…", command=imp).pack(side="left", padx=(0, 8))
        ctk.CTkButton(buttons, text="Close", command=win.destroy).pack(side="right")

        self._susi_fields = fields
        refresh()

    def _refresh_susi_cal_list(self):
        """Refresh calibration list UI."""
        if not hasattr(self, "_susi_mgr"):
            return

        for w in self._susi_list.winfo_children():
            w.destroy()

        try:
            cals = self._susi_mgr.list_calibrations()
            active_id = self._susi_mgr.get_active_id()
        except Exception as e:
            messagebox.showerror("Calibration DB", f"Failed to list calibrations:\n{e}")
            return

        if not cals:
            ctk.CTkLabel(self._susi_list, text="No calibrations saved yet.").pack(padx=10, pady=10, anchor="w")
            return

        for cal in cals:
            cal_id = cal["id"]
            name = cal.get("name", f"Calibration {cal_id}")
            is_active = (active_id == cal_id)

            row = ctk.CTkFrame(self._susi_list)
            row.pack(fill="x", padx=4, pady=3)

            label = f"✅ {name}" if is_active else name
            ctk.CTkLabel(row, text=label, anchor="w").pack(side="left", padx=8, pady=6, fill="x", expand=True)

            ctk.CTkButton(
                row, text="Set active", width=90,
                command=lambda cid=cal_id: self._set_active_susi_cal(cid)
            ).pack(side="right", padx=(6, 6), pady=6)

            ctk.CTkButton(
                row, text="Delete", width=70, fg_color="#cc4444", hover_color="#aa3333",
                command=lambda cid=cal_id: self._delete_susi_cal(cid)
            ).pack(side="right", pady=6)

    def _set_active_susi_cal(self, cal_id: int):
        try:
            self._susi_mgr.set_active(cal_id)
            self._refresh_susi_cal_list()
            messagebox.showinfo("Active calibration", "Active SUSI calibration updated.")
        except Exception as e:
            messagebox.showerror("Calibration DB", f"Failed to set active calibration:\n{e}")

    def _delete_susi_cal(self, cal_id: int):
        if not messagebox.askyesno("Delete calibration", "Delete this calibration? This cannot be undone."):
            return
        try:
            self._susi_mgr.delete_calibration(cal_id)
            self._refresh_susi_cal_list()
        except Exception as e:
            messagebox.showerror("Calibration DB", f"Failed to delete calibration:\n{e}")

    def _add_susi_calibration(self, fields):
        """Create/save a SUSI calibration."""
        name = fields["name"].get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Please enter a calibration name.")
            return

        try:
            lock_rate = float(fields["lock_rate"].get().strip())
            lte_rate = float(fields["lte_rate"].get().strip())
        except Exception:
            messagebox.showwarning("Invalid rates", "Enter numeric values for Lock and LTE rates.")
            return

        notes = fields["notes"].get().strip()

        try:
            cal_id = self._susi_mgr.add_calibration(
                name=name,
                created_at=datetime.utcnow().isoformat(),
                lock_growth_rate_ml_per_min=lock_rate,
                lte_growth_rate_ml_per_min=lte_rate,
                notes=notes
            )
            self._susi_mgr.set_active(cal_id)
            self._refresh_susi_cal_list()
            messagebox.showinfo("Saved", f"Calibration saved and set active:\n{name}")
        except Exception as e:
            messagebox.showerror("Calibration DB", f"Failed to save calibration:\n{e}")

    def _export_susi_calibrations(self):
        """Export all calibrations to a JSON file."""
        save_path = filedialog.asksaveasfilename(
            title="Export SUSI calibrations",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="susi_calibrations.json"
        )
        if not save_path:
            return
        try:
            self._susi_mgr.export_to_json(save_path)
            messagebox.showinfo("Exported", f"Saved: {save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{e}")

    def _import_susi_calibrations(self):
        """Import calibrations from a JSON file."""
        load_path = filedialog.askopenfilename(
            title="Import SUSI calibrations",
            filetypes=[("JSON files", "*.json")],
        )
        if not load_path:
            return
        try:
            count = self._susi_mgr.import_from_json(load_path)
            self._refresh_susi_cal_list()
            messagebox.showinfo("Imported", f"Imported {count} calibration(s).")
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import:\n{e}")

    def _clear_plots(self):
        for canvas in self._plot_canvases:
            try:
                canvas.get_tk_widget().destroy()
            except Exception:
                pass
        self._plot_canvases = []
        for w in self.plot_container.winfo_children():
            w.destroy()

    def _plot_file(self, path: Path):
        """Plot a file using plot tiles instead of raw canvases."""
        self._set_busy(True, f"Rendering {path.name}…")
        try:
            self._current_file = path
            self.current_file_label.configure(text=path.name)
            self._clear_plots()

            file_type = detect_analysis_type(path.name)
            df = load_labview_df(path)

            if file_type == "Flash":
                figs = plot_flash(df)
                titles = ["Flash temperature ramp", "Heater current / power"]
            elif file_type == "Termination":
                figs = plot_termination(df)
                titles = ["Termination sequence"]
            elif file_type == "SUSI":
                figs = plot_susi(df, auto_detect_thresholds=self.auto_detect_var.get())
                titles = ["SUSI growth", "Growth rate", "Overgrowth detection"]
            elif file_type == "Dose":
                figs = plot_dose(df)
                titles = ["Dose profile"]
            elif file_type == "Outgas":
                figs = plot_outgas(df)
                titles = ["Outgassing profile"]
            else:
                figs = plot_flash(df)
                titles = ["Process plot"]

            for i, fig in enumerate(figs):
                title = titles[i] if i < len(titles) else f"{file_type} plot {i+1}"
                subtitle = f"{path.name} • {file_type}"
                self._add_plot_tile(fig, title, subtitle)

        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot file:\n{e}")

        finally:
            self._set_busy(False)

        
    def _show_selected(self, path: Path):
        self._plot_file(path)

    def _save_current_plots(self):
        """Save plots currently displayed."""
        if not self._current_file:
            messagebox.showwarning("No File", "Select a file first.")
            return
        save_dir = filedialog.askdirectory(title="Select output folder")
        if not save_dir:
            return
        try:
            df = load_labview_df(self._current_file)
            file_type = detect_analysis_type(self._current_file.name)

            if file_type == "Flash":
                figs = plot_flash(df)
            elif file_type == "Termination":
                figs = plot_termination(df)
            elif file_type == "SUSI":
                figs = plot_susi(df, auto_detect_thresholds=self.auto_detect_var.get())
            elif file_type == "Dose":
                figs = plot_dose(df)
            elif file_type == "Outgas":
                figs = plot_outgas(df)
            else:
                figs = plot_flash(df)

            save_figures(figs, save_dir, prefix=self._current_file.stem)
            messagebox.showinfo("Saved", f"Saved figures to:\n{save_dir}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save plots:\n{e}")

    def _export_plots_folder(self):
        """Export plots for every file in selected folder."""
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No Folder", "Select a folder first.")
            return

        out_dir = filedialog.askdirectory(title="Select output folder for plots")
        if not out_dir:
            return

        try:
            export_plots_for_folder(folder, out_dir, auto_detect_thresholds=self.auto_detect_var.get())
            messagebox.showinfo("Exported", f"Exported plots to:\n{out_dir}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export plots:\n{e}")

    def _analyze_current_file_metrics(self):
        if not self._current_file:
            messagebox.showwarning("No File", "Select a file first.")
            return

        try:
            metrics = analyze_process_file(self._current_file)
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze file:\n{e}")
            return

        txt = self._format_metrics_for_display(metrics)

        win = ctk.CTkToplevel(self.tab)
        win.title("File Metrics")
        win.geometry("780x540")

        body = ctk.CTkFrame(win)
        body.pack(fill="both", expand=True, padx=12, pady=12)

        box = ctk.CTkTextbox(body, wrap="word")
        box.pack(fill="both", expand=True)
        box.insert("1.0", txt)
        box.configure(state="disabled")

        btn_row = ctk.CTkFrame(body)
        btn_row.pack(fill="x", pady=(10, 0))

        def save_to_excel():
            save_path = filedialog.asksaveasfilename(
                title="Save metrics to Excel",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                initialfile=f"{self._current_file.stem}_metrics.xlsx"
            )
            if not save_path:
                return
            try:
                export_batch_results_excel([metrics], save_path)
                messagebox.showinfo("Saved", f"Saved:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save Excel:\n{e}")

        def save_to_text():
            save_path = filedialog.asksaveasfilename(
                title="Save metrics to text",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                initialfile=f"{self._current_file.stem}_metrics.txt"
            )
            if not save_path:
                return
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(txt)
                messagebox.showinfo("Saved", f"Saved:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save text:\n{e}")

        ctk.CTkButton(btn_row, text="Save as Excel", command=save_to_excel).pack(side="left")
        ctk.CTkButton(btn_row, text="Save as text", command=save_to_text).pack(side="left", padx=8)
        ctk.CTkButton(btn_row, text="Close", command=win.destroy).pack(side="right")

    def _analyze_folder_metrics(self):
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No Folder", "Select a folder first.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save metrics summary",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="folder_metrics.xlsx"
        )
        if not save_path:
            return

        try:
            results = analyze_folder_metrics(folder)
            export_batch_results_excel(results, save_path)
            messagebox.showinfo("Saved", f"Saved:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed:\n{e}")

    def _format_metrics_for_display(self, metrics_dict):
        """Format a single metrics dict into a readable text block."""
        if not isinstance(metrics_dict, dict):
            return str(metrics_dict)

        lines = []
        lines.append("=" * 80)
        lines.append("PROCESS METRICS")
        lines.append("=" * 80)
        lines.append("")

        # Try to show the most common keys in a tidy way
        common_keys = [
            ("file", "File"),
            ("process_type", "Process type"),
            ("timestamp", "Timestamp"),
            ("duration_s", "Duration"),
            ("success", "Success"),
        ]
        for k, label in common_keys:
            if k in metrics_dict:
                v = metrics_dict[k]
                if k == "duration_s" and isinstance(v, (int, float)):
                    v = self.fmt_hms(v)
                lines.append(f"{label:<20}: {v}")

        lines.append("")
        lines.append("-" * 80)

        # SUSI-specific sections
        if metrics_dict.get("process_type", "").lower() in ("susi", "overgrowth"):
            lines.append("SUSI METRICS")
            lines.append("-" * 80)

            # Overgrowth detected?
            det = metrics_dict.get("overgrowth_detected", None)
            if det is not None:
                lines.append(f"Overgrowth detected: {'✓ Yes' if det else '✗ No'}")

            # If auto detection enabled
            if "overgrowth_thresholds" in metrics_dict:
                lines.append("")
                lines.append("Detected thresholds")
                thr = metrics_dict["overgrowth_thresholds"]
                for kk, vv in thr.items():
                    lines.append(f"  {kk:<24}: {vv}")

            # Deposition / rates
            for sec_name in ("lock", "lte"):
                key_prefix = f"{sec_name}_"
                ml_key = key_prefix + "deposition_ml"
                nm_key = key_prefix + "deposition_nm"
                rate_key = key_prefix + "growth_rate_ml_per_min"
                if ml_key in metrics_dict or nm_key in metrics_dict or rate_key in metrics_dict:
                    lines.append("")
                    lines.append(f"{sec_name.upper()} section")
                    ml_val = metrics_dict.get(ml_key, None)
                    nm_val = metrics_dict.get(nm_key, None)
                    rate_val = metrics_dict.get(rate_key, None)

                    if ml_val is not None:
                        nm_str = f" ({nm_val:.2f} nm)" if isinstance(nm_val, (int, float)) else ""
                        rate_str = f" @ {rate_val:.2f} ML/min" if rate_val is not None else ""
                        lines.append(f"    Deposition:         {ml_val:.1f} ML{nm_str}{rate_str}")
                    else:
                        lines.append("    Status:             ✗ Not detected")
                    lines.append("")

        else:
            lines.append("GENERIC METRICS")
            lines.append("-" * 80)
            import json
            lines.append(json.dumps(metrics_dict, indent=2))

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    def fmt_hms(self, seconds):
        try:
            seconds = float(seconds)
        except Exception:
            return str(seconds)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        if h > 0:
            return f"{h:d}h {m:02d}m {s:04.1f}s"
        if m > 0:
            return f"{m:d}m {s:04.1f}s"
        return f"{s:.1f}s"

    def _batch_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Ramp-Down Folder")
        if folder:
            self.batch_folder_var.set(folder)

    def _embed_figs(self, figs):
        """Embed figures using the same plot tile system."""
        self._clear_plots()
        for i, fig in enumerate(figs):
            title = f"Batch comparison {i+1}"
            subtitle = "Ramp-down analysis"
            self._add_plot_tile(fig, title, subtitle)

    def _batch_run(self):
        folder = self.batch_folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No Folder", "Select a folder first")
            return
        try:
            self._batch_results = process_folder(folder)
            for w in self.batch_files_frame.winfo_children():
                w.destroy()
            ok = 0
            for res in self._batch_results:
                btn = ctk.CTkButton(self.batch_files_frame, text=f"📄 {res['filename']}",
                                    anchor="w",
                                    command=lambda r=res: self._batch_show_per_file(r),
                                    height=28)
                btn.pack(fill="x", padx=3, pady=2)
                if res.get('success'):
                    ok += 1
            messagebox.showinfo("Done", f"Processed {len(self._batch_results)} files\nSuccessful: {ok}")
        except Exception as e:
            messagebox.showerror("Error", f"Batch analysis failed:\n{e}")

    def _batch_show_per_file(self, res):
        try:
            if not res.get('success'):
                raise ValueError(res.get('error', 'Unknown error'))
            fig = figure_per_file(res)
            self._embed_figs([fig])
        except Exception as e:
            messagebox.showerror("Plot Error", f"Unable to plot file:\n{e}")

    def _batch_show_comparisons(self):
        if not self._batch_results:
            messagebox.showwarning("No Results", "Run analysis first")
            return
        try:
            figs_dict = generate_comparison_figures(self._batch_results)
            figs = list(figs_dict.values())
            self._embed_figs(figs)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate comparison plots:\n{e}")

    def _batch_export_excel(self):
        if not self._batch_results:
            messagebox.showwarning("No Results", "Run analysis first")
            return
        save_path = filedialog.asksaveasfilename(title="Save Summary Excel",
                                                 defaultextension=".xlsx",
                                                 filetypes=[("Excel files","*.xlsx")],
                                                 initialfile="analysis_summary.xlsx")
        if not save_path:
            return
        try:
            export_summary_excel(self._batch_results, save_path)
            messagebox.showinfo("Exported", f"Saved: {save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export Excel:\n{e}")
            
    # Add this import at the top of labview_analysis_tab.py with your other imports
    # 

    def _analyze_and_export_html(self):
        """Run batch analysis, generate per-file plots, build a single
        HTML report, and open it in a browser."""
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return

        # Choose output HTML file
        default_name = f"labview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        out_path = filedialog.asksaveasfilename(
            title="Save HTML report",
            defaultextension=".html",
            filetypes=[("HTML files","*.html")],
            initialfile=default_name
        )
        if not out_path:
            return

        try:
            # 1) Run analysis over folder (your existing metrics)
            batch_results = analyze_folder_metrics(folder)

            # 2) Generate plots for each file into a folder next to the report
            out_dir = Path(out_path).with_suffix("")
            out_dir.mkdir(parents=True, exist_ok=True)

            export_plots_for_folder(folder, str(out_dir), auto_detect_thresholds=self.auto_detect_var.get())

            # 3) Build report content (uses your report builder)
            html = build_device_report(
                title="LabVIEW Process Summary",
                batch_results=batch_results,
                plots_folder=str(out_dir),
                generated_at=datetime.now().isoformat()
            )

            # 4) Save HTML
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)

            # 5) Open in browser
            webbrowser.open_new_tab(f"file://{os.path.abspath(out_path)}")

            messagebox.showinfo("Report saved", f"Saved HTML report:\n{out_path}")

        except Exception as e:
            messagebox.showerror("HTML report error", f"Failed to generate report:\n{e}")