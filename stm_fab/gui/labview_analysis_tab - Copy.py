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
        self._build_ui()

    def _build_ui(self):
        main = ctk.CTkFrame(self.tab)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # ========== HEADER ==========
        header = ctk.CTkLabel(main, text="LabVIEW Analysis & Export", 
                             font=ctk.CTkFont(size=24, weight="bold"))
        header.pack(pady=(5, 10))

        # ========== FOLDER SELECTION ==========
        folder_frame = ctk.CTkFrame(main)
        folder_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(folder_frame, text="📁 Data Folder:", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        self.folder_var = ctk.StringVar()
        self.folder_entry = ctk.CTkEntry(folder_frame, textvariable=self.folder_var, 
                                         placeholder_text="Select folder with .txt files")
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(folder_frame, text="Browse", command=self._browse, width=80).pack(side="left", padx=2)
        ctk.CTkButton(folder_frame, text="Scan", command=self._scan, width=80).pack(side="left", padx=2)

        # ========== ACTION BUTTONS (PROMINENT) ==========
        action_container = ctk.CTkFrame(main)
        action_container.pack(fill="x", padx=10, pady=10)
        
        # Row 1: PNG Export
        png_frame = ctk.CTkFrame(action_container)
        png_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(png_frame, text="🖼️  PNG Export:", 
                    font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=10)
        ctk.CTkButton(png_frame, text="💾 Save Current Plots", 
                     command=self._save_current_plots,
                     fg_color="#2E7D32", hover_color="#1B5E20",
                     width=180, height=35).pack(side="left", padx=5)
        ctk.CTkButton(png_frame, text="📂 Export All Plots (Folder)", 
                     command=self._export_plots_folder,
                     fg_color="#1976D2", hover_color="#0D47A1",
                     width=180, height=35).pack(side="left", padx=5)

        # NEW: HTML summary export button (next to PNG export buttons)
        ctk.CTkButton(
            png_frame,
            text="📑 Export HTML Summary",
            command=self._analyze_and_export_html,
            fg_color="#455A64",
            hover_color="#263238",
            width=220, height=35
        ).pack(side="left", padx=5)

        # Row 2: Analysis
        analysis_frame = ctk.CTkFrame(action_container)
        analysis_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(analysis_frame, text="📊 Analysis:", 
                    font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=10)
        ctk.CTkButton(analysis_frame, text="📈 Analyze Single File", 
                     command=self._analyze_current_file_metrics,
                     fg_color="#F57C00", hover_color="#E65100",
                     width=180, height=35).pack(side="left", padx=5)
        ctk.CTkButton(analysis_frame, text="📋 Analyze All Files (Folder)", 
                     command=self._analyze_folder_metrics,
                     fg_color="#D32F2F", hover_color="#B71C1C",
                     width=200, height=35).pack(side="left", padx=5)
 
        # ========== OPTIONS & CALIBRATION ==========
        options_frame = ctk.CTkFrame(main)
        options_frame.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkButton(
            options_frame,
            text="➕ Quick Add SUSI Calibration",
            command=self._quick_add_susi_cal,
            width=220
        ).pack(side="left", padx=10)

        # Auto-detect thresholds toggle
        self.auto_detect_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            options_frame,
            text="Auto-detect overgrowth thresholds",
            variable=self.auto_detect_var
        ).pack(side="left", padx=10)

        # SUSI Calibration manager button
        ctk.CTkButton(
            options_frame,
            text="🧪 SUSI Calibrations...",
            command=self._open_susi_calibration_manager,
            width=170
        ).pack(side="left", padx=10)

        # ========== MAIN CONTENT AREA ==========
        content = ctk.CTkFrame(main)
        content.pack(fill="both", expand=True, padx=10, pady=5)

        # Left panel: File list (narrower)
        left = ctk.CTkFrame(content, width=280)
        left.pack(side="left", fill="y", padx=5, pady=5)
        left.pack_propagate(False)
        
        file_header = ctk.CTkFrame(left)
        file_header.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(file_header, text="📄 Files", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
        self.file_count_label = ctk.CTkLabel(file_header, text="(0)", 
                                             font=ctk.CTkFont(size=11))
        self.file_count_label.pack(side="left", padx=5)
        
        self.file_list = ctk.CTkScrollableFrame(left, width=260)
        self.file_list.pack(fill="both", expand=True, padx=5, pady=5)

        # Right panel: Plots
        right = ctk.CTkFrame(content)
        right.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        plot_header = ctk.CTkFrame(right)
        plot_header.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(plot_header, text="📊 Plots", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
        self.current_file_label = ctk.CTkLabel(plot_header, text="", 
                                               font=ctk.CTkFont(size=11))
        self.current_file_label.pack(side="left", padx=10)
        
        self.plot_container = ctk.CTkScrollableFrame(right)
        self.plot_container.pack(fill="both", expand=True, padx=5, pady=5)

        self._plot_canvases = []
        

        
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
            row = ctk.CTkFrame(container, fg_color="transparent")
            row.pack(fill="x", pady=5)
            ctk.CTkLabel(row, text=label, width=140, anchor="w").pack(side="left")
            var = ctk.StringVar(value=(default_val if default_val is not None else ""))
            ent = ctk.CTkEntry(row, textvariable=var, placeholder_text=placeholder)
            ent.pack(side="left", fill="x", expand=True, padx=6)
            fields[key] = var

        from datetime import datetime
        add_row("Date (YYYY-MM-DD)", "date", default_val=datetime.now().strftime("%Y-%m-%d"))
        add_row("Sample Name", "sample", "e.g., Si_2025_Q4_001")
        add_row("Locking Rate (ML/min)", "lock_rate", "e.g., 0.85")
        add_row("LTE Rate (ML/min)", "lte_rate", "e.g., 0.42")

        def save():
            try:
                date = fields['date'].get().strip()
                sample = fields['sample'].get().strip()
                lock_rate = float(fields['lock_rate'].get().strip())
                lte_rate = float(fields['lte_rate'].get().strip())
                if not date or not sample:
                    raise ValueError("Date and Sample Name are required.")

                # Minimal method/units
                method = "QUICK_ADD"
                units = "ML/min"

                cid = mgr.add_calibration(
                    date=date,
                    sample_name=sample,
                    method=method,
                    locking_layer_rate=lock_rate,
                    lte_rate=lte_rate,
                    rate_units=units,
                    notes="Added via Quick Add",
                    set_as_active=True
                )
                messagebox.showinfo(
                    "Saved",
                    f"Added calibration (ID {cid}), set as ACTIVE.\n\n"
                    f"Locking: {lock_rate:.3f} ML/min\nLTE: {lte_rate:.3f} ML/min"
                )
                win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save calibration:\n{e}")

        btns = ctk.CTkFrame(container, fg_color="transparent")
        btns.pack(fill="x", pady=12)
        ctk.CTkButton(btns, text="Save & Set Active", command=save).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Cancel", command=win.destroy).pack(side="left", padx=6)


    def _scan(self):
        """Scan folder for .txt files, auto-convert autoflash files, and display all files."""
        for w in self.file_list.winfo_children(): 
            w.destroy()
        
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No Folder", "Please select a folder first")
            return
        
        folder_path = Path(folder)
        
        # Find all .txt files
        all_txt_files = list(folder_path.glob("*.txt"))
        
        if not all_txt_files:
            messagebox.showinfo("No Files", "No .txt files found")
            return
        
        # ⭐ NEW: Check for autoflash files and convert them
        autoflash_files = [f for f in all_txt_files if 'auto' in f.name.lower()]
        
        if autoflash_files:
            conversion_results = self._convert_autoflash_files(autoflash_files)
            
            # Show summary of conversions
            if conversion_results['success']:
                msg = f"✓ Converted {conversion_results['success']} autoflash file(s)\n\n"
                for old, new in conversion_results['conversions']:
                    msg += f"  {old.name} → {new.name}\n"
                messagebox.showinfo("Auto-Flash Conversion", msg)
            
            if conversion_results['errors']:
                err_msg = f"⚠ Failed to convert {len(conversion_results['errors'])} file(s):\n\n"
                for f, err in conversion_results['errors']:
                    err_msg += f"  {f.name}: {err}\n"
                messagebox.showwarning("Conversion Errors", err_msg)
        
        # ⭐ Rescan folder to get all files (including newly converted ones)
        files = [p for p in folder_path.glob("*.txt")]
        
        if not files:
            messagebox.showinfo("No Files", "No .txt files found after conversion")
            return
        
        self._files = files
        self.file_count_label.configure(text=f"({len(files)})")
        
        # Display all files with color coding
        for p in files:
            file_type = detect_analysis_type(p.name)
            colors = {
                'Flash': '#D32F2F',
                'Dose': '#1976D2', 
                'SUSI': '#388E3C',
                'Termination': '#F57C00',
                'Outgas': '#7B1FA2'
            }
            color = colors.get(file_type, '#616161')
            
            btn = ctk.CTkButton(
                self.file_list, 
                text=f"{file_type[:1]} {p.name[:35]}...",
                command=lambda path=p: self._plot_file(path), 
                anchor="w",
                fg_color=color,
                hover_color=color,
                height=32
            )
            btn.pack(fill="x", pady=2, padx=3)

    def _convert_autoflash_files(self, autoflash_files):
        """
        Convert autoflash files to standard format and rename them.
        
        Args:
            autoflash_files: List of Path objects pointing to autoflash files
            
        Returns:
            dict with 'success' count, 'conversions' list of (old, new) paths, and 'errors' list
        """
        results = {
            'success': 0,
            'conversions': [],
            'errors': []
        }
        
        for autoflash_file in autoflash_files:
            try:
                # Create new filename by removing 'autoflash' (case-insensitive)
                import re
                new_name = re.sub(r'autoflash[-_]?', '', autoflash_file.name, flags=re.IGNORECASE)
                new_name = re.sub(r'[-_]{2,}', '-', new_name)  # Clean up double separators
                new_path = autoflash_file.parent / new_name
                
                # Convert the file
                success = convert_autoflash(autoflash_file, new_path)
                
                if success:
                    # Delete the original autoflash file
                    autoflash_file.unlink()
                    
                    results['success'] += 1
                    results['conversions'].append((autoflash_file, new_path))
                else:
                    results['errors'].append((autoflash_file, "Conversion returned False"))
                    
            except Exception as e:
                results['errors'].append((autoflash_file, str(e)))
        
        return results
    


    # ======== SUSI CALIBRATION MANAGER UI ========

    def _open_susi_calibration_manager(self):
        """Open a simple UI to view/manage SUSI calibrations."""
        try:
            self._susi_mgr = SUSICalibrationManager()  # default DB path
        except Exception as e:
            messagebox.showerror("Calibration DB", f"Failed to open calibration DB:\n{e}")
            return

        win = ctk.CTkToplevel()
        win.title("SUSI Calibrations")
        win.geometry("860x520")

        header = ctk.CTkFrame(win)
        header.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(header, text="SUSI Growth Rate Calibrations",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")

        btns = ctk.CTkFrame(win)
        btns.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(btns, text="Refresh", width=100,
                      command=lambda: self._refresh_susi_cal_list(list_frame)).pack(side="left", padx=5)

        ctk.CTkButton(btns, text="Add...", width=100,
                      command=lambda: self._add_susi_calibration(win)).pack(side="left", padx=5)

        ctk.CTkButton(btns, text="Export JSON...", width=120,
                      command=self._export_susi_calibrations).pack(side="left", padx=5)

        ctk.CTkButton(btns, text="Import JSON...", width=120,
                      command=self._import_susi_calibrations).pack(side="left", padx=5)

        list_frame = ctk.CTkScrollableFrame(win)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # First load
        self._refresh_susi_cal_list(list_frame)

    def _refresh_susi_cal_list(self, frame):
        for w in frame.winfo_children():
            w.destroy()

        try:
            calibrations = self._susi_mgr.get_calibration_history()
            active = self._susi_mgr.get_active_calibration()
            active_id = active.calibration_id if active else None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibrations:\n{e}")
            return

        if not calibrations:
            ctk.CTkLabel(frame, text="No calibrations found.",
                         font=ctk.CTkFont(size=13)).pack(pady=8)
            return

        for cal in calibrations:
            row = ctk.CTkFrame(frame)
            row.pack(fill="x", pady=4, padx=6)

            left = ctk.CTkFrame(row, fg_color="transparent")
            left.pack(side="left", fill="x", expand=True)
            right = ctk.CTkFrame(row, fg_color="transparent")
            right.pack(side="right")

            flag = "✓ ACTIVE" if active_id == cal.calibration_id else ""
            ctk.CTkLabel(left, text=f"ID {cal.calibration_id} | {cal.date} | {cal.sample_name} | {cal.method}  {flag}",
                         font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
            ctk.CTkLabel(left, text=(f"Locking: {cal.locking_layer_rate:.3f} {cal.rate_units} | "
                                     f"LTE: {cal.lte_rate:.3f} {cal.rate_units}"),
                         font=ctk.CTkFont(size=11)).pack(anchor="w")
            if cal.notes:
                ctk.CTkLabel(left, text=f"Notes: {cal.notes}", font=ctk.CTkFont(size=10), text_color="gray").pack(anchor="w")

            ctk.CTkButton(right, text="Set Active",
                          command=lambda cid=cal.calibration_id: self._set_active_susi_cal(cid),
                          width=100).pack(pady=2)
            ctk.CTkButton(right, text="Delete",
                          command=lambda cid=cal.calibration_id: self._delete_susi_cal(cid),
                          width=100, fg_color="#B71C1C", hover_color="#7F0000").pack(pady=2)

    def _set_active_susi_cal(self, calibration_id: int):
        try:
            self._susi_mgr.set_active_calibration(calibration_id)
            messagebox.showinfo("Active Calibration", f"Set calibration ID {calibration_id} as active.")
            # Refresh any open list
            for w in self.tab.winfo_children():
                if isinstance(w, ctk.CTkToplevel):
                    self._refresh_susi_cal_list(w)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set active:\n{e}")

    def _delete_susi_cal(self, calibration_id: int):
        if not messagebox.askyesno("Confirm Delete", f"Delete calibration ID {calibration_id}?"):
            return
        try:
            self._susi_mgr.delete_calibration(calibration_id)
            messagebox.showinfo("Deleted", f"Calibration ID {calibration_id} deleted.")
            # Trigger a refresh: find the open window and refresh its list
            # Simple approach: re-open manager
            self._open_susi_calibration_manager()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete:\n{e}")

    def _add_susi_calibration(self, parent_win):
        dialog = ctk.CTkToplevel(parent_win)
        dialog.title("Add SUSI Calibration")
        dialog.geometry("460x420")

        container = ctk.CTkFrame(dialog)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        entries = {}

        def add_row(label, key, placeholder=""):
            row = ctk.CTkFrame(container, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(row, text=label, width=130, anchor="w").pack(side="left")
            var = ctk.StringVar()
            ent = ctk.CTkEntry(row, textvariable=var, placeholder_text=placeholder)
            ent.pack(side="left", fill="x", expand=True, padx=6)
            entries[key] = var

        add_row("Date (YYYY-MM-DD)", "date", datetime.now().strftime("%Y-%m-%d"))
        add_row("Sample Name", "sample_name", "e.g., Si_2025_Q4_001")
        add_row("Method", "method", "SIMS / MASK / STM / XRR / OTHER")
        add_row("Lock Rate", "lock_rate", "ML/min or nm/min")
        add_row("LTE Rate", "lte_rate", "ML/min or nm/min")
        add_row("Units", "units", "ML/min or nm/min")
        add_row("Notes", "notes", "")

        def save():
            try:
                date = entries['date'].get().strip()
                sample = entries['sample_name'].get().strip()
                method = entries['method'].get().strip() or "OTHER"
                lock_rate = float(entries['lock_rate'].get().strip())
                lte_rate = float(entries['lte_rate'].get().strip())
                units = entries['units'].get().strip() or "ML/min"
                notes = entries['notes'].get().strip()

                cid = self._susi_mgr.add_calibration(
                    date=date,
                    sample_name=sample,
                    method=method,
                    locking_layer_rate=lock_rate,
                    lte_rate=lte_rate,
                    rate_units=units,
                    notes=notes,
                    set_as_active=True
                )
                messagebox.showinfo("Saved", f"Added calibration (ID {cid}) and set active.")
                dialog.destroy()
                self._open_susi_calibration_manager()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add calibration:\n{e}")

        btns = ctk.CTkFrame(container, fg_color="transparent")
        btns.pack(fill="x", pady=10)
        ctk.CTkButton(btns, text="Save & Set Active", command=save).pack(side="left", padx=5)
        ctk.CTkButton(btns, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

    def _export_susi_calibrations(self):
        path = filedialog.asksaveasfilename(
            title="Export Calibrations to JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        try:
            self._susi_mgr.export_to_json(path)
            messagebox.showinfo("Exported", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")

    def _import_susi_calibrations(self):
        path = filedialog.askopenfilename(
            title="Import Calibrations from JSON",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        try:
            self._susi_mgr.import_from_json(path)
            messagebox.showinfo("Imported", f"Loaded calibrations from:\n{path}")
            self._open_susi_calibration_manager()
        except Exception as e:
            messagebox.showerror("Error", f"Import failed:\n{e}")

    def _clear_plots(self):
        for c in self._plot_canvases:
            c.get_tk_widget().destroy()
        self._plot_canvases.clear()

    def _plot_file(self, path: Path):
        try:
            self._current_file = path
            self.current_file_label.configure(text=f"Current: {path.name}")
            
            df = load_labview_df(str(path))
            mode = detect_analysis_type(path.name)
            if mode == 'Flash':
                figs = plot_flash(df)
            elif mode == 'Termination':
                figs = plot_termination(df)
            elif mode == 'SUSI':
                figs = plot_susi(df)
            elif mode == 'Dose':
                figs = plot_dose(df)
            elif mode == 'Outgas':
                figs = plot_outgas(df)
            else:
                figs = plot_flash(df)
            self._embed_figs(figs)
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot {path.name}:\n{e}")

    def _show_selected(self):
        messagebox.showinfo("Hint", "Click any file on the left to display plots.")

    def _save_current_plots(self):
        """Save currently displayed plots as PNG files"""
        if not self._plot_canvases:
            messagebox.showwarning("No Plots", "No plots currently displayed. Click a file first to show plots.")
            return
        
        # Ask for output directory
        out_dir = filedialog.askdirectory(title="Select Output Directory for PNG Files")
        if not out_dir:
            return
        
        try:
            # Get figures from canvases
            figs = [canvas.figure for canvas in self._plot_canvases]
            
            # Use current file name as base
            if self._current_file:
                base_name = self._current_file.stem
            else:
                base_name = "plot"
            
            # Add file type
            if self._current_file:
                file_type = detect_analysis_type(self._current_file.name).lower()
                base_name = f"{base_name}_{file_type}"
            
            # Save figures
            saved_files = save_figures(figs, out_dir, base_name, dpi=200)
            
            messagebox.showinfo("✓ Success", 
                              f"Saved {len(saved_files)} plot(s) to:\n{out_dir}\n\n"
                              f"Files:\n" + "\n".join([Path(f).name for f in saved_files[:3]]) +
                              (f"\n... and {len(saved_files)-3} more" if len(saved_files) > 3 else ""))
            
        except Exception as e:
            import traceback
            messagebox.showerror("Save Error", f"Failed to save plots:\n{e}\n\n{traceback.format_exc()}")
    
    def _export_plots_folder(self):
        """Batch export plots for all files in a folder"""
        # Get input folder
        in_dir = self.folder_var.get().strip()
        if not in_dir:
            in_dir = filedialog.askdirectory(title="Select Folder with LabVIEW Files")
            if not in_dir:
                return
        
        # Get output folder
        out_dir = filedialog.askdirectory(title="Select Output Directory for PNG Files")
        if not out_dir:
            return
        
        try:
            # Show progress message
            progress = ctk.CTkToplevel()
            progress.title("Exporting...")
            progress.geometry("400x100")
            ctk.CTkLabel(progress, text="Exporting plots...\nPlease wait...",
                        font=ctk.CTkFont(size=14)).pack(expand=True)
            progress.update()
            
            results = export_plots_for_folder(in_dir, out_dir, dpi=200)
            
            progress.destroy()
            
            total_plots = sum(len(files) for files in results.values())
            messagebox.showinfo("✓ Export Complete", 
                              f"Processed {len(results)} files\n"
                              f"Exported {total_plots} plots\n\n"
                              f"Output directory:\n{out_dir}")
            
        except Exception as e:
            import traceback
            if 'progress' in locals():
                progress.destroy()
            messagebox.showerror("Export Error", f"Failed to export plots:\n{e}\n\n{traceback.format_exc()}")
    
    def _analyze_current_file_metrics(self):
        """Analyze metrics for a selected file (with optional growth rate input)."""
        import tkinter as tk
        from tkinter import simpledialog

        if not hasattr(self, "_files") or not self._files:
            messagebox.showwarning("No Files", "Scan a folder first")
            return

        # Ask user to select a file
        file = filedialog.askopenfilename(
            title="Select LabVIEW File for Metrics Analysis",
            filetypes=[("LabVIEW Files", "*.txt")],
            initialdir=self.folder_var.get() or None
        )
        if not file:
            return

        # Ask if they want to use active DB calibration for growth rates
        use_db = messagebox.askyesno(
            "Growth Rates",
            "Use active SUSI calibration from database for deposition rates?\n\n"
            "Yes = use DB\nNo  = enter rates manually (ML/min)"
        )

        lock_rate = None
        lte_rate = None

        if use_db:
            try:
                mgr = SUSICalibrationManager()
                cal = mgr.get_active_calibration()
                if cal:
                    lock_rate = cal.get_rate_ML_min('locking_layer')
                    lte_rate = cal.get_rate_ML_min('lte')
                else:
                    # Offer to create one now
                    create_now = messagebox.askyesno(
                        "No Active SUSI Calibration",
                        "No active SUSI calibration found.\n\n"
                        "Would you like to create one now?"
                    )
                    if create_now:
                        self._quick_add_susi_cal()
                        # Re-fetch after creation
                        cal = mgr.get_active_calibration()
                        if cal:
                            lock_rate = cal.get_rate_ML_min('locking_layer')
                            lte_rate = cal.get_rate_ML_min('lte')
                        else:
                            messagebox.showwarning(
                                "Still Missing",
                                "No calibration found after creation.\nProceeding without rates."
                            )
                    else:
                        messagebox.showwarning(
                            "No Rates",
                            "Proceeding without rates (deposition amounts will be omitted)."
                        )
            except Exception as e:
                messagebox.showwarning(
                    "Calibration Error",
                    f"Failed to load active calibration.\nProceeding without rates.\n\n{e}"
                )

        else:
            try:
                lock_rate = simpledialog.askfloat(
                    "Locking Layer Rate",
                    "Enter Locking Layer rate (ML/min):",
                    minvalue=0.0
                )
                lte_rate = simpledialog.askfloat(
                    "LTE Rate",
                    "Enter LTE rate (ML/min):",
                    minvalue=0.0
                )
            except Exception:
                pass

        try:
            # Show progress
            progress = ctk.CTkToplevel()
            progress.title("Analyzing...")
            progress.geometry("300x80")
            ctk.CTkLabel(progress, text="Computing metrics...",
                        font=ctk.CTkFont(size=12)).pack(expand=True)
            progress.update()

            # Analyze the file
            result = analyze_process_file(
                file,
                locking_layer_rate_ML_min=lock_rate,
                lte_rate_ML_min=lte_rate,
                auto_detect_thresholds=self.auto_detect_var.get()
            )

            progress.destroy()

            if 'error' in result:
                messagebox.showerror("Analysis Error", result['error'])
                return

            # Format and display results
            metrics_text = self._format_metrics_for_display(
                result.get('metrics', {}),
                result.get('file_type', 'unknown'),
                Path(file).name
            )

            dialog = ctk.CTkToplevel()
            dialog.title(f"📊 Metrics: {Path(file).name}")
            dialog.geometry("800x900")

            header_frame = ctk.CTkFrame(dialog)
            header_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(header_frame, text=f"Process Metrics Analysis",
                        font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
            ctk.CTkLabel(header_frame, text=f"File: {Path(file).name}",
                        font=ctk.CTkFont(size=12)).pack()

            from tkinter import scrolledtext
            text = scrolledtext.ScrolledText(dialog, wrap="word", font=("Courier", 10))
            text.pack(fill="both", expand=True, padx=10, pady=10)
            text.insert("1.0", metrics_text)
            text.config(state="disabled")

            ctk.CTkButton(dialog, text="Close", command=dialog.destroy).pack(pady=10)

        except Exception as e:
            import traceback
            if 'progress' in locals():
                progress.destroy()
            messagebox.showerror("Metrics Error", 
                               f"Failed to analyze metrics:\n{e}\n\n{traceback.format_exc()}")

    def _analyze_folder_metrics(self):
        """Analyze metrics for all files in folder in process order (with growth rates prompt)"""
        from tkinter import simpledialog
        # Get folder
        folder = self.folder_var.get().strip()
        if not folder:
            folder = filedialog.askdirectory(title="Select Folder for Batch Analysis")
            if not folder:
                return

        # Temperature calibration prompt (optional)
        use_calibration = messagebox.askyesno(
            "Temperature Calibration",
            "Do you have a temperature calibration from a cooldown analysis?\n\n"
            "If Yes, you'll be asked to select the calibration file.\n"
            "If No, overgrowth analysis will only show currents (not temperatures)."
        )

        calibration = None
        if use_calibration:
            cal_path = filedialog.askopenfilename(
                title="Select Temperature Calibration File",
                initialdir=folder,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if cal_path:
                try:
                    import json
                    with open(cal_path, 'r') as f:
                        calibration = json.load(f)
                    model = calibration.get('model', 'unknown')
                    messagebox.showinfo(
                        "Calibration Loaded",
                        f"Successfully loaded calibration:\n\n"
                        f"Model: {model}\n"
                        f"Source: {calibration.get('source_file', 'N/A')}\n"
                        f"R² = {calibration.get('r_squared', 'N/A')}"
                    )
                except Exception as e:
                    messagebox.showerror(
                        "Calibration Error",
                        f"Failed to load calibration file:\n{e}\n\nProceeding without calibration."
                    )
                    calibration = None
            else:
                messagebox.showinfo("Info", "No calibration file selected.\nProceeding without calibration.")

        # Growth rates prompt: use active SUSI calibration or manual entry
        lock_rate = None
        lte_rate = None
        use_db = messagebox.askyesno(
            "Growth Rates",
            "Use active SUSI calibration for deposition rates?\n\n"
            "Yes = use DB (active calibration)\nNo  = enter rates manually (ML/min)"
        )

        if use_db:
            try:
                mgr = SUSICalibrationManager()
                cal = mgr.get_active_calibration()
                if cal:
                    lock_rate = cal.get_rate_ML_min('locking_layer')
                    lte_rate = cal.get_rate_ML_min('lte')
                    messagebox.showinfo(
                        "Using Active Calibration",
                        f"Locking Layer: {lock_rate:.3f} ML/min\nLTE: {lte_rate:.3f} ML/min"
                    )
                else:
                    create_now = messagebox.askyesno(
                        "No Active SUSI Calibration",
                        "No active SUSI calibration found.\n\n"
                        "Would you like to create one now?"
                    )
                    if create_now:
                        self._quick_add_susi_cal()
                        cal = mgr.get_active_calibration()
                        if cal:
                            lock_rate = cal.get_rate_ML_min('locking_layer')
                            lte_rate = cal.get_rate_ML_min('lte')
                            messagebox.showinfo(
                                "Using Newly Added Calibration",
                                f"Locking Layer: {lock_rate:.3f} ML/min\nLTE: {lte_rate:.3f} ML/min"
                            )
                        else:
                            messagebox.showwarning(
                                "Still Missing",
                                "No calibration found after creation. Please enter rates manually."
                            )
                    else:
                        messagebox.showwarning("No Active Calibration", "No active SUSI calibration found. Please enter rates.")
            except Exception as e:
                messagebox.showwarning("Calibration Error", f"Failed to load active calibration.\n{e}\nEnter rates manually.")

        if lock_rate is None or lte_rate is None:
            # Manual entry fallback
            try:
                lock_rate = simpledialog.askfloat(
                    "Locking Layer Rate (ML/min)",
                    "Enter Locking Layer rate (ML/min):\n\nTip: use your latest calibration value.",
                    minvalue=0.0
                )
                lte_rate = simpledialog.askfloat(
                    "LTE Rate (ML/min)",
                    "Enter LTE rate (ML/min):\n\nTip: use your latest calibration value.",
                    minvalue=0.0
                )
            except Exception:
                pass

        # Ask if they want to save JSON
        save_json = messagebox.askyesno(
            "Save Results",
            "Save detailed results to JSON file?"
        )

        try:
            # Show progress
            progress = ctk.CTkToplevel()
            progress.title("Batch Analysis")
            progress.geometry("420x160")

            title_label = ctk.CTkLabel(progress, 
                                      text="Analyzing all files in process order...",
                                      font=ctk.CTkFont(size=14, weight="bold"))
            title_label.pack(pady=10)

            status_label = ctk.CTkLabel(progress, text="Scanning folder...")
            status_label.pack(pady=5)

            progress_label = ctk.CTkLabel(progress, text="", font=ctk.CTkFont(size=11))
            progress_label.pack(pady=5)

            progress.update()

            # Scan folder
            txt_files = list(Path(folder).glob("*.txt"))
            total_files = len(txt_files)

            if total_files == 0:
                progress.destroy()
                messagebox.showwarning("No Files", f"No .txt files found in:\n{folder}")
                return

            status_label.configure(text=f"Found {total_files} files. Analyzing...")
            progress.update()

            # Decide DB fallback behavior
            # If we already have rates, we disable internal DB lookup to avoid surprises.
            use_active = False if (lock_rate is not None and lte_rate is not None) else True

            # Run analysis
            results = analyze_folder_metrics(
                folder,
                calibration=calibration,
                save_json=save_json,
                auto_detect_thresholds=self.auto_detect_var.get(),
                locking_layer_rate_ML_min=lock_rate,
                lte_rate_ML_min=lte_rate,
                use_active_susi_calibration=use_active
            )

            progress.destroy()

            if 'error' in results:
                messagebox.showerror("Analysis Error", results['error'])
                return

            # Format results
            formatted_text = format_batch_results(results)

            # Create results window
            results_window = ctk.CTkToplevel()
            results_window.title(f"📊 Batch Analysis Results")
            results_window.geometry("1000x900")

            # Header
            header_frame = ctk.CTkFrame(results_window)
            header_frame.pack(fill="x", padx=10, pady=10)

            ctk.CTkLabel(header_frame, 
                        text="Batch Process Metrics Analysis",
                        font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)

            stats = results['statistics']
            ctk.CTkLabel(header_frame,
                        text=f"{stats['successful_analyses']}/{stats['total_processes']} files analyzed successfully",
                        font=ctk.CTkFont(size=12)).pack()

            # Buttons
            button_frame = ctk.CTkFrame(header_frame)
            button_frame.pack(pady=5)

            def save_to_excel():
                save_path = filedialog.asksaveasfilename(
                    title="Save Excel Report",
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx")],
                    initialfile=f"batch_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                )
                if save_path:
                    try:
                        export_batch_results_excel(results, save_path)
                        messagebox.showinfo("Saved", f"Excel report saved to:\n{save_path}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to save Excel:\n{e}")

            def save_to_text():
                save_path = filedialog.asksaveasfilename(
                    title="Save Text Report",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt")],
                    initialfile=f"batch_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )
                if save_path:
                    try:
                        with open(save_path, 'w') as f:
                            f.write(formatted_text)
                        messagebox.showinfo("Saved", f"Text report saved to:\n{save_path}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to save text:\n{e}")

            ctk.CTkButton(button_frame, text="📊 Export to Excel", 
                         command=save_to_excel,
                         fg_color="#2E7D32", hover_color="#1B5E20").pack(side="left", padx=5)
            ctk.CTkButton(button_frame, text="📄 Save as Text", 
                         command=save_to_text,
                         fg_color="#1976D2", hover_color="#0D47A1").pack(side="left", padx=5)

            # Text display
            from tkinter import scrolledtext
            text_frame = ctk.CTkFrame(results_window)
            text_frame.pack(fill="both", expand=True, padx=10, pady=10)

            text = scrolledtext.ScrolledText(text_frame, wrap="word", font=("Courier", 10))
            text.pack(fill="both", expand=True)
            text.insert("1.0", formatted_text)
            text.config(state="disabled")

            # Close button
            ctk.CTkButton(results_window, text="Close", command=results_window.destroy).pack(pady=10)

            # Summary message
            summary_msg = (
                f"✓ Analysis Complete!\n\n"
                f"Analyzed: {stats['total_processes']} files\n"
                f"Success: {stats['successful_analyses']}\n"
                f"Failed: {stats['failed_analyses']}\n"
            )

            if save_json and 'saved_to' in results:
                summary_msg += f"\nJSON saved to:\n{results['saved_to']}"

            messagebox.showinfo("Batch Analysis Complete", summary_msg)

        except Exception as e:
            import traceback
            if 'progress' in locals():
                progress.destroy()
            messagebox.showerror(
                "Batch Analysis Error",
                f"Failed to analyze folder:\n{e}\n\n{traceback.format_exc()}"
            )

    def _format_metrics_for_display(self, metrics_dict, file_type, filename):
        """Format metrics for display"""

        def fmt_hms(seconds: float) -> str:
            try:
                seconds = int(round(float(seconds)))
            except Exception:
                return "0 h 0 min 0 s"
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            return f"{h} h {m} min {s} s"

        lines = []
        lines.append("=" * 80)
        lines.append(f"FILE: {filename}")
        lines.append(f"TYPE: {file_type.upper()}")
        lines.append("=" * 80)
        lines.append("")

        if file_type == 'dose':
            m = metrics_dict
            lines.append("DOSE ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"  Baseline Pressure:     {m.get('baseline_pressure_torr', 0):.2e} Torr")
            lines.append(f"  Peak Pressure:         {m.get('peak_pressure_torr', 0):.2e} Torr")
            lines.append(f"  Dose Threshold:        {m.get('dose_threshold_torr', 0):.2e} Torr")
            lines.append(f"  Dose Detected:         {'✓ YES' if m.get('dose_detected') else '✗ NO'}")
            lines.append("")
            if m.get('dose_detected'):
                lines.append("  DOSE PERIOD:")
                lines.append(f"    Duration:            {fmt_hms(m.get('dose_duration_s', 0.0))}")
                lines.append(f"    Start Time:          {m.get('dose_start_time_s', 0):.1f} s")
                lines.append(f"    End Time:            {m.get('dose_end_time_s', 0):.1f} s")
                lines.append(f"    Mean Pressure:       {m.get('mean_dose_pressure_torr', 0):.2e} Torr")
                lines.append("")
                lines.append("  MOLECULAR DELIVERY:")
                lines.append(f"    Exposure:            {m.get('exposure_langmuirs', 0):.2f} Langmuirs")
                lines.append(f"    Molecular Weight:    {m.get('molecular_weight_gmol', 0):.1f} g/mol")
                lines.append(f"    Temperature:         {m.get('temperature_K', 0):.1f} K")
                lines.append(f"    Integrated Dose:     {m.get('integrated_dose_cm2', 0):.2e} molecules/cm²")
                lines.append(f"    Mean Flux:           {m.get('mean_flux_cm2s', 0):.2e} molecules/cm²/s")

        elif file_type == 'flash':
            m = metrics_dict
            lines.append("FLASH ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"  Flash Count:           {m.get('flash_count', 0)}")
            lines.append(f"  Total Flash Time:      {fmt_hms(m.get('total_flash_time_s', 0.0))}")
            lines.append(f"  Peak Temperature:      {m.get('peak_temperature_C', 0):.1f} °C")
            lines.append(f"  Entry Threshold:       {m.get('flash_threshold_C', 0):.1f} °C")
            lines.append(f"  Exit Threshold:        {m.get('exit_threshold_C', 0):.1f} °C")
            lines.append("")
            flashes = m.get('flashes', [])
            if flashes:
                lines.append("  INDIVIDUAL FLASHES:")
                lines.append("-" * 80)
                for flash in flashes:
                    lines.append(f"  Flash #{flash['flash_number']}:")
                    lines.append(f"    Start:               {flash['start_time_s']:.1f} s")
                    lines.append(f"    End:                 {flash['end_time_s']:.1f} s")
                    lines.append(f"    Duration:            {fmt_hms(flash['duration_s'])}")
                    lines.append(f"    Peak Temperature:    {flash['peak_temp_C']:.1f} °C")
                    lines.append(f"    Peak Time:           {flash['peak_time_s']:.1f} s")
                    lines.append("")

        elif file_type == 'susi':
            if 'susi' in metrics_dict and 'overgrowth' in metrics_dict:
                lines.append("SUSI OPERATION")
                lines.append("-" * 80)
                susi = metrics_dict['susi']
                lines.append(f"  Operating Threshold:   {susi.get('operating_threshold_A', 0):.4f} A")
                lines.append(f"  Total Operating Time:  {fmt_hms(susi.get('total_operating_time_s', 0.0))}")
                lines.append(f"  Mean Current:          {susi.get('mean_operating_current_A', 0):.4f} A")
                lines.append(f"  Peak Current:          {susi.get('peak_current_A', 0):.4f} A")
                lines.append(f"  Segments:              {susi.get('number_of_segments', 0)}")
                lines.append("")

                lines.append("OVERGROWTH PHASE BREAKDOWN")
                lines.append("-" * 80)
                ovg = metrics_dict['overgrowth']
                lines.append(f"  Current Quantiles:")
                lines.append(f"    Q33 (RT/RTA boundary): {ovg.get('current_q33', 0):.4f} A")
                lines.append(f"    Q66 (RTA/LTE boundary): {ovg.get('current_q66', 0):.4f} A")
                lines.append("")
                lines.append(f"  Total Overgrowth Duration: {fmt_hms(ovg.get('total_duration_s', 0.0))}")
                if ovg.get('total_deposited_ML') is not None:
                    lines.append(f"  Total Deposited:           {ovg['total_deposited_ML']:.1f} ML "
                                 f"({ovg.get('total_deposited_nm', 0):.2f} nm)")
                lines.append("")

                for phase_name, phase_key in [('RT GROWTH', 'RT_growth'),
                                              ('RTA ANNEAL', 'RTA_anneal'),
                                              ('LTE GROWTH', 'LTE_growth')]:
                    phase = ovg.get(phase_key, {})
                    lines.append(f"  {phase_name}:")
                    if phase.get('detected'):
                        lines.append(f"    Duration:           {fmt_hms(phase.get('duration_s', 0.0))}")
                        st_min = phase.get('start_time_min', None)
                        en_min = phase.get('end_time_min', None)
                        if st_min is not None and en_min is not None:
                            lines.append(f"    Start / End:        {st_min:.1f} min → {en_min:.1f} min")
                        tag = " (estimated 10 ML)" if phase.get('estimated') else ""
                        lines.append(f"    Median Current:     {phase.get('median_current_A', 0):.4f} A{tag}")
                        if 'median_temperature_C' in phase:
                            lines.append(f"    Median Temperature: {phase.get('median_temperature_C', 0):.1f} °C")
                        if phase.get('deposited_ML') is not None:
                            ml_val = phase.get('deposited_ML')
                            nm_val = phase.get('deposited_nm')
                            rate_val = phase.get('deposition_rate_ML_min')
                            nm_str = f" ({nm_val:.2f} nm)" if nm_val is not None else ""
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




    def _batch_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Ramp-Down Folder")
        if folder:
            self.batch_folder_var.set(folder)

    def _embed_figs(self, figs):
        self._clear_plots()
        for fig in figs:
            canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, pady=6)
            self._plot_canvases.append(canvas)

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
        """Run batch analysis, generate per-file plots, build a single self-contained HTML report."""
        import threading

        # 1) Get folder
        folder = self.folder_var.get().strip()
        if not folder:
            folder = filedialog.askdirectory(title="Select Device Folder (with .txt files)")
            if not folder:
                return
            self.folder_var.set(folder)

        # Quick check for files
        txt_files = list(Path(folder).glob("*.txt"))
        if not txt_files:
            messagebox.showwarning("No Files", f"No .txt files found in:\n{folder}")
            return

        # 2) Output path
        default_name = f"device_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = filedialog.asksaveasfilename(
            title="Save Device HTML Report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
            initialdir=folder,
            initialfile=default_name
        )
        if not output_path:
            return

        # 3) Optional device metadata prompt
        device_meta = self._prompt_device_metadata()  # returns dict with device_name, notes (can be empty)

        # 4) Decide growth rates (reuse your batch flow prompting for DB/manual rates)
        from tkinter import simpledialog

        # Temperature calibration prompt (optional)
        use_calibration = messagebox.askyesno(
            "Temperature Calibration",
            "Do you have a temperature calibration from a cooldown analysis?\n\n"
            "If Yes, you'll be asked to select the calibration JSON file.\n"
            "If No, overgrowth analysis will only show currents (not temperatures)."
        )
        calibration = None
        if use_calibration:
            cal_path = filedialog.askopenfilename(
                title="Select Temperature Calibration File",
                initialdir=folder,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if cal_path:
                try:
                    import json
                    with open(cal_path, 'r') as f:
                        calibration = json.load(f)
                    model = calibration.get('model', 'unknown')
                    messagebox.showinfo(
                        "Calibration Loaded",
                        f"Successfully loaded calibration:\n\n"
                        f"Model: {model}\n"
                        f"Source: {calibration.get('source_file', 'N/A')}\n"
                        f"R² = {calibration.get('r_squared', 'N/A')}"
                    )
                except Exception as e:
                    messagebox.showerror(
                        "Calibration Error",
                        f"Failed to load calibration file:\n{e}\n\nProceeding without calibration."
                    )
                    calibration = None
            else:
                messagebox.showinfo("Info", "No calibration file selected.\nProceeding without calibration.")

        lock_rate = None
        lte_rate = None
        use_db = messagebox.askyesno(
            "Growth Rates",
            "Use active SUSI calibration for deposition rates?\n\n"
            "Yes = use DB (active calibration)\nNo  = enter rates manually (ML/min)"
        )

        if use_db:
            try:
                mgr = SUSICalibrationManager()
                cal = mgr.get_active_calibration()
                if cal:
                    lock_rate = cal.get_rate_ML_min('locking_layer')
                    lte_rate = cal.get_rate_ML_min('lte')
                    messagebox.showinfo(
                        "Using Active Calibration",
                        f"Locking Layer: {lock_rate:.3f} ML/min\nLTE: {lte_rate:.3f} ML/min"
                    )
                else:
                    create_now = messagebox.askyesno(
                        "No Active SUSI Calibration",
                        "No active SUSI calibration found.\n\n"
                        "Would you like to create one now?"
                    )
                    if create_now:
                        self._quick_add_susi_cal()
                        cal = mgr.get_active_calibration()
                        if cal:
                            lock_rate = cal.get_rate_ML_min('locking_layer')
                            lte_rate = cal.get_rate_ML_min('lte')
                            messagebox.showinfo(
                                "Using Newly Added Calibration",
                                f"Locking Layer: {lock_rate:.3f} ML/min\nLTE: {lte_rate:.3f} ML/min"
                            )
                        else:
                            messagebox.showwarning(
                                "Still Missing",
                                "No calibration found after creation. Please enter rates manually."
                            )
                    else:
                        messagebox.showwarning("No Active Calibration", "No active SUSI calibration found. Please enter rates.")
            except Exception as e:
                messagebox.showwarning("Calibration Error", f"Failed to load active calibration.\n{e}\nEnter rates manually.")

        if lock_rate is None or lte_rate is None:
            # Manual entry fallback
            try:
                lock_rate = simpledialog.askfloat(
                    "Locking Layer Rate (ML/min)",
                    "Enter Locking Layer rate (ML/min):\n\nTip: use your latest calibration value.",
                    minvalue=0.0
                )
                lte_rate = simpledialog.askfloat(
                    "LTE Rate (ML/min)",
                    "Enter LTE rate (ML/min):\n\nTip: use your latest calibration value.",
                    minvalue=0.0
                )
            except Exception:
                pass

        # 5) Show progress UI
        progress = ctk.CTkToplevel()
        progress.title("Building Report")
        progress.geometry("420x140")
        ctk.CTkLabel(progress, text="Analyzing and generating HTML report...",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=12)
        status_label = ctk.CTkLabel(progress, text="Running analysis...")
        status_label.pack(pady=5)
        progress.update()

        def work():
            try:
                # Decide DB fallback behavior for batch analysis
                use_active = False if (lock_rate is not None and lte_rate is not None) else True

                # 6) Batch analysis
                status_label.configure(text="Analyzing files...")
                progress.update()
                results = analyze_folder_metrics(
                    folder,
                    calibration=calibration,
                    save_json=False,
                    auto_detect_thresholds=self.auto_detect_var.get(),
                    locking_layer_rate_ML_min=lock_rate,
                    lte_rate_ML_min=lte_rate,
                    use_active_susi_calibration=use_active
                )
                if 'error' in results:
                    raise RuntimeError(results['error'])

                # 7) Build report (self-contained HTML with base64 PNGs)
                status_label.configure(text="Rendering HTML...")
                progress.update()
                build_device_report(
                    folder_path=folder,
                    results_dict=results,
                    output_html=output_path,
                    device_meta=device_meta,
                    dpi=150,           # static PNGs, inlined as base64
                    inline_images=True # single self-contained HTML
                )

                progress.destroy()
                messagebox.showinfo("Report Ready", f"Saved to:\n{output_path}")
                try:
                    webbrowser.open_new_tab(output_path)
                except Exception:
                    pass
            except Exception as e:
                import traceback
                progress.destroy()
                messagebox.showerror("Report Error", f"Failed to build report:\n{e}\n\n{traceback.format_exc()}")

        threading.Thread(target=work, daemon=True).start()
                
    def _prompt_device_metadata(self):
        """Prompt for optional device metadata to include in the HTML report."""
        win = ctk.CTkToplevel(self.tab)
        win.title("Device Metadata (Optional)")
        win.geometry("420x220")
        win.grab_set()

        container = ctk.CTkFrame(win)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        vars_ = {}

        def add_row(label, key, placeholder="", default_val=""):
            row = ctk.CTkFrame(container, fg_color="transparent")
            row.pack(fill="x", pady=6)
            ctk.CTkLabel(row, text=label, width=140, anchor="w").pack(side="left")
            var = ctk.StringVar(value=default_val)
            ent = ctk.CTkEntry(row, textvariable=var, placeholder_text=placeholder)
            ent.pack(side="left", fill="x", expand=True, padx=6)
            vars_[key] = var

        folder_name = Path(self.folder_var.get().strip()).name if self.folder_var.get().strip() else ""
        add_row("Device Name / ID", "device_name", "e.g., DEV-1234", folder_name)
        add_row("Notes", "notes", "optional notes")

        result = {"ok": False, "data": {}}

        def ok():
            result["ok"] = True
            result["data"] = {
                "device_name": vars_["device_name"].get().strip(),
                "notes": vars_["notes"].get().strip()
            }
            win.destroy()

        def cancel():
            win.destroy()

        btns = ctk.CTkFrame(container, fg_color="transparent")
        btns.pack(fill="x", pady=10)
        ctk.CTkButton(btns, text="OK", command=ok, width=100).pack(side="left", padx=5)
        ctk.CTkButton(btns, text="Skip", command=cancel, width=100).pack(side="left", padx=5)

        win.wait_window()
        return result["data"]