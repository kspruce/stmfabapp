"""
STM Fabrication Record Generator - Enhanced Version with Full Integration

Enhanced version with:
- Database persistence using SQLAlchemy
- LabVIEW file integration with process summaries
- Thermal budget tracking with live display
- Quality check system with dialog interface
- Cooldown curve analysis and calibration
- Sample-centric organization

Version: 2.0
Date: October 30, 2025
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from pathlib import Path
import threading
import sys
import numpy as np
from collections import defaultdict
from sqlalchemy.exc import IntegrityError

# Import new analysis modules
from stm_fab.db.models import init_database, Sample, Device, FabricationStep, STMScan, ProcessStep, ThermalBudget, CooldownCalibration, QualityCheck as DBQualityCheck
from stm_fab.db.operations import DatabaseOperations
from stm_fab.db.numpy_fix import patch_database_operations
from stm_fab.labview.labview_parser import LabVIEWParser
from stm_fab.labview.process_summarizer import ProcessSummarizer
from stm_fab.labview.bulk_import import (
    LabVIEWBulkImporter,
    add_bulk_import_button_to_setup_tab,
    show_bulk_import_dialog,
    create_labview_management_tab,
    refresh_labview_file_list,
)
from stm_fab.analysis.cooldown_analysis import CooldownAnalyzer
from stm_fab.analysis.html_report import build_device_report
from stm_fab.gui.device_management import DeviceManagementMixin
from stm_fab.gui.labview_analysis_tab import LabVIEWAnalysisTab
from stm_fab.reports.fabrication_record import (
    FABRICATION_STEPS,
    load_gwyddion_colormap,
    parse_sxm_metadata,
    generate_image_base64 as create_stm_image_base64,
    FabricationRecordGenerator
)
from stm_fab.quality.quality_checks import QualityCheckManager, QualityCheck, CheckResult, CheckType, CheckCategory
from stm_fab.thermal_budget import ThermalBudgetCalculator
from stm_fab.analysis.resistance_fit import fit_resistance_vs_temperature, current_for_temperature
from stm_fab.gui.cooldown_tab import CooldownTab
from stm_fab.gui.cooldown_comparison_tab import CooldownComparisonTab

patch_database_operations(DatabaseOperations)

# Configure CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class STMFabGUIEnhanced(DeviceManagementMixin):
    """
    Enhanced GUI application with full feature integration
    """

    def __init__(self, root):
        self.root = root
        self.root.title("STM Fabrication Record Generator - Enhanced v2.5")
        self.root.geometry("1600x1000")

        # Database
        self.db_session = init_database()

        # Enhanced components
        self.quality_manager = QualityCheckManager()
        self.thermal_calculator = ThermalBudgetCalculator()
        self.db_ops = DatabaseOperations(self.db_session)
        self.process_summarizer = ProcessSummarizer()
        self.cooldown_analyzer = None  # Created when needed

        # Current data
        self.current_sample = None
        self.current_device = None
        self.current_labview_files = []

        # GUI variables
        self.device_name = tk.StringVar()
        self.sample_name = tk.StringVar()
        self.scan_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.use_standard_steps = tk.BooleanVar(value=True)
        self.labview_folder = tk.StringVar()

        # Data storage
        self.custom_steps = []
        self.generator = None
        self.current_step_index = 0

        # Selection vars for DB tab
        self._device_select_vars = {}
        self._device_sample_map = {}

        # Per-step scan counts and button refs for UI display
        self._step_scan_counts = defaultdict(int)  # step_num -> count
        self._step_buttons = {}  # step_num -> button widget

        # Default output folder
        self.output_folder.set(str(Path.cwd() / 'output'))

        # Create menu bar
        self.create_menu_bar()

        # Create widgets
        self.create_widgets()
        self.load_database_tab()

    def create_menu_bar(self):
        """Create menu bar with analysis and view options"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Device", command=self.new_device)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Analyze Cooldown Curve", command=self.analyze_cooldown_from_flash)
        analysis_menu.add_command(label="Import LabVIEW Process", command=self.import_and_show_labview_process)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="View Thermal Budget", command=self.show_thermal_budget_breakdown)
        analysis_menu.add_command(label="Compare Cooldown Curves", command=self.open_cooldown_comparison_tab)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh Database", command=self.load_database_tab)
        view_menu.add_command(label="Refresh Thermal Budget", command=self.update_thermal_budget_display)

    def create_widgets(self):
        """Create main tab interface in the requested order"""
        self.tabview = ctk.CTkTabview(self.root, corner_radius=15)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # 1) Setup
        self.tabview.add("⚙️  Setup")

        # 2) Database
        self.tabview.add("💾  Database")

        # 3) LabVIEW Files — add explicitly before helper tries to access it
        self.tabview.add("📁  LabVIEW Files")

        # 4) Cooldown Analysis
        self.tabview.add("❄️  Cooldown Analysis")

        # LabVIEW Files management (helper expects "📁  LabVIEW Files" to exist)
        create_labview_management_tab(self, self.tabview)

        # 5) LabVIEW Analysis (created via its class at this point)
        self.labview_analysis_tab = LabVIEWAnalysisTab(self, self.tabview)

        # 6) Fabrication Steps
        self.tabview.add("🔬  Fabrication Steps")

        # 7) Generate Report
        self.tabview.add("📊  Generate Report")

        # 8) Cooldown Comparison via its class
        self.cooldown_comparison = CooldownComparisonTab(self, self.tabview)

        # 9) Thermal Budget
        self.tabview.add("🌡️  Thermal Budget")

        # Build content for tabs we manage directly
        self.create_setup_tab()
        self.create_database_tab()
        self.create_cooldown_tab()
        self.create_steps_tab()
        self.create_report_tab()
        self.create_thermal_budget_tab()

        # Status bar
        self.status_bar = ctk.CTkLabel(
            self.root,
            text="Ready - Enhanced v2.0 with full integration",
            height=35,
            corner_radius=0,
            fg_color=("gray75", "gray25")
        )
        self.status_bar.pack(side="bottom", fill="x")

    def apply_sample_paths_to_ui(self, sample):
        """Load stored paths from DB into the Setup tab and LabVIEW Analysis tab."""
        try:
            if not sample:
                return
            paths = self.db_ops.get_sample_paths(sample.sample_id)
            labview_path = paths.get("labview_folder_path") or ""
            scan_path = paths.get("scan_folder_path") or ""

            if labview_path:
                self.labview_folder.set(labview_path)
                # Also push to LabVIEW Analysis tab's folder field
                if hasattr(self, "labview_analysis_tab") and hasattr(self.labview_analysis_tab, "folder_var"):
                    self.labview_analysis_tab.folder_var.set(labview_path)

            if scan_path:
                self.scan_folder.set(scan_path)

        except Exception as e:
            # Non-fatal: just log/status
            self.update_status(f"Note: failed to apply sample paths - {e}")

    def open_cooldown_comparison_tab(self):
        """Navigate to the cooldown comparison tab"""
        try:
            self.tabview.set("🔥 Cooldown Comparison")
            return
        except Exception:
            pass
        try:
            for name in self.tabview._name_list:
                if "Comparison" in name:
                    self.tabview.set(name)
                    return
        except Exception:
            pass

    def create_cooldown_tab(self):
        """Create cooldown analysis tab"""
        cooldown_tab = self.tabview.tab("❄️  Cooldown Analysis")
        self.cooldown_widget = CooldownTab(parent=cooldown_tab, db_ops=self.db_ops)
        self.cooldown_widget.pack(fill="both", expand=True)
        self.cooldown_analyzer_tab = self.cooldown_widget

    def create_setup_tab(self):
        """Create setup/configuration tab"""
        setup_tab = self.tabview.tab("⚙️  Setup")
        scroll_frame = ctk.CTkScrollableFrame(setup_tab, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            scroll_frame,
            text="Configure Your Enhanced Fabrication Record",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(pady=(0, 10))

        ctk.CTkLabel(
            scroll_frame,
            text="Sample-centric organization with database persistence",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        ).pack(pady=(0, 30))

        # Sample Configuration
        sample_frame = ctk.CTkFrame(scroll_frame, corner_radius=15)
        sample_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            sample_frame,
            text="Sample Configuration",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=(20, 15))

        sample_container = ctk.CTkFrame(sample_frame, fg_color="transparent")
        sample_container.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            sample_container,
            text="🧪  Sample Name",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(0, 5))

        self.sample_entry = ctk.CTkEntry(
            sample_container,
            textvariable=self.sample_name,
            placeholder_text="e.g., Si_2025_Q4_001",
            height=40
        )
        self.sample_entry.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(
            sample_container,
            text="Enter a unique identifier for your sample",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(anchor="w")

        # Device Configuration
        config_frame = ctk.CTkFrame(scroll_frame, corner_radius=15)
        config_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            config_frame,
            text="Device Configuration",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=(20, 15))

        device_container = ctk.CTkFrame(config_frame, fg_color="transparent")
        device_container.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            device_container,
            text="🔬  Device Name",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(0, 5))

        self.device_entry = ctk.CTkEntry(
            device_container,
            textvariable=self.device_name,
            placeholder_text="e.g., QD_Device_001",
            height=40
        )
        self.device_entry.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(
            device_container,
            text="Enter a unique identifier for your device",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(anchor="w")

        # LabVIEW Files (setup area)
        labview_frame = ctk.CTkFrame(scroll_frame, corner_radius=15)
        labview_frame.pack(fill="x", pady=(0, 20))

        # Bulk import button
        add_bulk_import_button_to_setup_tab(self, scroll_frame)

        ctk.CTkLabel(
            labview_frame,
            text="LabVIEW Data Files",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=(20, 15))

        labview_container = ctk.CTkFrame(labview_frame, fg_color="transparent")
        labview_container.pack(fill="x", padx=20, pady=10)

        labview_input_frame = ctk.CTkFrame(labview_container, fg_color="transparent")
        labview_input_frame.pack(fill="x")

        self.labview_entry = ctk.CTkEntry(
            labview_input_frame,
            textvariable=self.labview_folder,
            placeholder_text="Select folder containing LabVIEW .txt files",
            height=40
        )
        self.labview_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        ctk.CTkButton(
            labview_input_frame,
            text="Browse",
            command=self.browse_labview_folder,
            width=100,
            height=40
        ).pack(side="right")

        ctk.CTkButton(
            labview_input_frame,
            text="Import Process",
            command=self.import_and_show_labview_process,
            width=140,
            height=40,
            fg_color="orange",
            hover_color="darkorange"
        ).pack(side="right", padx=(0, 5))

        # Scan Folder
        scan_container = ctk.CTkFrame(config_frame, fg_color="transparent")
        scan_container.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            scan_container,
            text="📁  Scan Folder",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(0, 5))

        scan_input_frame = ctk.CTkFrame(scan_container, fg_color="transparent")
        scan_input_frame.pack(fill="x")

        self.scan_entry = ctk.CTkEntry(
            scan_input_frame,
            textvariable=self.scan_folder,
            placeholder_text="Select folder containing .sxm files",
            height=40
        )
        self.scan_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        ctk.CTkButton(
            scan_input_frame,
            text="Browse",
            command=self.browse_scan_folder,
            width=100,
            height=40
        ).pack(side="right")

        # Initialize button
        init_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        init_frame.pack(fill="x", pady=20)

        self.init_button = ctk.CTkButton(
            init_frame,
            text="Initialize Device",
            command=self.initialize_device,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.init_button.pack(pady=10)

    def create_steps_tab(self):
        """Create fabrication steps tab"""
        steps_tab = self.tabview.tab("🔬  Fabrication Steps")

        # This will be populated when device is initialized
        self.steps_container = ctk.CTkScrollableFrame(steps_tab)
        self.steps_container.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            self.steps_container,
            text="Initialize a device to see fabrication steps",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        ).pack(pady=50)

    def create_report_tab(self):
        """Create report generation tab"""
        report_tab = self.tabview.tab("📊  Generate Report")

        scroll_frame = ctk.CTkScrollableFrame(report_tab, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            scroll_frame,
            text="Generate Final Report",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=20)

        # Generate button
        self.generate_button = ctk.CTkButton(
            scroll_frame,
            text="Generate HTML Report",
            command=self.generate_report,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled"
        )
        self.generate_button.pack(pady=20)

    def create_database_tab(self):
        """Create database viewing tab"""
        db_tab = self.tabview.tab("💾  Database")

        self.db_container = ctk.CTkScrollableFrame(db_tab)
        self.db_container.pack(fill="both", expand=True, padx=20, pady=20)

    def create_thermal_budget_tab(self):
        """Create thermal budget monitoring tab"""
        thermal_tab = self.tabview.tab("🌡️  Thermal Budget")

        scroll_frame = ctk.CTkScrollableFrame(thermal_tab, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Create thermal budget panel
        self.create_thermal_budget_panel(scroll_frame)

    def create_thermal_budget_panel(self, parent):
        """Create thermal budget status display panel"""
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.pack(fill="x", padx=10, pady=10)

        # Header with icon
        header_frame = ctk.CTkFrame(frame)
        header_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            header_frame,
            text="🔥 Thermal Budget Status",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(side="left")

        ctk.CTkButton(
            header_frame,
            text="View Details",
            width=120,
            command=self.show_thermal_budget_breakdown
        ).pack(side="right")

        # Progress bar container
        progress_frame = ctk.CTkFrame(frame)
        progress_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            progress_frame,
            text="Progress:",
            font=ctk.CTkFont(size=14)
        ).pack(side="left", padx=5)

        self.thermal_progress = ctk.CTkProgressBar(progress_frame, width=400)
        self.thermal_progress.pack(side="left", padx=5, fill="x", expand=True)
        self.thermal_progress.set(0)

        # Status label
        self.thermal_label = ctk.CTkLabel(
            frame,
            text="No thermal budget data",
            font=ctk.CTkFont(size=14)
        )
        self.thermal_label.pack(pady=10)

        # Warning message
        self.thermal_warning = ctk.CTkLabel(
            frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="orange"
        )
        self.thermal_warning.pack(pady=5)

        # Buttons
        button_frame = ctk.CTkFrame(frame)
        button_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(
            button_frame,
            text="🔄 Refresh Budget",
            command=self.update_thermal_budget_display,
            width=140,
            height=32
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="📊 View Breakdown",
            command=self.show_thermal_budget_breakdown,
            width=140,
            height=32
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="📈 Analyze Cooldown",
            command=self.analyze_cooldown_from_flash,
            width=160,
            height=32,
            fg_color="purple",
            hover_color="darkviolet"
        ).pack(side="left", padx=5)

        # Initial update
        self.update_thermal_budget_display()

    # ==================== THERMAL BUDGET FUNCTIONS ====================

    def update_thermal_budget_display(self):
        """Update thermal budget display with current data"""
        if not self.current_sample:
            self.thermal_label.configure(text="No sample selected")
            self.thermal_progress.set(0)
            self.thermal_warning.configure(text="")
            return

        # Get thermal budget from database
        budget = self.db_ops.get_thermal_budget(self.current_sample.sample_id)

        if not budget:
            self.thermal_label.configure(text="No thermal budget data for this sample")
            self.thermal_progress.set(0)
            self.thermal_warning.configure(text="")
            return

        # Compute effective and gross totals from contributions
        degas = getattr(budget, 'degas_contribution', 0) or 0
        flash = getattr(budget, 'flash_contribution', 0) or 0
        hterm = getattr(budget, 'hterm_contribution', 0) or 0
        incorporation = getattr(budget, 'incorporation_contribution', 0) or 0
        overgrowth = getattr(budget, 'overgrowth_contribution', 0) or 0
        other = getattr(budget, 'other_contribution', 0) or 0

        effective_total = incorporation + overgrowth
        gross_total = degas + flash + hterm + incorporation + overgrowth + other

        # Calculate progress (0 to 1) based on effective total only
        if getattr(budget, 'warning_threshold', 0) > 0:
            progress = min(effective_total / budget.warning_threshold, 1.0)
        else:
            progress = 0

        self.thermal_progress.set(progress)

        # Determine status and color using effective total only
        crit = getattr(budget, 'critical_threshold', 0) or 0
        warn = getattr(budget, 'warning_threshold', 0) or 0
        if crit > 0 and effective_total >= crit:
            status = 'critical'
        elif warn > 0 and effective_total >= warn:
            status = 'warning'
        else:
            status = 'normal'

        if status == 'normal':
            color = "green"
            warning_text = ""
        elif status == 'warning':
            color = "orange"
            warning_text = "⚠️ Approaching thermal budget limit"
        else:  # critical
            color = "red"
            warning_text = "🚨 CRITICAL: Thermal budget exceeded!"

        # Update labels
        self.thermal_label.configure(
            text=f"Counted: {effective_total:.2e} / {budget.warning_threshold:.2e} °C·s ({progress*100:.1f}%)\n"
                 f"Gross:   {gross_total:.2e} °C·s",
            text_color=color
        )

        self.thermal_warning.configure(text=warning_text, text_color=color)

    def show_thermal_budget_breakdown(self):
        """Show detailed thermal budget breakdown in a dialog"""
        if not self.current_sample:
            messagebox.showwarning("No Sample", "Please select or create a sample first")
            return

        budget = self.db_ops.get_thermal_budget(self.current_sample.sample_id)
        if not budget:
            messagebox.showinfo("No Data", "No thermal budget data available")
            return

        # Create breakdown window
        dialog = ctk.CTkToplevel(self.root)
        dialog.title(f"Thermal Budget Breakdown - {self.current_sample.sample_name}")
        dialog.geometry("700x700")

        # Main container
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        ctk.CTkLabel(
            main_frame,
            text="Thermal Budget Breakdown",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=15)

        # Compute effective/gross totals and status (effective-based)
        degas = getattr(budget, 'degas_contribution', 0) or 0
        flash = getattr(budget, 'flash_contribution', 0) or 0
        hterm = getattr(budget, 'hterm_contribution', 0) or 0
        incorporation = getattr(budget, 'incorporation_contribution', 0) or 0
        overgrowth = getattr(budget, 'overgrowth_contribution', 0) or 0
        other = getattr(budget, 'other_contribution', 0) or 0
        effective_total = incorporation + overgrowth
        gross_total = degas + flash + hterm + incorporation + overgrowth + other
        crit = getattr(budget, 'critical_threshold', 0) or 0
        warn = getattr(budget, 'warning_threshold', 0) or 0
        if crit > 0 and effective_total >= crit:
            status = 'critical'
        elif warn > 0 and effective_total >= warn:
            status = 'warning'
        else:
            status = 'normal'
        status_colors = {'normal': 'green', 'warning': 'orange', 'critical': 'red'}

        # Total budget display
        total_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        total_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            total_frame,
            text=f"Effective Total (counted): {effective_total:.2e} °C·s",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 2))

        ctk.CTkLabel(
            total_frame,
            text=f"Gross Total (all): {gross_total:.2e} °C·s",
            font=ctk.CTkFont(size=14)
        ).pack(pady=(0, 10))

        ctk.CTkLabel(
            total_frame,
            text=f"Status: {status.upper()}",
            text_color=status_colors.get(status, 'white'),
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)

        # Contributions table
        ctk.CTkLabel(
            main_frame,
            text="Process Contributions:",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=15)

        # Create scrollable frame for contributions
        contrib_frame = ctk.CTkScrollableFrame(main_frame, height=350)
        contrib_frame.pack(fill="both", expand=True, pady=10)

        contributions = [
            ('Degas/Outgas', degas, 'degas'),
            ('Flash Cleaning', flash, 'flash'),
            ('H-Termination', hterm, 'hterm'),
            ('Incorporation', incorporation, 'incorporation'),
            ('Overgrowth', overgrowth, 'overgrowth'),
            ('Other', other, 'other')
        ]
        counted_keys = {'incorporation', 'overgrowth'}

        any_nonzero = any(val > 0 for _, val, _ in contributions)

        if not any_nonzero:
            ctk.CTkLabel(
                contrib_frame,
                text="No contributions recorded for this sample yet.",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            ).pack(pady=10)
        else:
            for process_name, contribution, key in contributions:
                if contribution > 0:
                    # Show shares relative to the gross total for clarity
                    percentage = (contribution / gross_total * 100) if gross_total > 0 else 0

                    row_frame = ctk.CTkFrame(contrib_frame)
                    row_frame.pack(fill="x", pady=8, padx=5)

                    label_text = f"{process_name}{' (counted)' if key in counted_keys else ''}"
                    ctk.CTkLabel(
                        row_frame,
                        text=label_text,
                        width=200,
                        anchor="w",
                        font=ctk.CTkFont(size=13)
                    ).pack(side="left", padx=5)

                    progress = ctk.CTkProgressBar(row_frame, width=220)
                    progress.set(percentage / 100.0)
                    progress.pack(side="left", padx=5)

                    ctk.CTkLabel(
                        row_frame,
                        text=f"{contribution:.2e} °C·s ({percentage:.1f}%)",
                        width=220,
                        font=ctk.CTkFont(size=12)
                    ).pack(side="left", padx=5)

        # Close button
        ctk.CTkButton(
            main_frame,
            text="Close",
            command=dialog.destroy,
            width=120,
            height=35
        ).pack(pady=15)

    # ==================== QUALITY CHECK FUNCTIONS ====================

    def show_quality_check_dialog(self, step_id, step_name):
        """Show quality check dialog for a fabrication step"""
        # Get applicable checks for this step
        checks = self.quality_manager.get_checks_for_step(step_name)

        if not checks:
            messagebox.showinfo("No Checks", "No quality checks defined for this step")
            return

        # Create dialog
        dialog = ctk.CTkToplevel(self.root)
        dialog.title(f"Quality Checks - {step_name}")
        dialog.geometry("750x650")

        # Main container
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        ctk.CTkLabel(
            main_frame,
            text=f"Quality Checks for:\n{step_name}",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=15)

        # Scrollable frame for checks
        checks_frame = ctk.CTkScrollableFrame(main_frame, height=420)
        checks_frame.pack(fill="both", expand=True, pady=10)

        # Store check widgets for later retrieval
        check_widgets = {}

        for check in checks:
            # Check frame
            check_frame = ctk.CTkFrame(checks_frame, corner_radius=5)
            check_frame.pack(fill="x", pady=8, padx=5)

            # Check name and description
            header_frame = ctk.CTkFrame(check_frame)
            header_frame.pack(fill="x", padx=10, pady=8)

            category_emoji = {
                CheckCategory.CRITICAL: "🔴",
                CheckCategory.IMPORTANT: "🟡",
                CheckCategory.INFORMATIONAL: "🔵"
            }

            ctk.CTkLabel(
                header_frame,
                text=f"{category_emoji.get(check.category, '')} {check.name}",
                font=ctk.CTkFont(size=13, weight="bold")
            ).pack(anchor="w")

            ctk.CTkLabel(
                header_frame,
                text=check.description,
                font=ctk.CTkFont(size=10),
                text_color="gray"
            ).pack(anchor="w", pady=2)

            # Pass/Fail selection
            result_frame = ctk.CTkFrame(check_frame)
            result_frame.pack(fill="x", padx=10, pady=5)

            pass_var = tk.BooleanVar(value=True)

            ctk.CTkRadioButton(
                result_frame,
                text="✓ Pass",
                variable=pass_var,
                value=True
            ).pack(side="left", padx=10)

            ctk.CTkRadioButton(
                result_frame,
                text="✗ Fail",
                variable=pass_var,
                value=False
            ).pack(side="left", padx=10)

            # Notes field
            ctk.CTkLabel(
                check_frame,
                text="Notes:",
                font=ctk.CTkFont(size=10)
            ).pack(anchor="w", padx=10)

            notes_entry = ctk.CTkTextbox(check_frame, height=60)
            notes_entry.pack(fill="x", padx=10, pady=5)

            # Store widgets
            check_widgets[check.check_id] = {
                'check': check,
                'pass_var': pass_var,
                'notes': notes_entry
            }

        # Button frame
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=10)

        def save_checks():
            """Save all quality check results"""
            results = []

            for check_id, widgets in check_widgets.items():
                check = widgets['check']
                passed = widgets['pass_var'].get()
                notes = widgets['notes'].get("1.0", "end-1c")

                # Create result
                result = self.quality_manager.create_manual_check_result(
                    check=check,
                    passed=passed,
                    notes=notes,
                    operator=self.generator.operator_name if self.generator else "User"
                )
                results.append(result)

                # Save to database
                self.db_ops.add_quality_check(
                    step_id=step_id,
                    check_name=check.name,
                    passed=passed,
                    check_type=check.check_type.value,
                    category=check.category.value,
                    notes=notes,
                    checked_by=result.checked_by
                )

            # Optional summary
            _summary = self.quality_manager.generate_check_report(results)

            # Check if any critical checks failed
            critical_failed = any(
                not r.passed for r in results
                if r.check.category == CheckCategory.CRITICAL
            )

            if critical_failed:
                messagebox.showwarning(
                    "Critical Checks Failed",
                    "One or more critical quality checks failed!\n\n"
                    "Review the results before proceeding."
                )
            else:
                messagebox.showinfo(
                    "Quality Checks Complete",
                    "All quality checks completed successfully!"
                )

            dialog.destroy()

        ctk.CTkButton(
            button_frame,
            text="Save Results",
            command=save_checks,
            width=130,
            height=35,
            fg_color="green",
            hover_color="darkgreen"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            width=100,
            height=35
        ).pack(side="left", padx=5)

    # ==================== COOLDOWN ANALYSIS FUNCTIONS ====================

    def analyze_cooldown_from_flash(self):
        """Analyze cooldown curve from flash file"""
        if not self.current_sample:
            messagebox.showwarning("No Sample", "Please select a sample first")
            return

        # File dialog to select flash file
        flash_file = filedialog.askopenfilename(
            title="Select Flash File",
            filetypes=[("LabVIEW Files", "*.txt"), ("All Files", "*.*")]
        )

        if not flash_file:
            return

        try:
            self.update_status("Analyzing cooldown curve...")

            # Parse LabVIEW file
            parser = LabVIEWParser(flash_file)
            data = parser.parse()

            # Extract cooldown
            analyzer = CooldownAnalyzer()

            # Get temperature and current data
            channels = data.get('channels', {})
            temp = channels.get('Pyro_T', np.array([]))
            current = channels.get('TDK_I', np.array([]))

            # Get time from first column of data array
            if 'data' in data and data['data'] is not None and len(data['data']) > 0 and 'Time' in data['data'].columns:
                time = data['data']['Time'].values
            else:
                raise ValueError("No time data found in file")

            if len(temp) == 0 or len(current) == 0:
                raise ValueError("Temperature or current data not found in file")

            # Extract cooldown
            _cooldown = analyzer.extract_cooldown_curve(temp, current, time)

            # Create calibration
            calibration = analyzer.create_calibration(min_current=0.05, max_current=0.25)

            # Show results dialog
            self.show_cooldown_results(analyzer, flash_file)

            # Save to database
            self.db_ops.add_cooldown_calibration(
                sample_id=self.current_sample.sample_id,
                labview_file_path=flash_file,
                calibration_data=calibration
            )

            self.update_status("Cooldown calibration created successfully!")

            messagebox.showinfo(
                "Success",
                "Cooldown calibration created and saved to database!"
            )

        except Exception as e:
            self.update_status("Error analyzing cooldown")
            messagebox.showerror("Error", f"Failed to analyze cooldown:\n{str(e)}")

    def show_cooldown_results(self, analyzer, flash_file):
        """Show cooldown analysis results"""
        # Create results window
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Cooldown Analysis Results")
        dialog.geometry("900x750")

        # Main container
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        ctk.CTkLabel(
            main_frame,
            text="Cooldown Calibration Results",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=15)

        # File info
        ctk.CTkLabel(
            main_frame,
            text=f"File: {Path(flash_file).name}",
            font=ctk.CTkFont(size=11)
        ).pack()

        # Report text
        report_text = analyzer.generate_calibration_report()

        text_frame = ctk.CTkFrame(main_frame)
        text_frame.pack(fill="both", expand=True, pady=15)

        text_widget = ctk.CTkTextbox(text_frame, font=("Courier", 10))
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert("1.0", report_text)
        text_widget.configure(state="disabled")

        # Buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=10)

        ctk.CTkButton(
            button_frame,
            text="📊 Plot Cooldown Curve",
            command=lambda: analyzer.plot_cooldown_curve(),
            width=170,
            height=35
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="📈 Plot Calibration",
            command=lambda: analyzer.plot_calibration(),
            width=160,
            height=35
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Close",
            command=dialog.destroy,
            width=100,
            height=35
        ).pack(side="left", padx=5)

    # ==================== PROCESS SUMMARY FUNCTIONS ====================

    def import_and_show_labview_process(self):
        """Import LabVIEW file and show summary"""
        # File dialog
        filepath = filedialog.askopenfilename(
            title="Select LabVIEW Process File",
            filetypes=[("LabVIEW Files", "*.txt"), ("All Files", "*.*")]
        )

        if not filepath:
            return

        try:
            self.update_status("Parsing LabVIEW file...")

            # Parse file
            parser = LabVIEWParser(filepath)
            data = parser.parse()

            # Generate summary
            summary = self.process_summarizer.generate_summary(data)

            # Show summary dialog
            self.show_process_summary(summary, data, filepath)

            self.update_status("LabVIEW file parsed successfully")

        except Exception as e:
            self.update_status("Error parsing LabVIEW file")
            messagebox.showerror("Error", f"Failed to parse file:\n{str(e)}")

    def show_process_summary(self, summary_text, parsed_data, filepath):
        """Show process summary in a dialog"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Process Summary")
        dialog.geometry("900x750")

        # Main container
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        ctk.CTkLabel(
            main_frame,
            text="Process Summary",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=15)

        # File info
        ctk.CTkLabel(
            main_frame,
            text=f"File: {Path(filepath).name}",
            font=ctk.CTkFont(size=11)
        ).pack()

        # Summary text
        text_frame = ctk.CTkFrame(main_frame)
        text_frame.pack(fill="both", expand=True, pady=15)

        text_widget = ctk.CTkTextbox(text_frame, font=("Courier", 10))
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert("1.0", summary_text)
        text_widget.configure(state="disabled")

        # Action buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=10)

        def link_to_sample():
            """Link this process to current sample"""
            if not self.current_sample:
                messagebox.showwarning("No Sample", "Please select a sample first")
                return

            try:
                # Add to database
                self.db_ops.add_process_step(
                    sample_id=self.current_sample.sample_id,
                    process_type=parsed_data['file_type'],
                    labview_file_path=filepath,
                    parsed_data=parsed_data
                )

                messagebox.showinfo("Success", "Process linked to sample!")

                # Update thermal budget display
                self.update_thermal_budget_display()

                dialog.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to link process:\n{str(e)}")

        ctk.CTkButton(
            button_frame,
            text="Link to Current Sample",
            command=link_to_sample,
            width=180,
            height=35,
            fg_color="green",
            hover_color="darkgreen"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Close",
            command=dialog.destroy,
            width=100,
            height=35
        ).pack(side="left", padx=5)

    # ==================== FABRICATION STEPS: LINK SCANS ====================

    def populate_steps_tab(self):
        """Populate the fabrication steps tab"""
        # Clear existing content
        for widget in self.steps_container.winfo_children():
            widget.destroy()

        if not self.generator:
            return

        ctk.CTkLabel(
            self.steps_container,
            text="Fabrication Steps",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=15)

        self._step_buttons.clear()

        for step in FABRICATION_STEPS[:6]:
            step_frame = ctk.CTkFrame(self.steps_container, corner_radius=10)
            step_frame.pack(fill="x", pady=5, padx=10)

            ctk.CTkLabel(
                step_frame,
                text=f"Step {step['step_num']}: {step['name']}",
                font=ctk.CTkFont(size=13, weight="bold")
            ).pack(anchor="w", padx=10, pady=5)

            button_frame = ctk.CTkFrame(step_frame, fg_color="transparent")
            button_frame.pack(fill="x", padx=10, pady=5)

            if step.get('requires_scan', True):
                btn_text = self._add_scan_button_text(step['step_num'])
                add_btn = ctk.CTkButton(
                    button_frame,
                    text=btn_text,
                    width=130,
                    height=28,
                    command=lambda s=step: self.add_scan_to_step(s)
                )
                add_btn.pack(side="left", padx=2)
                self._step_buttons[step['step_num']] = add_btn

            ctk.CTkButton(
                button_frame,
                text="Quality Checks",
                command=lambda s=step: self.show_quality_check_dialog(0, s['name']),
                width=120,
                height=28,
                fg_color="orange",
                hover_color="darkorange"
            ).pack(side="left", padx=2)

    def _add_scan_button_text(self, step_num: int) -> str:
        count = self._step_scan_counts.get(step_num, 0)
        return f"Add Scan (+{count})" if count > 0 else "Add Scan"

    def add_scan_to_step(self, step_def: dict):
        """Attach an .sxm file to a fabrication step and persist to DB."""
        if not self.current_device:
            messagebox.showwarning("No Device", "Please initialize or select a device first.")
            return

        # File dialog
        initial_dir = self.scan_folder.get() or str(Path.cwd())
        filepath = filedialog.askopenfilename(
            title=f"Select .sxm file for Step {step_def['step_num']}",
            initialdir=initial_dir,
            filetypes=[("SXM files", "*.sxm"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            # Parse metadata
            try:
                metadata = parse_sxm_metadata(filepath) or {}
            except Exception:
                metadata = {}

            # Optional: create preview image base64
            try:
                image_b64 = create_stm_image_base64(filepath)
            except Exception:
                image_b64 = None

            # Ensure FabricationStep exists for this device + step_number
            step_row = self._get_or_create_fabrication_step(
                device_id=self.current_device.device_id,
                step_number=step_def['step_num'],
                step_name=step_def['name']
            )

            # Persist scan
            self._persist_scan_to_db(step_row, filepath, metadata, image_b64)

            # Optionally update step status to 'complete' after at least one scan
            try:
                if getattr(step_row, "status", "pending") != "complete":
                    self.db_ops.update_step_status(step_row.step_id, status="complete")
            except Exception:
                pass

            # Update UI
            self._step_scan_counts[step_def['step_num']] += 1
            if step_def['step_num'] in self._step_buttons:
                self._step_buttons[step_def['step_num']].configure(
                    text=self._add_scan_button_text(step_def['step_num'])
                )

            messagebox.showinfo(
                "Scan Linked",
                f"Linked scan to Step {step_def['step_num']}: {Path(filepath).name}"
            )
            self.status_bar.configure(
                text=f"Linked scan to Step {step_def['step_num']}: {Path(filepath).name}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to link scan:\n{e}")

    def _get_or_create_fabrication_step(self, device_id: int, step_number: int, step_name: str):
        """
        Find or create a FabricationStep in the database for the given device and step_number.
        """
        # Try to find an existing step by number from known steps
        try:
            steps = self.db_ops.get_device_steps(device_id)
            for s in steps:
                if getattr(s, "step_number", None) == step_number:
                    return s
        except Exception:
            pass

        # If not found, create one (note: DatabaseOperations.add_fabrication_step expects step_number)
        add_step = getattr(self.db_ops, "add_fabrication_step", None)
        if not callable(add_step):
            raise AttributeError("DatabaseOperations.add_fabrication_step is not available; please add it.")

        step_row = add_step(
            device_id=device_id,
            step_number=step_number,
            step_name=step_name,
            notes="",
            status="in_progress",
            requires_scan=True
        )
        return step_row

    def _persist_scan_to_db(self, step_row, file_path: str, metadata: dict, image_b64: str | None):
        """
        Persist an STM scan to the database and link it to the fabrication step.
        DatabaseOperations.add_stm_scan signature:
            add_stm_scan(step_id, filename, filepath, metadata, **kwargs)
        STMScan has 'image_data' column for base64 images.
        """
        filename = Path(file_path).name
        add_scan = getattr(self.db_ops, "add_stm_scan", None)
        if not callable(add_scan):
            raise AttributeError("DatabaseOperations.add_stm_scan is not available; please add it.")

        add_scan(
            step_id=getattr(step_row, "step_id", None),
            filename=filename,
            filepath=file_path,
            metadata=metadata,
            image_data=image_b64
        )

    # ==================== HELPER FUNCTIONS ====================

    def new_device(self):
        """Reset for new device"""
        self.sample_name.set("")
        self.device_name.set("")
        self.scan_folder.set("")
        self.current_sample = None
        self.current_device = None
        self.generator = None
        self._step_scan_counts.clear()
        self._step_buttons.clear()
        self.update_status("Ready for new device")


    def browse_labview_folder(self):
        folder = filedialog.askdirectory(title="Select LabVIEW Data Folder")
        if folder:
            self.labview_folder.set(folder)
            # Persist to DB if a sample is selected
            try:
                if self.current_sample:
                    self.db_ops.set_sample_paths(self.current_sample.sample_id,
                                                 labview_folder_path=folder)
                # Also push to LabVIEW Analysis tab input so Scan starts here
                if hasattr(self, "labview_analysis_tab"):
                    self.labview_analysis_tab.folder_var.set(folder)
            except Exception as e:
                self.update_status(f"Warning: failed to save labview folder ({e})")
    
    def browse_scan_folder(self):
        folder = filedialog.askdirectory(title="Select Scan Folder")
        if folder:
            self.scan_folder.set(folder)
            # Persist to DB if a sample is selected
            try:
                if self.current_sample:
                    self.db_ops.set_sample_paths(self.current_sample.sample_id,
                                                 scan_folder_path=folder)
            except Exception as e:
                self.update_status(f"Warning: failed to save scan folder ({e})")


    def initialize_device(self):
        """Initialize device in database"""
        sample_name = self.sample_name.get().strip()
        device_name = self.device_name.get().strip()

        if not sample_name or not device_name:
            messagebox.showwarning("Missing Info", "Please enter both sample and device names")
            return

        try:
            # Get or create sample
            self.current_sample = self.db_ops.create_sample(
                sample_name=sample_name,
                substrate_type="Si(100)",
                labview_folder_path=self.labview_folder.get() or None,
                scan_folder_path=self.scan_folder.get() or None
            )
            if not self.current_sample:
                self.current_sample = self.db_ops.create_sample(
                    sample_name=sample_name,
                    substrate_type="Si(100)"  # Default, can be customized
                )
                self.update_status(f"Created new sample: {sample_name}")
            else:
                self.update_status(f"Using existing sample: {sample_name}")

            # Get or create device
            self.current_device = self.db_ops.get_device_by_name(device_name)
            if not self.current_device:
                self.current_device = self.db_ops.create_device(
                    device_name=device_name,
                    sample_id=self.current_sample.sample_id,
                    operator="User"
                )
                self.update_status(f"Created new device: {device_name}")
            else:
                self.update_status(f"Using existing device: {device_name}")

            # After self.current_sample is set:
            self.apply_sample_paths_to_ui(self.current_sample)
        
            # Initialize generator (depends on scan_folder)
            self.initialize_generator()
        
            # Update thermal budget UI
            self.update_thermal_budget_display()

            # Enable generate button
            self.generate_button.configure(state="normal")

            messagebox.showinfo("Success", f"Device '{device_name}' initialized successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize device:\n{str(e)}")

    def initialize_generator(self):
        """Initialize the report generator"""
        if not self.device_name.get() or not self.scan_folder.get():
            messagebox.showwarning("Missing Information",
                                   "Please provide device name and scan folder")
            return

        try:
            self.generator = FabricationRecordGenerator(
                scan_folder=self.scan_folder.get(),
                device_name=self.device_name.get(),
                output_dir=self.output_folder.get()
            )

            _output_path = self.generator.save_report(output_format='html')

            # Populate steps tab
            self.populate_steps_tab()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize generator:\n{str(e)}")

    def generate_report(self):
        """Generate final HTML report"""
        if not self.generator:
            messagebox.showwarning("Not Initialized", "Please initialize device first")
            return

        try:
            self.update_status("Generating report...")

            # Generate using existing generator
            output_path = self.generator.save_report(output_format='html')

            self.update_status("Report generated successfully!")
            messagebox.showinfo("Success", f"Report saved to:\n{output_path}")

        except Exception as e:
            self.update_status("Error generating report")
            messagebox.showerror("Error", f"Failed to generate report:\n{str(e)}")

    def load_database_tab(self):
        """Load enhanced database view with management"""
        self.create_enhanced_database_tab()  # This method comes from the mixin

    def update_status(self, message):
        """Update status bar"""
        try:
            self.status_bar.configure(text=message)
            self.root.update_idletasks()
        except Exception:
            pass


def main():
    """Main entry point"""
    root = ctk.CTk()
    app = STMFabGUIEnhanced(root)
    root.mainloop()


if __name__ == "__main__":
    main()
