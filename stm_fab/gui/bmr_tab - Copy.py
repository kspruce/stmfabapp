"""
Batch Manufacturing Record (BMR) Tab for STM HDL Fabrication - Version 3

Uses JSON analysis files from LabVIEW Analysis Tab instead of direct LabVIEW files.

Author: Kieran Spruce
Date: November 2025
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog, simpledialog
import tkinter as tk
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import os

# Improved TableEditor using ttk.Treeview (robust headers, both scrollbars, in-place edit)
import tkinter as tk
from tkinter import ttk

class TableEditor(ctk.CTkFrame):
    def __init__(
        self,
        master,
        columns,
        default_rows=0,
        max_rows=999,
        data=None,
        entry_width=120,
        row_height=26,
        header_font=("Helvetica", 12, "bold"),
        header_bg="#334155",
        header_fg="#FFFFFF",
        select_bg="#1f6aa5",
    ):
        super().__init__(master)
        self.columns = columns  # list of dicts: {key,label,type}
        self.max_rows = max_rows
        self.entry_width = entry_width
        self._edit = None  # active editor: (entry, item_id, col_index)

        # Top button bar
        top = ctk.CTkFrame(self)
        top.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(top, text="+ Row", width=70, command=self.add_row).pack(side="left", padx=3)
        ctk.CTkButton(top, text="– Row", width=70, command=self.remove_row).pack(side="left", padx=3)

        # ttk style for Treeview
        self.style = ttk.Style(self)
        # Use a theme that allows heading bg/fg customizations
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(
            "BMR.Treeview",
            rowheight=row_height,
            background="white",
            fieldbackground="white",
            foreground="black",
        )
        # Selected row color
        self.style.map(
            "BMR.Treeview",
            background=[("selected", select_bg)],
            foreground=[("selected", "white")],
        )
        # Heading style (bold, contrasting)
        self.style.configure(
            "BMR.Treeview.Heading",
            font=header_font,
            background=header_bg,
            foreground=header_fg,
        )

        # Frame to host tree and scrollbars
        center = ctk.CTkFrame(self)
        center.pack(fill="both", expand=True)

        # Vertical scrollbar
        self.vsb = ttk.Scrollbar(center, orient="vertical")
        self.vsb.pack(side="right", fill="y")

        # Horizontal scrollbar (bottom)
        self.hsb = ttk.Scrollbar(self, orient="horizontal")
        self.hsb.pack(side="bottom", fill="x")

        # Build Treeview
        self.keys = [c.get("key") for c in columns]
        self.labels = [c.get("label", c.get("key", "")) for c in columns]

        self.tree = ttk.Treeview(
            center,
            columns=self.keys,
            show="headings",
            style="BMR.Treeview",
            xscrollcommand=self.hsb.set,
            yscrollcommand=self.vsb.set,
        )
        self.tree.pack(side="left", fill="both", expand=True)

        self.vsb.config(command=self.tree.yview)
        self.hsb.config(command=self.tree.xview)

        # Headings and columns
        for key, label in zip(self.keys, self.labels):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=entry_width, stretch=False, anchor="w")

        # Seed rows
        initial = data if (isinstance(data, list) and data) else [{}] * default_rows
        for row in initial:
            vals = [row.get(k, "") for k in self.keys]
            self.tree.insert("", "end", values=vals)

        # Double-click to edit cell
        self.tree.bind("<Double-1>", self._begin_edit)
        # Horizontal scroll with Shift + wheel
        self.tree.bind("<Shift-MouseWheel>", self._on_shift_wheel)
        # Linux Shift + Button-4/5
        self.tree.bind("<Shift-Button-4>", lambda e: self.tree.xview_scroll(-1, "units"))
        self.tree.bind("<Shift-Button-5>", lambda e: self.tree.xview_scroll(1, "units"))

    # ---------- API compatible with your code ----------

    def add_row(self):
        if len(self.tree.get_children()) >= self.max_rows:
            return
        self.tree.insert("", "end", values=["" for _ in self.keys])

    def remove_row(self):
        items = self.tree.get_children()
        if not items:
            return
        self.tree.delete(items[-1])

    def get_data(self):
        out = []
        for iid in self.tree.get_children():
            vals = self.tree.item(iid, "values")
            out.append({k: vals[i] if i < len(vals) else "" for i, k in enumerate(self.keys)})
        return out

    # ---------- cell editing ----------

    def _begin_edit(self, event):
        # Identify cell
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)  # '#1', '#2', ...
        if not row_id or not col_id:
            return
        col_index = int(col_id.replace("#", "")) - 1
        if col_index < 0 or col_index >= len(self.keys):
            return

        # Cell bbox (x,y,w,h) relative to tree
        bbox = self.tree.bbox(row_id, col_id)
        if not bbox:
            return
        x, y, w, h = bbox

        # Current value
        values = list(self.tree.item(row_id, "values"))
        current = values[col_index] if col_index < len(values) else ""

        # Destroy existing editor if any
        if self._edit and self._edit[0].winfo_exists():
            self._edit[0].destroy()
        # Place an Entry over the cell for editing
        entry = ttk.Entry(self.tree)
        entry.insert(0, str(current))
        entry.select_range(0, "end")
        entry.focus_set()
        entry.place(x=x, y=y, width=w, height=h)

        def commit(event=None):
            new_val = entry.get()
            # update values
            vals = list(self.tree.item(row_id, "values"))
            # ensure length
            while len(vals) < len(self.keys):
                vals.append("")
            vals[col_index] = new_val
            self.tree.item(row_id, values=vals)
            entry.destroy()
            self._edit = None

        def cancel(event=None):
            entry.destroy()
            self._edit = None

        entry.bind("<Return>", commit)
        entry.bind("<Escape>", cancel)
        entry.bind("<FocusOut>", commit)
        self._edit = (entry, row_id, col_index)

    # ---------- horizontal wheel support ----------

    def _on_shift_wheel(self, event):
        try:
            delta = int(event.delta)
        except Exception:
            delta = 0
        if delta != 0:
            self.tree.xview_scroll(-1 if delta > 0 else 1, "units")
            return "break"

class StepStatus(Enum):
    """Status of a manufacturing step"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BMRStepData:
    """Data structure for a single BMR step"""
    step_number: int
    step_name: str
    status: StepStatus = StepStatus.NOT_STARTED
    
    # Execution details
    operator_initials: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Process parameters
    parameters: Dict[str, Any] = None
    
    # Quality checks
    pre_check_pass: bool = False
    post_check_pass: bool = False
    quality_notes: str = ""
    
    # Issues and deviations
    deviations: List[str] = None
    corrective_actions: List[str] = None
    
    # References
    labview_file: str = ""
    sxm_scans: List[str] = None
    
    # Verification
    verified_by: str = ""
    verified_time: Optional[datetime] = None

    # NEW: step initials (explicit sign/initial) and time
    step_initials: str = ""
    step_initial_time: Optional[datetime] = None

    # NEW: per-item quality check results: { "check label": "Pass|Fail|N/A|''" }
    quality_check_results: Dict[str, str] = None


    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.deviations is None:
            self.deviations = []
        if self.corrective_actions is None:
            self.corrective_actions = []
        if self.sxm_scans is None:
            self.sxm_scans = []
        if self.quality_check_results is None:
            self.quality_check_results = {}


# Standard SET (Single Electron Transistor) Process Steps - CUSTOM VERSION
STANDARD_BMR_STEPS = [
    {
        "step_number": 1,
        "step_name": "Sample Preparation & Cleaning",
        "parameters": {
            "required": ["cleaning_method", "visual_inspection"],
            "optional": ["cleaning_duration", "solvent_used", "file_name"]
        },
        "quality_checks": ["Surface cleanliness", "No visible contamination"]
    },
    {
        "step_number": 2,
        "step_name": "Sample Loading & Pumpdown",
        "parameters": {
            "required": ["load_time", "pumpdown_duration", "base_pressure_mbar"],
            "optional": ["peak_mbe_pressure_mbar", "file_name"]
        },
        "quality_checks": ["Loadlock pressure < 5×10⁻⁶ mbar"]
    },
    {
        "step_number": 3,
        "step_name": "Degas",
        "parameters": {
            "required": [
                "resistive_degas_temp_c",
                "resistive_degas_duration_min",
                "resistive_degas_pressure_mbar",
                "direct_degas_temp_c",
                "direct_degas_duration_min",
                "direct_degas_pressure_mbar"
            ],
            "optional": ["file_name"]
        },
        "quality_checks": ["Target temperatures reached", "Pressure stable"]
    },
    {
        "step_number": 4,
        "step_name": "Flash Clean",
        "parameters": {
            "required": ["flash_temperature_c", "flash_duration_s", "flash_current_a"],
            "optional": ["number_of_flashes", "cooldown_time_s", "file_name"]
        },
        "quality_checks": ["Temperature profile correct", "Surface reconstructed"]
    },
    {
        "step_number": 5,
        "step_name": "H-Termination",
        "parameters": {
            "required": ["h_termination_temp_c", "h_dose_time_s", "h2_pressure_mbar"],
            "optional": ["dose_langmuirs", "file_name"]
        },
        "quality_checks": ["Temperature 330°C ± 5°C", "Uniform H coverage"]
    },
    {
        "step_number": 6,
        "step_name": "STM Imaging Pre-Lithography",
        "parameters": {
            "required": ["write_field"],
            "optional": ["scan_folder_location", "scan_numbers_of_interest", "file_name"]
        },
        "quality_checks": ["Atomically resolved", "Terrace quality acceptable"]
    },
    {
        "step_number": 7,
        "step_name": "STM Lithography",
        "parameters": {
            "required": [
                "fem_voltage_v",
                "fem_current_pa",
                "fem_speed",
                "apm_voltage_v",
                "apm_current_pa",
                "apm_speed"
            ],
            "optional": ["fem_pitch", "apm_pitch", "file_name"]
        },
        "quality_checks": ["Pattern complete", "Line quality good"]
    },
    {
        "step_number": 8,
        "step_name": "Post-Litho STM Imaging",
        "parameters": {
            "required": ["scan_area_nm", "bias_voltage_v", "setpoint_current_pa"],
            "optional": ["scan_folder_location", "scan_number", "feature_measurements", "file_name"]
        },
        "quality_checks": ["Pattern verified", "Dimensions within spec"]
    },
    {
        "step_number": 9,
        "step_name": "Dopant Dosing (PH3 or AsH3)",
        "parameters": {
            "required": ["dopant_species", "dose_pressure_mbar", "dose_duration_s"],
            "optional": ["dose_langmuirs", "background_pressure_mbar", "file_name"]
        },
        "quality_checks": ["Dose within spec", "Pressure stable during dose"]
    },
    {
        "step_number": 10,
        "step_name": "Dopant Incorporation",
        "parameters": {
            "required": ["incorporation_temp_c", "incorporation_time_s"],
            "optional": ["thermal_budget_delta_cs", "file_name"]
        },
        "quality_checks": ["Temperature 350°C ± 5°C", "Surface morphology acceptable"]
    },
    {
        "step_number": 11,
        "step_name": "Post-Incorporation STM Imaging",
        "parameters": {
            "required": ["scan_area_nm", "bias_voltage_v", "setpoint_current_pa"],
            "optional": ["feature_visibility", "file_name"]
        },
        "quality_checks": ["Pattern still visible", "No major defects"]
    },
    {
        "step_number": 12,
        "step_name": "Silicon Overgrowth - RT Phase",
        "parameters": {
            "required": ["growth_temp_c", "growth_time_s", "si_flux"],
            "optional": ["target_thickness_rt_nm", "growth_rate", "file_name"]
        },
        "quality_checks": ["Growth rate calibrated", "Uniform coverage"]
    },
    {
        "step_number": 13,
        "step_name": "Silicon Overgrowth - RTA Anneal",
        "parameters": {
            "required": ["anneal_temp_c", "anneal_time_s"],
            "optional": ["thermal_budget_delta_cs", "file_name"]
        },
        "quality_checks": ["Temperature profile correct", "No delamination"]
    },
    {
        "step_number": 14,
        "step_name": "Silicon Overgrowth - LTE Phase",
        "parameters": {
            "required": ["growth_temp_c", "growth_time_s", "total_thickness_nm"],
            "optional": ["growth_rate_lte", "file_name"]
        },
        "quality_checks": ["Target thickness achieved", "Final thermal budget acceptable"]
    }
]


class BMRTab:
    """
    Batch Manufacturing Record Tab
    
    Provides comprehensive manufacturing record keeping with:
    - Step-by-step data entry
    - Real-time validation
    - Quality check integration
    - JSON analysis file import
    - Digital signatures
    - PDF export for compliance
    """
    
    def __init__(self, parent_app, tabview):
        """
        Initialize the BMR tab
        
        Args:
            parent_app: Main application instance (STMFabGUIEnhanced)
            tabview: The CTkTabview to add this tab to
        """
        self.app = parent_app
        self.tabview = tabview
        
        # Get or create the BMR tab
        tab_name = "📋  Batch Record"
        
        if tab_name not in self.tabview._tab_dict:
            self.tabview.add(tab_name)
        
        self.tab = self.tabview.tab(tab_name)
        
        # Current BMR data
        self.current_device = None
        self.bmr_steps: List[BMRStepData] = []
        self.current_step_index = 0
        self.bmr_metadata = {
            "batch_number": "",
            "device_id": "",
            "operator": "",
            "start_date": None,
            "target_completion": None
        }
        
        self._current_bmr_id = None  # Track current BMR database ID
        # NEW: active step definitions used to render the form
        self.active_step_defs = STANDARD_BMR_STEPS[:]
        
        # UI widgets storage
        self.step_cards = {}
        self.parameter_widgets = {}
        
        # Build UI
        self.build_ui()
        
        # Autosave
        self._autosave_job = None
        self._autosave_dir = Path.cwd() / "bmr_autosaves"
        self._autosave_dir.mkdir(exist_ok=True)
        self._last_autosave_path = None
        self._schedule_autosave()

    def _schedule_autosave(self):
        """Schedule the next auto-save in 300,000 ms (5 minutes)."""
        if self._autosave_job:
            self.tab.after_cancel(self._autosave_job)
        self._autosave_job = self.tab.after(300000, self._do_autosave)

    def _do_autosave(self):
        """Perform auto-save if BMR has data."""
        if self.bmr_steps:
            batch_num = self.bmr_metadata.get("batch_number", "untitled")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self._autosave_dir / f"BMR_{batch_num}_autosave_{timestamp}.json"
            try:
                self.bmr_metadata["operator"] = self.operator_entry.get().strip()
                save_data = self._to_json_dict()
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2)
                self._last_autosave_path = filename
            except Exception:
                pass
        self._schedule_autosave()

    def _to_json_dict(self) -> dict:
        """Convert current BMR to JSON-serializable dict."""
        def iso_or_none(dt):
            return dt.isoformat() if dt else None
        
        meta = self.bmr_metadata.copy()
        meta["start_date"] = iso_or_none(meta.get("start_date"))
        meta["target_completion"] = iso_or_none(meta.get("target_completion"))
        
        steps_out = []
        for s in self.bmr_steps:
            steps_out.append({
                "step_number": s.step_number,
                "step_name": s.step_name,
                "status": s.status.value,
                "operator_initials": s.operator_initials or "",
                "start_time": iso_or_none(s.start_time),
                "end_time": iso_or_none(s.end_time),
                "parameters": dict(s.parameters or {}),
                "quality_notes": s.quality_notes or "",
                "pre_check_pass": s.pre_check_pass,
                "post_check_pass": s.post_check_pass,
                "deviations": list(s.deviations or []),
                "corrective_actions": list(s.corrective_actions or []),
                "labview_file": s.labview_file or "",
                "sxm_scans": list(s.sxm_scans or []),
                "verified_by": s.verified_by or "",
                "verified_time": iso_or_none(s.verified_time),
                "step_initials": s.step_initials or "",
                "step_initial_time": iso_or_none(s.step_initial_time),
                "quality_check_results": dict(s.quality_check_results or {}),

                
            })
        # NEW: include active step definitions so templates round-trip
        return {"metadata": meta, "step_definitions": self.active_step_defs, "steps": steps_out}


    def _load_from_json_obj(self, data: dict):
        """Load BMR from a JSON dict."""
        self.bmr_metadata = data.get("metadata", {}) or {}
        if self.bmr_metadata.get("start_date"):
            self.bmr_metadata["start_date"] = datetime.fromisoformat(self.bmr_metadata["start_date"])
        if self.bmr_metadata.get("target_completion"):
            self.bmr_metadata["target_completion"] = datetime.fromisoformat(self.bmr_metadata["target_completion"])

        # NEW: pick up template step definitions (fallback to built-in)
        step_defs = data.get("step_definitions")
        if step_defs and isinstance(step_defs, list):
            self.active_step_defs = step_defs
        else:
            self.active_step_defs = STANDARD_BMR_STEPS[:]

        self.bmr_steps = []
        for sd in data.get("steps", []):
            if sd.get("start_time"):
                sd["start_time"] = datetime.fromisoformat(sd["start_time"])
            if sd.get("end_time"):
                sd["end_time"] = datetime.fromisoformat(sd["end_time"])
            if sd.get("verified_time"):
                sd["verified_time"] = datetime.fromisoformat(sd["verified_time"])
            if sd.get("step_initial_time"):
                sd["step_initial_time"] = datetime.fromisoformat(sd["step_initial_time"])
            sd["status"] = StepStatus(sd["status"])
            self.bmr_steps.append(BMRStepData(**sd))


        # Update UI
        if hasattr(self, "batch_number_entry"):
            self.batch_number_entry.delete(0, "end")
            self.batch_number_entry.insert(0, self.bmr_metadata.get("batch_number", ""))
        if hasattr(self, "operator_entry"):
            self.operator_entry.delete(0, "end")
            self.operator_entry.insert(0, self.bmr_metadata.get("operator", ""))
        self.current_step_index = 0
        self.update_progress_panel()
        self.display_step(0)
   
    def initial_step(self, step_index: int, initials: Optional[str] = None):
        """Stamp initials and time for this step."""
        if step_index >= len(self.bmr_steps):
            return
        step = self.bmr_steps[step_index]
        value = (initials or "").strip() or self.operator_entry.get().strip() or step.operator_initials.strip()
        if not value:
            messagebox.showwarning("Missing Initials", "Enter initials or set Operator Initials.")
            return
        step.step_initials = value
        step.step_initial_time = datetime.now()
        # Update UI
        self.display_step(step_index)
        self.update_progress_panel()


    def _templates_dir(self) -> Path:
        """Return the templates directory, creating if needed."""
        tdir = Path.cwd() / "bmr_templates"
        tdir.mkdir(exist_ok=True)
        return tdir
    
    def build_ui(self):
        """Build the BMR tab interface"""
        # Clear existing
        for widget in self.tab.winfo_children():
            widget.destroy()
        
        # Main container with three columns
        main_container = ctk.CTkFrame(self.tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left column - BMR header and control (fixed width)
        left_frame = ctk.CTkFrame(main_container, width=300)
        left_frame.pack(side="left", fill="both", padx=(0, 5))
        left_frame.pack_propagate(False)
        
        # Middle column - Step details (expands)
        middle_frame = ctk.CTkFrame(main_container)
        middle_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # Right column - Progress tracker (fixed width)
        right_frame = ctk.CTkFrame(main_container, width=250)
        right_frame.pack(side="left", fill="both", padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Build panels
        self._build_left_panel(left_frame)
        self._build_middle_panel(middle_frame)
        self._build_right_panel(right_frame)
    
    def _build_left_panel(self, parent):
        """Build the left control panel with BMR header info"""
        scroll_container = ctk.CTkScrollableFrame(parent)
        scroll_container.pack(fill="both", expand=True)
        
        # Header
        ctk.CTkLabel(
            scroll_container,
            text="Batch Manufacturing\nRecord",
            font=ctk.CTkFont(size=18, weight="bold"),
            justify="center"
        ).pack(pady=15)
        
        # BMR Metadata Frame
        metadata_frame = ctk.CTkFrame(scroll_container)
        metadata_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            metadata_frame,
            text="BMR Information",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        # Batch number
        ctk.CTkLabel(metadata_frame, text="Batch Number:").pack(anchor="w", padx=10, pady=2)
        self.batch_number_entry = ctk.CTkEntry(metadata_frame, width=250)
        self.batch_number_entry.pack(padx=10, pady=2)
        
        # Device selection
        ctk.CTkLabel(metadata_frame, text="Device:").pack(anchor="w", padx=10, pady=2)
        self.device_label = ctk.CTkLabel(
            metadata_frame,
            text="No device selected",
            font=ctk.CTkFont(size=11),
            wraplength=240
        )
        self.device_label.pack(padx=10, pady=2)
        
        ctk.CTkButton(
            metadata_frame,
            text="Select Device",
            command=self.select_device_dialog,
            height=30
        ).pack(padx=10, pady=5)
        
        # Operator
        ctk.CTkLabel(metadata_frame, text="Operator Initials:").pack(anchor="w", padx=10, pady=2)
        self.operator_entry = ctk.CTkEntry(metadata_frame, width=250)
        self.operator_entry.pack(padx=10, pady=2)
        
        # Start date/time
        self.start_time_label = ctk.CTkLabel(
            metadata_frame,
            text=f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            font=ctk.CTkFont(size=10)
        )
        self.start_time_label.pack(pady=5)
        
        # Separator
        ctk.CTkFrame(scroll_container, height=2, fg_color="gray").pack(fill="x", padx=10, pady=15)
        
        # BMR Actions Section
        ctk.CTkLabel(
            scroll_container,
            text="BMR Actions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Initialize BMR
        ctk.CTkButton(
            scroll_container,
            text="🆕 New BMR (SET Process)",
            command=self.initialize_set_bmr,
            fg_color="#2fa572",
            hover_color="#1d6e4a",
            height=40
        ).pack(padx=10, pady=5, fill="x")
        
        # Load existing BMR
        ctk.CTkButton(
            scroll_container,
            text="📂 Load Existing BMR",
            command=self.load_bmr_dialog,
            height=35
        ).pack(padx=10, pady=5, fill="x")
        
        # Save BMR
        ctk.CTkButton(
            scroll_container,
            text="💾 Save BMR",
            command=self.save_bmr,
            height=35
        ).pack(padx=10, pady=5, fill="x")
        
        #import from json
        ctk.CTkButton(
            scroll_container,
            text="📋 Import ALL Steps",
            command=self.bulk_import_from_json,
            fg_color="#d4af37",
            hover_color="#9d7f23",
            height=35
        ).pack(padx=10, pady=5, fill="x")      
        
        # Export to PDF
        ctk.CTkButton(
            scroll_container,
            text="📄 Export to PDF",
            command=self.export_bmr_pdf,
            fg_color="#d4af37",
            hover_color="#9d7f23",
            height=35
        ).pack(padx=10, pady=5, fill="x")
        
        # Separator for Templates
        ctk.CTkFrame(scroll_container, height=2, fg_color="gray").pack(fill="x", padx=10, pady=15)
        
        # Template Actions Section
        ctk.CTkLabel(
            scroll_container,
            text="Templates",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Save as Template
        ctk.CTkButton(
            scroll_container,
            text="💾 Save as Template",
            command=self.save_as_template,
            height=35
        ).pack(padx=10, pady=5, fill="x")

        # Load Template
        ctk.CTkButton(
            scroll_container,
            text="📥 Load Template",
            command=self.load_template,
            height=35
        ).pack(padx=10, pady=5, fill="x")

        # Create Template
        ctk.CTkButton(
            scroll_container,
            text="🔧 Create Template…",
            command=self.create_template_stepwise,
            height=35
        ).pack(padx=10, pady=5, fill="x")

        # Manage Templates
        ctk.CTkButton(
            scroll_container,
            text="🧩 Manage Templates…",
            command=self.manage_templates,
            height=35
        ).pack(padx=10, pady=5, fill="x")
                
        # Separator
        ctk.CTkFrame(scroll_container, height=2, fg_color="gray").pack(fill="x", padx=10, pady=15)
        
        # Quality status
        self.quality_status_frame = ctk.CTkFrame(scroll_container)
        self.quality_status_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            self.quality_status_frame,
            text="Quality Status",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=5)
        
        self.quality_status_label = ctk.CTkLabel(
            self.quality_status_frame,
            text="No steps completed",
            font=ctk.CTkFont(size=10),
            wraplength=250
        )
        self.quality_status_label.pack(pady=5)
    
    def _build_middle_panel(self, parent):
        """Build the middle panel for step details"""
        # Header
        header_frame = ctk.CTkFrame(parent)
        header_frame.pack(fill="x", padx=10, pady=10)
        
        self.step_title_label = ctk.CTkLabel(
            header_frame,
            text="Select a step to begin",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.step_title_label.pack(side="left", padx=10)
        
        self.step_status_label = ctk.CTkLabel(
            header_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.step_status_label.pack(side="right", padx=10)
        
        # Scrollable content area
        self.step_content_frame = ctk.CTkScrollableFrame(parent)
        self.step_content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Navigation buttons
        nav_frame = ctk.CTkFrame(parent)
        nav_frame.pack(fill="x", padx=10, pady=10)
        
        self.prev_button = ctk.CTkButton(
            nav_frame,
            text="◀ Previous Step",
            command=self.previous_step,
            width=150,
            state="disabled"
        )
        self.prev_button.pack(side="left", padx=5)
        
        self.complete_button = ctk.CTkButton(
            nav_frame,
            text="✓ Complete Step",
            command=self.complete_current_step,
            fg_color="#2fa572",
            hover_color="#1d6e4a",
            width=150,
            state="disabled"
        )
        self.complete_button.pack(side="left", expand=True, padx=5)
        
        self.next_button = ctk.CTkButton(
            nav_frame,
            text="Next Step ▶",
            command=self.next_step,
            width=150,
            state="disabled"
        )
        self.next_button.pack(side="left", padx=5)
    
    def _build_right_panel(self, parent):
        """Build the right panel showing progress"""
        scroll_container = ctk.CTkScrollableFrame(parent)
        scroll_container.pack(fill="both", expand=True)
        
        # Header
        ctk.CTkLabel(
            scroll_container,
            text="Progress Tracker",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Progress overview
        self.progress_frame = ctk.CTkFrame(scroll_container)
        self.progress_frame.pack(fill="x", padx=10, pady=10)
        
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="0 / 0 Steps Complete",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, width=200)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)
        
        # Separator
        ctk.CTkFrame(scroll_container, height=2, fg_color="gray").pack(fill="x", padx=10, pady=10)
        
        # Steps list
        ctk.CTkLabel(
            scroll_container,
            text="All Steps",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        self.steps_list_frame = ctk.CTkFrame(scroll_container)
        self.steps_list_frame.pack(fill="both", expand=True, padx=10, pady=5)
    
    def initialize_set_bmr(self):
        """Initialize a new BMR for SET process"""
        if not self.current_device:
            messagebox.showwarning(
                "No Device",
                "Please select a device before initializing BMR."
            )
            return
        
        # Confirm
        if self.bmr_steps:
            if not messagebox.askyesno(
                "Confirm",
                "This will clear the current BMR data. Continue?"
            ):
                return

        # NEW: ensure we use the built-in standard definitions for a new standard run
        self.active_step_defs = STANDARD_BMR_STEPS[:]
        
        # Initialize steps from active definitions
        self.bmr_steps = []
        for step_template in self.active_step_defs:
            step_data = BMRStepData(
                step_number=step_template["step_number"],
                step_name=step_template["step_name"]
            )
            self.bmr_steps.append(step_data)
        
        # Set metadata
        self.bmr_metadata["start_date"] = datetime.now()
        self.bmr_metadata["device_id"] = self.current_device.device_id if self.current_device else ""
        
        # Update UI
        self.current_step_index = 0
        self.update_progress_panel()
        self.display_step(0)
        
        messagebox.showinfo(
            "BMR Initialized",
            f"Created BMR with {len(self.bmr_steps)} steps for SET process."
        )


    
    def select_device_dialog(self):
        """Show dialog to select a device and automatically load associated BMR if it exists"""
        try:
            from stm_fab.db.models import Device
            devices = self.app.db_session.query(Device).all()
            
            if not devices:
                messagebox.showinfo("No Devices", "No devices found in database.")
                return
            
            # Create selection dialog
            dialog = ctk.CTkToplevel(self.tab)
            dialog.title("Select Device")
            dialog.geometry("450x550")
            dialog.transient(self.tab)
            
            ctk.CTkLabel(
                dialog,
                text="Select Device",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=10)
            
            # Info label
            info_label = ctk.CTkLabel(
                dialog,
                text="Devices with existing BMRs are marked with 📋",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            info_label.pack(pady=5)
            
            # Scrollable list
            scroll_frame = ctk.CTkScrollableFrame(dialog, width=400, height=380)
            scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)
            
            selected_device = [None]
            
            def select_device(device):
                selected_device[0] = device
                dialog.destroy()
            
            # Get BMR counts for each device
            from stm_fab.db.models import BatchManufacturingRecord
            
            for device in devices:
                # Check if device has BMRs
                bmr_count = self.app.db_session.query(BatchManufacturingRecord)\
                    .filter_by(device_id=device.device_id).count()
                
                bmr_indicator = " 📋" if bmr_count > 0 else ""
                status_color = "#2fa572" if bmr_count > 0 else None
                
                device_btn = ctk.CTkButton(
                    scroll_frame,
                    text=f"{device.device_name}{bmr_indicator}\n({device.sample.sample_name})",
                    command=lambda d=device: select_device(d),
                    height=50,
                    fg_color=status_color if status_color else None
                )
                device_btn.pack(pady=5, fill="x")
            
            # Wait for dialog
            dialog.wait_window()
            
            if selected_device[0]:
                self.current_device = selected_device[0]
                self.device_label.configure(
                    text=f"{self.current_device.device_name}\n({self.current_device.sample.sample_name})"
                )
                self.bmr_metadata["device_id"] = self.current_device.device_id
                
                # **NEW: Automatically load BMR if one exists for this device**
                self._auto_load_bmr_for_device(self.current_device.device_id)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load devices: {e}")
    
    
    def _auto_load_bmr_for_device(self, device_id: int):
        """
        Automatically load the most recent BMR for a device
        
        Args:
            device_id: Device ID to load BMR for
        """
        try:
            from stm_fab.db.models import BatchManufacturingRecord
            
            # Get most recent BMR for this device (prefer in_progress, else most recent)
            bmr = self.app.db_session.query(BatchManufacturingRecord)\
                .filter_by(device_id=device_id, status='in_progress')\
                .order_by(BatchManufacturingRecord.created_at.desc())\
                .first()
            
            if not bmr:
                # Try to find any BMR (completed, failed, etc.)
                bmr = self.app.db_session.query(BatchManufacturingRecord)\
                    .filter_by(device_id=device_id)\
                    .order_by(BatchManufacturingRecord.created_at.desc())\
                    .first()
            
            if bmr:
                # Load from database
                self._load_bmr_from_database(bmr.bmr_id)
                
                # Show notification
                status_msg = f"({'in progress' if bmr.status == 'in_progress' else bmr.status})"
                messagebox.showinfo(
                    "BMR Loaded",
                    f"Loaded existing BMR for this device:\n\n"
                    f"Batch: {bmr.batch_number}\n"
                    f"Status: {status_msg}\n"
                    f"Operator: {bmr.operator}\n"
                    f"Progress: {bmr.calculate_completion():.0f}%"
                )
            else:
                # No BMR found - ask if user wants to create one
                create = messagebox.askyesno(
                    "No BMR Found",
                    f"No BMR exists for device '{self.current_device.device_name}'.\n\n"
                    "Would you like to create a new BMR for this device?"
                )
                if create:
                    self.initialize_set_bmr()
                    
        except Exception as e:
            print(f"Error auto-loading BMR: {e}")
            # Don't show error to user - this is a convenience feature
    
    
    def _load_bmr_from_database(self, bmr_id: int):
        """
        Load BMR data from database record
        
        Args:
            bmr_id: BMR ID to load
        """
        try:
            # Use database operations to load BMR
            bmr_data = self.app.db_ops.load_bmr_to_json(bmr_id)
            
            # Load into UI
            self._load_from_json_obj(bmr_data)
            
            # Store the BMR ID for later saves
            self._current_bmr_id = bmr_id
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load BMR from database:\n{e}")
    
    
    def save_bmr(self):
        """Save BMR to JSON file and database"""
        if not self.bmr_steps:
            messagebox.showwarning("No Data", "No BMR data to save.")
            return
        
        # Update operator from entry field
        self.bmr_metadata["operator"] = self.operator_entry.get().strip()
        
        # Convert to JSON dict
        save_data = self._to_json_dict()
        
        # Check if we should save to database
        if self.current_device and self.app.db_session:
            try:
                # Determine if this is a new BMR or updating existing
                bmr_id = getattr(self, '_current_bmr_id', None)
                
                if bmr_id:
                    # Update existing BMR in database
                    self._update_bmr_in_database(bmr_id, save_data)
                    msg = f"BMR updated in database (ID: {bmr_id})"
                else:
                    # Create new BMR in database
                    bmr_id = self._create_bmr_in_database(save_data)
                    self._current_bmr_id = bmr_id
                    msg = f"BMR created in database (ID: {bmr_id})"
                
                # Also save to JSON file
                json_path = self._save_bmr_json_file(save_data)
                
                # Link JSON file to database record
                self.app.db_ops.update_bmr(
                    bmr_id,
                    json_file_path=json_path
                )
                
                messagebox.showinfo(
                    "Save Successful",
                    f"{msg}\n\nJSON file: {json_path}"
                )
                
            except Exception as e:
                messagebox.showerror("Database Save Error", f"Failed to save to database:\n{e}\n\nTrying JSON only...")
                # Fallback to JSON only
                json_path = self._save_bmr_json_file(save_data)
                messagebox.showinfo("Save Successful", f"Saved to JSON:\n{json_path}")
        else:
            # No device selected or no database - save to JSON only
            json_path = self._save_bmr_json_file(save_data)
            messagebox.showinfo("Save Successful", f"Saved to JSON:\n{json_path}")
    
    
    def _save_bmr_json_file(self, save_data: dict) -> str:
        """
        Save BMR to JSON file
        
        Args:
            save_data: Dictionary containing BMR data
        
        Returns:
            Path to saved JSON file
        """
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"BMR_{self.bmr_metadata.get('batch_number', 'untitled')}.json"
        )
        
        if not filename:
            return ""
        
        # Save JSON
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)
        
        return filename
    
    
    def _create_bmr_in_database(self, bmr_data: dict) -> int:
        """
        Create new BMR record in database
        
        Args:
            bmr_data: Dictionary containing BMR data
        
        Returns:
            ID of created BMR
        """
        metadata = bmr_data.get('metadata', {})
        
        # Generate batch number if missing
        batch_number = metadata.get('batch_number', '').strip()
        if not batch_number:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_number = f"{self.current_device.device_name}_{timestamp}"
            metadata['batch_number'] = batch_number
            self.batch_number_entry.delete(0, 'end')
            self.batch_number_entry.insert(0, batch_number)
    
        # Convert ISO strings to datetime objects (if needed)
        start_date = metadata.get('start_date')
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date) if start_date else None
    
        target_completion = metadata.get('target_completion')
        if isinstance(target_completion, str):
            target_completion = datetime.fromisoformat(target_completion) if target_completion else None
    
        # NEW: normalize completion_date if present
        completion_date = metadata.get('completion_date')
        if isinstance(completion_date, str):
            completion_date = datetime.fromisoformat(completion_date) if completion_date else None
    
        bmr = self.app.db_ops.create_bmr(
            device_id=self.current_device.device_id,
            batch_number=batch_number,
            operator=metadata.get('operator', ''),
            process_type=metadata.get('process_type', 'SET'),
            start_date=start_date,
            target_completion=target_completion,
            completion_date=completion_date   # DatabaseOperations.update_bmr allows this; create_bmr ignores extras
        )
        
        # Create BMR steps
        for step_data in bmr_data.get('steps', []):
            start_time = step_data.get('start_time')
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time) if start_time else None
            
            end_time = step_data.get('end_time')
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time) if end_time else None
            
            self.app.db_ops.create_bmr_step(
                bmr_id=bmr.bmr_id,
                step_number=step_data['step_number'],
                step_name=step_data['step_name'],
                status=step_data.get('status', 'not_started'),
                operator_initials=step_data.get('operator_initials', ''),
                start_time=start_time,
                end_time=end_time,
                parameters_json=json.dumps(step_data.get('parameters', {})),
                quality_notes=step_data.get('quality_notes', ''),
                pre_check_pass=step_data.get('pre_check_pass', False),
                post_check_pass=step_data.get('post_check_pass', False),
                deviations_json=json.dumps(step_data.get('deviations', [])),
                corrective_actions_json=json.dumps(step_data.get('corrective_actions', [])),
                labview_file=step_data.get('labview_file', ''),
                sxm_scans_json=json.dumps(step_data.get('sxm_scans', []))
            )
        
        return bmr.bmr_id
    
    
    def _update_bmr_in_database(self, bmr_id: int, bmr_data: dict):
        """
        Update existing BMR record in database
    
        Args:
            bmr_id: BMR ID to update
            bmr_data: Dictionary containing updated BMR data
        """
        from datetime import datetime
    
        def to_dt(val):
            """Convert ISO string (or Z-terminated) to datetime, else passthrough or None."""
            if val is None or val == "":
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                s = val.strip()
                # Handle 'Z' timezone suffix if ever present
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                try:
                    return datetime.fromisoformat(s)
                except Exception:
                    # As a fallback you could use dateutil.parser.parse, but avoid extra deps
                    raise ValueError(f"Invalid datetime string: {val}")
            return None
    
        metadata = bmr_data.get('metadata', {}) or {}
    
        # Normalize BMR metadata datetime fields to real datetime (or None)
        start_date        = to_dt(metadata.get('start_date'))
        completion_date   = to_dt(metadata.get('completion_date'))
        target_completion = to_dt(metadata.get('target_completion'))
    
        # Update BMR metadata
        self.app.db_ops.update_bmr(
            bmr_id,
            operator=metadata.get('operator', ''),
            start_date=start_date,
            completion_date=completion_date,
            target_completion=target_completion,
            status=metadata.get('status', 'in_progress')
        )
    
        # Recreate steps
        from stm_fab.db.models import BMRStep
        existing_steps = self.app.db_session.query(BMRStep).filter_by(bmr_id=bmr_id).all()
    
        for step in existing_steps:
            self.app.db_session.delete(step)
        self.app.db_session.commit()
    
        # Insert updated steps, with normalized datetimes
        for step_data in bmr_data.get('steps', []):
            st = to_dt(step_data.get('start_time'))
            et = to_dt(step_data.get('end_time'))
            vt = to_dt(step_data.get('verified_time'))
    
            self.app.db_ops.create_bmr_step(
                bmr_id=bmr_id,
                step_number=step_data['step_number'],
                step_name=step_data['step_name'],
                status=step_data.get('status', 'not_started'),
                operator_initials=step_data.get('operator_initials', ''),
                start_time=st,
                end_time=et,
                parameters_json=json.dumps(step_data.get('parameters', {})),
                quality_notes=step_data.get('quality_notes', ''),
                pre_check_pass=bool(step_data.get('pre_check_pass', False)),
                post_check_pass=bool(step_data.get('post_check_pass', False)),
                deviations_json=json.dumps(step_data.get('deviations', [])),
                corrective_actions_json=json.dumps(step_data.get('corrective_actions', [])),
                labview_file=step_data.get('labview_file', ''),
                sxm_scans_json=json.dumps(step_data.get('sxm_scans', [])),
                verified_by=step_data.get('verified_by', ''),
                verified_time=vt
            )

    
    
    def export_bmr_pdf(self):
        """Export BMR to PDF format with database linking"""
        try:
            from stm_fab.reports.bmr_pdf_generator import (
                generate_blank_bmr_template,
                generate_filled_bmr,
            )
        except Exception as e:
            messagebox.showerror("PDF Generator Missing", f"Could not import PDF generator:\n{e}")
            return
    
        # Ask user whether to export a filled record or a blank template
        choice = messagebox.askyesno(
            "Export PDF",
            "Export a FILLED BMR PDF with current data?\n\n"
            "Yes = Filled PDF\nNo  = Blank PDF template"
        )
    
        # Pick output path
        pdf_path = filedialog.asksaveasfilename(
            title="Save BMR PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=(f"BMR_{self.bmr_metadata.get('batch_number','') or 'record'}.pdf")
        )
        if not pdf_path:
            return
    
        try:
            if choice:
                # Filled PDF
                self.bmr_metadata["operator"] = self.operator_entry.get().strip()
                data = self._to_json_dict()
    
                if "step_definitions" not in data:
                    step_defs = getattr(self, "active_step_defs", None) or STANDARD_BMR_STEPS[:]
                    data["step_definitions"] = step_defs
    
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
                    json.dump(data, tmp, indent=2)
                    tmp_path = tmp.name
    
                out = generate_filled_bmr(tmp_path, pdf_path)
    
                # Link PDF to database if BMR exists
                bmr_id = getattr(self, '_current_bmr_id', None)
                if bmr_id and self.app.db_session:
                    try:
                        self.app.db_ops.link_bmr_pdf(bmr_id, pdf_path)
                        messagebox.showinfo(
                            "PDF Export", 
                            f"Filled BMR PDF generated and linked to database:\n{out}"
                        )
                    except Exception as e:
                        messagebox.showinfo(
                            "PDF Export", 
                            f"Filled BMR PDF generated:\n{out}\n\n"
                            f"(Could not link to database: {e})"
                        )
                else:
                    messagebox.showinfo("PDF Export", f"Filled BMR PDF generated:\n{out}")
    
            else:
                # Blank PDF
                step_defs = getattr(self, "active_step_defs", None) or STANDARD_BMR_STEPS[:]
                out = generate_blank_bmr_template(pdf_path, step_definitions=step_defs)
                messagebox.showinfo("PDF Export", f"Blank BMR PDF generated:\n{out}")
    
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to generate PDF:\n{e}")
    
    
    def load_bmr_dialog(self):
        """Load BMR from JSON file or database"""
        # Ask user if they want to load from file or database
        choice = messagebox.askyesnocancel(
            "Load BMR",
            "Load BMR from:\n\n"
            "Yes = Database (for selected device)\n"
            "No = JSON file\n"
            "Cancel = Cancel"
        )
        
        if choice is None:  # Cancel
            return
        
        if choice:  # From database
            if not self.current_device:
                messagebox.showwarning(
                    "No Device Selected",
                    "Please select a device first to load its BMR from database."
                )
                return
            
            self._load_bmr_from_database_dialog()
        else:  # From file
            self._load_bmr_from_file()
    
    
    def _load_bmr_from_database_dialog(self):
        """Show dialog to select and load BMR from database"""
        try:
            from stm_fab.db.models import BatchManufacturingRecord
            
            # Get all BMRs for current device
            bmrs = self.app.db_session.query(BatchManufacturingRecord)\
                .filter_by(device_id=self.current_device.device_id)\
                .order_by(BatchManufacturingRecord.created_at.desc())\
                .all()
            
            if not bmrs:
                messagebox.showinfo(
                    "No BMRs Found",
                    f"No BMRs found for device '{self.current_device.device_name}'."
                )
                return
            
            # Create selection dialog
            dialog = ctk.CTkToplevel(self.tab)
            dialog.title("Select BMR")
            dialog.geometry("500x400")
            dialog.transient(self.tab)
            
            ctk.CTkLabel(
                dialog,
                text=f"BMRs for {self.current_device.device_name}",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=10)
            
            # Scrollable list
            scroll_frame = ctk.CTkScrollableFrame(dialog, width=450, height=280)
            scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)
            
            selected_bmr = [None]
            
            def select_bmr(bmr):
                selected_bmr[0] = bmr
                dialog.destroy()
            
            for bmr in bmrs:
                progress = bmr.calculate_completion()
                status_color = {
                    'in_progress': "#3b82f6",
                    'completed': "#2fa572",
                    'failed': "#dc2626",
                    'on_hold': "#f59e0b"
                }.get(bmr.status, None)
                
                bmr_btn = ctk.CTkButton(
                    scroll_frame,
                    text=f"{bmr.batch_number}\n"
                         f"Status: {bmr.status} | Progress: {progress:.0f}% | Operator: {bmr.operator}\n"
                         f"Created: {bmr.created_at.strftime('%Y-%m-%d %H:%M')}",
                    command=lambda b=bmr: select_bmr(b),
                    height=65,
                    fg_color=status_color if status_color else None
                )
                bmr_btn.pack(pady=5, fill="x")
            
            # Wait for dialog
            dialog.wait_window()
            
            if selected_bmr[0]:
                self._load_bmr_from_database(selected_bmr[0].bmr_id)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load BMRs from database:\n{e}")
    
    
    def _load_bmr_from_file(self):
        """Load BMR from JSON file"""
        filename = filedialog.askopenfilename(
            title="Load BMR",
            filetypes=[("JSON files", "*.json")],
            initialdir=Path.cwd()
        )
        
        if not filename:
            return
        
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._load_from_json_obj(data)
            
            # Clear current BMR ID since this is from file
            self._current_bmr_id = None
            
            messagebox.showinfo("Load Successful", f"Loaded BMR from:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Load Failed", f"Failed to load BMR:\n{e}")
    
    
    # Add this to __init__ method (after self.bmr_metadata initialization):
    # self._current_bmr_id = None  # Track current BMR database ID
        
    def display_step(self, step_index: int):
        """Display a specific step for data entry"""
        if not self.bmr_steps or step_index >= len(self.bmr_steps):
            return
        
        self.current_step_index = step_index
        step_data = self.bmr_steps[step_index]

        # NEW: render from active template definitions
        step_template = self.active_step_defs[step_index]
        param_defs = step_template.get("parameters", {"required": [], "optional": []})
        param_types = step_template.get("param_types", {})
        param_enums = step_template.get("param_enums", {})
        tables = step_template.get("tables", [])
        
        # Update header
        self.step_title_label.configure(
            text=f"Step {step_data.step_number}: {step_data.step_name}"
        )
        self.step_status_label.configure(
            text=f"Status: {step_data.status.value.replace('_', ' ').title()}"
        )
        
        # Clear content frame
        for widget in self.step_content_frame.winfo_children():
            widget.destroy()
        
        # Parameters section
        params_frame = ctk.CTkFrame(self.step_content_frame)
        params_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            params_frame,
            text="Process Parameters",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        # Create parameter entry fields (supports enum/file types)
        self.parameter_widgets[step_index] = {}

        for section in ["required", "optional"]:
            names = param_defs.get(section, [])
            if names:
                label_text = "Required:" if section == "required" else "Optional:"
                ctk.CTkLabel(
                    params_frame,
                    text=label_text,
                    font=ctk.CTkFont(size=12, weight="bold")
                ).pack(anchor="w", padx=10, pady=5)

                for param_name in names:
                    param_frame = ctk.CTkFrame(params_frame)
                    param_frame.pack(fill="x", padx=10, pady=2)

                    # Nice label formatting
                    display_name = param_name.replace("_", " ").title()
                    display_name = display_name.replace("Fem", "FEM").replace("Apm", "APM").replace("Ph3", "PH3").replace("Ash3", "AsH3")

                    ctk.CTkLabel(
                        param_frame,
                        text=display_name + ":",
                        width=200
                    ).pack(side="left", padx=5)

                    ptype = param_types.get(param_name, "string")
                    existing_val = step_data.parameters.get(param_name, "")

                    if ptype == "enum":
                        values = step_template.get("param_enums", {}).get(param_name, [])
                        var = tk.StringVar(value=str(existing_val) if existing_val else (values[0] if values else ""))
                        widget = ctk.CTkOptionMenu(param_frame, values=values, variable=var,
                                                   command=lambda _v, idx=step_index: self._mark_step_in_progress(idx), width=300)
                        widget.pack(side="left", padx=5)

                    elif ptype == "file":
                        file_frame = ctk.CTkFrame(param_frame)
                        file_frame.pack(side="left")
                        entry = ctk.CTkEntry(file_frame, width=260)
                        entry.pack(side="left", padx=3)
                        if existing_val:
                            entry.insert(0, str(existing_val))
                        def pick_file(entry_ref=entry, idx=step_index):
                            path = filedialog.askopenfilename()
                            if path:
                                entry_ref.delete(0, "end")
                                entry_ref.insert(0, path)
                                self._mark_step_in_progress(idx)
                        ctk.CTkButton(file_frame, text="Browse", width=70, command=pick_file).pack(side="left", padx=3)
                        widget = entry

                    else:
                        # default string/number entry
                        widget = ctk.CTkEntry(param_frame, width=300)
                        widget.pack(side="left", padx=5)
                        if existing_val != "":
                            widget.insert(0, str(existing_val))
                        widget.bind('<KeyRelease>', lambda e, idx=step_index: self._mark_step_in_progress(idx))
                        widget.bind('<FocusOut>', lambda e, idx=step_index: self._mark_step_in_progress(idx))

                    self.parameter_widgets[step_index][param_name] = widget

        # NEW: Tabulated data arrays
        if tables:
            tables_frame = ctk.CTkFrame(self.step_content_frame)
            tables_frame.pack(fill="x", padx=10, pady=10)

            ctk.CTkLabel(
                tables_frame,
                text="Tabulated Data",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=10, pady=5)

            self.parameter_widgets[step_index]["__tables__"] = {}
            for tbl in tables:
                tf = ctk.CTkFrame(tables_frame)
                tf.pack(fill="x", padx=10, pady=6)

                ctk.CTkLabel(
                    tf, text=tbl.get("title", tbl.get("name", "Table")),
                    font=ctk.CTkFont(size=12, weight="bold")
                ).pack(anchor="w", pady=3)

                editor = TableEditor(
                    tf,
                    columns=tbl.get("columns", []),
                    default_rows=tbl.get("default_rows", 0),
                    max_rows=tbl.get("max_rows", 999),
                    data=step_data.parameters.get(tbl["name"])
                )
                editor.pack(fill="x", padx=5, pady=3)
                self.parameter_widgets[step_index]["__tables__"][tbl["name"]] = editor


        # Quality checks section (per-item Pass/Fail/N/A)
        quality_frame = ctk.CTkFrame(self.step_content_frame)
        quality_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            quality_frame,
            text="Quality Checks",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)

        # NEW: store QC item widgets so we can save them later
        self.parameter_widgets[step_index]["__qc_items__"] = {}

        qc_items = step_template.get("quality_checks", [])
        for check in qc_items:
            row = ctk.CTkFrame(quality_frame)
            row.pack(fill="x", padx=10, pady=2)

            ctk.CTkLabel(row, text=check, width=320, anchor="w").pack(side="left", padx=5)

            options = ["", "Pass", "Fail", "N/A"]
            existing = step_data.quality_check_results.get(check, "")
            om = ctk.CTkOptionMenu(row, values=options, width=120)
            if existing:
                om.set(existing)
            else:
                om.set("")
            om.pack(side="left", padx=5)

            # Mark step in progress on change
            om.configure(command=lambda _v, idx=step_index: self._mark_step_in_progress(idx))

            self.parameter_widgets[step_index]["__qc_items__"][check] = om

        
        # Time tracking section
        time_frame = ctk.CTkFrame(self.step_content_frame)
        time_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            time_frame,
            text="Time Tracking",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        if step_data.start_time:
            ctk.CTkLabel(
                time_frame,
                text=f"Started: {step_data.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            ).pack(anchor="w", padx=10, pady=2)
        else:
            ctk.CTkButton(
                time_frame,
                text="▶ Start Step",
                command=lambda: self.start_step(step_index),
                width=150
            ).pack(anchor="w", padx=10, pady=5)
 
        # Signature / Initial section
        sign_frame = ctk.CTkFrame(self.step_content_frame)
        sign_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            sign_frame,
            text="Signature / Initial",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)

        row = ctk.CTkFrame(sign_frame)
        row.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(row, text="Initials:", width=100, anchor="w").pack(side="left", padx=5)

        initials_entry = ctk.CTkEntry(row, width=160)
        initials_entry.pack(side="left", padx=5)

        # Pre-fill with existing, else with operator entry
        if step_data.step_initials:
            initials_entry.insert(0, step_data.step_initials)
        else:
            if self.operator_entry.get().strip():
                initials_entry.insert(0, self.operator_entry.get().strip())

        def do_initial():
            self.initial_step(step_index, initials_entry.get())

        init_btn = ctk.CTkButton(row, text="Initial This Step", width=140, command=do_initial)
        init_btn.pack(side="left", padx=8)

        # Show stamped time if present
        stamped = ""
        if step_data.step_initial_time:
            stamped = f"Initialed at: {step_data.step_initial_time.strftime('%Y-%m-%d %H:%M:%S')}"
        stamp_lbl = ctk.CTkLabel(row, text=stamped, anchor="w")
        stamp_lbl.pack(side="left", padx=10)
        
        # JSON Import section
        import_frame = ctk.CTkFrame(self.step_content_frame)
        import_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            import_frame,
            text="LabVIEW Data Import",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)

        btn_frame = ctk.CTkFrame(import_frame)
        btn_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(
            btn_frame,
            text="📊 Import This Step",
            command=self.import_from_json_analysis,
            width=180,
            fg_color="#1f6aa5",
            hover_color="#144870"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="📋 Import ALL Steps",
            command=self.bulk_import_from_json,
            width=180,
            fg_color="#2fa572",
            hover_color="#1d6e4a"
        ).pack(side="left", padx=5)
        
        # Notes section
        notes_frame = ctk.CTkFrame(self.step_content_frame)
        notes_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            notes_frame,
            text="Quality Notes / Observations",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        notes_text = ctk.CTkTextbox(notes_frame, height=100)
        notes_text.pack(fill="x", padx=10, pady=5)
        if step_data.quality_notes:
            notes_text.insert("1.0", step_data.quality_notes)
        
        self.parameter_widgets[step_index]["quality_notes"] = notes_text
        notes_text.bind('<KeyRelease>', lambda e, idx=step_index: self._mark_step_in_progress(idx))
        
        # Update navigation buttons
        self.prev_button.configure(state="normal" if step_index > 0 else "disabled")
        self.next_button.configure(state="normal" if step_index < len(self.bmr_steps) - 1 else "disabled")
        self.complete_button.configure(state="normal")
    
    def start_step(self, step_index: int):
        """Mark step as started"""
        if step_index < len(self.bmr_steps):
            op = (self.operator_entry.get() or "").strip()
            self.bmr_steps[step_index].start_time = datetime.now()
            self.bmr_steps[step_index].status = StepStatus.IN_PROGRESS
            self.bmr_steps[step_index].operator_initials = op
            # Keep metadata operator in sync for PDFs
            if op:
                self.bmr_metadata["operator"] = op
            self.display_step(step_index)
            self.update_progress_panel()

    
    def complete_current_step(self):
        """Mark current step as complete after validation"""
        if not self.bmr_steps:
            return
        
        step_data = self.bmr_steps[self.current_step_index]
        
        # Validate required fields using active template
        step_template = self.active_step_defs[self.current_step_index]
        missing_required = []
        
        if "required" in step_template["parameters"]:
            for param in step_template["parameters"]["required"]:
                if param in self.parameter_widgets[self.current_step_index]:
                    widget = self.parameter_widgets[self.current_step_index][param]
                    value = widget.get()
                    if not value or value.strip() == "":
                        missing_required.append(param)
        
        if missing_required:
            messagebox.showwarning(
                "Missing Required Fields",
                f"Please fill in the following required fields:\n\n" +
                "\n".join(f"• {f.replace('_', ' ').title()}" for f in missing_required)
            )
            return
        
        # Save parameter data
        widgets_map = self.parameter_widgets[self.current_step_index]

        # Notes
        if "quality_notes" in widgets_map:
            step_data.quality_notes = widgets_map["quality_notes"].get("1.0", "end-1c")

        # Scalars (entries/option menus) — skip special buckets
        for param_name, widget in widgets_map.items():
            if param_name in ("quality_notes", "__tables__", "__qc_items__"):
                continue
            try:
                step_data.parameters[param_name] = widget.get()
            except Exception:
                pass

        # Tables -> arrays of rows
        table_editors = widgets_map.get("__tables__", {})
        for tname, editor in table_editors.items():
            step_data.parameters[tname] = editor.get_data()

        # Per-item QC results
        qc_items = widgets_map.get("__qc_items__", {})
        qc_map = {}
        for label, om in qc_items.items():
            # CTkOptionMenu.get() returns the selected text ("", "Pass", "Fail", "N/A")
            try:
                qc_map[label] = (om.get() or "").strip()
            except Exception:
                qc_map[label] = ""
        step_data.quality_check_results = qc_map

        # Compute aggregate: post_check_pass = True iff all non-N/A items are "Pass"
        non_na = [v for v in qc_map.values() if v and v.upper() != "N/A"]
        step_data.post_check_pass = bool(non_na) and all(v.lower() == "pass" for v in non_na)

        # Auto-initial if not set yet (robust fallback + prompt)
        if not (step_data.step_initials or "").strip():
            # Try Operator field (left panel)
            op = (self.operator_entry.get() or "").strip()
            # Fall back to step operator_initials (set on Start Step)
            if not op:
                op = (step_data.operator_initials or "").strip()
            # Fall back to metadata operator (used by PDF header)
            if not op:
                op = (self.bmr_metadata.get("operator") or "").strip()
            # As a last resort, prompt the user
            if not op:
                op = simpledialog.askstring("Initials required", "Enter your initials to initial this step:")
                if not op:
                    messagebox.showwarning("Missing Initials", "Initials are required to complete this step.")
                    return
            step_data.step_initials = op
            step_data.step_initial_time = datetime.now()


        
        # Mark as complete
        step_data.end_time = datetime.now()
        step_data.status = StepStatus.COMPLETED
        # keep previously computed step_data.post_check_pass from per-item QC

        
        # Update UI
        self.update_progress_panel()
        
        # Auto-advance to next step
        if self.current_step_index < len(self.bmr_steps) - 1:
            self.next_step()
        else:
            messagebox.showinfo(
                "BMR Complete",
                "All steps have been completed! Don't forget to export to PDF."
            )

    def _mark_step_in_progress(self, step_index: int):
        """Mark step as in-progress if any field has data."""
        if step_index >= len(self.bmr_steps):
            return
        
        step_data = self.bmr_steps[step_index]
        
        # Only auto-mark if step is NOT_STARTED
        if step_data.status != StepStatus.NOT_STARTED:
            return
        
        # Check parameters dict first
        if step_data.parameters:
            has_content = any(v for v in step_data.parameters.values() if v)
            if has_content:
                step_data.status = StepStatus.IN_PROGRESS
                if not step_data.start_time:
                    step_data.start_time = datetime.now()
                if not step_data.operator_initials:
                    step_data.operator_initials = self.operator_entry.get()
                
                self.step_status_label.configure(
                    text=f"Status: {step_data.status.value.replace('_', ' ').title()}"
                )
                self.update_progress_panel()
                return
        
        # Check if any parameter widget has content
        widgets = self.parameter_widgets.get(step_index, {})
        has_content = False
        
        for param_name, widget in widgets.items():
            if param_name == "quality_notes":
                content = widget.get("1.0", "end-1c").strip()
                if content:
                    has_content = True
                    break
            else:
                content = widget.get().strip()
                if content:
                    has_content = True
                    break
        
        if has_content:
            step_data.status = StepStatus.IN_PROGRESS
            if not step_data.start_time:
                step_data.start_time = datetime.now()
            if not step_data.operator_initials:
                step_data.operator_initials = self.operator_entry.get()
            
            self.step_status_label.configure(
                text=f"Status: {step_data.status.value.replace('_', ' ').title()}"
            )
            self.update_progress_panel()
    
    def next_step(self):
        """Navigate to next step"""
        if self.current_step_index < len(self.bmr_steps) - 1:
            self.display_step(self.current_step_index + 1)
    
    def previous_step(self):
        """Navigate to previous step"""
        if self.current_step_index > 0:
            self.display_step(self.current_step_index - 1)
    
    def update_progress_panel(self):
        """Update the progress panel display"""
        if not self.bmr_steps:
            return
        
        # Calculate progress
        completed = sum(1 for s in self.bmr_steps if s.status == StepStatus.COMPLETED)
        total = len(self.bmr_steps)
        progress = completed / total if total > 0 else 0
        
        # Update labels
        self.progress_label.configure(text=f"{completed} / {total} Steps Complete")
        self.progress_bar.set(progress)
        
        # Update quality status
        failed = sum(1 for s in self.bmr_steps if s.status == StepStatus.FAILED)
        if failed > 0:
            quality_text = f"⚠️ {failed} step(s) failed"
        elif completed == total:
            quality_text = "✓ All steps passed"
        else:
            quality_text = f"In progress: {completed}/{total}"
        
        self.quality_status_label.configure(text=quality_text)

        # FIX: Clear old step buttons before recreating
        for child in self.steps_list_frame.winfo_children():
            child.destroy()
        
        # Rebuild steps list
        for idx, step in enumerate(self.bmr_steps):
            if step.status == StepStatus.COMPLETED:
                fg_color = "#2fa572"
                status_indicator = "✓ "
            elif step.status == StepStatus.IN_PROGRESS:
                fg_color = "#d4af37"
                status_indicator = "⚠ "
            elif step.status == StepStatus.FAILED:
                fg_color = "#c9302c"
                status_indicator = "✗ "
            else:
                fg_color = "#5a5a5a"
                status_indicator = ""
            
            btn = ctk.CTkButton(
                self.steps_list_frame,
                text=f"{status_indicator}{step.step_number}. {step.step_name}",
                command=lambda i=idx: self.display_step(i),
                fg_color=fg_color,
                height=35,
                anchor="w"
            )
            btn.pack(fill="x", padx=5, pady=2)
         
    def bulk_import_from_json(self):
        """Import JSON data for ALL applicable steps at once."""
        if not self.bmr_steps:
            messagebox.showwarning("No BMR", "Initialize a BMR first.")
            return
        
        # File selection
        json_path = filedialog.askopenfilename(
            title="Select LabVIEW Analysis JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(Path.home() / "device_record" / "labview_data")
        )
        
        if not json_path:
            return
        
        try:
            with open(json_path, 'r') as f:
                analysis_data = json.load(f)
            
            filled_steps = []
            
            # Loop through all steps and try to fill from JSON
            for step_index, step_data in enumerate(self.bmr_steps):
                step_name = step_data.step_name.lower()
                
                # Get mapped parameters for this step
                mapped = self._map_json_to_bmr_params(step_name, analysis_data)
                
                if mapped:
                    # Store parameters in step data
                    for param_name, value in mapped.items():
                        step_data.parameters[param_name] = str(value)
                    
                    # Mark as in progress
                    if step_data.status == StepStatus.NOT_STARTED:
                        step_data.status = StepStatus.IN_PROGRESS
                        if not step_data.start_time:
                            step_data.start_time = datetime.now()
                        if not step_data.operator_initials:
                            step_data.operator_initials = self.operator_entry.get()
                    
                    filled_steps.append(f"Step {step_data.step_number}: {step_data.step_name}")
            
            # Update UI
            self.update_progress_panel()
            if self.current_step_index < len(self.bmr_steps):
                self.display_step(self.current_step_index)
            
            if filled_steps:
                steps_list = "\n".join(f"  • {s}" for s in filled_steps)
                messagebox.showinfo(
                    "Bulk Import Complete", 
                    f"Filled parameters for {len(filled_steps)} steps:\n\n{steps_list}"
                )
            else:
                messagebox.showinfo(
                    "No Data Found",
                    "No matching data found in JSON for any steps."
                )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import JSON:\n{e}")

    
    def import_from_json_analysis(self):
        """Import parameters from LabVIEW analysis JSON file for current step."""
        if not self.bmr_steps:
            messagebox.showwarning("No BMR", "Initialize a BMR first.")
            return
        
        # File selection
        json_path = filedialog.askopenfilename(
            title="Select LabVIEW Analysis JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(Path.home() / "device_record" / "labview_data")
        )
        
        if not json_path:
            return
        
        try:
            with open(json_path, 'r') as f:
                analysis_data = json.load(f)
            
            step_data = self.bmr_steps[self.current_step_index]
            step_name = step_data.step_name.lower()
            widgets = self.parameter_widgets.get(self.current_step_index, {})
            
            # Map JSON data to BMR parameters based on step type
            mapped = self._map_json_to_bmr_params(step_name, analysis_data)
            

            # Fill in the widgets
            for param_name, value in mapped.items():
                if param_name in widgets:
                    widget = widgets[param_name]
                    # Entry widget
                    if hasattr(widget, 'delete'):
                        widget.delete(0, "end")
                        widget.insert(0, str(value))
                    # OptionMenu
                    elif hasattr(widget, 'set'):
                        widget.set(str(value))
            # Note: table arrays would need a bit more logic; left as manual or add later

            
            self._mark_step_in_progress(self.current_step_index)
            
            messagebox.showinfo("Success", f"Imported data from JSON analysis\n{len(mapped)} parameters filled")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import JSON:\n{e}")
    
    def _map_json_to_bmr_params(self, step_name: str, analysis_data: dict) -> dict:
        """Map JSON analysis data to BMR parameter fields."""
        out = {}
        files = analysis_data.get('files', [])
        
        # Step 2: Loading & Pumpdown
        if 'loading' in step_name or 'pumpdown' in step_name:
            for file_data in files:
                if file_data.get('file_type') == 'outgas':
                    metrics = file_data.get('metrics', {})
                    out['base_pressure_mbar'] = f"{metrics.get('base_pressure_mbar', ''):.2e}"
                    out['peak_mbe_pressure_mbar'] = f"{metrics.get('peak_pressure_mbar', ''):.2e}"
                    
                    # Duration
                    total_s = metrics.get('total_duration_s', 0)
                    out['pumpdown_duration'] = f"{total_s / 3600:.1f} hours"
                    
                    out['file_name'] = file_data.get('filename', '')
                    break
        
        # Step 3: Degas
        elif 'degas' in step_name:
            for file_data in files:
                if file_data.get('file_type') == 'outgas':
                    metrics = file_data.get('metrics', {})
                    
                    # Pressures
                    out['resistive_degas_pressure_mbar'] = f"{metrics.get('peak_pressure_mbar', ''):.2e}"
                    out['direct_degas_pressure_mbar'] = f"{metrics.get('final_pressure_mbar', ''):.2e}"
                    
                    # Duration estimates (split total time)
                    total_duration_s = metrics.get('total_duration_s', 0)
                    time_to_peak_s = metrics.get('peak_time_s', 0)
                    time_after_peak_s = total_duration_s - time_to_peak_s
                    
                    out['resistive_degas_duration_min'] = f"{time_to_peak_s / 60:.1f}"
                    out['direct_degas_duration_min'] = f"{time_after_peak_s / 60:.1f}"
                    
                    # Temperatures - typically need manual entry
                    out['resistive_degas_temp_c'] = "TBD"
                    out['direct_degas_temp_c'] = "TBD"
                    
                    out['file_name'] = file_data.get('filename', '')
                    break
        
        # Step 4: Flash Clean
        elif 'flash' in step_name:
            for file_data in files:
                if file_data.get('file_type') == 'flash':
                    metrics = file_data.get('metrics', {})
                    flashes = metrics.get('flashes', [])
                    
                    if flashes:
                        # Use peak across all flashes
                        peak_temp = metrics.get('peak_temperature_C', 0)
                        out['flash_temperature_c'] = f"{peak_temp:.1f}"
                        
                        # Average flash duration
                        avg_duration = metrics.get('total_flash_time_s', 0) / max(len(flashes), 1)
                        out['flash_duration_s'] = f"{avg_duration:.1f}"
                        
                        out['number_of_flashes'] = str(metrics.get('flash_count', len(flashes)))
                    
                    out['flash_current_a'] = "TBD"  # Not in JSON
                    out['file_name'] = file_data.get('filename', '')
                    break
        
        # Step 5: H-Termination
        elif 'termin' in step_name or 'h-term' in step_name:
            for file_data in files:
                if file_data.get('file_type') == 'hterm':
                    metrics = file_data.get('metrics', {})
                    
                    # Temperature - standard value (not in pressure measurement file)
                    out['h_termination_temp_c'] = "330"
                    
                    out['h_dose_time_s'] = f"{metrics.get('dose_duration_s', ''):.1f}"
                    out['h2_pressure_mbar'] = f"{metrics.get('mean_dose_pressure_mbar', ''):.2e}"
                    out['dose_langmuirs'] = f"{metrics.get('exposure_langmuirs', ''):.1f}"
                    out['file_name'] = file_data.get('filename', '')
                    break
        
        # Step 9: Dopant Dosing
        elif 'dose' in step_name or 'dopant' in step_name:
            for file_data in files:
                if file_data.get('file_type') == 'dose':
                    metrics = file_data.get('metrics', {})
                    filename = file_data.get('filename', '').lower()
                    
                    # Detect dopant species from filename or molecular weight
                    mw = metrics.get('molecular_weight_gmol', 0)
                    if 'ph3' in filename or abs(mw - 34.0) < 1:
                        out['dopant_species'] = 'PH3'
                    elif 'ash3' in filename or 'arsenic' in filename or abs(mw - 77.95) < 1:
                        out['dopant_species'] = 'AsH3'
                    else:
                        out['dopant_species'] = 'Unknown'
                    
                    out['dose_duration_s'] = f"{metrics.get('dose_duration_s', ''):.1f}"
                    
                    # Pressure - convert from Torr to mbar
                    mean_pressure_torr = metrics.get('mean_dose_pressure_torr', 0)
                    out['dose_pressure_mbar'] = f"{mean_pressure_torr * 1.33322:.2e}"
                    
                    out['dose_langmuirs'] = f"{metrics.get('exposure_langmuirs', ''):.2f}"
                    
                    baseline_torr = metrics.get('baseline_pressure_torr', 0)
                    out['background_pressure_mbar'] = f"{baseline_torr * 1.33322:.2e}"
                    
                    out['file_name'] = file_data.get('filename', '')
                    break
        
        # Step 10: Incorporation
        elif 'incorporation' in step_name or 'incorp' in step_name:
            # Temperature typically controlled separately - standard value
            out['incorporation_temp_c'] = "350"
            # Time must be entered manually
        
        # Step 12: RT Growth / Silicon Overgrowth - RT Phase
        elif ('rt' in step_name and ('growth' in step_name or 'phase' in step_name)) or \
             ('overgrowth' in step_name and 'rt' in step_name):
            for file_data in files:
                if file_data.get('file_type') == 'susi':
                    metrics = file_data.get('metrics', {})
                    overgrowth = metrics.get('overgrowth', {})
                    rt_growth = overgrowth.get('RT_growth', {})
                    
                    if rt_growth.get('detected'):
                        # Temperature from calibration
                        temp_c = rt_growth.get('median_temperature_C', 0)
                        temp_k = rt_growth.get('median_temperature_K', 273.15)
                        
                        # Use calibrated temp if available, otherwise assume RT
                        if temp_c > 0:
                            out['growth_temp_c'] = f"{temp_c:.1f}"
                        else:
                            # Convert from K if C is zero
                            out['growth_temp_c'] = f"{temp_k - 273.15:.1f}"
                        
                        out['growth_time_s'] = f"{rt_growth.get('duration_s', ''):.1f}"
                        out['si_flux'] = "TBD"  # Not directly available
                        out['target_thickness_rt_nm'] = f"{rt_growth.get('deposited_nm', ''):.2f}"
                        out['growth_rate'] = f"{rt_growth.get('deposition_rate_ML_min', ''):.2f}"
                        out['file_name'] = file_data.get('filename', '')
                    break
        
        # Step 13: RTA Anneal
        elif 'rta' in step_name or 'anneal' in step_name:
            for file_data in files:
                if file_data.get('file_type') == 'susi':
                    metrics = file_data.get('metrics', {})
                    overgrowth = metrics.get('overgrowth', {})
                    rta = overgrowth.get('RTA_anneal', {})
                    
                    if rta.get('detected'):
                        out['anneal_time_s'] = f"{rta.get('duration_s', ''):.1f}"
                        
                        # Temperature from calibration
                        temp_c = rta.get('median_temperature_C', 0)
                        temp_k = rta.get('median_temperature_K', 273.15)
                        
                        if temp_c > 0:
                            out['anneal_temp_c'] = f"{temp_c:.1f}"
                        else:
                            # Use Kelvin if Celsius not available
                            out['anneal_temp_c'] = f"{temp_k - 273.15:.1f}"
                        
                        out['file_name'] = file_data.get('filename', '')
                    break
        
        # Step 14: LTE Growth / Silicon Overgrowth - LTE Phase
        elif 'lte' in step_name or ('overgrowth' in step_name and 'lte' in step_name.lower()):
            for file_data in files:
                if file_data.get('file_type') == 'susi':
                    metrics = file_data.get('metrics', {})
                    overgrowth = metrics.get('overgrowth', {})
                    lte = overgrowth.get('LTE_growth', {})
                    
                    if lte.get('detected'):
                        out['growth_time_s'] = f"{lte.get('duration_s', ''):.1f}"
                        
                        # Temperature from calibration
                        temp_c = lte.get('median_temperature_C', 0)
                        temp_k = lte.get('median_temperature_K', 273.15)
                        
                        if temp_c > 0:
                            out['growth_temp_c'] = f"{temp_c:.1f}"
                        else:
                            out['growth_temp_c'] = f"{temp_k - 273.15:.1f}"
                        
                        # Total thickness from overgrowth summary
                        total_nm = overgrowth.get('total_deposited_nm', 0)
                        out['total_thickness_nm'] = f"{total_nm:.2f}"
                        
                        out['growth_rate_lte'] = f"{lte.get('deposition_rate_ML_min', ''):.2f}"
                        out['file_name'] = file_data.get('filename', '')
                    break
        
        return out
    

    def save_as_template(self):
        """Save current BMR as a template"""
        if not self.bmr_steps:
            messagebox.showwarning("No Data", "Initialize or load a BMR first.")
            return
        name = simpledialog.askstring("Save Template", "Template name:")
        if not name:
            return
        tpl_path = self._templates_dir() / f"{name}.json"
        with open(tpl_path, "w", encoding="utf-8") as f:
            json.dump(self._to_json_dict(), f, indent=2)
        messagebox.showinfo("Saved", f"Template saved:\n{tpl_path}")

    def load_template(self):
        """Load a BMR template"""
        path = filedialog.askopenfilename(
            title="Load BMR Template",
            filetypes=[("JSON files", "*.json")],
            initialdir=str(self._templates_dir())
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._load_from_json_obj(data)
            messagebox.showinfo("Loaded", f"Template loaded:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load template:\n{e}")

    def manage_templates(self):
        """Open Template Manager window"""
        win = ctk.CTkToplevel(self.tab)
        win.title("Template Manager")
        win.geometry("760x560")
        win.grab_set()

        header = ctk.CTkFrame(win)
        header.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(header, text="BMR Templates", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")

        list_frame = ctk.CTkScrollableFrame(win)
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0,10))

        def refresh():
            for w in list_frame.winfo_children():
                w.destroy()
            for p in sorted(self._templates_dir().glob("*.json")):
                row = ctk.CTkFrame(list_frame)
                row.pack(fill="x", pady=4, padx=2)
                ctk.CTkLabel(row, text=p.name, width=380, anchor="w").pack(side="left", padx=6)
                ctk.CTkButton(row, text="Edit…",  width=80, command=lambda path=p: self._edit_template_dialog(path)).pack(side="left", padx=4)
                ctk.CTkButton(row, text="Rename", width=80, command=lambda path=p: self._rename_template(path, refresh)).pack(side="left", padx=4)
                ctk.CTkButton(row, text="Duplicate", width=90, command=lambda path=p: self._duplicate_template(path, refresh)).pack(side="left", padx=4)
                ctk.CTkButton(row, text="Delete", width=80, fg_color="#B71C1C", hover_color="#7F0000",
                              command=lambda path=p: self._delete_template(path, refresh)).pack(side="left", padx=4)

        refresh()

        footer = ctk.CTkFrame(win)
        footer.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(footer, text="Close", command=win.destroy, width=120).pack(side="right", padx=5)
        ctk.CTkButton(footer, text="Refresh", command=refresh, width=120).pack(side="right", padx=5)

    def _edit_template_dialog(self, path: Path):
        """Open basic editor for template"""
        messagebox.showinfo("Edit Template", f"Open template in text editor:\n{path}")

    def _rename_template(self, path: Path, refresh_cb):
        """Rename template"""
        new_name = simpledialog.askstring("Rename", f"New name for {path.stem}:", initialvalue=path.stem)
        if new_name and new_name != path.stem:
            new_path = path.parent / f"{new_name}.json"
            path.rename(new_path)
            refresh_cb()

    def _duplicate_template(self, path: Path, refresh_cb):
        """Duplicate template"""
        new_name = simpledialog.askstring("Duplicate", f"Name for copy of {path.stem}:", initialvalue=f"{path.stem}_copy")
        if new_name:
            new_path = path.parent / f"{new_name}.json"
            import shutil
            shutil.copy(path, new_path)
            refresh_cb()

    def _delete_template(self, path: Path, refresh_cb):
        """Delete template"""
        if messagebox.askyesno("Confirm Delete", f"Delete template:\n{path.name}?"):
            path.unlink()
            refresh_cb()

    def create_template_stepwise(self):
        """Create a new BMR template step-by-step"""
        win = ctk.CTkToplevel(self.tab)
        win.title("Create BMR Template")
        win.geometry("900x700")
        win.grab_set()
        
        # Header
        header = ctk.CTkFrame(win)
        header.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(header, text="Create New Template", 
                    font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        
        # Template name
        name_frame = ctk.CTkFrame(win)
        name_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(name_frame, text="Template Name:", width=150).pack(side="left", padx=5)
        template_name_entry = ctk.CTkEntry(name_frame, width=400)
        template_name_entry.pack(side="left", padx=5)
        
        # Process type selection
        process_frame = ctk.CTkFrame(win)
        process_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(process_frame, text="Process Type:", width=150).pack(side="left", padx=5)
        process_var = tk.StringVar(value="SET")
        ctk.CTkOptionMenu(process_frame, values=["SET", "Custom"], 
                         variable=process_var, width=200).pack(side="left", padx=5)
        
        # Steps area
        steps_frame = ctk.CTkScrollableFrame(win, height=400)
        steps_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Store step data
        step_widgets = []
        
        def add_step_widget(step_data=None):
            """Add a step configuration widget"""
            step_frame = ctk.CTkFrame(steps_frame)
            step_frame.pack(fill="x", pady=5, padx=5)
            
            widgets = {}
            
            # Step number and name
            header_frame = ctk.CTkFrame(step_frame)
            header_frame.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(header_frame, text=f"Step {len(step_widgets)+1}:", 
                        font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
            
            widgets['name'] = ctk.CTkEntry(header_frame, width=400, 
                                          placeholder_text="Step name")
            widgets['name'].pack(side="left", padx=5)
            
            if step_data:
                widgets['name'].insert(0, step_data.get('step_name', ''))
            
            # Remove button
            ctk.CTkButton(header_frame, text="✕", width=30, fg_color="red",
                         command=lambda: remove_step(step_frame, widgets)).pack(side="right", padx=5)
            
            # Parameters section
            params_frame = ctk.CTkFrame(step_frame)
            params_frame.pack(fill="x", padx=10, pady=5)
            
            # Required parameters
            req_frame = ctk.CTkFrame(params_frame)
            req_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(req_frame, text="Required params:", width=120).pack(side="left", padx=5)
            widgets['required'] = ctk.CTkEntry(req_frame, width=500,
                                               placeholder_text="param1, param2, param3")
            widgets['required'].pack(side="left", padx=5)
            
            if step_data and 'parameters' in step_data:
                req_params = step_data['parameters'].get('required', [])
                widgets['required'].insert(0, ', '.join(req_params))
            
            # Optional parameters
            opt_frame = ctk.CTkFrame(params_frame)
            opt_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(opt_frame, text="Optional params:", width=120).pack(side="left", padx=5)
            widgets['optional'] = ctk.CTkEntry(opt_frame, width=500,
                                               placeholder_text="param4, param5")
            widgets['optional'].pack(side="left", padx=5)
            
            if step_data and 'parameters' in step_data:
                opt_params = step_data['parameters'].get('optional', [])
                widgets['optional'].insert(0, ', '.join(opt_params))
            
            # Quality checks
            qc_frame = ctk.CTkFrame(params_frame)
            qc_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(qc_frame, text="Quality checks:", width=120).pack(side="left", padx=5)
            widgets['quality'] = ctk.CTkEntry(qc_frame, width=500,
                                             placeholder_text="check1, check2")
            widgets['quality'].pack(side="left", padx=5)
            
            if step_data and 'quality_checks' in step_data:
                widgets['quality'].insert(0, ', '.join(step_data['quality_checks']))
            
            step_widgets.append(widgets)
        
        def remove_step(frame, widgets):
            """Remove a step widget"""
            frame.destroy()
            if widgets in step_widgets:
                step_widgets.remove(widgets)
            renumber_steps()
        
        def renumber_steps():
            """Update step numbers after deletion"""
            for i, frame in enumerate(steps_frame.winfo_children()):
                if isinstance(frame, ctk.CTkFrame):
                    header = frame.winfo_children()[0]
                    label = header.winfo_children()[0]
                    if isinstance(label, ctk.CTkLabel):
                        label.configure(text=f"Step {i+1}:")
        
        def load_standard_steps():
            """Load standard SET steps as template"""
            for widget_set in step_widgets[:]:
                step_widgets.remove(widget_set)
            for widget in steps_frame.winfo_children():
                widget.destroy()
            
            for step in STANDARD_BMR_STEPS:
                add_step_widget(step)
        
        def save_template():
            """Save the configured template"""
            name = template_name_entry.get().strip()
            if not name:
                messagebox.showwarning("No Name", "Please enter a template name.")
                return
            
            if not step_widgets:
                messagebox.showwarning("No Steps", "Please add at least one step.")
                return
            
            # Build step definitions (used to render the form)
            step_definitions = []
            for i, widgets in enumerate(step_widgets):
                step_name = widgets['name'].get().strip()
                if not step_name:
                    messagebox.showwarning("Missing Name", f"Step {i+1} needs a name.")
                    return
                
                req_text = widgets['required'].get().strip()
                opt_text = widgets['optional'].get().strip()
                qc_text = widgets['quality'].get().strip()
                
                required = [p.strip() for p in req_text.split(',') if p.strip()]
                optional = [p.strip() for p in opt_text.split(',') if p.strip()]
                quality_checks = [q.strip() for q in qc_text.split(',') if q.strip()]
                
                step_definitions.append({
                    "step_number": i + 1,
                    "step_name": step_name,
                    "parameters": {
                        "required": required,
                        "optional": optional
                    },
                    "quality_checks": quality_checks
                })
            
            # Create template BMR data
            template_data = {
                "metadata": {
                    "template_name": name,
                    "process_type": process_var.get(),
                    "created_date": datetime.now().isoformat(),
                    "batch_number": "",
                    "device_id": "",
                    "operator": "",
                    "start_date": None,
                    "target_completion": None
                },
                "step_definitions": step_definitions,  # NEW: include definitions
                "steps": []                             # empty run-time steps (filled below)
            }
            
            # Initialize empty step data
            for step in step_definitions:
                step_data = {
                    "step_number": step["step_number"],
                    "step_name": step["step_name"],
                    "status": "not_started",
                    "operator_initials": "",
                    "start_time": None,
                    "end_time": None,
                    "parameters": {},
                    "quality_notes": "",
                    "pre_check_pass": False,
                    "post_check_pass": False,
                    "deviations": [],
                    "corrective_actions": [],
                    "labview_file": "",
                    "sxm_scans": [],
                    "verified_by": "",
                    "verified_time": None
                }
                template_data["steps"].append(step_data)
            
            # Save to templates directory
            tpl_path = self._templates_dir() / f"{name}.json"
            try:
                with open(tpl_path, "w", encoding="utf-8") as f:
                    json.dump(template_data, f, indent=2)
                messagebox.showinfo("Success", f"Template saved:\n{tpl_path}")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save template:\n{e}")
        
        # Button bar
        btn_frame = ctk.CTkFrame(win)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(btn_frame, text="➕ Add Step", 
                     command=lambda: add_step_widget(),
                     width=120).pack(side="left", padx=5)
        
        ctk.CTkButton(btn_frame, text="📋 Load SET Steps", 
                     command=load_standard_steps,
                     width=140).pack(side="left", padx=5)
        
        ctk.CTkButton(btn_frame, text="💾 Save Template", 
                     command=save_template,
                     fg_color="#2fa572", hover_color="#1d6e4a",
                     width=140).pack(side="right", padx=5)
        
        ctk.CTkButton(btn_frame, text="Cancel", 
                     command=win.destroy,
                     width=100).pack(side="right", padx=5)
        
        # Start with one empty step
        add_step_widget()
