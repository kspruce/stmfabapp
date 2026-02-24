import customtkinter as ctk
import tkinter as tk
from typing import List, Dict, Any, Optional


class PlanBuilderDialog(ctk.CTkToplevel):
    """
    Wizard-style dialog for creating an entire fabrication plan in one go.
    Steps are kept in-memory until confirmed.
    """

    def __init__(self, parent, db_ops, device_id: int,
                 initial_steps: Optional[List[Dict[str, Any]]] = None,
                 on_complete=None):
        super().__init__(parent)

        self.db_ops = db_ops
        self.device_id = device_id
        self.on_complete = on_complete

        self.title("Create Fabrication Plan")
        self.geometry("900x600")
        self.grab_set()

        # In-memory draft steps
        self.steps: List[Dict[str, Any]] = initial_steps[:] if initial_steps else []

        self._build_ui()
        self._render_steps()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.main = ctk.CTkFrame(self)
        self.main.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            self.main,
            text="Fabrication Plan Builder",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(
            self.main,
            text="Define all fabrication steps for this device before starting.",
            text_color="gray"
        ).pack(anchor="w", pady=(0, 15))

        # Table header
        header = ctk.CTkFrame(self.main)
        header.pack(fill="x")

        ctk.CTkLabel(header, text="#", width=30).pack(side="left")
        ctk.CTkLabel(header, text="Step name").pack(side="left", padx=5)
        ctk.CTkLabel(header, text="Requires scan", width=120).pack(side="right", padx=10)

        # Scroll area
        self.scroll = ctk.CTkScrollableFrame(self.main)
        self.scroll.pack(fill="both", expand=True, pady=10)

        # Controls
        controls = ctk.CTkFrame(self.main)
        controls.pack(fill="x", pady=10)

        ctk.CTkButton(
            controls,
            text="+ Add step",
            command=self._add_step,
            width=120
        ).pack(side="left")

        ctk.CTkButton(
            controls,
            text="Create plan",
            command=self._commit,
            fg_color="green",
            hover_color="darkgreen"
        ).pack(side="right")

        ctk.CTkButton(
            controls,
            text="Cancel",
            command=self.destroy
        ).pack(side="right", padx=10)

    # ------------------------------------------------------------------
    # Step handling
    # ------------------------------------------------------------------

    def _add_step(self):
        self.steps.append({
            "name": "",
            "purpose": "",
            "requires_scan": True
        })
        self._render_steps(focus_last=True)

    def _render_steps(self, focus_last=False):
        for w in self.scroll.winfo_children():
            w.destroy()

        for idx, step in enumerate(self.steps):
            self._render_step_row(idx, step)

        if focus_last and self.scroll.winfo_children():
            last = self.scroll.winfo_children()[-1]
            entry = last.winfo_children()[1]
            entry.focus_set()

    def _render_step_row(self, index: int, step: Dict[str, Any]):
        row = ctk.CTkFrame(self.scroll)
        row.pack(fill="x", pady=2)

        # Index
        ctk.CTkLabel(row, text=str(index + 1), width=30).pack(side="left")

        # Name entry
        name_var = tk.StringVar(value=step["name"])
        name_entry = ctk.CTkEntry(row, textvariable=name_var)
        name_entry.pack(side="left", fill="x", expand=True, padx=5)

        def update_name(*_):
            step["name"] = name_var.get()

        name_var.trace_add("write", update_name)

        # Requires scan
        scan_var = tk.BooleanVar(value=step.get("requires_scan", True))
        scan_chk = ctk.CTkCheckBox(
            row,
            text="",
            variable=scan_var,
            width=120
        )
        scan_chk.pack(side="right", padx=10)

        def update_scan(*_):
            step["requires_scan"] = scan_var.get()

        scan_var.trace_add("write", update_scan)

        # Delete
        del_btn = ctk.CTkButton(
            row,
            text="✕",
            width=30,
            fg_color="transparent",
            text_color="red",
            command=lambda: self._delete_step(index)
        )
        del_btn.pack(side="right")

    def _delete_step(self, index: int):
        del self.steps[index]
        self._render_steps()

    # ------------------------------------------------------------------
    # Commit to database
    # ------------------------------------------------------------------

    def _commit(self):
        cleaned = [
            s for s in self.steps
            if s.get("name", "").strip()
        ]

        if not cleaned:
            ctk.CTkMessageBox(
                title="No steps",
                message="You must define at least one step.",
                icon="warning"
            )
            return

        step_defs = []
        for i, s in enumerate(cleaned, start=1):
            step_defs.append({
                "step_num": i,
                "name": s["name"],
                "purpose": s.get("purpose", ""),
                "requires_scan": s.get("requires_scan", True)
            })

        self.db_ops.initialize_device_steps(
            device_id=self.device_id,
            step_definitions=step_defs,
            overwrite=True
        )

        if self.on_complete:
            self.on_complete()

        self.destroy()
