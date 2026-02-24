"""
fabrication_steps_tab.py - GUI Tab for Fabrication Steps Management

REORGANIZED VERSION:
- Simplified device selection (no initialization)
- HTML report generation integrated into this tab
- Improved scan management with clear step selection
- Add scans directly within step cards
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog, simpledialog, ttk
import tkinter as tk
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from stm_fab.gui.plan_builder import PlanBuilderDialog



class FabricationStepsTab:
    """
    Tab for managing fabrication steps for devices
    
    Streamlined workflow:
    - Select existing device
    - View/manage fabrication steps
    - Add scans to steps
    - Generate HTML reports
    """
    
    def __init__(self, parent_app, tabview):
        """
        Initialize the fabrication steps tab
        
        Args:
            parent_app: Main application instance (STMFabGUIEnhanced)
            tabview: The CTkTabview to add this tab to
        """
        self.app = parent_app
        self.tabview = tabview
        self.selected_step_id = None

        # Get or create the Fabrication Steps tab
        tab_name = "🔬  Fabrication Steps"
        
        # Check if tab already exists, if not create it
        if tab_name not in self.tabview._tab_dict:
            self.tabview.add(tab_name)
        
        self.tab = self.tabview.tab(tab_name)
        
        # Current selections
        self.current_device = None
        self.selected_step = None
        self.step_widgets = {}  # step_id -> widget references
        
        # Build UI
        self.build_ui()
    
    def build_ui(self):
        """Build the fabrication steps tab interface"""
        # Clear existing
        for widget in self.tab.winfo_children():
            widget.destroy()
        
        # Main container with two columns - use pack for compatibility
        main_container = ctk.CTkFrame(self.tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left column - Device selection and actions (fixed width)
        left_frame = ctk.CTkFrame(main_container, width=280)
        left_frame.pack(side="left", fill="both", padx=(0, 5))
        left_frame.pack_propagate(False)  # Maintain fixed width
        
        # Right column - Steps display (expands)
        right_frame = ctk.CTkFrame(main_container)
        right_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        # Build panels
        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)
    
    def _build_left_panel(self, parent):
        """Build the left control panel"""
        # Make left panel scrollable in case of small screens
        scroll_container = ctk.CTkScrollableFrame(parent)
        scroll_container.pack(fill="both", expand=True)
        
        # Header
        ctk.CTkLabel(
            scroll_container,
            text="Device Selection",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Device info display
        self.device_info_frame = ctk.CTkFrame(scroll_container)
        self.device_info_frame.pack(fill="x", padx=10, pady=5)
        
        self.device_label = ctk.CTkLabel(
            self.device_info_frame,
            text="No device selected",
            font=ctk.CTkFont(size=12),
            wraplength=240
        )
        self.device_label.pack(pady=10)
        
        # Select device button
        ctk.CTkButton(
            scroll_container,
            text="📱 Select Device",
            command=self.select_device_dialog,
            height=35,
            fg_color="#1f6aa5",
            hover_color="#144870"
        ).pack(padx=10, pady=5, fill="x")
        
        # Separator
        ctk.CTkFrame(scroll_container, height=2, fg_color="gray").pack(fill="x", padx=10, pady=15)
        
        # Actions section
        ctk.CTkLabel(
            scroll_container,
            text="Actions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Add custom step button
        ctk.CTkButton(
            scroll_container,
            text="➕ Add Custom Step",
            command=self.add_custom_step_dialog,
            fg_color="#2fa572",
            hover_color="#1d6e4a",
            height=35
        ).pack(padx=10, pady=5, fill="x")
        
        # Initialize standard protocol (kept for convenience)
        ctk.CTkButton(
            scroll_container,
            text="📋 Initialize SET Protocol",
            command=self.initialize_standard_protocol,
            fg_color="#5a5a5a",
            hover_color="#404040",
            height=30
        ).pack(padx=10, pady=5, fill="x")
        
        # Generate HTML report button
        ctk.CTkButton(
            scroll_container,
            text="📄 Generate HTML Report",
            command=self.generate_html_report,
            fg_color="#8b4513",
            hover_color="#654321",
            height=35
        ).pack(padx=10, pady=5, fill="x")
        
        # Separator
        ctk.CTkFrame(scroll_container, height=2, fg_color="gray").pack(fill="x", padx=10, pady=15)
        
        # Statistics
        ctk.CTkLabel(
            scroll_container,
            text="Progress",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        self.stats_frame = ctk.CTkFrame(scroll_container)
        self.stats_frame.pack(fill="x", padx=10, pady=5)
        
        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="Select a device",
            font=ctk.CTkFont(size=10),
            justify="left"
        )
        self.stats_label.pack(pady=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(scroll_container)
        self.progress_bar.pack(padx=10, pady=5, fill="x")
        self.progress_bar.set(0)
        
        # Separator
        ctk.CTkFrame(scroll_container, height=2, fg_color="gray").pack(fill="x", padx=10, pady=15)
        
        # Refresh button at bottom
        ctk.CTkButton(
            scroll_container,
            text="🔄 Refresh",
            command=self.refresh_steps,
            height=30
        ).pack(padx=10, pady=10, fill="x")
    
    def _build_right_panel(self, parent):
        """Build the right panel with steps display"""
        # Header with selected step indicator
        header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        header_frame.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(
            header_frame,
            text="Fabrication Steps",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            header_frame,
            text="🧱 Create / Edit Plan",
            command=self.open_plan_builder,
            width=160,
            height=32
        ).pack(side="right", padx=10)
                
        
        # Selected step indicator
        self.selected_step_label = ctk.CTkLabel(
            header_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.selected_step_label.pack(side="left", padx=10)
        
        # Scrollable steps container
        self.steps_scroll = ctk.CTkScrollableFrame(
            parent,
            corner_radius=10
        )
        self.steps_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Initial placeholder
        self.placeholder_label = ctk.CTkLabel(
            self.steps_scroll,
            text="Select a device to view its fabrication steps",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.placeholder_label.pack(pady=50)
    
    def select_device_dialog(self):
        """Show dialog to select a device"""
        # Create selection dialog
        dialog = ctk.CTkToplevel(self.app.root)
        dialog.title("Select Device")
        dialog.geometry("650x600")
        dialog.transient(self.app.root)
        dialog.grab_set()
        
        # Configure grid for proper expansion
        dialog.grid_rowconfigure(0, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        
        # Main frame
        main_frame = ctk.CTkFrame(dialog)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Configure main frame grid
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Header
        ctk.CTkLabel(
            main_frame,
            text="Select Device for Fabrication Steps",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=10, sticky="ew")
        
        # Scrollable frame for devices
        scroll_frame = ctk.CTkScrollableFrame(main_frame)
        scroll_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Load devices grouped by sample
        all_devices = self.app.db_ops.list_devices()
        
        if not all_devices:
            ctk.CTkLabel(
                scroll_frame,
                text="No devices found. Create a device first.",
                text_color="gray"
            ).pack(pady=20)
        else:
            # Group by sample
            devices_by_sample = {}
            for device in all_devices:
                sample_name = device.sample.sample_name
                if sample_name not in devices_by_sample:
                    devices_by_sample[sample_name] = []
                devices_by_sample[sample_name].append(device)
            
            # Display grouped devices
            for sample_name, devices in devices_by_sample.items():
                # Sample header
                sample_frame = ctk.CTkFrame(scroll_frame)
                sample_frame.pack(fill="x", padx=5, pady=5)
                
                ctk.CTkLabel(
                    sample_frame,
                    text=f"Sample: {sample_name}",
                    font=ctk.CTkFont(size=13, weight="bold")
                ).pack(anchor="w", padx=10, pady=5)
                
                # Device buttons
                for device in devices:
                    def select_device(d=device):
                        self.current_device = d
                        self.load_device_steps()
                        self.update_device_info()
                        self.update_statistics()
                        dialog.destroy()
                    
                    device_btn = ctk.CTkButton(
                        sample_frame,
                        text=f"  {device.device_name}  ({device.overall_status})",
                        command=select_device,
                        height=30,
                        anchor="w"
                    )
                    device_btn.pack(fill="x", padx=20, pady=2)
        
        # Close button
        ctk.CTkButton(
            main_frame,
            text="Cancel",
            command=dialog.destroy,
            width=100
        ).grid(row=2, column=0, pady=10)
    
    def load_device_steps(self):
        """Load and display fabrication steps for current device"""
        if not self.current_device:
            return
        
        # Clear steps display - destroy all widgets including placeholder
        for widget in self.steps_scroll.winfo_children():
            widget.destroy()
        
        self.step_widgets.clear()
        self.placeholder_label = None  # Clear reference
        
        # Get steps from database
        steps = self.app.db_ops.get_device_steps(self.current_device.device_id)
        
        if not steps:
            # Show "no steps" message
            no_steps_label = ctk.CTkLabel(
                self.steps_scroll,
                text="No steps defined.\nClick 'Initialize SET Protocol' or 'Add Custom Step'",
                font=ctk.CTkFont(size=13),
                text_color="gray"
            )
            no_steps_label.pack(pady=50)
            return
        
        # Display each step
        for step in steps:
            self._create_step_card(step)
    
    def _create_step_card(self, step):
        """Create a card widget for a fabrication step"""
        # Main step frame
        step_frame = ctk.CTkFrame(
            self.steps_scroll,
            corner_radius=10,
            border_width=2,
            border_color="gray"
        )
        step_frame.pack(fill="x", padx=5, pady=5)
        
        # Store reference
        self.step_widgets[step.step_id] = {
            'frame': step_frame,
            'step': step
        }
        
        # Make frame clickable for selection
        def select_step(event=None):
            self.selected_step = step
            self._update_step_selection()
        
        step_frame.bind("<Button-1>", select_step)
        
        # Header row with step info and controls
        header_frame = ctk.CTkFrame(step_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=5)
        
        # Left side - step info
        info_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True)
        
        # Step number and name
        step_title = ctk.CTkLabel(
            info_frame,
            text=f"Step {step.step_number}: {step.step_name}",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        step_title.pack(anchor="w")
        step_title.bind("<Button-1>", select_step)
        
        # Status indicator
        status_colors = {
            'pending': 'orange',
            'complete': 'green',
            'in_progress': 'blue',
            'skipped': 'gray',
            'failed': 'red'
        }
        status_color = status_colors.get(step.status, 'gray')
        
        status_label = ctk.CTkLabel(
            info_frame,
            text=f"● {step.status.upper()}",
            font=ctk.CTkFont(size=11),
            text_color=status_color,
            anchor="w"
        )
        status_label.pack(anchor="w")
        status_label.bind("<Button-1>", select_step)
        
        # Right side - action buttons
        btn_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        btn_frame.pack(side="right")
        # Edit step button
        ctk.CTkButton(
            btn_frame,
            text="✏ Edit",
            command=lambda s=step: self.edit_step_dialog(s),
            width=60,
            height=28
        ).pack(side="left", padx=2)

        
        # Add scan button (direct action)
        if step.requires_scan:
            ctk.CTkButton(
                btn_frame,
                text="📷 Add Scan",
                command=lambda: self.add_scan_to_specific_step(step),
                width=100,
                height=28,
                fg_color="#2fa572",
                hover_color="#1d6e4a"
            ).pack(side="left", padx=2)
        
        # Status dropdown
        status_var = tk.StringVar(value=step.status)
        status_menu = ctk.CTkOptionMenu(
            btn_frame,
            variable=status_var,
            values=['pending', 'in_progress', 'complete', 'skipped', 'failed'],
            command=lambda val: self.update_step_status(step, val),
            width=110,
            height=28
        )
        status_menu.pack(side="left", padx=2)
        
        # Delete button
        ctk.CTkButton(
            btn_frame,
            text="🗑",
            command=lambda: self.delete_step_confirm(step),
            width=30,
            height=28,
            fg_color="red",
            hover_color="darkred"
        ).pack(side="left", padx=2)
        
        # Purpose (if available)
        if step.purpose:
            purpose_label = ctk.CTkLabel(
                step_frame,
                text=step.purpose,
                font=ctk.CTkFont(size=11),
                text_color="gray",
                wraplength=700,
                anchor="w",
                justify="left"
            )
            purpose_label.pack(anchor="w", padx=10, pady=(0, 5))
            purpose_label.bind("<Button-1>", select_step)
        
        # Scans section
        scans = self.app.db_ops.get_step_scans(step.step_id)
        if scans:
            scans_frame = ctk.CTkFrame(step_frame)
            scans_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            ctk.CTkLabel(
                scans_frame,
                text=f"📷 Scans ({len(scans)}):",
                font=ctk.CTkFont(size=11, weight="bold")
            ).pack(anchor="w", padx=5, pady=2)
            
            for scan in scans:
                scan_item = ctk.CTkFrame(scans_frame, fg_color="transparent")
                scan_item.pack(fill="x", padx=5, pady=1)
                
                ctk.CTkLabel(
                    scan_item,
                    text=f"  • {scan.filename}",
                    font=ctk.CTkFont(size=10),
                    anchor="w"
                ).pack(side="left")
                
                # Remove scan button
                ctk.CTkButton(
                    scan_item,
                    text="×",
                    command=lambda s=scan: self.remove_scan_confirm(s, step),
                    width=20,
                    height=20,
                    fg_color="red",
                    hover_color="darkred"
                ).pack(side="right", padx=2)
                
                # View details button
                ctk.CTkButton(
                    scan_item,
                    text="ℹ",
                    command=lambda s=scan: self.view_scan_details(s),
                    width=20,
                    height=20
                ).pack(side="right", padx=2)
    
    def _update_step_selection(self):
        """Update visual indication of selected step"""
        # Reset all borders
        for step_id, widgets in self.step_widgets.items():
            widgets['frame'].configure(border_color="gray", border_width=2)
        
        # Highlight selected step
        if self.selected_step and self.selected_step.step_id in self.step_widgets:
            self.step_widgets[self.selected_step.step_id]['frame'].configure(
                border_color="#1f6aa5",
                border_width=3
            )
            self.selected_step_label.configure(
                text=f"Selected: Step {self.selected_step.step_number}"
            )
        else:
            self.selected_step_label.configure(text="")
    
    def add_scan_to_specific_step(self, step):
        """Add scan to a specific step directly"""
        self.selected_step = step
        self.add_scan_to_step()
    
    def add_scan_to_step(self):
        """Add a scan file or image to the currently selected step"""
        if not self.selected_step:
            messagebox.showwarning("No Step Selected",
                                 "Please select a fabrication step first")
            return
        
        # File dialog to select .sxm file OR image file
        filetypes = [
            ("All Supported Files", "*.sxm *.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
            ("STM Scan Files", "*.sxm"),
            ("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
            ("All Files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title=f"Select scan or image for Step {self.selected_step.step_number}",
            filetypes=filetypes
        )
        
        if not filepath:
            return
        
        try:
            file_path = Path(filepath)
            file_extension = file_path.suffix.lower()
            
            # Check if it's an image file or .sxm
            if file_extension in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                # Handle as image file
                self._add_image_file_to_step(file_path)
            elif file_extension == '.sxm':
                # Handle as STM scan file
                self._add_sxm_file_to_step(file_path)
            else:
                messagebox.showwarning("Unsupported File Type",
                                     f"File type {file_extension} is not supported.\n"
                                     "Please select .sxm or image files (.png, .jpg, .tif, etc.)")
                return
            
            self.refresh_steps()
            messagebox.showinfo("Success",
                              f"File added to Step {self.selected_step.step_number}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add file:\n{str(e)}")
    
    def _add_image_file_to_step(self, file_path: Path):
        """Add a pre-processed image file to a step (screenshots, edited images, device plans)"""
        from PIL import Image
        import base64
        from io import BytesIO
        
        # Read and convert image to base64
        try:
            img = Image.open(file_path)
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes buffer as PNG
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            
            # Encode as base64
            image_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Create metadata dictionary for the image
            metadata = {
                'File Type': 'Image File',
                'Original Filename': file_path.name,
                'Image Size': f"{img.width} x {img.height} pixels",
                'Format': img.format if img.format else file_path.suffix.upper().strip('.'),
                'Mode': img.mode,
                'File Size': f"{file_path.stat().st_size / 1024:.1f} KB"
            }
            
            # Add to database as STMScan (reusing the same table structure)
            scan = self.app.db_ops.add_stm_scan(
                step_id=self.selected_step.step_id,
                filename=file_path.name,
                filepath=str(file_path),
                metadata=metadata,
                image_data=image_base64
            )
            
        except Exception as e:
            raise Exception(f"Failed to process image file: {str(e)}")
    


    def _add_sxm_file_to_step(self, file_path: Path):
        """Add an STM .sxm file to a step (with processing and Gwyddion colormap)"""
        try:
            import pySPM
            from stm_fab.reports.fabrication_record import (
                parse_sxm_metadata,
                generate_image_base64,
                load_gwyddion_colormap
            )

            # Parse metadata
            metadata = parse_sxm_metadata(str(file_path))

            from stm_fab.reports.fabrication_record import load_gwyddion_colormap, generate_image_base64, parse_sxm_metadata

            # Load Gwyddion colormap (ensure resource exists at stm_fab/resources/gwyddionnet.pymap)
            cmap = load_gwyddion_colormap(r"C:\Projects\stm_app2\stm_fab\resources\gwyddionnet.pymap")

            # Generate processed Base64 image
            image_base64 = generate_image_base64(
                str(file_path),
                colormap=cmap,
                plane_level=True,
                line_subtract=True,    # row-wise (default)
                line_axis='row',
                line_method='median'
            )

            # Add to database
            scan = self.app.db_ops.add_stm_scan(
                step_id=self.selected_step.step_id,
                filename=file_path.name,
                filepath=str(file_path),
                metadata=metadata,
                image_data=image_base64
            )

        except Exception as e:
            # Minimal metadata fallback
            metadata = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'File Type': 'STM Scan (.sxm)'
            }
            scan = self.app.db_ops.add_stm_scan(
                step_id=self.selected_step.step_id,
                filename=file_path.name,
                filepath=str(file_path),
                metadata=metadata
            )

    def open_plan_builder(self):
        """Open the fabrication plan builder for the current device"""
        if not self.current_device:
            messagebox.showwarning(
                "No Device Selected",
                "Please select a device first."
            )
            return

        device_id = self.current_device.device_id
        existing_steps = self.app.db_ops.get_device_steps(device_id)

        # If steps exist, confirm overwrite
        if existing_steps:
            confirm = messagebox.askyesno(
                "Overwrite Fabrication Plan?",
                (
                    f"This device already has {len(existing_steps)} fabrication steps.\n\n"
                    "Opening the Plan Builder will REPLACE all existing steps.\n\n"
                    "This action cannot be undone.\n\n"
                    "Continue?"
                ),
                icon="warning"
            )
            if not confirm:
                return

        def refresh_after_plan():
            self.refresh_steps()
            self.app.update_status("Fabrication plan created / updated")

        PlanBuilderDialog(
            parent=self.app.root,
            db_ops=self.app.db_ops,
            device_id=device_id,
            initial_steps=[
                {
                    "name": s.step_name,
                    "purpose": s.purpose or "",
                    "requires_scan": s.requires_scan
                }
                for s in existing_steps
            ],
            on_complete=refresh_after_plan
        )

    
    def initialize_standard_protocol(self):
        """Initialize device with standard SET protocol"""
        if not self.current_device:
            messagebox.showwarning("No Device",
                                 "Please select a device first")
            return
        
        # Check if device already has steps
        existing = self.app.db_ops.get_device_steps(self.current_device.device_id)
        if existing:
            response = messagebox.askyesno(
                "Steps Exist",
                f"Device already has {len(existing)} steps.\n"
                "Initialize anyway? (Will add to existing steps)",
                icon='warning'
            )
            if not response:
                return
        
        try:
            # Get standard protocol from fabrication_steps module
            from stm_fab.fabrication_steps import get_standard_set_steps
            standard_steps = get_standard_set_steps()
            
            # Add each step to database
            for step_def in standard_steps:
                self.app.db_ops.add_fabrication_step(
                    device_id=self.current_device.device_id,
                    step_number=step_def.step_num,
                    step_name=step_def.name,
                    purpose=step_def.purpose,
                    requires_scan=step_def.requires_scan,
                    notes=step_def.note
                )
            
            self.refresh_steps()
            self.update_statistics()
            messagebox.showinfo("Success",
                              f"Initialized {len(standard_steps)} standard SET steps")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize protocol:\n{str(e)}")
    
    def add_custom_step_dialog(self):
        """Dialog to add a custom fabrication step"""
        if not self.current_device:
            messagebox.showwarning("No Device",
                                 "Please select a device first")
            return
        
        dialog = ctk.CTkToplevel(self.app.root)
        dialog.title("Add Custom Step")
        dialog.geometry("550x600")
        dialog.transient(self.app.root)
        dialog.grab_set()
        
        # Configure grid
        dialog.grid_rowconfigure(0, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        
        # Main scrollable frame
        main_scroll = ctk.CTkScrollableFrame(dialog)
        main_scroll.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        ctk.CTkLabel(
            main_scroll,
            text="Add Custom Fabrication Step",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Step number
        ctk.CTkLabel(main_scroll, text="Step Number:").pack(anchor="w", pady=(10, 2))
        step_num_entry = ctk.CTkEntry(main_scroll, placeholder_text="Enter step number")
        step_num_entry.pack(fill="x", pady=(0, 10))
        
        # Auto-suggest next number
        existing = self.app.db_ops.get_device_steps(self.current_device.device_id)
        next_num = max([s.step_number for s in existing], default=0) + 1
        step_num_entry.insert(0, str(next_num))
        
        # Step name
        ctk.CTkLabel(main_scroll, text="Step Name:").pack(anchor="w", pady=(10, 2))
        name_entry = ctk.CTkEntry(main_scroll, placeholder_text="Enter step name")
        name_entry.pack(fill="x", pady=(0, 10))
        
        # Purpose
        ctk.CTkLabel(main_scroll, text="Purpose/Description:").pack(anchor="w", pady=(10, 2))
        purpose_entry = ctk.CTkTextbox(main_scroll, height=100)
        purpose_entry.pack(fill="x", pady=(0, 10))
        
        # Requires scan
        requires_scan_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            main_scroll,
            text="Requires STM Scan",
            variable=requires_scan_var
        ).pack(anchor="w", pady=5)
        
        # Notes
        ctk.CTkLabel(main_scroll, text="Notes (optional):").pack(anchor="w", pady=(10, 2))
        notes_entry = ctk.CTkTextbox(main_scroll, height=80)
        notes_entry.pack(fill="x", pady=(0, 10))
        
        # Buttons
        btn_frame = ctk.CTkFrame(main_scroll, fg_color="transparent")
        btn_frame.pack(pady=15)
        
        def save_step():
            try:
                step_num = int(step_num_entry.get())
                name = name_entry.get().strip()
                purpose = purpose_entry.get("1.0", "end-1c").strip()
                notes = notes_entry.get("1.0", "end-1c").strip()
                
                if not name:
                    messagebox.showwarning("Missing Info", "Please enter a step name")
                    return
                
                # Add to database
                self.app.db_ops.add_fabrication_step(
                    device_id=self.current_device.device_id,
                    step_number=step_num,
                    step_name=name,
                    purpose=purpose,
                    requires_scan=requires_scan_var.get(),
                    notes=notes
                )
                
                dialog.destroy()
                self.refresh_steps()
                self.update_statistics()
                messagebox.showinfo("Success", "Custom step added")
                
            except ValueError:
                messagebox.showerror("Invalid Input", "Step number must be a number")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add step:\n{str(e)}")
        
        ctk.CTkButton(
            btn_frame,
            text="Add Step",
            command=save_step,
            width=120
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=dialog.destroy,
            width=120
        ).pack(side="left", padx=5)
    ##
    
    def edit_step_dialog(self, step):
        """Dialog to edit an existing fabrication step"""
        dialog = ctk.CTkToplevel(self.app.root)
        dialog.title(f"Edit Step {step.step_number}")
        dialog.geometry("550x650")
        dialog.transient(self.app.root)
        dialog.grab_set()

        # Scrollable container
        scroll = ctk.CTkScrollableFrame(dialog)
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        is_complete = step.status == "complete"

        # Warning label (once, at top)
        if is_complete:
            ctk.CTkLabel(
                scroll,
                text="⚠ This step is COMPLETE. Core fields are locked.",
                text_color="orange",
                font=ctk.CTkFont(size=11)
            ).pack(pady=(0, 10))

        # Title
        ctk.CTkLabel(
            scroll,
            text=f"Edit Step {step.step_number}",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        # Step name
        ctk.CTkLabel(scroll, text="Step Name").pack(anchor="w", pady=(10, 2))
        name_entry = ctk.CTkEntry(scroll)
        name_entry.pack(fill="x")
        name_entry.insert(0, step.step_name)

        if is_complete:
            name_entry.configure(state="disabled")

        # Purpose
        ctk.CTkLabel(scroll, text="Purpose / Description").pack(anchor="w", pady=(10, 2))
        purpose_box = ctk.CTkTextbox(scroll, height=120)
        purpose_box.pack(fill="x")
        purpose_box.insert("1.0", step.purpose or "")

        if is_complete:
            purpose_box.configure(state="disabled")

        # Requires scan
        requires_scan_var = tk.BooleanVar(value=step.requires_scan)
        scan_cb = ctk.CTkCheckBox(
            scroll,
            text="Requires STM Scan",
            variable=requires_scan_var
        )
        scan_cb.pack(anchor="w", pady=10)

        if is_complete:
            scan_cb.configure(state="disabled")

        # Notes (intentionally NOT locked – append-only at DB layer)
        ctk.CTkLabel(scroll, text="Notes").pack(anchor="w", pady=(10, 2))
        notes_box = ctk.CTkTextbox(scroll, height=100)
        notes_box.pack(fill="x")
        notes_box.insert("1.0", step.notes or "")

        # Buttons
        btn_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_frame.pack(pady=20)


        def save_changes():
            try:
                update_kwargs = {
                    "step_id": step.step_id,
                    "notes": notes_box.get("1.0", "end-1c").strip(),
                    "operator": getattr(self.app, "current_user", None)
                }

                # Only allow core field edits if NOT complete
                if not is_complete:
                    update_kwargs.update({
                        "step_name": name_entry.get().strip(),
                        "purpose": purpose_box.get("1.0", "end-1c").strip(),
                        "requires_scan": requires_scan_var.get()
                    })

                self.app.db_ops.update_fabrication_step(**update_kwargs)

                dialog.destroy()
                self.refresh_steps()
                self.update_statistics()
                messagebox.showinfo("Saved", "Step updated successfully")

            except Exception as e:
                messagebox.showerror("Error", str(e))


        ctk.CTkButton(
            btn_frame,
            text="Save Changes",
            command=save_changes,
            width=140
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=dialog.destroy,
            width=140
        ).pack(side="left", padx=5)


    
    def update_step_status(self, step, new_status):
        """Update step status"""
        try:
            self.app.db_ops.update_fabrication_step(
                step.step_id,
                status=new_status
            )
            step.status = new_status
            self.refresh_steps()
            self.update_statistics()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update status:\n{str(e)}")
    
    def delete_step_confirm(self, step):
        """Confirm and delete a step"""
        response = messagebox.askyesno(
            "Delete Step",
            f"Delete step {step.step_number}: {step.step_name}?\n\n"
            "This will also delete all associated scans.",
            icon='warning'
        )
        
        if response:
            try:
                self.app.db_ops.delete_fabrication_step(step.step_id)
                self.refresh_steps()
                self.update_statistics()
                messagebox.showinfo("Success", "Step deleted")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete step:\n{str(e)}")
    
    def remove_scan_confirm(self, scan, step):
        """Confirm and remove a scan"""
        response = messagebox.askyesno(
            "Remove Scan",
            f"Remove scan '{scan.filename}' from step {step.step_number}?",
            icon='warning'
        )
        
        if response:
            try:
                self.app.db_ops.delete_stm_scan(scan.scan_id)
                self.refresh_steps()
                messagebox.showinfo("Success", "Scan removed")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to remove scan:\n{str(e)}")
    
    def view_scan_details(self, scan):
        """Show detailed scan information"""
        dialog = ctk.CTkToplevel(self.app.root)
        dialog.title(f"Scan Details - {scan.filename}")
        dialog.geometry("650x750")
        
        # Configure grid
        dialog.grid_rowconfigure(0, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        
        main_frame = ctk.CTkFrame(dialog)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Configure main frame grid
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Filename
        ctk.CTkLabel(
            main_frame,
            text=scan.filename,
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=10, sticky="ew")
        
        # Scrollable metadata
        scroll = ctk.CTkScrollableFrame(main_frame)
        scroll.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Display metadata
        metadata = scan.metadata_json or {}
        
        if not metadata:
            ctk.CTkLabel(
                scroll,
                text="No metadata available",
                text_color="gray"
            ).pack(pady=20)
        else:
            for key, value in metadata.items():
                item_frame = ctk.CTkFrame(scroll)
                item_frame.pack(fill="x", pady=2, padx=5)
                
                ctk.CTkLabel(
                    item_frame,
                    text=f"{key}:",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    width=200,
                    anchor="w"
                ).pack(side="left", padx=5)
                
                ctk.CTkLabel(
                    item_frame,
                    text=str(value),
                    font=ctk.CTkFont(size=11),
                    anchor="w",
                    wraplength=350
                ).pack(side="left", padx=5, fill="x", expand=True)
        
        # Close button
        ctk.CTkButton(
            main_frame,
            text="Close",
            command=dialog.destroy,
            width=100
        ).grid(row=2, column=0, pady=10)
    
    def generate_html_report(self):
        """Generate HTML fabrication report for current device"""
        if not self.current_device:
            messagebox.showwarning("No Device",
                                 "Please select a device first")
            return
        
        try:
            # Import the report generator
            from stm_fab.reports.fabrication_record import create_fabrication_record
            
            # Ask for output location
            default_filename = f"{self.current_device.device_name}_fabrication_record.html"
            filepath = filedialog.asksaveasfilename(
                title="Save HTML Report",
                defaultextension=".html",
                filetypes=[("HTML Files", "*.html"), ("All Files", "*.*")],
                initialfile=default_filename
            )
            
            if not filepath:
                return
            
            # Generate report
            self.app.update_status("Generating HTML report...")
            
            output_path = create_fabrication_record(
                session=self.app.db_session,
                device_id=self.current_device.device_id,
                output_path=filepath
            )
            
            self.app.update_status("HTML report generated")
            
            # Ask if user wants to open it
            response = messagebox.askyesno(
                "Report Generated",
                f"Report saved to:\n{output_path}\n\nOpen in browser?",
                icon='question'
            )
            
            if response:
                import webbrowser
                webbrowser.open(f'file://{output_path}')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report:\n{str(e)}")
            self.app.update_status("Report generation failed")
    
    def update_device_info(self):
        """Update device information display"""
        if self.current_device:
            info_text = f"{self.current_device.device_name}\n"
            info_text += f"Sample: {self.current_device.sample.sample_name}\n"
            info_text += f"Status: {self.current_device.overall_status}"
            self.device_label.configure(text=info_text)
        else:
            self.device_label.configure(text="No device selected")
    
    def update_statistics(self):
        """Update statistics display"""
        if not self.current_device:
            self.stats_label.configure(text="Select a device")
            self.progress_bar.set(0)
            return
        
        try:
            stats = self.app.db_ops.get_step_completion_stats(self.current_device.device_id)
            
            stats_text = f"Total: {stats['total_steps']} steps\n"
            stats_text += f"✓ Complete: {stats['completed']}\n"
            stats_text += f"⚙ In Progress: {stats['in_progress']}\n"
            stats_text += f"⏳ Pending: {stats['pending']}"
            
            self.stats_label.configure(text=stats_text)
            
            # Update progress bar
            completion = stats['completion_percentage'] / 100.0
            self.progress_bar.set(completion)
            
            # Update device completion in database
            if stats['total_steps'] > 0:
                self.current_device.completion_percentage = stats['completion_percentage']
                if stats['completed'] == stats['total_steps']:
                    self.current_device.overall_status = 'complete'
                elif stats['completed'] > 0:
                    self.current_device.overall_status = 'in_progress'
                self.app.db_session.commit()
            
        except Exception as e:
            self.stats_label.configure(text=f"Error:\n{str(e)}")
    
    def refresh_steps(self):
        """Refresh the steps display"""
        if self.current_device:
            self.load_device_steps()
            self.update_statistics()
        self.app.update_status("Steps refreshed")
