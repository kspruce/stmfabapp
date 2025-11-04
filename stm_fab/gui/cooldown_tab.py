"""
cooldown_tab_ctk.py - CustomTkinter version of Cooldown Analysis Tab

A CustomTkinter-based tab for analyzing flash cooldown curves and creating
temperature-current calibrations for BMR processing.

This version integrates seamlessly with CustomTkinter applications.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit, brentq
from scipy.interpolate import interp1d
import json
from pathlib import Path
import traceback


class CooldownTab(ctk.CTkFrame):
    """
    CustomTkinter tab for cooldown analysis and temperature-current calibration
    """
    
    def __init__(self, parent, db_ops=None):
        super().__init__(parent)
        self.db_ops = db_ops
        
        # Data storage
        self.raw_data = None
        self.cooldown_data = None
        self.calibration = None
        self.file_path = None
        
        # Fitting parameters (R = a*exp(b*T) + c)
        self.a = None
        self.b = None
        self.c = None
        
        # Span selector for interactive region selection
        self.span_selector = None
        self.selected_range = None
        
        # Pyrometer correction (for setpoints only)
        self.pyro_enable_var = tk.BooleanVar(value=True)     # apply by default
        self.pyro_offset_var = tk.DoubleVar(value=-70.0)      # +70°C offset
        self.pyro_threshold_var = tk.DoubleVar(value=600.0)  # 600°C threshold

        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create scrollable frame
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            self.scroll_frame,
            text="Flash Cooldown Analysis & T-I Calibration",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Create sections
        self.create_file_section()
        self.create_plot_section()
        self.create_control_section()
        self.create_results_section()
        self.create_setpoints_section()
        self.create_export_section()
        
        # Status bar at bottom
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready. Load a flash file to begin.",
            font=ctk.CTkFont(size=12),
            text_color=("blue", "cyan"),
            anchor="w"
        )
        self.status_label.pack(side="bottom", fill="x", padx=10, pady=5)
    
    def create_file_section(self):
        """Create file loading section"""
        file_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
        file_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            file_frame,
            text="1. Load Flash File",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        button_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.file_label = ctk.CTkLabel(
            button_frame,
            text="No file loaded",
            text_color="gray",
            anchor="w"
        )
        self.file_label.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        browse_btn = ctk.CTkButton(
            button_frame,
            text="Browse...",
            command=self.load_flash_file,
            width=100,
            height=32
        )
        browse_btn.pack(side="right")
    
    def create_plot_section(self):
        """Create matplotlib plotting area"""
        plot_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
        plot_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        ctk.CTkLabel(
            plot_frame,
            text="Data Visualization",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('#2b2b2b')  # Dark theme
        
        self.axes = []
        for i in range(4):
            ax = self.fig.add_subplot(2, 2, i+1)
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            self.axes.append(ax)
        
        self.axes[0].set_title("Temperature vs Time")
        self.axes[0].set_xlabel("Time (s)")
        self.axes[0].set_ylabel("Temperature (°C)")
        self.axes[0].grid(True, alpha=0.3, color='gray')
        
        self.axes[1].set_title("Current vs Time")
        self.axes[1].set_xlabel("Time (s)")
        self.axes[1].set_ylabel("Current (A)")
        self.axes[1].grid(True, alpha=0.3, color='gray')
        
        self.axes[2].set_title("Temperature vs Current")
        self.axes[2].set_xlabel("Current (A)")
        self.axes[2].set_ylabel("Temperature (°C)")
        self.axes[2].grid(True, alpha=0.3, color='gray')
        
        self.axes[3].set_title("Fit Residuals")
        self.axes[3].set_xlabel("Current (A)")
        self.axes[3].set_ylabel("Residual (°C)")
        self.axes[3].grid(True, alpha=0.3, color='gray')
        
        self.fig.tight_layout(pad=2.0)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=(0, 15))
    
    def create_control_section(self):
        """Create analysis control section"""
        control_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
        control_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            control_frame,
            text="2. Analysis Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        # Channel selection for display
        channel_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        channel_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(
            channel_frame,
            text="Display Channel:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(0, 10))
        
        self.channel_var = tk.StringVar(value="Temperature")
        self.channel_menu = ctk.CTkOptionMenu(
            channel_frame,
            variable=self.channel_var,
            values=["Temperature", "Current", "Resistance"],
            command=self.on_channel_change,
            width=150
        )
        self.channel_menu.pack(side="left")
        
        # Instructions for cursor selection
        info_label = ctk.CTkLabel(
            channel_frame,
            text="  ℹ️ Drag on plot to select fitting region",
            font=ctk.CTkFont(size=11),
            text_color="cyan"
        )
        info_label.pack(side="left", padx=10)
        
        # Current range
        range_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        range_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(
            range_frame,
            text="Calibration Current Range:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(0, 10))
        
        ctk.CTkLabel(range_frame, text="Min:").pack(side="left", padx=(0, 5))
        self.min_current_var = tk.DoubleVar(value=0.03)
        min_entry = ctk.CTkEntry(range_frame, textvariable=self.min_current_var, width=80)
        min_entry.pack(side="left", padx=(0, 5))
        ctk.CTkLabel(range_frame, text="A").pack(side="left", padx=(0, 15))
        
        ctk.CTkLabel(range_frame, text="Max:").pack(side="left", padx=(0, 5))
        self.max_current_var = tk.DoubleVar(value=0.30)
        max_entry = ctk.CTkEntry(range_frame, textvariable=self.max_current_var, width=80)
        max_entry.pack(side="left", padx=(0, 5))
        ctk.CTkLabel(range_frame, text="A").pack(side="left")
        
        # Model selection - Updated to show R(T) model
        model_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        model_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(
            model_frame,
            text="Fit Model:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(0, 10))
        
        self.model_var = tk.StringVar(value="R(T) Exponential")
        model_menu = ctk.CTkOptionMenu(
            model_frame,
            variable=self.model_var,
            values=[
                "R(T) Exponential",  # R = a*exp(b*T) + c
                "Polynomial (degree 3)",
                "Polynomial (degree 4)",
                "Polynomial (degree 5)"
            ],
            width=200
        )
        model_menu.pack(side="left")
        
        # Model formula display
        self.formula_label = ctk.CTkLabel(
            model_frame,
            text="R = a·exp(b·T) + c",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="yellow"
        )
        self.formula_label.pack(side="left", padx=10)
        
        # Analyze button
        self.analyze_btn = ctk.CTkButton(
            control_frame,
            text="3. Extract & Fit Calibration",
            command=self.analyze_cooldown,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green",
            hover_color="darkgreen",
            state="disabled"
        )
        self.analyze_btn.pack(fill="x", padx=15, pady=(0, 15))
    
    def create_results_section(self):
        """Create results display section"""
        results_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
        results_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            results_frame,
            text="4. Calibration Results",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        self.results_text = ctk.CTkTextbox(
            results_frame,
            height=100,
            font=ctk.CTkFont(family="Courier", size=11)
        )
        self.results_text.pack(fill="x", padx=15, pady=(0, 10))
        
        # Add data table for context
        ctk.CTkLabel(
            results_frame,
            text="Data Context (around target):",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(10, 5))
        
        # Create frame for treeview with scrollbar
        table_container = ctk.CTkFrame(results_frame, fg_color="transparent")
        table_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Style for the treeview
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Treeview",
                       background="#2b2b2b",
                       foreground="white",
                       fieldbackground="#2b2b2b",
                       borderwidth=0)
        style.configure("Custom.Treeview.Heading",
                       background="#1f1f1f",
                       foreground="white",
                       borderwidth=1)
        style.map('Custom.Treeview',
                 background=[('selected', '#4a4a4a')])
        
        # Create treeview
        columns = ("Time", "Temperature", "Resistance", "Current")
        self.results_table = ttk.Treeview(
            table_container,
            columns=columns,
            show='headings',
            height=8,
            style="Custom.Treeview"
        )
        
        # Configure columns
        self.results_table.heading("Time", text="Time (s)")
        self.results_table.heading("Temperature", text="T (°C)")
        self.results_table.heading("Resistance", text="R (Ω)")
        self.results_table.heading("Current", text="I (A)")
        
        self.results_table.column("Time", width=100, anchor="center")
        self.results_table.column("Temperature", width=100, anchor="center")
        self.results_table.column("Resistance", width=100, anchor="center")
        self.results_table.column("Current", width=100, anchor="center")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=self.results_table.yview)
        self.results_table.configure(yscrollcommand=scrollbar.set)
        
        self.results_table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_setpoints_section(self):
        """Create setpoints calculation section"""
        setpoints_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
        setpoints_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            setpoints_frame,
            text="5. Process Setpoints",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
 
        # Pyrometer offset controls (setpoints only)
        pyro_frame = ctk.CTkFrame(setpoints_frame, fg_color="transparent")
        pyro_frame.pack(fill="x", padx=15, pady=(0, 8))
        
        pyro_chk = ctk.CTkCheckBox(
            pyro_frame,
            text="Apply pyrometer offset for setpoints (true T < threshold)",
            variable=self.pyro_enable_var
        )
        pyro_chk.pack(side="left", padx=(0, 10))
        
        ctk.CTkLabel(pyro_frame, text="Offset (°C):").pack(side="left", padx=(0, 5))
        ctk.CTkEntry(pyro_frame, textvariable=self.pyro_offset_var, width=80).pack(side="left", padx=(0, 15))
        
        ctk.CTkLabel(pyro_frame, text="Threshold (°C):").pack(side="left", padx=(0, 5))
        ctk.CTkEntry(pyro_frame, textvariable=self.pyro_threshold_var, width=80).pack(side="left")
                
        
        # Create table-like structure with labels
        table_frame = ctk.CTkFrame(setpoints_frame, fg_color="transparent")
        table_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        # Headers
        header_frame = ctk.CTkFrame(table_frame, fg_color=("gray75", "gray25"))
        header_frame.pack(fill="x", pady=(0, 2))
        
        ctk.CTkLabel(
            header_frame,
            text="Process",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=150
        ).pack(side="left", padx=5, pady=5)
        
        ctk.CTkLabel(
            header_frame,
            text="Target Temp (°C)",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=120
        ).pack(side="left", padx=5, pady=5)
        
        ctk.CTkLabel(
            header_frame,
            text="Required Current (A)",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=150
        ).pack(side="left", padx=5, pady=5)
        
        # Store entries for later
        self.setpoint_entries = []
        
        # Standard processes
        processes = [
            ("H-Termination", "340"),
            ("Incorporation", "350"),
            ("RTA", "470"),
            ("LTE", "250"),
            ("Custom", "")
        ]
        
        for process, temp in processes:
            row_frame = ctk.CTkFrame(table_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=1)
            
            # Process name
            ctk.CTkLabel(
                row_frame,
                text=process,
                width=150,
                anchor="w"
            ).pack(side="left", padx=5, pady=3)
            
            # Temperature entry
            temp_var = tk.StringVar(value=temp)
            temp_entry = ctk.CTkEntry(
                row_frame,
                textvariable=temp_var,
                width=120
            )
            temp_entry.pack(side="left", padx=5, pady=3)
            
            # Current result
            current_var = tk.StringVar(value="")
            current_label = ctk.CTkLabel(
                row_frame,
                textvariable=current_var,
                width=150,
                anchor="w"
            )
            current_label.pack(side="left", padx=5, pady=3)
            
            self.setpoint_entries.append((process, temp_var, current_var))
        
        # Calculate button
        calc_btn = ctk.CTkButton(
            setpoints_frame,
            text="Calculate Setpoints",
            command=self.calculate_setpoints,
            height=32
        )
        calc_btn.pack(padx=15, pady=(0, 15))
    
    def create_export_section(self):
        """Create export buttons section"""
        export_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10, fg_color="transparent")
        export_frame.pack(fill="x", pady=(0, 10))
        
        export_cal_btn = ctk.CTkButton(
            export_frame,
            text="Export Calibration (JSON)",
            command=self.export_calibration,
            height=32,
            width=200
        )
        export_cal_btn.pack(side="left", padx=5)
        
        export_fig_btn = ctk.CTkButton(
            export_frame,
            text="Save Figures (PNG)",
            command=self.save_figures,
            height=32,
            width=200
        )
        export_fig_btn.pack(side="left", padx=5)
    
    def load_flash_file(self):
        """Load a LabVIEW flash file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Flash LabVIEW File",
                filetypes=[("LabVIEW Files", "*.txt *.lvm"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
            
            self.update_status("Loading file...", "orange")
            
            # Parse the file
            self.raw_data = self.parse_labview_file(file_path)
            self.file_path = file_path
            
            # Update UI
            self.file_label.configure(
                text=f"Loaded: {Path(file_path).name}",
                text_color=("green", "lightgreen")
            )
            self.analyze_btn.configure(state="normal")
            
            # Plot raw data
            self.plot_raw_data()
            
            self.update_status(
                f"File loaded successfully. {len(self.raw_data)} data points.",
                "green"
            )
            
        except Exception as e:
            messagebox.showerror(
                "Error Loading File",
                f"Failed to load flash file:\n\n{str(e)}\n\n{traceback.format_exc()}"
            )
            self.update_status("Error loading file.", "red")
    
    def parse_labview_file(self, file_path):
        """Parse LabVIEW flash file and calculate resistance"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find data start
        data_start_idx = None
        for i, line in enumerate(lines):
            if 'X_Value' in line:
                data_start_idx = i + 1
                break
        
        if data_start_idx is None:
            raise ValueError("Could not find X_Value header in file")
        
        # Parse data
        parsed_data = []
        for line in lines[data_start_idx:]:
            if line.strip() == '' or line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            
            try:
                time = float(parts[0])
                voltage = float(parts[3]) if len(parts) > 3 else None  # Try to get voltage
                current = float(parts[5])
                temp = float(parts[7])
                
                # Calculate resistance: R = V / I
                if voltage is not None and current != 0:
                    resistance = voltage / current
                else:
                    # If no voltage, estimate from typical values or use placeholder
                    resistance = None
                
                parsed_data.append({
                    'Time': time, 
                    'Current': current, 
                    'Temperature': temp,
                    'Voltage': voltage,
                    'Resistance': resistance
                })
            except (ValueError, IndexError):
                continue
        
        if not parsed_data:
            raise ValueError("No valid data found in file")
        
        df = pd.DataFrame(parsed_data)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # If we don't have resistance data, we can't do R(T) fitting
        # Print warning but don't fail
        if df['Resistance'].isna().all():
            print("WARNING: No resistance data available. Will use Temperature-Current relationship instead.")
            # Drop the resistance column if all NaN
            df = df.drop(columns=['Resistance', 'Voltage'])
        else:
            df = df.dropna()
        
        return df
    
    def plot_raw_data(self):
        """Plot raw data with interactive span selector"""
        if self.raw_data is None:
            return
        
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        # Determine which channel to plot
        selected_channel = self.channel_var.get()
        
        # Plot selected channel vs time on first plot
        if selected_channel == "Temperature":
            y_data = self.raw_data['Temperature']
            ylabel = "Temperature (°C)"
            color = 'cyan'
        elif selected_channel == "Current":
            y_data = self.raw_data['Current']
            ylabel = "Current (A)"
            color = 'red'
        elif selected_channel == "Resistance" and 'Resistance' in self.raw_data.columns:
            y_data = self.raw_data['Resistance']
            ylabel = "Resistance (Ω)"
            color = 'green'
        else:
            # Default to temperature
            y_data = self.raw_data['Temperature']
            ylabel = "Temperature (°C)"
            color = 'cyan'
        
        # Plot main channel
        self.axes[0].plot(
            self.raw_data['Time'],
            y_data,
            color=color, linewidth=1.0, alpha=0.8
        )
        self.axes[0].set_title(f"{selected_channel} vs Time (Drag to select region)", color='white')
        self.axes[0].set_xlabel("Time (s)", color='white')
        self.axes[0].set_ylabel(ylabel, color='white')
        self.axes[0].grid(True, alpha=0.3, color='gray')
        
        # Add span selector for interactive region selection
        if self.span_selector is not None:
            self.span_selector.disconnect_events()
        
        self.span_selector = SpanSelector(
            self.axes[0],
            self.on_select_region,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='yellow'),
            interactive=True,
            drag_from_anywhere=True
        )
        
        # Current vs time on second plot
        self.axes[1].plot(
            self.raw_data['Time'],
            self.raw_data['Current'],
            'r-', linewidth=1.0, alpha=0.7
        )
        self.axes[1].set_title("Current vs Time", color='white')
        self.axes[1].set_xlabel("Time (s)", color='white')
        self.axes[1].set_ylabel("Current (A)", color='white')
        self.axes[1].grid(True, alpha=0.3, color='gray')
        
        # Temperature vs Current scatter on third plot
        self.axes[2].scatter(
            self.raw_data['Current'],
            self.raw_data['Temperature'],
            s=1, alpha=0.3, c='cyan'
        )
        self.axes[2].set_title("Temperature vs Current (Raw)", color='white')
        self.axes[2].set_xlabel("Current (A)", color='white')
        self.axes[2].set_ylabel("Temperature (°C)", color='white')
        self.axes[2].grid(True, alpha=0.3, color='gray')
        
        self.canvas.draw()
    
    def on_select_region(self, xmin, xmax):
        """Callback when user selects a region with span selector"""
        self.selected_range = (xmin, xmax)
        self.update_status(f"Region selected: {xmin:.2f}s to {xmax:.2f}s. Click 'Extract & Fit' to analyze.", "cyan")
        
        # Highlight selected region on plot
        for ax in [self.axes[0], self.axes[1]]:
            # Remove previous highlights
            for patch in [p for p in ax.patches if hasattr(p, 'is_highlight')]:
                patch.remove()
            
            # Add new highlight
            ylim = ax.get_ylim()
            rect = ax.axvspan(xmin, xmax, alpha=0.2, color='yellow')
            rect.is_highlight = True
        
        self.canvas.draw()
    
    def on_channel_change(self, selected_channel):
        """Update plot when channel selection changes"""
        self.plot_raw_data()
    
    def analyze_cooldown(self):
        """Analyze cooldown and fit calibration"""
        try:
            self.update_status("Analyzing cooldown...", "orange")
            
            # Extract cooldown
            self.extract_cooldown_curve()
            
            # Fit calibration
            self.fit_calibration()
            
            # Plot
            self.plot_calibration()
            
            # Display results
            self.display_results()
            
            self.update_status("Analysis complete!", "green")
            
        except Exception as e:
            messagebox.showerror(
                "Analysis Error",
                f"Failed to analyze cooldown:\n\n{str(e)}\n\n{traceback.format_exc()}"
            )
            self.update_status("Analysis failed.", "red")
    
    def extract_cooldown_curve(self):
        """Extract cooldown from peak onwards or from selected region"""
        # Use selected region if available
        if self.selected_range is not None:
            xmin, xmax = self.selected_range
            mask = (self.raw_data['Time'] >= xmin) & (self.raw_data['Time'] <= xmax)
            selected_data = self.raw_data[mask].copy()
            
            if len(selected_data) < 10:
                raise ValueError(f"Insufficient data points ({len(selected_data)}) in selected region")
            
            # Use selected region data
            cooldown_df = selected_data.copy()
        else:
            # Default behavior: extract from peak onwards
            peak_idx = self.raw_data['Temperature'].idxmax()
            cooldown_df = self.raw_data.loc[peak_idx:].copy()
        
        # Sort and remove duplicates
        cooldown_df = cooldown_df.sort_values('Current', ascending=False)
        cooldown_df = cooldown_df.drop_duplicates(subset='Current', keep='first')
        
        # Apply current range filter
        min_i = self.min_current_var.get()
        max_i = self.max_current_var.get()
        
        cooldown_df = cooldown_df[
            (cooldown_df['Current'] >= min_i) &
            (cooldown_df['Current'] <= max_i)
        ]
        
        if len(cooldown_df) < 10:
            raise ValueError(f"Insufficient data points ({len(cooldown_df)}) in range {min_i}-{max_i} A")
        
        self.cooldown_data = cooldown_df
    
    def fit_calibration(self):
        """Fit calibration model - supports both R(T) and T(I) models"""
        current = self.cooldown_data['Current'].values
        temp_celsius = self.cooldown_data['Temperature'].values
        
        # Check if we have resistance data for R(T) fitting
        has_resistance = 'Resistance' in self.cooldown_data.columns
        if has_resistance:
            resistance = self.cooldown_data['Resistance'].values
        
        model_text = self.model_var.get()
        
        if "R(T) Exponential" in model_text and has_resistance:
            # Correct R(T) model: R = a*exp(b*T) + c
            # Convert temperature to Kelvin as per documentation
            temp_kelvin = temp_celsius + 273.15
            
            def r_vs_t_model(T_K, a, b, c):
                """R = a*exp(b*T) + c where T is in Kelvin"""
                return a * np.exp(b * T_K) + c
            
            try:
                # Initial guess for parameters
                p0 = [
                    resistance.max() - resistance.min(),  # a: amplitude
                    -0.01,  # b: negative for decreasing R with T
                    resistance.min()  # c: offset
                ]
                popt, pcov = curve_fit(r_vs_t_model, temp_kelvin, resistance, p0=p0, maxfev=10000)
                self.a, self.b, self.c = popt
                
                resistance_fit = r_vs_t_model(temp_kelvin, *popt)
                
                self.calibration = {
                    'model': 'R(T)_exponential',
                    'coefficients': {'a': self.a, 'b': self.b, 'c': self.c},
                    'coefficients_list': popt.tolist(),
                    'function': lambda T_C: r_vs_t_model(T_C + 273.15, *popt),
                    'current': current,
                    'temperature': temp_celsius,
                    'resistance': resistance,
                    'resistance_fit': resistance_fit,
                    'uses_kelvin': True
                }
                
                # Calculate residuals
                residuals = resistance - resistance_fit
                
            except Exception as e:
                messagebox.showerror("Fit Error", 
                    f"R(T) exponential fit failed: {str(e)}\n\nTry polynomial model instead.")
                return
                
        elif "Polynomial" in model_text:
            # Polynomial fit for T(I) relationship
            degree = int(model_text.split("degree ")[1].split(")")[0])
            coeffs = np.polyfit(current, temp_celsius, degree)
            poly_func = np.poly1d(coeffs)
            temp_fit = poly_func(current)
            
            self.calibration = {
                'model': f'polynomial_deg{degree}',
                'coefficients': coeffs.tolist(),
                'function': poly_func,
                'current': current,
                'temperature': temp_celsius,
                'temperature_fit': temp_fit,
                'uses_kelvin': False
            }
            
            if has_resistance:
                self.calibration['resistance'] = resistance
            
            residuals = temp_celsius - temp_fit
            
        elif "R(T) Exponential" in model_text and not has_resistance:
            # User selected R(T) but no resistance data available
            messagebox.showwarning(
                "No Resistance Data",
                "R(T) model requires resistance data, but none found in file.\n\n"
                "Falling back to Temperature-Current polynomial fitting."
            )
            # Fall back to polynomial
            degree = 3
            coeffs = np.polyfit(current, temp_celsius, degree)
            poly_func = np.poly1d(coeffs)
            temp_fit = poly_func(current)
            
            self.calibration = {
                'model': f'polynomial_deg{degree}_fallback',
                'coefficients': coeffs.tolist(),
                'function': poly_func,
                'current': current,
                'temperature': temp_celsius,
                'temperature_fit': temp_fit,
                'uses_kelvin': False
            }
            
            residuals = temp_celsius - temp_fit
        else:
            messagebox.showerror("Error", f"Unknown model type: {model_text}")
            return
        
        # Calculate fit quality
        ss_res = np.sum(residuals**2)
        if 'R(T)' in self.calibration['model']:
            ss_tot = np.sum((resistance - np.mean(resistance))**2)
        else:
            ss_tot = np.sum((temp_celsius - np.mean(temp_celsius))**2)
        
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals**2))
        
        self.calibration['r_squared'] = r_squared
        self.calibration['rmse'] = rmse
        self.calibration['residuals'] = residuals
    
    def plot_calibration(self):
        """Plot calibration fit"""
        # Clear third and fourth plots
        self.axes[2].clear()
        self.axes[3].clear()
        
        # Determine what to plot based on model
        is_r_t_model = 'R(T)' in self.calibration['model']
        
        if is_r_t_model and 'resistance' in self.calibration:
            # Plot R vs T for R(T) model
            x_data = self.calibration['temperature']
            y_data = self.calibration['resistance']
            y_fit = self.calibration['resistance_fit']
            xlabel = "Temperature (°C)"
            ylabel = "Resistance (Ω)"
            title = "R(T) Calibration Fit"
            
            # Create smooth curve for visualization
            temp_smooth = np.linspace(x_data.min(), x_data.max(), 200)
            resistance_smooth = self.calibration['function'](temp_smooth)
            
        else:
            # Plot T vs I for polynomial model
            x_data = self.calibration['current']
            y_data = self.calibration['temperature']
            y_fit = self.calibration['temperature_fit']
            xlabel = "Current (A)"
            ylabel = "Temperature (°C)"
            title = "T(I) Calibration Fit"
            
            # Create smooth curve
            current_smooth = np.linspace(x_data.min(), x_data.max(), 200)
            temp_smooth = self.calibration['function'](current_smooth)
            resistance_smooth = temp_smooth  # For consistent variable naming
        
        # Plot calibration on third plot
        self.axes[2].scatter(
            x_data, y_data,
            s=20, alpha=0.6, c='cyan', label='Data', zorder=2
        )
        
        if is_r_t_model:
            self.axes[2].plot(
                temp_smooth, resistance_smooth,
                'r-', linewidth=2,
                label=f"R = a·exp(b·T) + c\nR²={self.calibration['r_squared']:.4f}",
                zorder=3
            )
        else:
            self.axes[2].plot(
                current_smooth, resistance_smooth,
                'r-', linewidth=2,
                label=f"Fit (R²={self.calibration['r_squared']:.4f})",
                zorder=3
            )
        
        self.axes[2].set_title(title, color='white')
        self.axes[2].set_xlabel(xlabel, color='white')
        self.axes[2].set_ylabel(ylabel, color='white')
        self.axes[2].legend(loc='best')
        self.axes[2].grid(True, alpha=0.3, color='gray')
        
        # Plot residuals on fourth plot
        self.axes[3].scatter(
            x_data,
            self.calibration['residuals'],
            s=20, alpha=0.6, c='orange'
        )
        self.axes[3].axhline(y=0, color='white', linestyle='--', linewidth=1)
        
        # Determine residual units
        if is_r_t_model:
            residual_unit = "Ω"
        else:
            residual_unit = "°C"
        
        self.axes[3].set_title(
            f"Residuals (RMSE={self.calibration['rmse']:.3f} {residual_unit})",
            color='white'
        )
        self.axes[3].set_xlabel(xlabel, color='white')
        self.axes[3].set_ylabel(f"Residual ({residual_unit})", color='white')
        self.axes[3].grid(True, alpha=0.3, color='gray')
        
        self.canvas.draw()
    
    def display_results(self):
        """Display calibration results and populate data table"""
        self.results_text.delete("1.0", "end")
    
        lines = []
        lines.append("=" * 60)
        lines.append("COOLDOWN CALIBRATION RESULTS")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Model: {self.calibration['model']}")
    
        # Display model-specific information
        if 'R(T)' in self.calibration['model']:
            lines.append("")
            lines.append("Model: R = a·exp(b·T) + c  (T in Kelvin)")
            lines.append("Parameters:")
            lines.append(f"  a = {self.a:.6e}")
            lines.append(f"  b = {self.b:.6e} K⁻¹")
            lines.append(f"  c = {self.c:.6f} Ω")
    
        lines.append("")
        lines.append("Fit Quality:")
        lines.append(f"  R² = {self.calibration['r_squared']:.6f}")
    
        if 'R(T)' in self.calibration['model']:
            lines.append(f"  RMSE = {self.calibration['rmse']:.4f} Ω")
        else:
            lines.append(f"  RMSE = {self.calibration['rmse']:.2f} °C")
    
        lines.append("")
        lines.append("Calibration Range:")
        lines.append(
            f"  Current: {self.calibration['current'].min():.4f} - "
            f"{self.calibration['current'].max():.4f} A"
        )
        lines.append(
            f"  Temperature: {self.calibration['temperature'].min():.1f} - "
            f"{self.calibration['temperature'].max():.1f} °C"
        )
    
        if 'resistance' in self.calibration:
            lines.append(
                f"  Resistance: {self.calibration['resistance'].min():.2f} - "
                f"{self.calibration['resistance'].max():.2f} Ω"
            )
    
        lines.append(f"  Data Points: {len(self.calibration['current'])}")
        lines.append("")
        lines.append("=" * 60)
    
        self.results_text.insert("1.0", "\n".join(lines))
    
        # Populate data table
        self.populate_data_table()

    
    def populate_data_table(self):
        """Populate the data table with calibration data context"""
        # Clear existing items
        for item in self.results_table.get_children():
            self.results_table.delete(item)
        
        # Get data
        df = self.cooldown_data.copy()
        
        # Add fitted values for display
        if 'R(T)' in self.calibration['model'] and 'resistance' in self.calibration:
            df['R_fit'] = self.calibration['resistance_fit']
        
        # Sort by time for display
        df = df.sort_values('Time')
        
        # Determine columns to display
        has_resistance = 'Resistance' in df.columns
        
        # Insert data rows
        for idx, row in df.iterrows():
            time_str = f"{row['Time']:.2f}"
            temp_str = f"{row['Temperature']:.2f}"
            current_str = f"{row['Current']:.4f}"
            
            if has_resistance and not pd.isna(row['Resistance']):
                resistance_str = f"{row['Resistance']:.4f}"
            else:
                resistance_str = "N/A"
            
            self.results_table.insert('', 'end', values=(
                time_str, temp_str, resistance_str, current_str
            ))
    
    def calculate_setpoints(self):
        """Calculate required currents for target temperatures"""
        if self.calibration is None:
            messagebox.showwarning(
                "No Calibration",
                "Please run cooldown analysis first."
            )
            return
        
        try:
            for process, temp_var, current_var in self.setpoint_entries:
                temp_str = temp_var.get().strip()
                if not temp_str:
                    continue
                
                target_temp = float(temp_str)
                current = self.get_current_for_temperature(target_temp)
                current_var.set(f"{current:.4f}")
            
            self.update_status("Setpoints calculated!", "green")
            
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Failed:\n{str(e)}")
    
  
    def get_current_for_temperature(self, target_temp_true):
        """
        Get current for a true target temperature (setpoints).
        Applies pyrometer offset ONLY here (does not alter fitting/plots).
    
        For true temperatures below threshold, the pyrometer reads +offset hotter.
        Therefore the effective measured target = true + offset (below threshold),
        else effective measured target = true.
        """
    
        # Compute effective measured target temperature
        if self.pyro_enable_var.get() and float(target_temp_true) < float(self.pyro_threshold_var.get()):
            effective_measured_target = float(target_temp_true) + float(self.pyro_offset_var.get())
        else:
            effective_measured_target = float(target_temp_true)
    
        # If we have a T(I) polynomial, use root-finding against measured temp
        if 'polynomial' in str(self.calibration.get('model', '')).lower():
            func = lambda I: float(self.calibration['function'](I)) - effective_measured_target
    
            min_i = float(self.calibration['current'].min())
            max_i = float(self.calibration['current'].max())
    
            try:
                return float(brentq(func, min_i, max_i))
            except ValueError:
                # Fall back to measured T -> I interpolation from data
                temp_sorted_idx = np.argsort(self.calibration['temperature'])
                temp_sorted = self.calibration['temperature'][temp_sorted_idx]
                current_sorted = self.calibration['current'][temp_sorted_idx]
                inv_interp = interp1d(
                    temp_sorted, current_sorted,
                    kind='linear', fill_value='extrapolate', bounds_error=False
                )
                return float(inv_interp(effective_measured_target))
    
        # R(T) model case: invert measured Temperature -> Current by interpolation only
        temp_sorted_idx = np.argsort(self.calibration['temperature'])
        temp_sorted = self.calibration['temperature'][temp_sorted_idx]
        current_sorted = self.calibration['current'][temp_sorted_idx]
        inv_interp = interp1d(
            temp_sorted, current_sorted,
            kind='linear', fill_value='extrapolate', bounds_error=False
        )
        return float(inv_interp(effective_measured_target))


    def export_calibration(self):
        """Export calibration to JSON"""
        if self.calibration is None:
            messagebox.showwarning("No Calibration", "Please run analysis first.")
            return
        
        try:
            save_path = filedialog.asksaveasfilename(
                title="Save Calibration",
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")]
            )
            
            if not save_path:
                return
            
            export_data = {
                'source_file': self.file_path,
                'model': self.calibration['model'],
                'coefficients': self.calibration['coefficients'],
                'r_squared': self.calibration['r_squared'],
                'rmse': self.calibration['rmse'],
                'calibration_range': {
                    'current_min': float(self.calibration['current'].min()),
                    'current_max': float(self.calibration['current'].max()),
                    'temp_min': float(self.calibration['temperature'].min()),
                    'temp_max': float(self.calibration['temperature'].max())
                }
            }
            
            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Success", f"Calibration saved to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed:\n{str(e)}")
    
    def save_figures(self):
        """Save figures to file"""
        try:
            save_path = filedialog.asksaveasfilename(
                title="Save Figures",
                defaultextension=".png",
                filetypes=[("PNG Images", "*.png"), ("PDF Files", "*.pdf")]
            )
            
            if not save_path:
                return
            
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#2b2b2b')
            messagebox.showinfo("Success", f"Figures saved to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed:\n{str(e)}")
    
    def update_status(self, message, color="blue"):
        """Update status label"""
        color_map = {
            "blue": ("blue", "cyan"),
            "green": ("green", "lightgreen"),
            "orange": ("orange", "yellow"),
            "red": ("red", "pink")
        }
        self.status_label.configure(text=message, text_color=color_map.get(color, ("blue", "cyan")))