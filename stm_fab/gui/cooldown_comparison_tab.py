# gui/cooldown_comparison_tab.py
"""
Cooldown Comparison Tab - Compare cooldown curves from multiple flash files
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

from stm_fab.analysis.batch_rampdown import (
    process_folder, 
    process_file as bp_process_file, 
    figure_per_file, 
    generate_comparison_figures, 
    export_summary_excel
)


from stm_fab.scripts.convert_autoheater_to_flash_format import convert_file as convert_autoflash


class CooldownComparisonTab:
    """Tab for comparing cooldown curves from multiple flash files"""
    
    def __init__(self, parent_gui, tabview):
        self.gui = parent_gui
        self.tab = tabview.add("🔥 Cooldown Comparison")
        self._batch_results = None
        self._plot_canvases = []
        self._build_ui()
    
    def _build_ui(self):
        """Build the UI for cooldown comparison"""
        main = ctk.CTkFrame(self.tab)
        main.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header = ctk.CTkLabel(
            main, 
            text="Cooldown Curve Comparison", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.pack(pady=10)
        
        description = ctk.CTkLabel(
            main,
            text="Compare cooldown curves from multiple flash files to analyze consistency and variations",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        description.pack(pady=(0, 20))
        
        # Control panel
        control_panel = ctk.CTkFrame(main, corner_radius=15)
        control_panel.pack(fill="x", pady=(0, 10))
        
        # Folder selection
        folder_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        folder_frame.pack(fill="x", padx=15, pady=15)
        
        ctk.CTkLabel(
            folder_frame,
            text="Flash Files Folder:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 5))
        
        folder_input_frame = ctk.CTkFrame(folder_frame, fg_color="transparent")
        folder_input_frame.pack(fill="x")
        
        self.folder_var = ctk.StringVar()
        self.folder_entry = ctk.CTkEntry(
            folder_input_frame, 
            textvariable=self.folder_var,
            placeholder_text="Select folder containing flash .txt files",
            height=35
        )
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ctk.CTkButton(
            folder_input_frame,
            text="📁 Browse",
            command=self._browse_folder,
            width=100,
            height=35
        ).pack(side="left", padx=2)
        
        # Action buttons
        action_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        action_frame.pack(fill="x", padx=15, pady=(5, 15))
        
        ctk.CTkButton(
            action_frame,
            text="▶️ Analyze Files",
            command=self._analyze_files,
            width=150,
            height=40,
            fg_color="green",
            hover_color="darkgreen",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            action_frame,
            text="📊 Show Comparisons",
            command=self._show_comparisons,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            action_frame,
            text="📈 Individual Files",
            command=self._show_individual,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            action_frame,
            text="💾 Export Excel",
            command=self._export_excel,
            width=120,
            height=40,
            fg_color="orange",
            hover_color="darkorange"
        ).pack(side="left", padx=5)
        
        # Content area with two panels
        content = ctk.CTkFrame(main)
        content.pack(fill="both", expand=True, pady=10)
        
        # Left panel: File list
        left_panel = ctk.CTkFrame(content, corner_radius=15)
        left_panel.pack(side="left", fill="y", padx=(0, 10), pady=5)
        
        ctk.CTkLabel(
            left_panel,
            text="Analyzed Files",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        # File list scrollable frame
        self.file_list_frame = ctk.CTkScrollableFrame(
            left_panel,
            width=350,
            height=600,
            corner_radius=10
        )
        self.file_list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 15))
        
        # Right panel: Plots
        right_panel = ctk.CTkFrame(content, corner_radius=15)
        right_panel.pack(side="left", fill="both", expand=True, pady=5)
        
        ctk.CTkLabel(
            right_panel,
            text="Visualization",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        # Plot container
        self.plot_container = ctk.CTkScrollableFrame(
            right_panel,
            corner_radius=10
        )
        self.plot_container.pack(fill="both", expand=True, padx=10, pady=(0, 15))
    
    def _browse_folder(self):
        """Browse for folder containing flash files"""
        folder = filedialog.askdirectory(title="Select Folder with Flash Files")
        if folder:
            self.folder_var.set(folder)
    
    # ==================== MODIFIED _analyze_files METHOD ====================
    
    def _analyze_files(self):
        """Analyze all flash files in the selected folder (with auto-conversion)"""
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No Folder", "Please select a folder first")
            return
        
        folder_path = Path(folder)
        
        # ⭐ NEW: Check for autoflash files and convert them BEFORE analysis
        all_txt_files = list(folder_path.glob("*.txt"))
        autoflash_files = [f for f in all_txt_files if 'autoflash' in f.name.lower()]
        
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
        
        try:
            # Show progress
            self.gui.update_status("Analyzing flash files...")
            
            # ⭐ Process all files in folder (now includes converted files)
            self._batch_results = process_folder(folder)
            
            # Clear and populate file list
            for widget in self.file_list_frame.winfo_children():
                widget.destroy()
            
            successful = 0
            failed = 0
            

            for result in self._batch_results:
                if result.get('success'):
                    successful += 1
                    # Only show files that contain "flash" in the filename
                    if 'flash' in result.get('filename', '').lower():
                        # Create file card
                        file_card = self._create_file_card(result)
                        file_card.pack(fill="x", pady=5, padx=5)
                else:
                    failed += 1
                    # Don't show error cards for non-flash files
                    if 'flash' in result.get('filename', '').lower():
                        # Create error card
                        error_card = self._create_error_card(result)
                        error_card.pack(fill="x", pady=5, padx=5)
            
            # Show summary
            summary = f"Analysis Complete\n\n✓ Successful: {successful}\n✗ Failed: {failed}"
            messagebox.showinfo("Analysis Complete", summary)
            
            self.gui.update_status(f"Analyzed {len(self._batch_results)} files: {successful} successful, {failed} failed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze files:\n{str(e)}")
            self.gui.update_status("Error analyzing files")
    
    # ==================== NEW HELPER METHOD ====================
    
    def _get_flash_results(self):
        """Get only successful results with 'flash' in filename"""
        if not self._batch_results:
            return []
        
        return [
            result for result in self._batch_results 
            if result.get('success') and 'flash' in result.get('filename', '').lower()
        ]    
    
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
    
    # ==================== END OF MODIFIED SECTION ====================
    
    def _create_file_card(self, result: Dict[str, Any]) -> ctk.CTkFrame:
        """Create a card widget for a successfully analyzed file"""
        card = ctk.CTkFrame(self.file_list_frame, corner_radius=10)
        
        # Header with filename
        header_frame = ctk.CTkFrame(card, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            header_frame,
            text="✓",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="green"
        ).pack(side="left", padx=(0, 5))
        
        ctk.CTkLabel(
            header_frame,
            text=result['sample_name'],
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        ).pack(side="left", fill="x", expand=True)
        
        # Date if available
        if result.get('date'):
            ctk.CTkLabel(
                card,
                text=f"📅 {result['date'].strftime('%d-%m-%Y')}",
                font=ctk.CTkFont(size=10),
                text_color="gray",
                anchor="w"
            ).pack(anchor="w", padx=10)
        
        # Key metrics
        metrics_text = (
            f"Peak: {result['temp_at_max']:.0f}°C @ {result['power_at_max']:.1f}W\n"
            f"2W: {result['temp_at_2w']:.0f}°C" if result['temp_at_2w'] else "2W: N/A"
        )
        
        ctk.CTkLabel(
            card,
            text=metrics_text,
            font=ctk.CTkFont(size=10),
            anchor="w",
            justify="left"
        ).pack(anchor="w", padx=10, pady=5)
        
        # Button to show individual plot
        ctk.CTkButton(
            card,
            text="View Details",
            command=lambda r=result: self._show_file_plot(r),
            width=100,
            height=28
        ).pack(padx=10, pady=(0, 10))
        
        return card
    
    def _create_error_card(self, result: Dict[str, Any]) -> ctk.CTkFrame:
        """Create a card widget for a failed file"""
        card = ctk.CTkFrame(self.file_list_frame, corner_radius=10, fg_color=("gray85", "gray20"))
        
        # Header with filename
        header_frame = ctk.CTkFrame(card, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            header_frame,
            text="✗",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="red"
        ).pack(side="left", padx=(0, 5))
        
        ctk.CTkLabel(
            header_frame,
            text=result['filename'],
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        ).pack(side="left", fill="x", expand=True)
        
        # Error message
        error_text = result.get('error', 'Unknown error')
        if len(error_text) > 60:
            error_text = error_text[:57] + "..."
        
        ctk.CTkLabel(
            card,
            text=f"Error: {error_text}",
            font=ctk.CTkFont(size=10),
            text_color="red",
            anchor="w",
            wraplength=300
        ).pack(anchor="w", padx=10, pady=(0, 10))
        
        return card
    
    def _show_file_plot(self, result: Dict[str, Any]):
        """Show detailed plot for a single file"""
        try:
            fig = figure_per_file(result)
            self._clear_plots()
            self._embed_figure(fig)
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot file:\n{str(e)}")
    
    def _show_comparisons(self):
        """Show comparison plots for all analyzed files"""
        if not self._batch_results:
            messagebox.showwarning("No Data", "Please analyze files first")
            return
        
        # Filter for flash files only
        flash_results = self._get_flash_results()
        
        if not flash_results:
            messagebox.showwarning("No Flash Files", "No successful flash files found to compare")
            return
        
        try:
            self.gui.update_status("Generating comparison plots...")
            
            # Generate all comparison figures with flash files only
            figs_dict = generate_comparison_figures(flash_results)
            
            # Clear existing plots
            self._clear_plots()
            
            # Embed all comparison figures
            for fig_name, fig in figs_dict.items():
                self._embed_figure(fig)
            
            self.gui.update_status(f"Showing {len(figs_dict)} comparison plots")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate comparison plots:\n{str(e)}")
            self.gui.update_status("Error generating comparisons")
    
    def _show_individual(self):
        """Show individual plots for flash files"""
        if not self._batch_results:
            messagebox.showwarning("No Data", "Please analyze files first")
            return
        
        # Filter for flash files only
        flash_results = self._get_flash_results()
        
        if not flash_results:
            messagebox.showwarning("No Flash Files", "No successful flash files found to plot")
            return
        
        try:
            self.gui.update_status("Generating individual plots...")
            
            # Clear existing plots
            self._clear_plots()
            
            # Generate and embed individual plots
            count = 0
            for result in flash_results:
                    try:
                        fig = figure_per_file(result)
                        self._embed_figure(fig)
                        count += 1
                    except Exception as e:
                        print(f"Error plotting {result['filename']}: {e}")
            
            self.gui.update_status(f"Showing {count} individual plots")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate individual plots:\n{str(e)}")
            self.gui.update_status("Error generating plots")
    
    def _export_excel(self):
        """Export comparison data to Excel"""
        if not self._batch_results:
            messagebox.showwarning("No Data", "Please analyze files first")
            return
        
        # Filter for flash files only
        flash_results = self._get_flash_results()
        
        if not flash_results:
            messagebox.showwarning("No Flash Files", "No successful flash files found to export")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save Comparison Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="cooldown_comparison.xlsx"
        )
        
        if not save_path:
            return
        
        try:
            self.gui.update_status("Exporting to Excel...")
            export_summary_excel(flash_results, save_path)
            messagebox.showinfo("Success", f"Data exported to:\n{save_path}")
            self.gui.update_status("Export complete")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export Excel:\n{str(e)}")
            self.gui.update_status("Export failed")
    
    def _clear_plots(self):
        """Clear all existing plots"""
        for canvas in self._plot_canvases:
            canvas.get_tk_widget().destroy()
        self._plot_canvases.clear()
        
        # Also destroy any remaining children in plot container
        for widget in self.plot_container.winfo_children():
            widget.destroy()
    
    def _embed_figure(self, fig):
        """Embed a matplotlib figure in the plot container"""
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
        self._plot_canvases.append(canvas)