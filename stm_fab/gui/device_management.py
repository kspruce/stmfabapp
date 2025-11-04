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


class DeviceManagementMixin:
    """
    Mixin class to add device management features to the main GUI
    
    Usage: Add this as a parent class to STMFabGUIEnhanced
    """
    
    def create_enhanced_database_tab(self):
        """Enhanced database tab with management buttons"""
        # Clear existing
        for widget in self.db_container.winfo_children():
            widget.destroy()
        
        # Header with refresh button
        header_frame = ctk.CTkFrame(self.db_container, fg_color="transparent")
        header_frame.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(
            header_frame,
            text="Database Manager",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            header_frame,
            text="🔄 Refresh",
            command=self.refresh_database_view,
            width=100,
            height=30
        ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            header_frame,
            text="View Thermal Budget",
            command=self.show_thermal_budget_selector,
            width=180,
            height=30,
            fg_color="orange",
            hover_color="darkorange"
        ).pack(side="right", padx=5)
        
        # Scrollable frame for samples and devices
        scroll_frame = ctk.CTkScrollableFrame(
            self.db_container,
            height=600,
            corner_radius=10
        )
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Get samples
        samples = self.db_ops.list_samples()
        
        if not samples:
            ctk.CTkLabel(
                scroll_frame,
                text="No samples in database",
                font=ctk.CTkFont(size=14),
                text_color="gray"
            ).pack(pady=30)
            return
        
        # Display samples with management options
        for sample in samples:
            self._create_sample_widget(scroll_frame, sample)
    
    def _create_sample_widget(self, parent, sample):
        """Create widget for a single sample with management buttons"""
        sample_frame = ctk.CTkFrame(parent, corner_radius=10)
        sample_frame.pack(fill="x", pady=5, padx=5)
        
        # Sample header
        header_frame = ctk.CTkFrame(sample_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=5)
        
        # Sample name and info
        info_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True)
        
        ctk.CTkLabel(
            info_frame,
            text=f"🧪 {sample.sample_name}",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            info_frame,
            text=f"{sample.substrate_type} | Created: {sample.creation_date.strftime('%Y-%m-%d')}",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w")
        
        # Sample management buttons
        button_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_frame.pack(side="right")
        
        ctk.CTkButton(
            button_frame,
            text="Rename",
            command=lambda: self.rename_sample_dialog(sample),
            width=80,
            height=28,
            fg_color="blue",
            hover_color="darkblue"
        ).pack(side="left", padx=2)
        
        ctk.CTkButton(
            button_frame,
            text="Delete",
            command=lambda: self.delete_sample_dialog(sample),
            width=80,
            height=28,
            fg_color="red",
            hover_color="darkred"
        ).pack(side="left", padx=2)
        
        # Devices for this sample
        devices = self.db_ops.list_devices(sample_id=sample.sample_id)
        
        if devices:
            device_container = ctk.CTkFrame(sample_frame)
            device_container.pack(fill="x", padx=20, pady=(0, 10))
            
            for device in devices:
                self._create_device_widget(device_container, device)
        else:
            ctk.CTkLabel(
                sample_frame,
                text="No devices yet",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            ).pack(padx=30, pady=5)
    
    def _create_device_widget(self, parent, device):
        """Create widget for a single device with management buttons"""
        device_frame = ctk.CTkFrame(parent)
        device_frame.pack(fill="x", pady=2, padx=5)
        
        # Device info
        info_frame = ctk.CTkFrame(device_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        status_emoji = {
            'in_progress': '🔄',
            'complete': '✅',
            'failed': '❌'
        }
        
        ctk.CTkLabel(
            info_frame,
            text=f"{status_emoji.get(device.overall_status, '🔬')} {device.device_name}",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        ctk.CTkLabel(
            info_frame,
            text=f"{device.completion_percentage:.0f}%",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left", padx=5)
        
        # Device management buttons
        button_frame = ctk.CTkFrame(device_frame, fg_color="transparent")
        button_frame.pack(side="right", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Select",
            command=lambda: self.select_device(device),
            width=70,
            height=25,
            fg_color="green",
            hover_color="darkgreen"
        ).pack(side="left", padx=1)
        
        ctk.CTkButton(
            button_frame,
            text="Rename",
            command=lambda: self.rename_device_dialog(device),
            width=70,
            height=25
        ).pack(side="left", padx=1)
        
        ctk.CTkButton(
            button_frame,
            text="Delete",
            command=lambda: self.delete_device_dialog(device),
            width=70,
            height=25,
            fg_color="red",
            hover_color="darkred"
        ).pack(side="left", padx=1)
    
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


