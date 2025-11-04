"""
labview_bulk_import.py - Bulk import LabVIEW process files (FIXED VERSION)

Provides functionality to:
1. Import all .txt files from a folder
2. Parse and link them to a sample/device
3. Show summary of all imported files
4. Update thermal budget automatically

FIXED: Added NumPy array to JSON conversion to prevent serialization errors
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from pathlib import Path
from typing import List, Dict, Any
import traceback
import numpy as np
from stm_fab.labview.labview_parser import LabVIEWParser

def convert_numpy_to_python(obj: Any) -> Any:
    """
    Recursively convert NumPy arrays and other non-serializable objects to Python types
    
    This function is needed because SQLAlchemy JSON fields cannot serialize NumPy arrays.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


class LabVIEWBulkImporter:
    """Handle bulk import of LabVIEW files"""
    
    def __init__(self, db_ops, labview_parser, thermal_calculator, process_summarizer):
        self.db_ops = db_ops
        self.labview_parser = labview_parser
        self.thermal_calculator = thermal_calculator
        self.process_summarizer = process_summarizer
    
    def find_labview_files(self, folder_path: str) -> List[Path]:
        """Find all .txt files in a folder"""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        # Find all .txt files
        txt_files = list(folder.glob("*.txt"))
        
        return sorted(txt_files)
    
    def parse_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Parse multiple LabVIEW files
        
        Returns:
            Dictionary mapping file path to parsed data
        """
        results = {}
        errors = {}
        
        for filepath in file_paths:
            try:
                #from labview_parser import LabVIEWParser
                parser = LabVIEWParser(str(filepath))
                data = parser.parse()
                
                # CRITICAL FIX: Convert NumPy arrays to Python lists
                data = convert_numpy_to_python(data)
                
                results[str(filepath)] = {
                    'data': data,
                    'filename': filepath.name,
                    'success': True
                }
            except Exception as e:
                errors[str(filepath)] = {
                    'filename': filepath.name,
                    'error': str(e),
                    'success': False
                }
        
        return {
            'parsed': results,
            'errors': errors,
            'total': len(file_paths),
            'success_count': len(results),
            'error_count': len(errors)
        }
    
    def import_to_sample(self, sample_id: int, parsed_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import all parsed files to a sample
        
        Returns:
            Summary of import operation
        """
        imported = []
        failed = []
        
        for filepath, result in parsed_results['parsed'].items():
            try:
                data = result['data']
                
                # CRITICAL FIX: Ensure data is converted before passing to database
                data = convert_numpy_to_python(data)
                
                # Add to database
                process_step = self.db_ops.add_process_step(
                    sample_id=sample_id,
                    process_type=data.get('file_type', 'unknown'),
                    labview_file_path=filepath,
                    parsed_data=data
                )
                
                imported.append({
                    'filename': result['filename'],
                    'process_type': data.get('file_type', 'unknown'),
                    'process_id': process_step.process_id
                })
                
            except Exception as e:
                failed.append({
                    'filename': result['filename'],
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        
        return {
            'imported': imported,
            'failed': failed,
            'import_count': len(imported),
            'fail_count': len(failed)
        }


class LabVIEWImportDialog:
    """GUI dialog for bulk importing LabVIEW files"""
    
    def __init__(self, parent, sample, importer, update_callback=None):
        self.parent = parent
        self.sample = sample
        self.importer = importer
        self.update_callback = update_callback
        
        self.selected_files = []
        self.parsed_results = None
        
        self.create_dialog()
    
    def create_dialog(self):
        """Create the import dialog"""
        self.dialog = ctk.CTkToplevel(self.parent)
        self.dialog.title(f"Import LabVIEW Files - {self.sample.sample_name}")
        self.dialog.geometry("1000x700")
        
        # Main frame
        main_frame = ctk.CTkFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(
            main_frame,
            text=f"Bulk Import LabVIEW Files",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=10)
        
        ctk.CTkLabel(
            main_frame,
            text=f"Sample: {self.sample.sample_name}",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        ).pack()
        
        # Folder selection
        folder_frame = ctk.CTkFrame(main_frame)
        folder_frame.pack(fill="x", pady=15, padx=10)
        
        ctk.CTkLabel(
            folder_frame,
            text="Select Folder:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=10)
        
        self.folder_var = tk.StringVar()
        folder_entry = ctk.CTkEntry(
            folder_frame,
            textvariable=self.folder_var,
            width=400
        )
        folder_entry.pack(side="left", padx=5)
        
        ctk.CTkButton(
            folder_frame,
            text="Browse",
            command=self.browse_folder,
            width=100
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            folder_frame,
            text="Scan Folder",
            command=self.scan_folder,
            width=100,
            fg_color="blue",
            hover_color="darkblue"
        ).pack(side="left", padx=5)
        
        # File list
        list_frame = ctk.CTkFrame(main_frame)
        list_frame.pack(fill="both", expand=True, pady=10, padx=10)
        
        ctk.CTkLabel(
            list_frame,
            text="Files Found:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        # Scrollable file list
        self.file_listbox = ctk.CTkScrollableFrame(list_frame, height=300)
        self.file_listbox.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Status
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Select a folder to scan for .txt files",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.status_label.pack(pady=5)
        
        # Action buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=10)
        
        self.import_button = ctk.CTkButton(
            button_frame,
            text="Import All Files",
            command=self.import_files,
            width=150,
            height=40,
            fg_color="green",
            hover_color="darkgreen",
            state="disabled"
        )
        self.import_button.pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.dialog.destroy,
            width=100,
            height=40
        ).pack(side="left", padx=5)
    
    def browse_folder(self):
        """Browse for folder"""
        folder = filedialog.askdirectory(title="Select LabVIEW Data Folder")
        if folder:
            self.folder_var.set(folder)
    
    def scan_folder(self):
        """Scan folder for .txt files"""
        folder = self.folder_var.get()
        if not folder:
            messagebox.showwarning("No Folder", "Please select a folder first")
            return
        
        try:
            # Find files
            files = self.importer.find_labview_files(folder)
            
            if not files:
                messagebox.showinfo("No Files", "No .txt files found in the selected folder")
                return
            
            # Store files
            self.selected_files = files
            
            # Display files
            for widget in self.file_listbox.winfo_children():
                widget.destroy()
            
            for file in files:
                file_frame = ctk.CTkFrame(self.file_listbox)
                file_frame.pack(fill="x", pady=2, padx=5)
                
                ctk.CTkLabel(
                    file_frame,
                    text=f"📄 {file.name}",
                    font=ctk.CTkFont(size=11),
                    anchor="w"
                ).pack(side="left", padx=10, pady=5)
            
            # Update status
            self.status_label.configure(
                text=f"Found {len(files)} file(s) - Ready to import",
                text_color="green"
            )
            
            # Enable import button
            self.import_button.configure(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan folder:\n{str(e)}")
            self.status_label.configure(text="Error scanning folder", text_color="red")
    
    def import_files(self):
        """Parse and import all selected files"""
        if not self.selected_files:
            messagebox.showwarning("No Files", "No files selected for import")
            return
        
        try:
            # Disable button during import
            self.import_button.configure(state="disabled")
            self.status_label.configure(text="Parsing files...", text_color="blue")
            self.dialog.update()
            
            # Parse files
            parsed_results = self.importer.parse_files(self.selected_files)
            
            # Show parse results
            if parsed_results['error_count'] > 0:
                error_msg = f"Failed to parse {parsed_results['error_count']} file(s):\n\n"
                for filepath, error_info in list(parsed_results['errors'].items())[:5]:
                    error_msg += f"• {error_info['filename']}: {error_info['error']}\n"
                if parsed_results['error_count'] > 5:
                    error_msg += f"\n... and {parsed_results['error_count'] - 5} more errors"
                
                result = messagebox.askyesno(
                    "Parse Errors",
                    error_msg + "\n\nContinue importing successfully parsed files?",
                    icon='warning'
                )
                if not result:
                    self.import_button.configure(state="normal")
                    self.status_label.configure(text="Import cancelled", text_color="gray")
                    return
            
            if parsed_results['success_count'] == 0:
                messagebox.showerror("No Files Parsed", "No files were successfully parsed")
                self.import_button.configure(state="normal")
                self.status_label.configure(text="No files to import", text_color="red")
                return
            
            # Import to database
            self.status_label.configure(text="Importing to database...", text_color="blue")
            self.dialog.update()
            
            import_results = self.importer.import_to_sample(
                self.sample.sample_id,
                parsed_results
            )
            
            # Show results
            self.show_results(parsed_results, import_results)
            
            # Callback to update parent UI
            if self.update_callback:
                self.update_callback()
            
            self.dialog.destroy()
            
        except Exception as e:
            self.status_label.configure(text="Error during import", text_color="red")
            error_details = f"Failed to import files:\n\n{str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Import Error", error_details)
            self.import_button.configure(state="normal")
    
    def show_results(self, parsed_results, import_results):
        """Show import results"""
        success = import_results['import_count']
        failed = import_results['fail_count']
        parse_errors = parsed_results['error_count']
        
        message = f"Import Complete!\n\n"
        message += f"Successfully imported: {success} file(s)\n"
        
        if failed > 0:
            message += f"Failed to import: {failed} file(s)\n"
        
        if parse_errors > 0:
            message += f"Failed to parse: {parse_errors} file(s)\n"
        
        message += f"\nThermal budget has been updated."
        
        # List imported files
        if import_results['imported']:
            message += "\n\nImported:\n"
            for item in import_results['imported']:
                message += f"  • {item['filename']} ({item['process_type']})\n"
        
        # List errors
        if import_results['failed']:
            message += "\n\nFailed:\n"
            for item in import_results['failed']:
                message += f"  • {item['filename']}: {item['error']}\n"
        
        if parse_errors > 0:
            message += "\n\nParse Errors:\n"
            for filepath, error_info in parsed_results['errors'].items():
                message += f"  • {error_info['filename']}: {error_info['error']}\n"
        
        messagebox.showinfo("Import Results", message)


# Integration functions for main GUI

def add_bulk_import_button_to_setup_tab(gui_instance, setup_container):
    """
    Add bulk import button to the setup tab
    
    Call this in create_setup_tab after the LabVIEW folder selection
    """
    import_frame = ctk.CTkFrame(setup_container)
    import_frame.pack(fill="x", pady=10)
    
    ctk.CTkButton(
        import_frame,
        text="📁 Bulk Import LabVIEW Files",
        command=lambda: show_bulk_import_dialog(gui_instance),
        height=40,
        font=ctk.CTkFont(size=13, weight="bold"),
        fg_color="orange",
        hover_color="darkorange"
    ).pack(fill="x", padx=20, pady=10)
    
    ctk.CTkLabel(
        import_frame,
        text="Import all .txt files from a folder and link to current sample",
        font=ctk.CTkFont(size=11),
        text_color="gray"
    ).pack(padx=20)


def show_bulk_import_dialog(gui_instance):
    """Show the bulk import dialog"""
    if not gui_instance.current_sample:
        messagebox.showwarning(
            "No Sample Selected",
            "Please create or select a sample first"
        )
        return
    
    # Create importer
    importer = LabVIEWBulkImporter(
        db_ops=gui_instance.db_ops,
        labview_parser=gui_instance.labview_parser if hasattr(gui_instance, 'labview_parser') else None,
        thermal_calculator=gui_instance.thermal_calculator,
        process_summarizer=gui_instance.process_summarizer
    )
    
    # Show dialog
    dialog = LabVIEWImportDialog(
        parent=gui_instance.root,
        sample=gui_instance.current_sample,
        importer=importer,
        update_callback=lambda: gui_instance.update_thermal_budget_display()
    )


def create_labview_management_tab(gui_instance, tabview):
    """
    Create a dedicated LabVIEW management tab
    
    Add this to create_widgets():
        tabview.add("📁  LabVIEW Files")
        create_labview_management_tab(self, tabview)
    """
    labview_tab = tabview.tab("📁  LabVIEW Files")
    
    main_frame = ctk.CTkFrame(labview_tab)
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Title
    ctk.CTkLabel(
        main_frame,
        text="LabVIEW Process Files",
        font=ctk.CTkFont(size=24, weight="bold")
    ).pack(pady=10)
    
    # Current sample info
    info_frame = ctk.CTkFrame(main_frame)
    info_frame.pack(fill="x", pady=10)
    
    sample_label = ctk.CTkLabel(
        info_frame,
        text="No sample selected",
        font=ctk.CTkFont(size=14),
        text_color="gray"
    )
    sample_label.pack(pady=10)
    
    # Store reference for updates
    gui_instance._labview_sample_label = sample_label
    
    # Bulk import button
    ctk.CTkButton(
        main_frame,
        text="📁 Bulk Import from Folder",
        command=lambda: show_bulk_import_dialog(gui_instance),
        height=50,
        font=ctk.CTkFont(size=14, weight="bold"),
        fg_color="orange",
        hover_color="darkorange"
    ).pack(fill="x", pady=20, padx=50)
    
    # Single file import
    ctk.CTkButton(
        main_frame,
        text="📄 Import Single File",
        command=gui_instance.import_and_show_labview_process,
        height=40,
        font=ctk.CTkFont(size=13)
    ).pack(fill="x", pady=10, padx=50)
    
    # File list
    ctk.CTkLabel(
        main_frame,
        text="Imported Process Files:",
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(anchor="w", padx=20, pady=(20, 10))
    
    file_list = ctk.CTkScrollableFrame(main_frame, height=300)
    file_list.pack(fill="both", expand=True, padx=20, pady=10)
    
    # Store reference for updates
    gui_instance._labview_file_list = file_list
    
    # Refresh button
    ctk.CTkButton(
        main_frame,
        text="🔄 Refresh",
        command=lambda: refresh_labview_file_list(gui_instance),
        width=100
    ).pack(pady=10)


def refresh_labview_file_list(gui_instance):
    """Refresh the LabVIEW file list"""
    if not hasattr(gui_instance, '_labview_file_list'):
        return
    
    # Update sample label
    if hasattr(gui_instance, '_labview_sample_label'):
        if gui_instance.current_sample:
            gui_instance._labview_sample_label.configure(
                text=f"Current Sample: {gui_instance.current_sample.sample_name}",
                text_color="white"
            )
        else:
            gui_instance._labview_sample_label.configure(
                text="No sample selected",
                text_color="gray"
            )
    
    # Clear file list
    for widget in gui_instance._labview_file_list.winfo_children():
        widget.destroy()
    
    if not gui_instance.current_sample:
        ctk.CTkLabel(
            gui_instance._labview_file_list,
            text="Select a sample to view files",
            text_color="gray"
        ).pack(pady=20)
        return
    
    # Get process steps for sample
    processes = gui_instance.db_ops.get_sample_processes(gui_instance.current_sample.sample_id)
    
    if not processes:
        ctk.CTkLabel(
            gui_instance._labview_file_list,
            text="No LabVIEW files imported yet",
            text_color="gray"
        ).pack(pady=20)
        return
    
    # Display files
    for process in processes:
        file_frame = ctk.CTkFrame(gui_instance._labview_file_list)
        file_frame.pack(fill="x", pady=2, padx=5)
        
        filename = Path(process.labview_file_path).name if process.labview_file_path else "Unknown"
        
        ctk.CTkLabel(
            file_frame,
            text=f"📄 {filename}",
            font=ctk.CTkFont(size=12),
            anchor="w"
        ).pack(side="left", padx=10, pady=5)
        
        ctk.CTkLabel(
            file_frame,
            text=f"{process.process_type}",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left", padx=5)