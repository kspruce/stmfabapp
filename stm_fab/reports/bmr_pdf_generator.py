"""
Batch Manufacturing Record PDF Generator

Generates GMP-compliant printable BMR templates for STM HDL fabrication.
Supports both blank templates and filled records from electronic BMR data.

Author: STM Fab Team
Date: November 2025
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, 
    Spacer, PageBreak, KeepTogether, Frame, PageTemplate
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

def _fmt_ts(ts: Optional[str]) -> str:
    """Format ISO timestamp into a compact human-readable string."""
    if not ts:
        return "______"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)

def _to_key(name: str) -> str:
    """Convert a display label into a snake_case key (fallback if 'key' missing)."""
    import re
    k = name.strip().lower()
    k = re.sub(r'[^a-z0-9]+', '_', k)
    k = re.sub(r'_+', '_', k).strip('_')
    return k

def _cb(flag: bool) -> str:
    """ASCII checkbox: [x] or [ ] to avoid missing Unicode glyphs."""
    return "[x]" if flag else "[ ]"



class BMRPDFGenerator:
    """
    Generate printable BMR forms in PDF format
    
    Features:
    - GMP-compliant layout with signature blocks
    - Checkboxes for quality verification
    - Parameter tables with units
    - Deviation tracking sections
    - Multi-page support with headers/footers
    """
    
    def __init__(self, output_path: str, page_size=letter):
        """
        Initialize PDF generator
        
        Args:
            output_path: Path for output PDF file
            page_size: Page size (letter or A4)
        """
        self.output_path = output_path
        self.page_size = page_size
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=page_size,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        
        # Styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Page elements
        self.elements = []
        
        # Tracking
        self.current_page = 1
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='BMRTitle',
            parent=self.styles['Title'],
            fontSize=20,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=6,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Step header
        self.styles.add(ParagraphStyle(
            name='StepHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=6,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        # Small text
        self.styles.add(ParagraphStyle(
            name='SmallText',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#7f8c8d')
        ))
    
    def add_header_section(self, batch_number: str = "", device_name: str = "", 
                          sample_name: str = "", operator: str = ""):
        """Add BMR header with batch information"""
        # Title
        title = Paragraph("BATCH MANUFACTURING RECORD", self.styles['BMRTitle'])
        self.elements.append(title)
        self.elements.append(Spacer(1, 0.2*inch))
        # Keep operator for downstream fallback in steps
        self.header_operator = operator or ""

        # Subtitle
        subtitle = Paragraph(
            "STM Hydrogen Desorption Lithography Process",
            self.styles['Heading2']
        )
        self.elements.append(subtitle)
        self.elements.append(Spacer(1, 0.3*inch))
        
        # Header information table
        header_data = [
            ['Batch Number:', batch_number or '____________________', 
             'Date Started:', '____________________'],
            ['Device ID:', device_name or '____________________', 
             'Target Completion:', '____________________'],
            ['Sample:', sample_name or '____________________', 
             'Actual Completion:', '____________________'],
            ['Primary Operator:', operator or '____________________', 
             'QC Review:', '____________________']
        ]
        
        header_table = Table(header_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
        header_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('FONT', (2, 0), (2, -1), 'Helvetica-Bold', 10),
            ('FONT', (1, 0), (1, -1), 'Helvetica', 10),
            ('FONT', (3, 0), (3, -1), 'Helvetica', 10),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        self.elements.append(header_table)
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_signature_page(self):
        """Add final signature and review page"""
        self.elements.append(PageBreak())

        # Title
        title = Paragraph("FINAL REVIEW AND APPROVAL", self.styles['SectionHeader'])
        self.elements.append(title)
        self.elements.append(Spacer(1, 0.2*inch))

        # Review checklist
        review_data = [
            ['<b>Review Item</b>', '<b>Complete</b>', '<b>Initials</b>'],
            ['All process steps completed as specified', '[ ]', '______'],
            ['All quality checks passed', '[ ]', '______'],
            ['All parameters within specification', '[ ]', '______'],
            ['All deviations documented and approved', '[ ]', '______'],
            ['All required signatures obtained', '[ ]', '______'],
            ['LabVIEW process files linked/attached', '[ ]', '______'],
            ['STM scan files linked/attached', '[ ]', '______']
        ]

        review_table = Table(review_data, colWidths=[4.5*inch, 1*inch, 1.5*inch])
        review_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 0), (2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]))
        self.elements.append(review_table)
        self.elements.append(Spacer(1, 0.3*inch))

        # Final signatures
        sig_title = Paragraph("<b>Final Approvals:</b>", self.styles['Normal'])
        self.elements.append(sig_title)
        self.elements.append(Spacer(1, 0.1*inch))

        final_sig_data = [
            ['', '', ''],
            ['Quality Control Review:', '', ''],
            ['Signature:', '________________________________', 'Date: ________________'],
            ['Printed Name:', '________________________________', ''],
            ['', '', ''],
            ['Process Engineer Review:', '', ''],
            ['Signature:', '________________________________', 'Date: ________________'],
            ['Printed Name:', '________________________________', ''],
            ['', '', ''],
            ['Principal Investigator Approval:', '', ''],
            ['Signature:', '________________________________', 'Date: ________________'],
            ['Printed Name:', '________________________________', '']
        ]

        final_sig_table = Table(final_sig_data, colWidths=[2*inch, 3*inch, 2*inch])
        final_sig_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('FONT', (0, 1), (0, 1), 'Helvetica-Bold', 11),
            ('FONT', (0, 5), (0, 5), 'Helvetica-Bold', 11),
            ('FONT', (0, 9), (0, 9), 'Helvetica-Bold', 11),
            ('SPAN', (0, 1), (2, 1)),
            ('SPAN', (0, 5), (2, 5)),
            ('SPAN', (0, 9), (2, 9)),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
        ]))
        self.elements.append(final_sig_table)
        self.elements.append(Spacer(1, 0.4*inch))

        # Footer note
        footer_note = Paragraph(
            "<i>This batch manufacturing record must be completed in its entirety and "
            "approved by all designated reviewers before the device can proceed to "
            "electrical testing or further processing.</i>",
            self.styles['SmallText']
        )
        self.elements.append(footer_note)

    def add_signature_page(self):
        """Add final signature and review page"""
        self.elements.append(PageBreak())

        title = Paragraph("FINAL REVIEW AND APPROVAL", self.styles['SectionHeader'])
        self.elements.append(title)
        self.elements.append(Spacer(1, 0.2*inch))

        review_data = [
            ['<b>Review Item</b>', '<b>Complete</b>', '<b>Initials</b>'],
            ['All process steps completed as specified', '[ ]', '______'],
            ['All quality checks passed', '[ ]', '______'],
            ['All parameters within specification', '[ ]', '______'],
            ['All deviations documented and approved', '[ ]', '______'],
            ['All required signatures obtained', '[ ]', '______'],
            ['LabVIEW process files linked/attached', '[ ]', '______'],
            ['STM scan files linked/attached', '[ ]', '______']
        ]
        review_table = Table(review_data, colWidths=[4.5*inch, 1*inch, 1.5*inch])
        review_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 0), (2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]))
        self.elements.append(review_table)
        self.elements.append(Spacer(1, 0.3*inch))

        sig_title = Paragraph("<b>Final Approvals:</b>", self.styles['Normal'])
        self.elements.append(sig_title)
        self.elements.append(Spacer(1, 0.1*inch))

        final_sig_data = [
            ['', '', ''],
            ['Quality Control Review:', '', ''],
            ['Signature:', '________________________________', 'Date: ________________'],
            ['Printed Name:', '________________________________', ''],
            ['', '', ''],
            ['Process Engineer Review:', '', ''],
            ['Signature:', '________________________________', 'Date: ________________'],
            ['Printed Name:', '________________________________', ''],
            ['', '', ''],
            ['Principal Investigator Approval:', '', ''],
            ['Signature:', '________________________________', 'Date: ________________'],
            ['Printed Name:', '________________________________', '']
        ]
        final_sig_table = Table(final_sig_data, colWidths=[2*inch, 3*inch, 2*inch])
        final_sig_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('FONT', (0, 1), (0, 1), 'Helvetica-Bold', 11),
            ('FONT', (0, 5), (0, 5), 'Helvetica-Bold', 11),
            ('FONT', (0, 9), (0, 9), 'Helvetica-Bold', 11),
            ('SPAN', (0, 1), (2, 1)),
            ('SPAN', (0, 5), (2, 5)),
            ('SPAN', (0, 9), (2, 9)),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
        ]))
        self.elements.append(final_sig_table)
        self.elements.append(Spacer(1, 0.4*inch))

        footer_note = Paragraph(
            "<i>This batch manufacturing record must be completed in its entirety and "
            "approved by all designated reviewers before the device can proceed to "
            "electrical testing or further processing.</i>",
            self.styles['SmallText']
        )
        self.elements.append(footer_note)

    def generate(self):
        """Generate the PDF document"""
        def add_page_number(canvas, doc):
            page_num = canvas.getPageNumber()
            text = f"Page {page_num}"
            canvas.saveState()
            canvas.setFont('Helvetica', 9)
            canvas.drawRightString(7.5*inch, 0.5*inch, text)
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.grey)
            canvas.drawString(0.75*inch, 0.5*inch,
                              f"STM HDL BMR - Generated: {datetime.now().strftime('%Y-%m-%d')}")
            canvas.restoreState()

        self.doc.build(self.elements, onFirstPage=add_page_number, onLaterPages=add_page_number)
        return self.output_path



    def add_step(self, step_number: int, step_name: str, 
                 parameters: List[Dict[str, str]], 
                 quality_checks: List[str],
                 filled_data: Optional[Dict] = None,
                 tables_def: Optional[List[Dict[str, Any]]] = None):
        # Step header
        step_header = Paragraph(f"<b>Step {step_number}: {step_name}</b>", self.styles['StepHeader'])
        self.elements.append(step_header)
        self.elements.append(Spacer(1, 0.10*inch))

        # Time box with operator fallback
        header_operator = getattr(self, "header_operator", "")
        op_val = (filled_data or {}).get('operator_initials') or header_operator or '______'
        start_val = _fmt_ts((filled_data or {}).get('start_time'))
        end_val = _fmt_ts((filled_data or {}).get('end_time'))
        time_data = [[
            'Operator Initials:', Paragraph(str(op_val), self.styles['Normal']),
            'Start Time:', Paragraph(start_val, self.styles['Normal']),
            'End Time:', Paragraph(end_val, self.styles['Normal'])
        ]]
        time_table = Table(time_data, colWidths=[1.2*inch, 1.2*inch, 0.9*inch, 1.2*inch, 0.8*inch, 1.2*inch])
        time_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
            ('FONT', (0, 0), (0, 0), 'Helvetica-Bold', 9),
            ('FONT', (2, 0), (2, 0), 'Helvetica-Bold', 9),
            ('FONT', (4, 0), (4, 0), 'Helvetica-Bold', 9),
            ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
            ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
            ('ALIGN', (4, 0), (4, 0), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
        ]))
        self.elements.append(time_table)
        self.elements.append(Spacer(1, 0.12*inch))

        # Parameters: two columns (Parameter | Value)
        actuals = (filled_data or {}).get('parameters', {})
        table_names = {t.get('name') for t in (tables_def or []) if isinstance(t, dict)}
        param_rows = [['Parameter', 'Value']]
        for p in parameters or []:
            label = p.get('name', '')
            key = p.get('key') or label
            if key in table_names:
                continue
            val = actuals.get(key, '')
            display_value = '____________________' if str(val).strip() == '' else str(val)
            param_rows.append([label, display_value])
        if len(param_rows) > 1:
            param_table = Table(param_rows, colWidths=[3.5*inch, 3.5*inch])
            param_table.setStyle(TableStyle([
                ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
                ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
            ]))
            self.elements.append(param_table)
            self.elements.append(Spacer(1, 0.10*inch))

        # Tables (arrays), e.g., flash_table, seven_step_table, overgrowth_events_table
        if tables_def:
            for tbl in tables_def:
                name = tbl.get('name')
                title = tbl.get('title', name or 'Table')
                cols = tbl.get('columns', [])
                rows = actuals.get(name, [])
                if not rows and not name:
                    continue
                self.elements.append(Paragraph(f"<b>{title}:</b>", self.styles['Normal']))
                self.elements.append(Spacer(1, 0.04*inch))
                header_row = [c.get('label', c.get('key', '')) for c in cols]
                data_rows = [header_row] + [[str(r.get(c.get('key'), '')) for c in cols] for r in rows]
                col_count = max(1, len(cols))
                total_w = 7.0 * inch
                sub = Table(data_rows, colWidths=[total_w / col_count] * col_count)
                sub.setStyle(TableStyle([
                    ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8),
                    ('FONT', (0, 1), (-1, -1), 'Helvetica', 8),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#bdc3c7')),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3)
                ]))
                self.elements.append(sub)
                self.elements.append(Spacer(1, 0.10*inch))

        # Quality checks: per-item Pass/Fail/N/A
        if quality_checks:
            self.elements.append(Paragraph("<b>Quality Checks:</b>", self.styles['Normal']))
            self.elements.append(Spacer(1, 0.04*inch))
            qcr = (filled_data or {}).get('quality_check_results', {}) or {}
            post_ok = bool((filled_data or {}).get('post_check_pass', False))
            qc_rows = [['Check', 'Pass', 'Fail', 'N/A']]
            for check in quality_checks:
                val = str(qcr.get(check, '')).strip().lower()
                qc_rows.append([
                    check,
                    _cb(val == 'pass' or (val == '' and post_ok)),
                    _cb(val == 'fail'),
                    _cb(val in ('na', 'n/a'))
                ])
            qc_table = Table(qc_rows, colWidths=[3.9*inch, 1.03*inch, 1.03*inch, 0.94*inch])
            qc_table.setStyle(TableStyle([
                ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
                ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
            ]))
            self.elements.append(qc_table)
            self.elements.append(Spacer(1, 0.12*inch))

        # Notes box
        notes_text = (filled_data or {}).get('quality_notes') or ''
        notes_title = Paragraph("<b>Notes / Observations:</b>", self.styles['Normal'])
        notes_table = Table([[notes_title], [notes_text or "\n\n\n"]], colWidths=[7*inch])
        notes_table.setStyle(TableStyle([
            ('FONT', (0, 1), (0, 1), 'Helvetica', 9),
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
        ]))
        self.elements.append(notes_table)
        self.elements.append(Spacer(1, 0.18*inch))

        # Signature / Initial box
        initials_val = (filled_data or {}).get('step_initials', '') or op_val
        initials_time = _fmt_ts((filled_data or {}).get('step_initial_time'))

        completed_by = op_val
        completed_time = _fmt_ts((filled_data or {}).get('end_time'))

        sig = [
            ['Initialed by:', initials_val or '______________________', 'Date/Time:', initials_time or '______________________'],
            ['Completed by:', completed_by or '______________________', 'Date/Time:', completed_time or '______________________']
        ]
        sig_table = Table(sig, colWidths=[1.2*inch, 2*inch, 1*inch, 2*inch])

        if filled_data:
            if op_val:
                sig[0][1] = op_val
            if (filled_data or {}).get('end_time'):
                sig[0][3] = _fmt_ts((filled_data or {})['end_time'])
            if (filled_data or {}).get('verified_by'):
                sig[1][1] = (filled_data or {})['verified_by']
            if (filled_data or {}).get('verified_time'):
                sig[1][3] = _fmt_ts((filled_data or {})['verified_time'])
        sig_table = Table(sig, colWidths=[1.2*inch, 2*inch, 1*inch, 2*inch])
        sig_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 9),
            ('FONT', (2, 0), (2, -1), 'Helvetica-Bold', 9),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
        ]))
        self.elements.append(sig_table)
        self.elements.append(Spacer(1, 0.28*inch))


# Standard SET process parameters for each step
SET_PROCESS_PARAMETERS = {
    1: [  # Sample Preparation
        {'name': 'Cleaning Method', 'spec': 'Acetone + IPA ultrasonic'},
        {'name': 'Inspection Result', 'spec': 'No visible contamination'}
    ],
    2: [  # Loading & Pumpdown
        {'name': 'Load Time', 'spec': 'As required'},
        {'name': 'Base Pressure (mbar)', 'spec': '< 1×10⁻⁹'}
    ],
    3: [  # Outgas
        {'name': 'Temperature (°C)', 'spec': '200-300'},
        {'name': 'Duration (min)', 'spec': '30-120'},
        {'name': 'Pressure During (mbar)', 'spec': '< 5×10⁻⁹'}
    ],
    4: [  # Flash Clean
        {'name': 'Flash Temperature (°C)', 'spec': '1200 ± 50'},
        {'name': 'Flash Duration (s)', 'spec': '5-10'},
        {'name': 'Flash Current (A)', 'spec': 'Per calibration'}
    ],
    5: [  # H-Termination
        {'name': 'Temperature (°C)', 'spec': '330 ± 5'},
        {'name': 'H₂ Pressure (mbar)', 'spec': '1×10⁻⁶'},
        {'name': 'Dose Time (s)', 'spec': 'As required'},
        {'name': 'Dose (Langmuirs)', 'spec': '~ 1000 L'}
    ],
    6: [  # Pre-Litho Imaging
        {'name': 'Scan Area (nm)', 'spec': 'As required'},
        {'name': 'Bias (V)', 'spec': 'Typically -2.0 to +2.0'},
        {'name': 'Setpoint (pA)', 'spec': '10-100'}
    ],
    7: [  # Lithography
        {'name': 'Pattern Design', 'spec': 'Per device spec'},
        {'name': 'Litho Voltage (V)', 'spec': '6-8 V typical'},
        {'name': 'Desorption Speed', 'spec': 'Per pattern'}
    ],
    8: [  # Post-Litho Imaging
        {'name': 'Scan Area (nm)', 'spec': 'Cover pattern'},
        {'name': 'Feature Measurements', 'spec': 'Within design spec'}
    ],
    9: [  # Dopant Dosing
        {'name': 'Dopant Species', 'spec': 'PH₃ or AsH₃'},
        {'name': 'Dose Pressure (mbar)', 'spec': '1×10⁻⁷ to 1×10⁻⁶'},
        {'name': 'Dose Duration (s)', 'spec': 'As required'},
        {'name': 'Dose (Langmuirs)', 'spec': 'Per design'}
    ],
    10: [  # Incorporation
        {'name': 'Temperature (°C)', 'spec': '350 ± 5'},
        {'name': 'Duration (s)', 'spec': '60-180'},
        {'name': 'Thermal Budget Δ (°C·s)', 'spec': 'Track cumulative'}
    ],
    11: [  # Post-Incorp Imaging
        {'name': 'Scan Area (nm)', 'spec': 'Cover pattern'},
        {'name': 'Pattern Visibility', 'spec': 'Features visible'}
    ],
    12: [  # RT Growth
        {'name': 'Growth Temp (°C)', 'spec': '25 (RT)'},
        {'name': 'Growth Time (s)', 'spec': 'Per rate'},
        {'name': 'Target Thickness (nm)', 'spec': '~ 7 nm'}
    ],
    13: [  # RTA
        {'name': 'Anneal Temp (°C)', 'spec': '550'},
        {'name': 'Anneal Time (s)', 'spec': '60'},
        {'name': 'Thermal Budget Δ (°C·s)', 'spec': 'Track cumulative'}
    ],
    14: [  # LTE Growth
        {'name': 'Growth Temp (°C)', 'spec': '250'},
        {'name': 'Growth Time (s)', 'spec': 'Per rate'},
        {'name': 'Total Thickness (nm)', 'spec': '~ 20 nm'}
    ]
}


def generate_blank_bmr_template(output_path: str,
                                step_definitions: Optional[List[Dict[str, Any]]] = None) -> str:
    generator = BMRPDFGenerator(output_path)
    generator.add_header_section()
    def build_param_rows(defn: Dict[str, Any]) -> List[Dict[str, str]]:
        params = defn.get("parameters", {})
        names = (params.get("required", []) or []) + (params.get("optional", []) or [])
        table_names = {t.get('name') for t in (defn.get('tables') or [])}
        rows = []
        for n in names:
            if n in table_names:
                continue
            rows.append({'name': n.replace('_', ' ').title(), 'key': n})
        return rows
    if step_definitions and isinstance(step_definitions, list):
        for i, step in enumerate(step_definitions):
            step_num = step.get('step_number', i + 1)
            generator.add_step(
                step_number=step_num,
                step_name=step.get('step_name', f"Step {step_num}"),
                parameters=build_param_rows(step),
                quality_checks=step.get('quality_checks', []),
                filled_data=None,
                tables_def=step.get('tables')
            )
            if (i + 1) % 2 == 0 and i < len(step_definitions) - 1:
                generator.elements.append(PageBreak())
    else:
        # Fallback generic layout based on SET_PROCESS_PARAMETERS
        steps_numbers = sorted(SET_PROCESS_PARAMETERS.keys())
        for i, step_num in enumerate(steps_numbers):
            params = [{**p, 'key': (p.get('key') or p.get('name', ''))} for p in SET_PROCESS_PARAMETERS.get(step_num, [])]
            generator.add_step(
                step_number=step_num,
                step_name=f"Step {step_num}",
                parameters=params,
                quality_checks=[],
                filled_data=None,
                tables_def=None
            )
            if (i + 1) % 2 == 0 and i < len(steps_numbers) - 1:
                generator.elements.append(PageBreak())
    generator.add_signature_page()
    return generator.generate()

def generate_filled_bmr(bmr_json_path: str, output_path: str) -> str:
    with open(bmr_json_path, 'r', encoding='utf-8') as f:
        bmr_data = json.load(f)

    generator = BMRPDFGenerator(output_path)
    # Header
    md = bmr_data.get('metadata', {})
    generator.add_header_section(
        batch_number=md.get('batch_number', ''),
        device_name=md.get('device_id', ''),
        operator=md.get('operator', '')
    )

    step_definitions = bmr_data.get('step_definitions')
    steps_data = bmr_data.get('steps', [])

    def build_param_rows_from_defs(defn: Dict[str, Any]) -> List[Dict[str, str]]:
        params = defn.get("parameters", {})
        names = (params.get("required", []) or []) + (params.get("optional", []) or [])
        table_names = {t.get('name') for t in (defn.get('tables') or [])}
        rows = []
        for n in names:
            if n in table_names:
                continue
            rows.append({'name': n.replace('_', ' ').title(), 'key': n})
        return rows

    if step_definitions and isinstance(step_definitions, list):
        for i in range(min(len(step_definitions), len(steps_data))):
            step_def = step_definitions[i]
            sd = steps_data[i]
            step_num = step_def.get('step_number', i + 1)
            filled_info = {
                'status': sd.get('status', ''),
                'operator_initials': sd.get('operator_initials', ''),
                'start_time': sd.get('start_time', ''),
                'end_time': sd.get('end_time', ''),
                'parameters': sd.get('parameters', {}),
                'quality_notes': sd.get('quality_notes', ''),
                'post_check_pass': sd.get('post_check_pass', False),
                'quality_check_results': sd.get('quality_check_results', {}),
                'verified_by': sd.get('verified_by', ''),
                'verified_time': sd.get('verified_time', ''),
                # NEW
                'step_initials': sd.get('step_initials', ''),
                'step_initial_time': sd.get('step_initial_time', '')
            }

            generator.add_step(
                step_number=step_num,
                step_name=step_def.get('step_name', f"Step {step_num}"),
                parameters=build_param_rows_from_defs(step_def),
                quality_checks=step_def.get('quality_checks', []),
                filled_data=filled_info,
                tables_def=step_def.get('tables')
            )
            if (i + 1) % 2 == 0 and i < len(steps_data) - 1:
                generator.elements.append(PageBreak())
    else:
        # Fallback: no step_definitions
        for i, sd in enumerate(steps_data):
            step_num = sd.get('step_number', i + 1)
            filled_info = {
                'status': sd.get('status', ''),
                'operator_initials': sd.get('operator_initials', ''),
                'start_time': sd.get('start_time', ''),
                'end_time': sd.get('end_time', ''),
                'parameters': sd.get('parameters', {}),
                'quality_notes': sd.get('quality_notes', ''),
                'post_check_pass': sd.get('post_check_pass', False),
                'quality_check_results': sd.get('quality_check_results', {}),
                'verified_by': sd.get('verified_by', ''),
                'verified_time': sd.get('verified_time', '')
            }
            params = [{**p, 'key': (p.get('key') or p.get('name', ''))} for p in SET_PROCESS_PARAMETERS.get(step_num, [])]
            generator.add_step(
                step_number=step_num,
                step_name=f"Step {step_num}",
                parameters=params,
                quality_checks=[],
                filled_data=filled_info,
                tables_def=None
            )
            if (i + 1) % 2 == 0 and i < len(steps_data) - 1:
                generator.elements.append(PageBreak())

    generator.add_signature_page()
    return generator.generate()

if __name__ == "__main__":
    # Example usage
    print("Generating blank BMR template...")
    output = generate_blank_bmr_template("BMR_Template_SET_Process.pdf")
    print(f"✓ Generated: {output}")
