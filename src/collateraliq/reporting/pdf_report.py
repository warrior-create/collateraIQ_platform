"""
PDF Report Generator for Daily Margin Summaries.

Uses reportlab to generate a structured PDF document.
"""
from __future__ import annotations
import pandas as pd
from datetime import date
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(as_of: date, db_path: str, out_path: str = "margin_report.pdf"):
    """
    Generate a formatted PDF report of the daily margin run.
    """
    from collateraliq.data.db import get_engine
    engine = get_engine(db_path)
    with engine.begin() as conn:
        df = pd.read_sql(f"""
            SELECT e.client_id, c.name, e.exposure_inr, 
                   COALESCE(i.im_total_inr, 0) as im_total_inr, 
                   e.collateralised_exposure_inr
            FROM exposures e
            JOIN clients c ON e.client_id = c.client_id
            LEFT JOIN im_results i ON e.client_id = i.client_id AND e.date = i.date AND i.product_type IS NULL
            WHERE e.date = '{as_of}'
        """, conn)
        
    doc = SimpleDocTemplate(out_path, pagesize=landscape(letter))
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    elements.append(Paragraph(f"CollateralIQ Daily Margin Report - {as_of}", styles['Title']))
    elements.append(Spacer(1, 12))
    
    if df.empty:
        elements.append(Paragraph("No data available for this date.", styles['Normal']))
        doc.build(elements)
        return
        
    # Format data for table
    table_data = [["Client ID", "Name", "Exposure (INR)", "IM Held (INR)", "Collateralised Exp (INR)"]]
    
    for _, row in df.iterrows():
        table_data.append([
            row["client_id"],
            row["name"][:20],
            f"{row['exposure_inr']:,.2f}",
            f"{row['im_total_inr']:,.2f}",
            f"{row['collateralised_exposure_inr']:,.2f}"
        ])
        
    # Create Table
    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#002B36')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#FDF6E3')),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    
    elements.append(t)
    doc.build(elements)
