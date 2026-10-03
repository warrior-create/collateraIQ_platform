"""
Excel and PDF Reporting generation.
"""

from __future__ import annotations
import pandas as pd

def generate_excel_report(as_of: str, db_path: str, out_path: str = "report.xlsx"):
    """Generate daily Excel summary report."""
    from collateraliq.data.db import get_engine
    engine = get_engine(db_path)
    with engine.begin() as conn:
        exposures = pd.read_sql(f"SELECT * FROM exposures WHERE date = '{as_of}'", conn)
        im = pd.read_sql(f"SELECT * FROM im_results WHERE date = '{as_of}'", conn)
        
    with pd.ExcelWriter(out_path) as writer:
        exposures.to_excel(writer, sheet_name="Exposures", index=False)
        im.to_excel(writer, sheet_name="Initial Margin", index=False)
