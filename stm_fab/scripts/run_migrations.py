# scripts/one_off_add_sample_paths.py
from sqlalchemy import create_engine, inspect, text

# Adjust if your DB URL differs
DB_URL = "sqlite:///stm_fab_records.db"

engine = create_engine(DB_URL)

with engine.begin() as conn:
    insp = inspect(conn)
    tables = insp.get_table_names()
    if 'samples' not in tables:
        raise RuntimeError("Table 'samples' not found. Create DB first (init_database).")

    cols = [c['name'] for c in insp.get_columns('samples')]
    if 'labview_folder_path' not in cols:
        conn.execute(text("ALTER TABLE samples ADD COLUMN labview_folder_path VARCHAR(500)"))
        print("Added column: labview_folder_path")
    else:
        print("Column 'labview_folder_path' already exists")

    if 'scan_folder_path' not in cols:
        conn.execute(text("ALTER TABLE samples ADD COLUMN scan_folder_path VARCHAR(500)"))
        print("Added column: scan_folder_path")
    else:
        print("Column 'scan_folder_path' already exists")
