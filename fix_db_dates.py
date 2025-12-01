import sqlite3
import pandas as pd

# Path to your database
DB_PATH = r"c:/Users/PeterHall/OneDrive - AMPYR IDEA UK Ltd/Python scripts/Inverter data - Juggle/solar_assets.db"

# Connect to the database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get all table names
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
tables = [t[0] for t in tables]

# Patterns to detect date columns
date_patterns = ["date", "month", "period", "time"]

def format_dates_in_table(table):
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    changed = False
    for col in df.columns:
        if any(pat in col.lower() for pat in date_patterns):
            try:
                new_col = pd.to_datetime(df[col], errors='coerce').dt.strftime('%b-%y')
                if not new_col.equals(df[col]):
                    df[col] = new_col
                    changed = True
            except Exception:
                pass
    if changed:
        df.to_sql(table, conn, if_exists='replace', index=False)
        print(f"Updated date fields in table: {table}")
    else:
        print(f"No date fields updated in table: {table}")

for t in tables:
    format_dates_in_table(t)

conn.close()
print("All tables processed.")
