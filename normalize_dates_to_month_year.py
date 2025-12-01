import sqlite3
import pandas as pd

# Path to your database
DB_PATH = r"c:/Users/PeterHall/OneDrive - AMPYR IDEA UK Ltd/Python scripts/Inverter data - Juggle/solar_assets.db"
TABLE = "your_table_name_here"  # <-- Replace with your actual table name
DATE_COL = "Date"  # <-- Replace with your actual date column name

conn = sqlite3.connect(DB_PATH)

# Read the table
print(f"Reading table: {TABLE}")
df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)

# Normalize all date values to 'Apr-25' format
print(f"Normalizing column: {DATE_COL}")
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.strftime('%b-%y')

# Save back to the database
print(f"Saving normalized table: {TABLE}")
df.to_sql(TABLE, conn, if_exists='replace', index=False)

conn.close()
print("Done. All dates in column '{DATE_COL}' are now in 'Apr-25' format.")
