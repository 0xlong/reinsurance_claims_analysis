import os
import duckdb

# Define the data directories
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
DB_PATH = os.path.join(DATA_DIR, 'insurance.db')

def load_into_duckdb():
    """Load the raw CSV files into DuckDB tables."""
    freq_path = os.path.join(RAW_DIR, 'freMTPL2freq.csv')
    sev_path = os.path.join(RAW_DIR, 'freMTPL2sev.csv')
    
    print(f"Connecting to DuckDB at {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)
    
    # Load frequency data (policies)
    print("Loading frequency data into DuckDB table 'raw_policies'...")
    conn.execute(f"CREATE TABLE IF NOT EXISTS raw_policies AS SELECT * FROM read_csv_auto('{freq_path}')")
    
    # Load severity data (claims)
    print("Loading severity data into DuckDB table 'raw_claims'...")
    conn.execute(f"CREATE TABLE IF NOT EXISTS raw_claims AS SELECT * FROM read_csv_auto('{sev_path}')")
    
    # Verify tables
    tables = conn.execute("SHOW TABLES").fetchall()
    print("DuckDB Tables created:")
    for table in tables:
        print(f" - {table[0]}")
        
    conn.close()
    print("DuckDB loading complete.")

if __name__ == "__main__":
    print("Starting Phase 1: Data Ingestion (Loading from local CSVs)...")
    load_into_duckdb()
    print("Phase 1 Data Ingestion completed successfully.")
