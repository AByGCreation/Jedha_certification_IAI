#=============================== LIBRARIES ==============================
#=================== DATA MANIPULATION ====================
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
#=================== CONFIG IMPORT ====================
import config as cfg #local_connectionURL, localDB_connectionURL, localDB_tableName, neonDB_connectionURL, neonDB_fraudTableName, HF_connectionCSV

def dataSourceLoader(inputDBFormat: str) -> pd.DataFrame|bool:
    """Load data from specified source format into a DataFrame.

    Args:
        inputDBFormat (str): The format of the data source ("csv", "db", "neon", "HF").

    Returns:
        pd.DataFrame: Loaded DataFrame or False on error.
    """
    dfRaw = pd.DataFrame()
    conn = None
    engine = None

    try:
        if inputDBFormat == "csv":
            dfRaw = pd.read_csv(cfg.local_connectionURL)
        elif inputDBFormat == "db":
            conn = sqlite3.connect(cfg.localDB_connectionURL)
            query = f"SELECT * FROM {cfg.localDB_tableName}"
            dfRaw = pd.read_sql_query(query, conn)
        elif inputDBFormat == "neon":
            engine = create_engine(cfg.neonDB_connectionURL)
            query = f"SELECT * FROM {cfg.neonDB_fraudTableName}"
            dfRaw = pd.read_sql_query(query, engine)
        elif inputDBFormat == "HF_CSV":
            dfRaw = pd.read_csv(cfg.HF_connectionCSV)
        else:
            print("❌ Invalid inputDBFormat. Please choose 'csv', 'db', 'neon', or 'HF_CSV'.")
            return False
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return False
    finally:
        # Ensure connections are properly closed
        if conn is not None:
            conn.close()
        if engine is not None:
            engine.dispose()


    dfRaw = dfRaw.astype({col: "float64" for col in dfRaw.select_dtypes(include=["int"]).columns})

    return dfRaw

