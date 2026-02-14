import sqlite3               
from datetime import datetime  
import pandas as pd  

DB_FILE = "cellula_database.db"
# Create a new table
def initialize_database():

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_type TEXT,
                user_input TEXT,
                prediction TEXT
            )
        """)

#  Inserts a new record into the database.
def save_to_database(input_type, user_input, prediction):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_inputs
            (timestamp, input_type, user_input, prediction)
            VALUES (?, ?, ?, ?)
        """, (timestamp, input_type, user_input, prediction))

# View all records in the table 
def view_all_records():

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_inputs")
        records = cursor.fetchall()
    return records

# View all records in the table as pandas Data frame 
def view_all_records_dataframe():

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("SELECT * FROM user_inputs", conn)
    return df


