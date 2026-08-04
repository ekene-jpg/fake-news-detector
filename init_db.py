"""
Sets up the SQLite database per Chapter 3.12 Database Design:

  News_Dataset(ID, Title, News_Text, Label)
  Prediction_History(Prediction_ID, Input_Text, Result, Date)

Usage:
    python init_db.py
"""
import sqlite3
import pandas as pd
import os

DB_PATH = "news_detection.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS News_Dataset (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Title TEXT,
            News_Text TEXT,
            Label TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS Prediction_History (
            Prediction_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Input_Text TEXT,
            Result TEXT,
            Date TEXT
        )
    """)

    conn.commit()

    # Populate News_Dataset from the training CSV, if present and table is empty.
    cur.execute("SELECT COUNT(*) FROM News_Dataset")
    count = cur.fetchone()[0]
    csv_path = "data/news_dataset.csv"
    if count == 0 and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        label_text = df['label'].map({0: 'Fake', 1: 'Real'})
        rows = list(zip(df['title'], df['text'], label_text))
        cur.executemany(
            "INSERT INTO News_Dataset (Title, News_Text, Label) VALUES (?, ?, ?)",
            rows
        )
        conn.commit()
        print(f"Populated News_Dataset with {len(rows)} rows from {csv_path}")
    else:
        print(f"News_Dataset already has {count} rows, or {csv_path} not found — skipped populating.")

    conn.close()
    print(f"Database ready at {DB_PATH}")

if __name__ == "__main__":
    init_db()
