# user_mapping.py
import sqlite3
from uuid import uuid4

conn = sqlite3.connect("user_mapping.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_mapping (
    user_id TEXT PRIMARY KEY,
    uuid TEXT UNIQUE
)
""")
conn.commit()

def get_or_create_uuid(user_id: str) -> str:
    cursor.execute("SELECT uuid FROM user_mapping WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    if row:
        return row[0]
    new_uuid = str(uuid4())
    cursor.execute("INSERT INTO user_mapping (user_id, uuid) VALUES (?, ?)", (user_id, new_uuid))
    conn.commit()
    return new_uuid

def get_uuid(user_id: str) -> str | None:
    cursor.execute("SELECT uuid FROM user_mapping WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    return row[0] if row else None
