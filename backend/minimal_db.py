# minimal_db.py
import sqlite3
import uuid
from typing import Dict, Optional
from user_types import VectorType

class SimpleUserDB:
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create the simple table"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_vectors (
                    user_id TEXT NOT NULL,
                    vector_type TEXT NOT NULL,
                    uuid TEXT NOT NULL,
                    PRIMARY KEY (user_id, vector_type)
                )
            ''')
    
    def get_user_uuids(self, user_id: str) -> Optional[Dict[VectorType, str]]:
        """Get existing UUIDs for user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT vector_type, uuid FROM user_vectors WHERE user_id = ?',
                (user_id,)
            )
            rows = cursor.fetchall()
            
            if len(rows) == 4:  # Must have all 4 vectors
                return {
                    VectorType(row[0]): row[1] for row in rows
                }
            return None
    
    def save_user_uuids(self, user_id: str, uuids: Dict[VectorType, str]):
        """Save UUIDs for user"""
        with sqlite3.connect(self.db_path) as conn:
            for vector_type, uuid_str in uuids.items():
                conn.execute(
                    'INSERT OR REPLACE INTO user_vectors (user_id, vector_type, uuid) VALUES (?, ?, ?)',
                    (user_id, vector_type.value, uuid_str)
                )
