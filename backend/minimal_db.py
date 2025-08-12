# minimal_db.py
import sqlite3
import uuid
from typing import Dict, Optional, List
from user_types import VectorType

class SimpleUserDB:
    def __init__(self, db_path: str = "C:/Users/siddh/_code_/Research_IT/backend/users.db"):
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
                );
            ''')
    
    def get_user_uuids(self, user_id: str) -> Optional[Dict[VectorType, str]]:
        """Get existing UUIDs for user, ensuring all are present."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT vector_type, uuid FROM user_vectors WHERE user_id = ?',
                (user_id,)
            )
            rows = cursor.fetchall()
            
            # First, build the dictionary from whatever rows are found
            if not rows:
                return None
                
            # Use a set to handle potential duplicate vector_types in the DB
            # and ensure we only process unique ones.
            processed_rows = {row[0]: row[1] for row in rows}
            
            # Now, create the final dictionary using the VectorType enum
            uuids = {VectorType(vt): uuid_str for vt, uuid_str in processed_rows.items()}

            # THE CRITICAL FIX: Check if the CONSTRUCTED dictionary is complete.
            # It should contain all four members of the VectorType enum.
            if len(uuids) == 4 and all(vt in uuids for vt in VectorType):
                return uuids
                
            # If the data is incomplete or corrupted (e.g., missing 'complete' type),
            # return None. This will correctly trigger the creation of a new,
            # complete set of UUIDs for the user.
            return None
    
    def save_user_uuids(self, user_id: str, uuids: Dict[VectorType, str]):
        """Save UUIDs for user"""
        with sqlite3.connect(self.db_path) as conn:
            for vector_type, uuid_str in uuids.items():
                conn.execute(
                    'INSERT OR REPLACE INTO user_vectors (user_id, vector_type, uuid) VALUES (?, ?, ?)',
                    (user_id, vector_type.value, uuid_str)
                )
                
    # minimal_db.py (inside class SimpleUserDB)

    def list_all_user_ids(self) -> List[str]:
        """
        Return every distinct user_id we have stored vectors for.
        The feed service uses this during startup to re-hydrate the in-memory
        UserEmbeddingManager.users dictionary.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT user_id FROM user_vectors"
            )
            rows = cursor.fetchall()
            return [row[0] for row in rows]

