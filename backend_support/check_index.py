import os
from qdrant_client import QdrantClient
import logging

# --- Configuration ---
# Ensure these match your main script and credentials


def check_collection_indexes():
    """Connects to Qdrant and prints the payload index information for a collection."""
    
    logging.info(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    
    try:
        logging.info(f"Fetching information for collection '{COLLECTION_NAME}'...")
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        
        print("\n" + "="*50)
        print(f"🔬 Current Payload Indexes for '{COLLECTION_NAME}'")
        print("="*50)

        # The payload_schema attribute holds the index info
        payload_schema = collection_info.payload_schema
        
        if not payload_schema:
            print("No payload indexes are currently set up for this collection.")
            print("The 'arxiv_id' field needs a 'keyword' index.")
            return

        # Check for our specific required index
        arxiv_id_indexed = False
        for field_name, schema_info in payload_schema.items():
            print(f"  - Field: '{field_name}'")
            print(f"    Data Type: {schema_info.data_type}")
            print(f"    Index Parameters: {schema_info.params}")
            if field_name == 'arxiv_id':
                arxiv_id_indexed = True

        print("="*50)
        if arxiv_id_indexed:
            print("\n✅ The 'arxiv_id' field is indexed. Your /similar endpoint should work.")
        else:
            print("\n❌ CRITICAL: The 'arxiv_id' field is NOT indexed. This is causing the error.")
            print("   You need to run the `create_index.py` script from the previous step.")

            
    except Exception as e:
        logging.error(f"❌ An error occurred: {e}")
        logging.info("Please ensure your collection name and credentials are correct.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    check_collection_indexes()