from qdrant_client import QdrantClient, models
import logging

# --- Configuration ---
# These details should match your Qdrant setup

FIELD_TO_INDEX = "arxiv_id"

def setup_payload_index():
    """Connects to Qdrant and creates a payload index on the specified field."""
    
    logging.info(f"Connecting to Qdrant at {QDRANT_URL}...")
    # Increased timeout for potentially long-running operations on a remote server
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    
    logging.info(f"Attempting to create a 'keyword' payload index on the field '{FIELD_TO_INDEX}'...")
    
    try:
        # This is the core command that tells Qdrant to build the index.
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=FIELD_TO_INDEX,
            field_schema=models.PayloadSchemaType.KEYWORD, # As requested by the error message
            wait=True  # This makes the script wait until the operation is finished.
        )
        logging.info(f"✅ Success! The index on field '{FIELD_TO_INDEX}' has been created.")
        logging.info("You can now restart your main application.")
        
    except Exception as e:
        # This can happen if the index already exists, which is not a problem.
        logging.error(f"❌ An error occurred: {e}")
        logging.info("If the error message indicates that the index already exists, you can safely ignore this message.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    setup_payload_index()