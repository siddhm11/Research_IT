import os
import kagglehub
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# --- Step 1: Download Dataset from KaggleHub ---
print("📥 Downloading dataset from KaggleHub...")
dataset_path = kagglehub.dataset_download("siddhm11/arxivdata")
print("✅ Dataset downloaded to:", dataset_path)


#print("Files in dataset directory:")
#for root, dirs, files in os.walk(dataset_path):
 #   for f in files:
  #      print(os.path.join(root, f))


# --- Step 2: Load CSV File ---
# --- Step 2: Load CSV File ---
csv_file = os.path.join(dataset_path, "arxiv_comprehensive_papers.csv")
if not os.path.exists(csv_file):
    raise FileNotFoundError("❌ CSV file not found in the downloaded dataset folder.")


print("📊 Reading dataset...")
df = pd.read_csv(csv_file)

# Keep only necessary columns and clean data
df = df[['id', 'title', 'authors', 'abstract']].fillna('')

# --- Step 3: Connect to DynamoDB ---
print("🔗 Connecting to DynamoDB...")
dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')  # change region if needed
table_name = "ArxivPapers"
table = dynamodb.Table(table_name)

# --- Step 4: Upload to DynamoDB using batch_writer ---
from tqdm import tqdm
import time

batch_size = 25
items = df.to_dict(orient='records')
total = len(items)

print(f"Uploading {total} items...")

with table.batch_writer(overwrite_by_pkeys=['id']) as batch:
    for i in tqdm(range(0, total, batch_size)):
        chunk = items[i:i+batch_size]
        for row in chunk:
            try:
                item = {
                    'id': str(row['id']),
                    'title': row['title'],
                    'authors': row['authors'],
                    'abstract': row['abstract']
                }
                batch.put_item(Item=item)
            except ClientError as e:
                print(f"❌ Failed at index {i}: {e.response['Error']['Message']}")



