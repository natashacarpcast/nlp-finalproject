import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from numpy import dtype
import time

# Load the dataset
dtype_dict = {'Unnamed: 0': dtype('int64'),
 'id': dtype('O'),
 'created': dtype('O'),
 'author': dtype('O'),
 'author_flair_text': dtype('O'),
 'score': dtype('int64'),
 'num_comments': dtype('int64'),
 'link': dtype('O'),
 'cleaned_title': dtype('O'),
 'cleaned_bodytext': dtype('float64'),
 'title+bodytext': dtype('O'),
 'word_count': dtype('int64'),
 'in_english': dtype('bool'),
 'vader_score': dtype('O'),
 'vader_neg': dtype('float64'),
 'vader_neu': dtype('float64'),
 'vader_pos': dtype('float64'),
 'vader_compound': dtype('float64'),
 'year': dtype('int64'),
 'month': dtype('int64')}


# Constants
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "life_pro_tips"
DATA_FILE_PATH = "data/cleaned_submissions.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize Qdrant client
qdrant_client = QdrantClient(QDRANT_URL)

# Load Data
df = pd.read_csv(DATA_FILE_PATH, dtype=dtype_dict,low_memory=False)

# Ensure required columns exist
if not {"cleaned_title", "author", "created"}.issubset(df.columns):
    raise ValueError("Missing required columns in dataset.")

# Extract Year and Month for formatting
df["year"] = pd.to_datetime(df["created"]).dt.year
df["month"] = pd.to_datetime(df["created"]).dt.month
df["date"] = df["year"].astype(str) + "/" + df["month"].astype(str).str.zfill(2)

# Initialize Model
print("Loading Sentence Transformer model...")
model = SentenceTransformer(MODEL_NAME)

# Check if Collection Exists
collections = qdrant_client.get_collections()
collection_exists = COLLECTION_NAME in [collection.name for collection in collections.collections]


# If collection exists, clear it
if collection_exists:
    print(f"Collection '{COLLECTION_NAME}' already exists. Deleting and recreating...")
    qdrant_client.delete_collection(COLLECTION_NAME)

# Create Collection in Qdrant
print(f"Creating collection '{COLLECTION_NAME}'...")
qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

# Function to encode and insert vectors in batches
def process_and_upload(df, batch_size=100):
    total = len(df)
    
    for i in range(0, total, batch_size):
        batch = df.iloc[i:i+batch_size]
        print(f"Processing batch {i} to {min(i+batch_size, total)}...")

        # Encode text
        batch_vectors = model.encode(batch["cleaned_title"].tolist(), convert_to_tensor=False)

        # Create Qdrant points
        points = [
            models.PointStruct(
                id=int(index),
                payload={
                    "author": row["author"],
                    "cleaned_title": row["cleaned_title"],
                    "date": row["date"]
                },
                vector=vector.tolist(),
            )
            for (index, row), vector in zip(batch.iterrows(), batch_vectors)
        ]

        # Insert into Qdrant
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Inserted {len(points)} vectors.")

# Run the process
start_time = time.time()
process_and_upload(df, batch_size=100)
end_time = time.time()

print(f"Vector upload complete. Time taken: {end_time - start_time:.2f} seconds")


