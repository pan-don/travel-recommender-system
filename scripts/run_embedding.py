import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure src is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embedder import DestinationEmbedder

def main():
    csv_path = "data/processed/destinations_clean.csv"

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Check if necessary columns exist
    required_cols = ['Place_Id', 'Place_Name', 'Category', 'City', 'Description']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # Combine fields as specified: '{name}. {category} di {city}. {description}'
    # Clean up and capitalize where needed
    def create_embedding_text(row):
        name = str(row['Place_Name']).title()
        category = str(row['Category']).title()
        city = str(row['City']).title()
        desc = str(row['Description']).capitalize()
        return f"{name}. {category} di {city}. {desc}"

    print("Constructing combined text field for embedding...")
    df['embedding_text'] = df.apply(create_embedding_text, axis=1)

    texts_to_embed = df['embedding_text'].tolist()
    ids = df['Place_Id'].tolist()

    print(f"Total documents to embed: {len(texts_to_embed)}")

    embedder = DestinationEmbedder()

    start_time = time.time()
    embeddings = embedder.embed_batch(texts_to_embed, batch_size=32)
    embedding_time = time.time() - start_time

    print(f"\n--- Embedding Statistics ---")
    print(f"Time taken to embed: {embedding_time:.2f} seconds")
    print(f"Raw embeddings shape: {embeddings.shape}")
    print(f"Raw embeddings dtype: {embeddings.dtype}")

    # Normalize embeddings
    normalized_embeddings = embedder.normalize(embeddings)

    print(f"Normalized embeddings shape: {normalized_embeddings.shape}")
    print(f"Normalized embeddings dtype: {normalized_embeddings.dtype}")

    # Compute sample similarity
    if len(normalized_embeddings) >= 2:
        sim = np.dot(normalized_embeddings[0], normalized_embeddings[1])
        print(f"Sample similarity between id {ids[0]} and {ids[1]}: {sim:.4f}")
    print("----------------------------\n")

    # Save the embeddings and mapping
    embedder.save(normalized_embeddings, ids)
    print("Embedding process completed successfully.")

if __name__ == "__main__":
    main()
