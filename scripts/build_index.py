import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indexer import FAISSIndexer

def main():
    emb_path = "data/embeddings/embeddings.npy"
    if not os.path.exists(emb_path):
        print("Embeddings not found. Run run_embedding.py first.")
        return

    embeddings = np.load(emb_path)
    print(f"Loaded embeddings with shape {embeddings.shape}")

    indexer = FAISSIndexer()
    index = indexer.build_flat(embeddings)

    indexer.save_index(index, "data/index/travel_lens.faiss")
    print("Saved FAISS Flat index.")

if __name__ == "__main__":
    main()
