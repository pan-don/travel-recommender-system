import json
import numpy as np
import pandas as pd
import random

def main():
    emb_path = "data/embeddings/embeddings.npy"
    map_path = "data/embeddings/id_map.json"
    csv_path = "data/processed/destinations_clean.csv"

    print(f"Loading embeddings from '{emb_path}'...")
    embeddings = np.load(emb_path)

    print(f"Loading id map from '{map_path}'...")
    with open(map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    print(f"Loading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    # Create mapping from Place_Id to Place_Name
    id_to_name = dict(zip(df['Place_Id'].astype(str), df['Place_Name']))

    # Select 3 random pairs
    ids = list(id_map.keys())
    if len(ids) < 2:
        print("Not enough embeddings to compare.")
        return

    random.seed(42) # Set seed for reproducibility or comment out for true random
    pairs = []
    for _ in range(3):
        pair = random.sample(ids, 2)
        pairs.append(pair)

    print("\n" + "="*50)
    print("EMBEDDING SIMILARITY VERIFICATION REPORT")
    print("="*50)

    for id1, id2 in pairs:
        idx1 = id_map[id1]
        idx2 = id_map[id2]

        vec1 = embeddings[idx1]
        vec2 = embeddings[idx2]

        # Calculate cosine similarity using dot product (since vectors are already L2 normalized)
        similarity = np.dot(vec1, vec2)

        name1 = id_to_name.get(id1, f"Unknown (ID: {id1})").title()
        name2 = id_to_name.get(id2, f"Unknown (ID: {id2})").title()

        print(f"\nComparing:")
        print(f"A: {name1}")
        print(f"B: {name2}")
        print(f"Cosine Similarity: {similarity:.4f}")

        if similarity > 0.8:
            print("Verdict: Very High Similarity (Likely identical or very similar concepts)")
        elif similarity > 0.6:
            print("Verdict: High Similarity (Share many common attributes)")
        elif similarity > 0.4:
            print("Verdict: Moderate Similarity (Some commonalities)")
        else:
            print("Verdict: Low Similarity (Distinctly different concepts)")

    print("\n" + "="*50)

if __name__ == "__main__":
    main()
