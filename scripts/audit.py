import pandas as pd
import numpy as np
import os

def run_audit(filepath):
    print("=" * 50)
    print(f"DATASET AUDIT REPORT: {filepath}")
    print("=" * 50)

    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    df = pd.read_csv(filepath)

    # 1. Total row count
    total_rows = len(df)
    print(f"Total row count: {total_rows}")
    if total_rows >= 100:
        print("✓ Row count is sufficient (>= 100).")
    else:
        print("✗ Row count is insufficient (< 100).")

    # 2. Null/missing values per column
    print("\n--- Null/Missing Values ---")
    null_counts = df.isnull().sum()
    print(null_counts)

    # 3. Average description length
    print("\n--- Description Length Analysis ---")
    # Handling potential null descriptions by filling with empty string
    desc_words = df['Description'].fillna('').apply(lambda x: len(str(x).split()))
    avg_length = desc_words.mean()
    min_length = desc_words.min()
    max_length = desc_words.max()
    print(f"Average description length: {avg_length:.2f} words")
    print(f"Min description length: {min_length} words")
    print(f"Max description length: {max_length} words")

    if 50 <= avg_length <= 200:
        print("✓ Average length is ideal (50-200 words).")
    elif avg_length < 50:
        print("✗ Average length is too short (< 50 words). Enrichment recommended.")
    else:
        print("! Average length is quite long (> 200 words). May need truncation depending on context window.")

    # 4. Duplicate entries
    print("\n--- Duplicate Check ---")
    # Check duplicates based on ID or Name
    if 'Place_Id' in df.columns:
        duplicate_ids = df.duplicated(subset=['Place_Id']).sum()
        print(f"Duplicate Place_Id count: {duplicate_ids}")

    duplicate_names = df.duplicated(subset=['Place_Name']).sum()
    print(f"Duplicate Place_Name count: {duplicate_names}")

    # 5. Distribution of 'category' and 'city' columns
    print("\n--- Data Distribution ---")
    if 'Category' in df.columns:
        print("\nCategory Distribution:")
        print(df['Category'].value_counts())

    if 'City' in df.columns:
        print("\nCity Distribution:")
        print(df['City'].value_counts())

    print("\n" + "=" * 50)
    print("AUDIT COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    raw_filepath = "data/raw/tourism_with_image_clean.csv"
    run_audit(raw_filepath)
