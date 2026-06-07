import pandas as pd
import re
import os

def clean_text(text):
    """
    Cleans text by lowercasing, stripping whitespace, and removing special chars.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    # Remove special characters but keep letters, numbers, and basic punctuation
    text = re.sub(r'[^\w\s.,!?\-]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_dataset(input_filepath, output_filepath):
    print(f"Loading dataset from {input_filepath}...")
    df = pd.read_csv(input_filepath)

    # Check if necessary columns exist
    required_columns = ['Place_Name', 'Category', 'City', 'Description']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    print("Cleaning text fields...")
    for col in required_columns:
        df[col] = df[col].apply(clean_text)

    print("Creating combined embedding input field...")
    # Creates a combined text field: "{name}. {category} di {city}. {description}"
    df['embedding_text'] = df.apply(
        lambda row: f"{row['Place_Name'].title()}. {row['Category'].title()} di {row['City'].title()}. {row['Description'].capitalize()}",
        axis=1
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    print(f"Saving processed dataset to {output_filepath}...")
    df.to_csv(output_filepath, index=False)
    print("Done!")

if __name__ == "__main__":
    raw_filepath = "data/raw/tourism_with_image_clean.csv"
    processed_filepath = "data/processed/destinations_clean.csv"

    preprocess_dataset(raw_filepath, processed_filepath)
