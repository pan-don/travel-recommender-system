import json
import numpy as np
import torch
import time
import os
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DestinationEmbedder:
    """
    A class for embedding destination text descriptions using sentence-transformers.
    This embedder uses the 'paraphrase-multilingual-MiniLM-L12-v2' model and produces
    normalized 384-dimensional float32 vectors suitable for FAISS IndexFlatIP.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initializes the DestinationEmbedder.

        Args:
            model_name (str): The name of the sentence-transformers model to use.
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """
        Loads the embedding model with caching.
        It uses GPU if available, otherwise gracefully falls back to CPU.
        """
        if self.model is None:
            print(f"Loading model '{self.model_name}' on {self.device}...")
            start_time = time.time()
            # The library handles caching automatically to ~/.cache/torch/sentence_transformers
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embeds a single text string into a vector.

        Args:
            text (str): The query or text to embed.

        Returns:
            np.ndarray: A 1D float32 numpy array representing the embedded text.
        """
        if self.model is None:
            self.load_model()

        # Output shape is (384,)
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds a batch of texts into vectors.

        Args:
            texts (List[str]): A list of text strings to embed.
            batch_size (int): The number of texts to embed in each batch.

        Returns:
            np.ndarray: A 2D float32 numpy array of shape (len(texts), 384).
        """
        if self.model is None:
            self.load_model()

        print(f"Embedding {len(texts)} documents using batch size of {batch_size}...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Applies L2 normalization to vectors for cosine similarity computation using Inner Product.

        Args:
            vectors (np.ndarray): The 2D numpy array of vectors to normalize.

        Returns:
            np.ndarray: The normalized 2D float32 numpy array.
        """
        # Calculate L2 norm for each vector along axis 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero by setting 0 norms to 1
        norms = np.where(norms == 0, 1e-10, norms)
        normalized_vectors = vectors / norms
        return normalized_vectors.astype(np.float32)

    def save(self, embeddings: np.ndarray, ids: List[Union[int, str]],
             emb_path: str = "data/embeddings/embeddings.npy",
             map_path: str = "data/embeddings/id_map.json") -> None:
        """
        Saves the normalized embeddings and the metadata mapping (id -> index) to disk.

        Args:
            embeddings (np.ndarray): The embeddings to save.
            ids (List[Union[int, str]]): A list of unique IDs corresponding to each embedding.
            emb_path (str): The file path where embeddings will be saved (.npy).
            map_path (str): The file path where the ID to index mapping will be saved (.json).
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        os.makedirs(os.path.dirname(map_path), exist_ok=True)

        # Save embeddings
        np.save(emb_path, embeddings)

        # Save mapping (id -> index)
        # Using string keys since JSON requires string keys for dictionaries
        id_map = {str(id_val): idx for idx, id_val in enumerate(ids)}
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(id_map, f, indent=4)

        print(f"Saved {len(embeddings)} embeddings to '{emb_path}'")
        print(f"Saved mapping for {len(ids)} IDs to '{map_path}'")
