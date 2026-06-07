import json
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Any
from sklearn.decomposition import PCA

from src.embedder import DestinationEmbedder
from src.indexer import FAISSIndexer

class QueryEngine:
    """
    A search engine class that provides various vector retrieval patterns using FAISS.
    """
    def __init__(self,
                 index_path: str = "data/index/travel_lens.faiss",
                 metadata_path: str = "data/processed/destinations_clean.csv",
                 embeddings_path: str = "data/embeddings/embeddings.npy",
                 id_map_path: str = "data/embeddings/id_map.json"):
        """
        Initializes the QueryEngine by loading the index, metadata, embeddings, and embedder.
        Also prepares a secondary PCA-reduced index for Q7.

        Args:
            index_path (str): Path to the saved FAISS index.
            metadata_path (str): Path to the processed dataset containing metadata.
            embeddings_path (str): Path to the saved numpy embeddings.
            id_map_path (str): Path to the JSON mapping of IDs to indices.
        """
        self.indexer = FAISSIndexer()
        self.index = self.indexer.load_index(index_path)

        self.metadata = pd.read_csv(metadata_path)
        # Convert Place_Id to string for consistent lookup
        self.metadata['Place_Id'] = self.metadata['Place_Id'].astype(str)

        self.embeddings = np.load(embeddings_path)
        with open(id_map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        # Reverse map for quick index-to-ID lookups
        self.index_to_id = {idx: id_val for id_val, idx in self.id_map.items()}

        self.embedder = DestinationEmbedder()
        self.embedder.load_model()

        # Prepare PCA index for Q7
        self._build_pca_index()

    def _build_pca_index(self) -> None:
        """
        Builds a secondary FAISS index with PCA dimensionality reduction (384 -> 64) for Q7.
        """
        self.pca = PCA(n_components=64)
        reduced_embeddings = self.pca.fit_transform(self.embeddings).astype(np.float32)

        # Normalizing after PCA is necessary for Inner Product to represent Cosine Similarity
        reduced_embeddings = self.embedder.normalize(reduced_embeddings)

        self.pca_index = faiss.IndexFlatIP(64)
        self.pca_index.add(reduced_embeddings)

    def _format_result(self, rank: int, idx: int, score: float) -> Dict[str, Any]:
        """
        Helper method to format a single search result.

        Args:
            rank (int): The rank of the result.
            idx (int): The FAISS index of the result.
            score (float): The similarity score.

        Returns:
            Dict[str, Any]: A dictionary containing destination metadata and score.
        """
        place_id = self.index_to_id.get(idx)
        if place_id is None:
            return {}

        # Extract row from metadata
        row = self.metadata[self.metadata['Place_Id'] == place_id]
        if row.empty:
            return {}

        row = row.iloc[0]

        return {
            "rank": rank,
            "id": place_id,
            "name": str(row.get('Place_Name', '')),
            "city": str(row.get('City', '')),
            "category": str(row.get('Category', '')),
            "score": float(score)
        }

    def q1_semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Q1 — SEMANTIC SEARCH (baseline)
        Process: embed query -> FAISS search -> return top-k results.

        Args:
            query (str): The natural language text query.
            k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Top k matching destinations.

        Example:
            engine.q1_semantic_search("taman hiburan untuk keluarga", k=5)
        """
        query_vec = self.embedder.embed_single(query).reshape(1, -1)
        query_vec = self.embedder.normalize(query_vec)

        distances, indices = self.indexer.search(self.index, query_vec, k=k)

        results = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx != -1:
                results.append(self._format_result(rank, idx, score))

        return results

    def q2_category_filter(self, query: str, category: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Q2 — CATEGORY FILTER
        Process: embed -> search -> post-filter by metadata['category'].

        Args:
            query (str): The natural language text query.
            category (str): The exact category name to filter by.
            k (int): Target number of final results.

        Returns:
            List[Dict[str, Any]]: Top matching destinations filtered by category.

        Example:
            engine.q2_category_filter("tempat bersejarah", "Budaya", k=3)
        """
        # Search a larger pool to allow for post-filtering
        search_k = min(50, len(self.embeddings))

        query_vec = self.embedder.embed_single(query).reshape(1, -1)
        query_vec = self.embedder.normalize(query_vec)

        distances, indices = self.indexer.search(self.index, query_vec, k=search_k)

        results = []
        rank = 1
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            formatted = self._format_result(rank, idx, score)
            if not formatted:
                continue

            if formatted['category'].lower() == category.lower():
                formatted['rank'] = rank
                results.append(formatted)
                rank += 1

            if len(results) >= k:
                break

        return results

    def q3_city_semantic_filter(self, query: str, allowed_cities: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        Q3 — CITY + SEMANTIC FILTER
        Process: embed -> search top-50 -> filter by city in allowed_cities.

        Args:
            query (str): The natural language text query.
            allowed_cities (List[str]): List of allowed city names.
            k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Top matching destinations filtered by city.

        Example:
            engine.q3_city_semantic_filter("pantai indah", ["Bali", "Lombok"], k=5)
        """
        search_k = 50
        query_vec = self.embedder.embed_single(query).reshape(1, -1)
        query_vec = self.embedder.normalize(query_vec)

        distances, indices = self.indexer.search(self.index, query_vec, k=search_k)

        allowed_cities_lower = [city.lower() for city in allowed_cities]

        results = []
        rank = 1
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            formatted = self._format_result(rank, idx, score)
            if not formatted:
                continue

            if formatted['city'].lower() in allowed_cities_lower:
                formatted['rank'] = rank
                results.append(formatted)
                rank += 1

            if len(results) >= k:
                break

        return results

    def q4_broad_semantic_search(self, query: str, k: int = 50) -> List[Dict[str, Any]]:
        """
        Q4 — BROAD SEMANTIC SEARCH
        Process: embed -> search top-50.
        Note: Originally intended for rating filtering, but modified to a broad search
        as per user constraints (ignoring non-existent rating/price fields).

        Args:
            query (str): The natural language text query.
            k (int): Number of results to return (default 50 for broad search).

        Returns:
            List[Dict[str, Any]]: Top 50 matching destinations.

        Example:
            engine.q4_broad_semantic_search("taman nasional", k=50)
        """
        return self.q1_semantic_search(query, k=k)

    def q5_item_based_search(self, destination_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Q5 — ITEM-BASED (find similar to existing)
        Process: retrieve stored embedding by id -> use as query vector -> exclude self.

        Args:
            destination_id (str): The ID of the query destination.
            k (int): Number of similar items to return.

        Returns:
            List[Dict[str, Any]]: Top similar destinations (excluding the query item).

        Example:
            engine.q5_item_based_search("12", k=5)
        """
        if destination_id not in self.id_map:
            raise ValueError(f"Destination ID {destination_id} not found in index.")

        target_idx = self.id_map[destination_id]
        query_vec = self.embeddings[target_idx].reshape(1, -1)

        # Search k+1 to account for excluding self
        distances, indices = self.indexer.search(self.index, query_vec, k=k+1)

        results = []
        rank = 1
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            # Exclude self
            if self.index_to_id[idx] == destination_id:
                continue

            formatted = self._format_result(rank, idx, score)
            if formatted:
                results.append(formatted)
                rank += 1

            if len(results) >= k:
                break

        return results

    def q6_multi_query_ensemble(self, queries: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        Q6 — MULTI-QUERY ENSEMBLE
        Process: embed 3 queries -> search separately -> merge results -> deduplicate -> re-rank by avg score.
        If an item is not returned by all queries, missing scores are treated as 0.0 for the average.

        Args:
            queries (List[str]): List of exactly 3 query strings.
            k (int): Target number of top results to return.

        Returns:
            List[Dict[str, Any]]: Ensemble ranked destinations.

        Example:
            engine.q6_multi_query_ensemble(["gunung berapi", "hiking trail", "pemandangan alam"], k=5)
        """
        if len(queries) != 3:
            raise ValueError("Must provide exactly 3 queries for the ensemble.")

        num_queries = len(queries)
        item_scores = {}

        for query in queries:
            query_vec = self.embedder.embed_single(query).reshape(1, -1)
            query_vec = self.embedder.normalize(query_vec)

            # Fetch a decent amount per query to ensure overlap
            distances, indices = self.indexer.search(self.index, query_vec, k=15)

            for score, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                if idx not in item_scores:
                    item_scores[idx] = []
                item_scores[idx].append(score)

        # Calculate average scores (missing queries = 0.0)
        avg_scores = []
        for idx, scores in item_scores.items():
            avg_score = sum(scores) / num_queries
            avg_scores.append((idx, avg_score))

        # Sort by average score descending
        avg_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, avg_score) in enumerate(avg_scores[:k], start=1):
            formatted = self._format_result(rank, idx, avg_score)
            if formatted:
                results.append(formatted)

        return results

    def q7_pca_reduced_query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Q7 — PCA-REDUCED QUERY
        Process: embed -> PCA transform -> search PCA index (384 -> 64 dim).

        Args:
            query (str): The natural language text query.
            k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Top matching destinations using the reduced index.

        Example:
            engine.q7_pca_reduced_query("museum seni", k=5)
        """
        query_vec = self.embedder.embed_single(query).reshape(1, -1)

        # Transform using PCA
        reduced_vec = self.pca.transform(query_vec).astype(np.float32)
        reduced_vec = self.embedder.normalize(reduced_vec)

        distances, indices = self.indexer.search(self.pca_index, reduced_vec, k=k)

        results = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx != -1:
                results.append(self._format_result(rank, idx, score))

        return results
