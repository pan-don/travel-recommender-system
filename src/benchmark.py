import time
import numpy as np
import pandas as pd
import faiss
from typing import Callable, Any, Dict, List, Tuple
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import nltk
import re
import concurrent.futures

from src.query_engine import QueryEngine

# Ensure nltk resources are downloaded
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')


class QueryBenchmark:
    """
    A benchmarking class to measure latency and evaluate retrieval performance of vector search queries.
    """
    def __init__(self, engine: QueryEngine):
        self.engine = engine
        self.indonesian_stopwords = set(stopwords.words('indonesian'))

    def measure(self, func: Callable, *args, n_runs: int = 20, **kwargs) -> Tuple[Dict[str, float], Any]:
        """
        Measures the execution time of a function over multiple runs to reduce variance.

        Args:
            func (Callable): The function to measure.
            *args: Positional arguments for the function.
            n_runs (int): Number of times to run the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Tuple[Dict[str, float], Any]: Latency stats (avg, min, max, std) and the result of the last run.
        """
        latencies = []
        result = None
        for _ in range(n_runs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # Convert to ms

        stats = {
            "avg_ms": float(np.mean(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "std_ms": float(np.std(latencies))
        }
        return stats, result

    def precision_at_k(self, results: List[Dict[str, Any]], relevant_ids: List[str], k: int) -> float:
        """
        Calculates Precision@K to measure result overlap/degradation.

        Args:
            results (List[Dict[str, Any]]): The list of search results.
            relevant_ids (List[str]): Ground truth (or pseudo-ground truth) IDs.
            k (int): Cutoff rank.

        Returns:
            float: Precision@K score.
        """
        if not results or k <= 0:
            return 0.0

        top_k_results = results[:k]
        top_k_ids = [res['id'] for res in top_k_results]

        relevant_set = set(relevant_ids)
        hits = sum(1 for rid in top_k_ids if rid in relevant_set)

        return hits / min(k, len(top_k_ids))

    def _preprocess_query(self, text: str) -> str:
        """
        Q1 optimization: lowercase, remove stopwords, strip, and truncate.
        """
        text = text.lower().strip()
        words = text.split()
        filtered_words = [w for w in words if w not in self.indonesian_stopwords]
        cleaned_text = " ".join(filtered_words)
        return cleaned_text[:128]

    def _benchmark_q1(self, query: str, n_runs: int) -> Dict[str, Any]:
        """Q1 Optimization: text preprocessing before embedding."""
        k = 5
        # Before
        before_stats, before_results = self.measure(self.engine.q1_semantic_search, query, k=k, n_runs=n_runs)
        pseudo_relevant_ids = [res['id'] for res in before_results]

        # After
        optimized_query = self._preprocess_query(query)
        after_stats, after_results = self.measure(self.engine.q1_semantic_search, optimized_query, k=k, n_runs=n_runs)

        precision = self.precision_at_k(after_results, pseudo_relevant_ids, k)

        return {
            "query_type": "Q1 Semantic Search",
            "optimization": "Text Preprocessing",
            "before_ms": before_stats["avg_ms"],
            "after_ms": after_stats["avg_ms"],
            "improvement_%": ((before_stats["avg_ms"] - after_stats["avg_ms"]) / before_stats["avg_ms"]) * 100,
            "metric_name": "Precision@5 (Overlap)",
            "metric_val": precision
        }

    def _benchmark_q2(self, query: str, category: str, n_runs: int) -> Dict[str, Any]:
        """Q2 Optimization: pre-filter index (subset FAISS search)."""
        k = 5
        # Before: Search full index -> filter post-hoc
        before_stats, before_results = self.measure(self.engine.q2_category_filter, query, category, k=k, n_runs=n_runs)
        pseudo_relevant_ids = [res['id'] for res in before_results]

        # After: Build category sub-indexes and search the relevant one
        # Build sub-indexes dynamically
        category_indices = {}
        for idx, row in self.engine.metadata.iterrows():
            cat = str(row.get('Category', '')).lower()
            if cat not in category_indices:
                category_indices[cat] = []

            # Map the DataFrame index to the FAISS index if possible.
            # In QueryEngine, self.embeddings has the same row order as self.metadata typically,
            # but let's be safe and use Place_Id -> faiss index
            place_id = str(row['Place_Id'])
            if place_id in self.engine.id_map:
                faiss_idx = self.engine.id_map[place_id]
                category_indices[cat].append((faiss_idx, place_id))

        target_cat = category.lower()
        if target_cat in category_indices:
            cat_data = category_indices[target_cat]
            cat_embeddings = np.array([self.engine.embeddings[faiss_idx] for faiss_idx, _ in cat_data], dtype=np.float32)
            # Normalization is assumed to be preserved, but let's make sure
            cat_embeddings = self.engine.embedder.normalize(cat_embeddings)

            sub_index = faiss.IndexFlatIP(self.engine.embeddings.shape[1])
            sub_index.add(cat_embeddings)

            # Map local sub_index ids back to place_ids
            local_id_map = {i: pid for i, (_, pid) in enumerate(cat_data)}

            def optimized_q2(q: str, sub_idx, local_map, req_k: int):
                q_vec = self.engine.embedder.embed_single(q).reshape(1, -1)
                q_vec = self.engine.embedder.normalize(q_vec)
                D, I = sub_idx.search(q_vec, k=req_k)
                res = []
                rank = 1
                for s, i in zip(D[0], I[0]):
                    if i != -1:
                        pid = local_map.get(i)
                        if pid:
                            # We just need the ID to check overlap for benchmarking
                            res.append({"id": pid, "score": s, "rank": rank})
                            rank += 1
                return res

            after_stats, after_results = self.measure(optimized_q2, query, sub_index, local_id_map, k, n_runs=n_runs)
            precision = self.precision_at_k(after_results, pseudo_relevant_ids, k)
        else:
            after_stats = {"avg_ms": 0.0}
            precision = 0.0

        return {
            "query_type": "Q2 Category Filter",
            "optimization": "Per-Category Sub-Indexes",
            "before_ms": before_stats["avg_ms"],
            "after_ms": after_stats["avg_ms"],
            "improvement_%": ((before_stats["avg_ms"] - after_stats["avg_ms"]) / before_stats["avg_ms"]) * 100 if before_stats["avg_ms"] > 0 else 0,
            "metric_name": "Precision@5 (Overlap)",
            "metric_val": precision
        }

    def _benchmark_q3(self, query: str, allowed_cities: List[str], n_runs: int) -> Dict[str, Any]:
        """Q3 Optimization: reduce search space with tighter k expansion."""
        k = 5
        # Before (search top-50, filter) - the current implementation does this.
        before_stats, before_results = self.measure(self.engine.q3_city_semantic_filter, query, allowed_cities, k=k, n_runs=n_runs)
        pseudo_relevant_ids = [res['id'] for res in before_results]

        # After (search top-15, filter)
        def optimized_q3(q: str, cities: List[str], req_k: int):
            search_k = 15 # Tighter k
            q_vec = self.engine.embedder.embed_single(q).reshape(1, -1)
            q_vec = self.engine.embedder.normalize(q_vec)
            distances, indices = self.engine.indexer.search(self.engine.index, q_vec, k=search_k)
            allowed_cities_lower = [c.lower() for c in cities]
            res = []
            rank = 1
            for score, idx in zip(distances[0], indices[0]):
                if idx == -1: continue
                fmt = self.engine._format_result(rank, idx, score)
                if fmt and fmt['city'].lower() in allowed_cities_lower:
                    fmt['rank'] = rank
                    res.append(fmt)
                    rank += 1
                if len(res) >= req_k: break
            return res

        after_stats, after_results = self.measure(optimized_q3, query, allowed_cities, req_k=k, n_runs=n_runs)
        precision = self.precision_at_k(after_results, pseudo_relevant_ids, k)

        return {
            "query_type": "Q3 City Filter",
            "optimization": "Tighter K Expansion (top-15)",
            "before_ms": before_stats["avg_ms"],
            "after_ms": after_stats["avg_ms"],
            "improvement_%": ((before_stats["avg_ms"] - after_stats["avg_ms"]) / before_stats["avg_ms"]) * 100,
            "metric_name": "Precision@5 (Overlap)",
            "metric_val": precision
        }

    def _benchmark_q4(self, query: str, n_runs: int) -> Dict[str, Any]:
        """Q4 Optimization: City-partitioned index instead of broad semantic search."""
        k = 5
        # In the original query engine Q4 is Broad Semantic Search, returning top 50.
        # We modify the benchmark logic as requested:
        # Before: search full index -> filter by city post-hoc (simulate it here since engine.q4 doesn't filter)

        # We will use a dummy city to filter by
        target_city = "jakarta"

        def before_q4(q: str, city: str, req_k: int):
            # simulate post-hoc filtering on top-50
            search_k = 50
            q_vec = self.engine.embedder.embed_single(q).reshape(1, -1)
            q_vec = self.engine.embedder.normalize(q_vec)
            D, I = self.engine.indexer.search(self.engine.index, q_vec, k=search_k)
            res = []
            for s, i in zip(D[0], I[0]):
                if i != -1:
                    fmt = self.engine._format_result(0, i, s)
                    if fmt and fmt.get('city', '').lower() == city.lower():
                        res.append(fmt)
                    if len(res) >= req_k: break
            return res

        before_stats, before_results = self.measure(before_q4, query, target_city, req_k=k, n_runs=n_runs)
        pseudo_relevant_ids = [res['id'] for res in before_results]

        # After: City-partitioned index
        city_indices = {}
        for idx, row in self.engine.metadata.iterrows():
            c = str(row.get('City', '')).lower()
            if c not in city_indices:
                city_indices[c] = []
            place_id = str(row['Place_Id'])
            if place_id in self.engine.id_map:
                faiss_idx = self.engine.id_map[place_id]
                city_indices[c].append((faiss_idx, place_id))

        if target_city in city_indices:
            city_data = city_indices[target_city]
            city_embeddings = np.array([self.engine.embeddings[faiss_idx] for faiss_idx, _ in city_data], dtype=np.float32)
            city_embeddings = self.engine.embedder.normalize(city_embeddings)

            sub_index = faiss.IndexFlatIP(self.engine.embeddings.shape[1])
            sub_index.add(city_embeddings)

            local_id_map = {i: pid for i, (_, pid) in enumerate(city_data)}

            def optimized_q4(q: str, sub_idx, local_map, req_k: int):
                q_vec = self.engine.embedder.embed_single(q).reshape(1, -1)
                q_vec = self.engine.embedder.normalize(q_vec)
                D, I = sub_idx.search(q_vec, k=req_k)
                res = []
                for s, i in zip(D[0], I[0]):
                    if i != -1:
                        pid = local_map.get(i)
                        if pid:
                            res.append({"id": pid, "score": s})
                return res

            after_stats, after_results = self.measure(optimized_q4, query, sub_index, local_id_map, req_k=k, n_runs=n_runs)
            precision = self.precision_at_k(after_results, pseudo_relevant_ids, k)
        else:
            after_stats = {"avg_ms": 0.0}
            precision = 0.0

        return {
            "query_type": "Q4 Broad Search",
            "optimization": "City-partitioned Index",
            "before_ms": before_stats["avg_ms"],
            "after_ms": after_stats["avg_ms"],
            "improvement_%": ((before_stats["avg_ms"] - after_stats["avg_ms"]) / before_stats["avg_ms"]) * 100 if before_stats["avg_ms"] > 0 else 0,
            "metric_name": "Precision@5 (Overlap)",
            "metric_val": precision
        }

    def _benchmark_q5(self, destination_id: str, n_runs: int) -> Dict[str, Any]:
        """Q5 Optimization: simulate cache miss vs RAM cache."""
        k = 5
        embeddings_path = "data/embeddings/embeddings.npy"

        # Before (Simulated Disk Read)
        def before_q5(dest_id: str, req_k: int):
            # Simulate a cold/no-cache scenario
            disk_embeddings = np.load(embeddings_path)
            if dest_id not in self.engine.id_map:
                return []
            target_idx = self.engine.id_map[dest_id]
            query_vec = disk_embeddings[target_idx].reshape(1, -1)

            distances, indices = self.engine.indexer.search(self.engine.index, query_vec, k=req_k+1)

            res = []
            for s, i in zip(distances[0], indices[0]):
                if i != -1 and self.engine.index_to_id.get(i) != dest_id:
                    res.append({"id": self.engine.index_to_id.get(i)})
            return res[:req_k]

        before_stats, before_results = self.measure(before_q5, destination_id, req_k=k, n_runs=n_runs)
        pseudo_relevant_ids = [res['id'] for res in before_results]

        # After (RAM cached)
        after_stats, after_results = self.measure(self.engine.q5_item_based_search, destination_id, k=k, n_runs=n_runs)
        precision = self.precision_at_k(after_results, pseudo_relevant_ids, k)

        return {
            "query_type": "Q5 Item-Based",
            "optimization": "RAM Caching (Simulated)",
            "before_ms": before_stats["avg_ms"],
            "after_ms": after_stats["avg_ms"],
            "improvement_%": ((before_stats["avg_ms"] - after_stats["avg_ms"]) / before_stats["avg_ms"]) * 100,
            "metric_name": "Precision@5 (Overlap)",
            "metric_val": precision
        }

    def _benchmark_q6(self, queries: List[str], n_runs: int) -> Dict[str, Any]:
        """Q6 Optimization: Parallel query execution."""
        k = 5
        # Before
        before_stats, before_results = self.measure(self.engine.q6_multi_query_ensemble, queries, k=k, parallel=False, n_runs=n_runs)
        pseudo_relevant_ids = [res['id'] for res in before_results]

        # After
        after_stats, after_results = self.measure(self.engine.q6_multi_query_ensemble, queries, k=k, parallel=True, n_runs=n_runs)
        precision = self.precision_at_k(after_results, pseudo_relevant_ids, k)

        return {
            "query_type": "Q6 Multi-Query",
            "optimization": "Parallel Execution",
            "before_ms": before_stats["avg_ms"],
            "after_ms": after_stats["avg_ms"],
            "improvement_%": ((before_stats["avg_ms"] - after_stats["avg_ms"]) / before_stats["avg_ms"]) * 100,
            "metric_name": "Precision@5 (Overlap)",
            "metric_val": precision
        }

    def _benchmark_q7(self, query: str, n_runs: int) -> Dict[str, Any]:
        """Q7 Optimization: PCA n_components selection based on explained variance."""
        k = 5
        # Before: Search the full-dimensional index (384 dims) as the baseline
        before_stats, before_results = self.measure(self.engine.q1_semantic_search, query, k=k, n_runs=n_runs)
        pseudo_relevant_ids = [res['id'] for res in before_results]

        # Dynamically test PCA components
        components_to_test = [32, 64, 128, 256]
        optimal_n = None
        optimal_index = None
        optimal_pca = None

        for n in components_to_test:
            pca = PCA(n_components=n)
            pca.fit(self.engine.embeddings)
            variance_sum = pca.explained_variance_ratio_.sum()
            if variance_sum >= 0.95:
                optimal_n = n
                optimal_pca = pca
                break

        # If none hit 0.95, pick the max
        if optimal_n is None:
            optimal_n = 256
            optimal_pca = PCA(n_components=optimal_n).fit(self.engine.embeddings)

        # Build optimal PCA index
        reduced_embeddings = optimal_pca.transform(self.engine.embeddings).astype(np.float32)
        reduced_embeddings = self.engine.embedder.normalize(reduced_embeddings)

        optimal_index = faiss.IndexFlatIP(optimal_n)
        optimal_index.add(reduced_embeddings)

        def optimized_q7(q: str, pca_model, pca_idx, req_k: int):
            q_vec = self.engine.embedder.embed_single(q).reshape(1, -1)
            reduced_vec = pca_model.transform(q_vec).astype(np.float32)
            reduced_vec = self.engine.embedder.normalize(reduced_vec)

            D, I = pca_idx.search(reduced_vec, k=req_k)
            res = []
            for s, i in zip(D[0], I[0]):
                if i != -1:
                    pid = self.engine.index_to_id.get(i)
                    if pid:
                        res.append({"id": pid, "score": s})
            return res

        after_stats, after_results = self.measure(optimized_q7, query, optimal_pca, optimal_index, req_k=k, n_runs=n_runs)
        precision = self.precision_at_k(after_results, pseudo_relevant_ids, k)

        return {
            "query_type": "Q7 PCA Reduced",
            "optimization": f"Optimal PCA ({optimal_n} dims)",
            "before_ms": before_stats["avg_ms"],
            "after_ms": after_stats["avg_ms"],
            "improvement_%": ((before_stats["avg_ms"] - after_stats["avg_ms"]) / before_stats["avg_ms"]) * 100,
            "metric_name": "Precision@5 (Overlap)",
            "metric_val": precision
        }

    def run_full_benchmark(self, test_queries: Dict[str, Any], n_runs: int = 20) -> pd.DataFrame:
        """
        Runs benchmarks for all 7 queries and returns a DataFrame of results.

        Args:
            test_queries (Dict[str, Any]): Dictionary containing test data.
            n_runs (int): Number of runs for averaging.

        Returns:
            pd.DataFrame: A DataFrame containing the benchmark results.
        """
        results = []

        print("Running Q1 Benchmark...")
        results.append(self._benchmark_q1(test_queries['q1'], n_runs))

        print("Running Q2 Benchmark...")
        results.append(self._benchmark_q2(test_queries['q2']['query'], test_queries['q2']['category'], n_runs))

        print("Running Q3 Benchmark...")
        results.append(self._benchmark_q3(test_queries['q3']['query'], test_queries['q3']['cities'], n_runs))

        print("Running Q4 Benchmark...")
        results.append(self._benchmark_q4(test_queries['q4'], n_runs))

        print("Running Q5 Benchmark...")
        results.append(self._benchmark_q5(test_queries['q5'], n_runs))

        print("Running Q6 Benchmark...")
        results.append(self._benchmark_q6(test_queries['q6'], n_runs))

        print("Running Q7 Benchmark...")
        results.append(self._benchmark_q7(test_queries['q7'], n_runs))

        df = pd.DataFrame(results)
        # Format the columns for nicer output
        df['before_ms'] = df['before_ms'].round(2)
        df['after_ms'] = df['after_ms'].round(2)
        df['improvement_%'] = df['improvement_%'].round(2)
        df['metric_val'] = df['metric_val'].round(2)

        return df
