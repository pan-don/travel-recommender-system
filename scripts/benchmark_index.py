import os
import sys
import time
import tracemalloc
import numpy as np
import faiss
from tabulate import tabulate

# Ensure src is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indexer import FAISSIndexer

def benchmark_index(indexer, build_func, vectors, index_name, queries, k=5):
    """
    Helper function to benchmark build time, memory usage, and search time.
    """
    # 1. Measure Memory and Build Time
    tracemalloc.start()
    start_time = time.time()

    index = build_func(vectors)

    build_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory usage in KB
    memory_kb = peak / 1024.0

    # 2. Measure Search Time over multiple queries
    search_times = []
    # Using the same vectors as queries to simulate 50 random query evaluations
    for query in queries:
        query_2d = np.expand_dims(query, axis=0) # Search requires 2D array
        start_search = time.time()
        indexer.search(index, query_2d, k)
        search_times.append(time.time() - start_search)

    avg_search_time_ms = (sum(search_times) / len(search_times)) * 1000.0

    return [index_name, f"{build_time:.4f}", f"{memory_kb:.2f}", f"{avg_search_time_ms:.4f}"]

def main():
    emb_path = "data/embeddings/embeddings.npy"
    if not os.path.exists(emb_path):
        print(f"Error: Embeddings not found at {emb_path}. Please run run_embedding.py first.")
        return

    print(f"Loading embeddings from '{emb_path}'...")
    vectors = np.load(emb_path)

    num_vectors, dim = vectors.shape
    print(f"Dataset Size: {num_vectors} vectors of dimension {dim}.")

    indexer = FAISSIndexer(dimension=dim)

    # Simulate 50 queries by picking the first 50 vectors (or fewer if dataset is small)
    num_queries = min(50, num_vectors)
    queries = vectors[:num_queries]

    results = []

    print("\nBenchmarking indexes (this may take a moment)...\n")

    # 1. Benchmark FlatIP
    res_flat = benchmark_index(
        indexer,
        indexer.build_flat,
        vectors,
        "IndexFlatIP (Exact)",
        queries
    )
    results.append(res_flat)

    # 2. Benchmark IVF
    # For small datasets (~400 entries), nlist should be small (e.g., sqrt(N))
    # N = ~437 -> sqrt(437) ~ 21. We use 10 as requested by the prompt.
    res_ivf = benchmark_index(
        indexer,
        lambda v: indexer.build_ivf(v, nlist=10),
        vectors,
        "IndexIVFFlat (Approx, nlist=10)",
        queries
    )
    results.append(res_ivf)

    # 3. Benchmark HNSW
    res_hnsw = benchmark_index(
        indexer,
        lambda v: indexer.build_hnsw(v, M=16, ef=200),
        vectors,
        "IndexHNSWFlat (Graph, M=16)",
        queries
    )
    results.append(res_hnsw)

    # Print the table
    headers = ["Index Type", "Build Time (s)", "Peak Memory (KB)", "Avg Search Time (ms)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("\n")

    # Print the academic justification
    print("=" * 80)
    print("ACADEMIC JUSTIFICATION & INDEX RECOMMENDATION")
    print("=" * 80)
    print("""
For a dataset of this size (~100-500 entries), IndexFlatIP is unequivocally the recommended choice. While IndexIVFFlat and IndexHNSWFlat offer logarithmic or sub-linear search complexities beneficial for massive datasets, they introduce significant memory overhead and complexity (such as training phases and graph construction) that are unjustified for less than a thousand vectors. In modern hardware, an exhaustive exact search (brute-force) over 500 vectors of 384 dimensions takes merely fractions of a millisecond, often outperforming the constant-time overhead required to traverse an HNSW graph or project into IVF Voronoi cells. Furthermore, IndexFlatIP avoids approximation errors (guaranteeing 100% recall), inherently supports dynamic ID mapping, and seamlessly accommodates metadata filtering frameworks (like IDSelector) without the complexities associated with filtering graph-based index structures.
""")
    print("=" * 80)

    # Finally, save the recommended index
    best_index = indexer.build_flat(vectors)
    indexer.save_index(best_index, "data/index/travel_destinations.faiss")
    print("\nSaved recommended IndexFlatIP to 'data/index/travel_destinations.faiss'")

if __name__ == "__main__":
    main()
