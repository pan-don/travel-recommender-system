import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tabulate import tabulate
from src.query_engine import QueryEngine
from src.benchmark import QueryBenchmark

def get_test_queries():
    """
    Returns a dictionary of test queries to be used in the benchmark.
    """
    return {
        "q1": "taman hiburan keluarga dengan wahana permainan",
        "q2": {"query": "tempat bersejarah", "category": "budaya"},
        "q3": {"query": "pantai indah pasir putih", "cities": ["Bali", "Lombok", "Yogyakarta"]},
        "q4": "pusat perbelanjaan dan oleh-oleh",
        "q5": "1", # Using Place_Id 1 (Monumen Nasional) as the target
        "q6": ["gunung berapi", "hiking trail", "pemandangan alam sunrise"],
        "q7": "museum seni dan galeri lukisan"
    }

def print_academic_explanations():
    """
    Prints academic-ready explanation paragraphs for each optimization.
    """
    explanations = {
        "Q1 (Text Preprocessing)": (
            "By applying standard natural language processing (NLP) techniques such as lowercasing, stopword removal, "
            "and length truncation before passing the query to the embedding model, we reduce noise in the dense vector representation. "
            "This optimization lowers the computational cost of embedding generation (due to shorter token sequences) and often "
            "improves semantic retrieval accuracy by focusing the model's attention on core semantic keywords rather than syntactic filler."
        ),
        "Q2 (Category Pre-Filtering via Sub-Indexes)": (
            "Vector search over a global index inherently retrieves candidates regardless of hard metadata constraints, requiring "
            "inefficient post-hoc filtering. By partitioning the embedding space into smaller, category-specific sub-indexes (IndexFlatIP), "
            "the search space is drastically reduced at query time. This pre-filtering strategy yields significant latency reductions "
            "proportional to the size of the partition relative to the global dataset, while maintaining perfect recall within the specified category."
        ),
        "Q3 (Search Space Reduction via Tightened K Expansion)": (
            "In hybrid search architectures (vector similarity combined with metadata filtering), expanding the top-K retrieval pool "
            "(e.g., retrieving K=50 to find 5 valid items) is a common heuristic to account for post-filter drop-offs. "
            "However, if the embedding model is highly attuned to the target domain, a tighter K-expansion (e.g., K=15) can be utilized. "
            "This optimization trades a negligible amount of recall for a measurable decrease in both FAISS distance computation time and post-retrieval processing overhead."
        ),
        "Q4 (Index Partitioning by Geographic Metadata)": (
            "Similar to the categorical pre-filtering approach, partitioning the vector index by high-cardinality metadata attributes "
            "(such as City) converts an O(N) exhaustive distance calculation across the global dataset into an O(N_partition) operation. "
            "When queried for broad semantic concepts within a specific geographic boundary, routing the query exclusively to the localized "
            "index eliminates irrelevant vector comparisons, substantially improving query throughput without compromising result relevance."
        ),
        "Q5 (In-Memory Embedding Caching)": (
            "Disk I/O latency constitutes a significant bottleneck in item-based recommendation scenarios where source embeddings "
            "must be retrieved dynamically to serve as query vectors. By pre-loading and caching the full embedding matrix in Random Access Memory (RAM) "
            "during system initialization, we eliminate the persistent storage read penalty. This architectural shift shifts the bottleneck from I/O throughput "
            "to memory bandwidth, resulting in orders of magnitude improvements in latency for item-to-item similarity operations."
        ),
        "Q6 (Parallel Multi-Query Ensemble Execution)": (
            "Multi-query ensemble methods, which aggregate retrieved candidates from several distinct semantic vectors to improve overall recall, "
            "are typically bottlenecked by sequential embedding generation and index traversal. By leveraging multi-threading (ThreadPoolExecutor) "
            "to parallelize the embedding and search phases, the total latency is reduced to approximately the latency of the single slowest query, "
            "plus a minor synchronization overhead. This optimization is crucial for maintaining interactive response times in complex query fusion pipelines."
        ),
        "Q7 (Dimensionality Reduction via PCA)": (
            "Dense embeddings (e.g., 384 dimensions) incur high computational costs during the inner-product calculations of exact search (IndexFlatIP). "
            "By applying Principal Component Analysis (PCA) to project the embeddings into a lower-dimensional subspace, we mathematically trade exact "
            "spatial fidelity for computational efficiency. By selecting an optimal number of components that preserves ≥95% of the explained variance, "
            "we achieve a near-lossless compression of the semantic space, yielding faster retrieval times while maintaining high precision relative to the full-dimensional baseline."
        )
    }

    print("\n" + "="*80)
    print("ACADEMIC EXPLANATIONS FOR OPTIMIZATIONS")
    print("="*80)
    for title, text in explanations.items():
        print(f"\n### {title}\n{text}")

def main():
    print("Initializing QueryEngine...")
    engine = QueryEngine()

    print("Initializing QueryBenchmark...")
    benchmark = QueryBenchmark(engine)

    test_queries = get_test_queries()

    print("Running Full Benchmark Suite (20 runs per query)...")
    results_df = benchmark.run_full_benchmark(test_queries, n_runs=20)

    # Ensure results directory exists
    os.makedirs("data/results", exist_ok=True)

    csv_path = "data/results/benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults exported to: {csv_path}")

    print("\n" + "="*80)
    print("VECTOR SEARCH OPTIMIZATION BENCHMARK RESULTS")
    print("="*80)

    # Format table for terminal output
    table = tabulate(
        results_df,
        headers="keys",
        tablefmt="github",
        showindex=False
    )
    print(table)

    # Calculate and print overall improvement summary
    avg_improvement = results_df['improvement_%'].mean()
    print(f"\nOverall Average Latency Improvement: {avg_improvement:.2f}%")

    # Print academic explanations
    print_academic_explanations()

if __name__ == "__main__":
    main()
