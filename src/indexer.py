import os
import faiss
import numpy as np

class FAISSIndexer:
    """
    A class to build, save, load, and search FAISS indexes.
    Provides implementations for Flat, IVF, and HNSW indexes.
    """

    def __init__(self, dimension: int = 384):
        """
        Initializes the FAISSIndexer.

        Args:
            dimension (int): The dimension of the embeddings (default 384 for MiniLM).
        """
        self.dimension = dimension

    def build_flat(self, vectors: np.ndarray) -> faiss.IndexFlatIP:
        """
        Builds an IndexFlatIP for exact search using Inner Product (equivalent to
        Cosine Similarity when vectors are L2 normalized). This is the baseline index.

        Args:
            vectors (np.ndarray): A 2D numpy array of L2 normalized float32 embeddings.

        Returns:
            faiss.IndexFlatIP: The built index.
        """
        index = faiss.IndexFlatIP(self.dimension)
        index.add(vectors)
        return index

    def build_ivf(self, vectors: np.ndarray, nlist: int = 10) -> faiss.IndexIVFFlat:
        """
        Builds an IndexIVFFlat for approximate nearest neighbor search.

        Academic note on 'nlist':
        The nlist parameter defines the number of Voronoi cells (clusters) the vector space
        is partitioned into during training. When searching, the algorithm only scans
        the nearest `nprobe` cells. A higher nlist means finer clustering and faster
        search times (if nprobe is kept small), but requires more memory for cluster
        centroids and more training time. If nlist is too high compared to the dataset
        size, the clusters become too sparse or empty, degrading search accuracy.

        Args:
            vectors (np.ndarray): A 2D numpy array of L2 normalized float32 embeddings.
            nlist (int): The number of clusters to partition the space into.

        Returns:
            faiss.IndexIVFFlat: The built and trained index.
        """
        quantizer = faiss.IndexFlatIP(self.dimension)
        # Using METRIC_INNER_PRODUCT since vectors are L2 normalized.
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        # IVF indexes must be trained before adding vectors
        if not index.is_trained:
            index.train(vectors)

        index.add(vectors)
        return index

    def build_hnsw(self, vectors: np.ndarray, M: int = 16, ef: int = 200) -> faiss.IndexHNSWFlat:
        """
        Builds an IndexHNSWFlat (Hierarchical Navigable Small World) graph-based index.

        Academic note on 'M' and 'ef':
        'M' specifies the maximum number of outgoing connections (edges) in the graph per node.
        A higher M improves recall and search speed but increases memory consumption linearly.
        'ef' (or efConstruction/efSearch) dictates the size of the dynamic list of nearest
        neighbors used during the graph construction and search phases. Increasing ef
        provides higher accuracy (recall) at the cost of significantly higher build and
        search times. HNSW is generally the fastest for search among approximate methods
        but consumes the most memory.

        Args:
            vectors (np.ndarray): A 2D numpy array of L2 normalized float32 embeddings.
            M (int): Number of connections per node.
            ef (int): Size of the dynamic list for the nearest neighbors.

        Returns:
            faiss.IndexHNSWFlat: The built index.
        """
        index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef
        index.hnsw.efSearch = ef
        index.add(vectors)
        return index

    def save_index(self, index: faiss.Index, path: str) -> None:
        """
        Saves the FAISS index to the specified path.

        Args:
            index (faiss.Index): The built FAISS index.
            path (str): The file path where the index will be saved (e.g., .faiss format).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)
        # print(f"Index successfully saved to {path}")

    def load_index(self, path: str) -> faiss.Index:
        """
        Loads a saved FAISS index from disk.

        Args:
            path (str): The file path to the saved FAISS index.

        Returns:
            faiss.Index: The loaded FAISS index.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index file not found at {path}")
        return faiss.read_index(path)

    def search(self, index: faiss.Index, query_vec: np.ndarray, k: int = 5) -> tuple:
        """
        Searches the FAISS index for the k-nearest neighbors to the query vector.

        Args:
            index (faiss.Index): The FAISS index to search.
            query_vec (np.ndarray): A 2D numpy array of the query vectors.
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            tuple: A tuple containing:
                - distances (np.ndarray): The distances (similarities) to the nearest neighbors.
                - indices (np.ndarray): The IDs of the nearest neighbors.
        """
        distances, indices = index.search(query_vec, k)
        return distances, indices
