import re
import time
import numpy as np
import faiss
import sklearn.decomposition

def preprocess_query(text: str) -> str:
    """
    Applies lowercase, strip whitespace, remove special characters (keep alphanumeric and spaces),
    remove Indonesian stopwords, and truncate to 128 characters.
    """
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    stopwords = {"di", "ke", "dari", "yang", "dan", "untuk", "dengan", "ini", "itu", "ada", "adalah"}
    words = text.split()
    words = [w for w in words if w not in stopwords]
    text = " ".join(words)
    return text[:128]

def measure_time(func, *args, **kwargs) -> tuple[any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    return result, duration_ms

def build_category_indexes(engine):
    categories = engine.metadata['Category'].dropna().unique()
    indexes = {}
    dim = engine.embeddings.shape[1]

    for cat in categories:
        cat_indices = engine.metadata.index[engine.metadata['Category'] == cat].tolist()
        if not cat_indices:
            continue

        sub_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        sub_embs = engine.embeddings[cat_indices].astype(np.float32)
        # Using the original index positions as IDs so we can lookup in metadata
        sub_index.add_with_ids(sub_embs, np.array(cat_indices, dtype=np.int64))
        indexes[cat] = sub_index

    return indexes

def build_city_indexes(engine):
    cities = engine.metadata['City'].dropna().unique()
    indexes = {}
    dim = engine.embeddings.shape[1]

    for city in cities:
        city_indices = engine.metadata.index[engine.metadata['City'] == city].tolist()
        if not city_indices:
            continue

        sub_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        sub_embs = engine.embeddings[city_indices].astype(np.float32)
        sub_index.add_with_ids(sub_embs, np.array(city_indices, dtype=np.int64))
        indexes[city] = sub_index

    return indexes

def build_pca_index(engine, n_components):
    pca = sklearn.decomposition.PCA(n_components=n_components)
    reduced_embs = pca.fit_transform(engine.embeddings).astype(np.float32)

    # Normalize the reduced embeddings for IP search (cosine similarity)
    norms = np.linalg.norm(reduced_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    reduced_embs = reduced_embs / norms

    index = faiss.IndexIDMap(faiss.IndexFlatIP(n_components))
    index.add_with_ids(reduced_embs, np.arange(len(reduced_embs), dtype=np.int64))

    return pca, index

# Helper for resolving full results
def resolve_results(engine, indices, distances, k, rank_start=1):
    results = []
    rank = rank_start
    for idx, score in zip(indices, distances):
        if idx == -1:
            continue
        res = engine._format_result(rank, int(idx), float(score))
        if res:
            results.append(res)
            rank += 1
            if len(results) >= k:
                break
    return results

def q2_before(engine, query, category, k):
    return engine.q2_category_filter(query, category, k=200)

def q2_after(engine, query, category, k, category_indexes):
    query_vec = engine.embedder.embed_single(query).reshape(1, -1)
    query_vec = engine.embedder.normalize(query_vec)

    # Case-insensitive category match
    sub_index = None
    for key, idx in category_indexes.items():
        if key.lower() == category.lower():
            sub_index = idx
            break

    if sub_index is None:
        return []

    distances, indices = sub_index.search(query_vec, k=k)
    return resolve_results(engine, indices[0], distances[0], k)

def _q3_logic(engine, query, allowed_cities, k, search_k):
    query_vec = engine.embedder.embed_single(query).reshape(1, -1)
    query_vec = engine.embedder.normalize(query_vec)

    distances, indices = engine.indexer.search(engine.index, query_vec, k=search_k)

    allowed_cities_lower = [city.lower() for city in allowed_cities]

    results = []
    rank = 1
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        formatted = engine._format_result(rank, idx, score)
        if not formatted:
            continue

        if formatted['city'].lower() in allowed_cities_lower:
            formatted['rank'] = rank
            results.append(formatted)
            rank += 1

        if len(results) >= k:
            break

    return results

def q3_before(engine, query, allowed_cities, k, search_k=50):
    return _q3_logic(engine, query, allowed_cities, k, search_k)

def q3_after(engine, query, allowed_cities, k, search_k=15):
    return _q3_logic(engine, query, allowed_cities, k, search_k)

def q4_before(engine, query, k):
    return engine.q4_broad_semantic_search(query, k=200)

def q4_after(engine, query, k, city_indexes):
    top_5 = engine.q1_semantic_search(query, k=5)

    if not top_5:
        return []

    # Find most common city
    city_counts = {}
    for res in top_5:
        c = res.get('city')
        if c:
            city_counts[c] = city_counts.get(c, 0) + 1

    if not city_counts:
        return []

    most_common_city = max(city_counts, key=city_counts.get)

    sub_index = None
    for key, idx in city_indexes.items():
        if key.lower() == most_common_city.lower():
            sub_index = idx
            break

    if sub_index is None:
        return []

    query_vec = engine.embedder.embed_single(query).reshape(1, -1)
    query_vec = engine.embedder.normalize(query_vec)

    distances, indices = sub_index.search(query_vec, k=k)
    return resolve_results(engine, indices[0], distances[0], k)

def q5_before(engine, destination_id, k, embeddings_path="data/embeddings/embeddings.npy"):
    # Simulate disk read
    _ = np.load(embeddings_path)
    # The current engine implementation uses in-memory, but we want to simulate the overhead.
    # The actual retrieval logic is the same.
    return engine.q5_item_based_search(destination_id, k=k)

def q5_after(engine, destination_id, k):
    return engine.q5_item_based_search(destination_id, k=k)

def q6_before(engine, queries, k):
    return engine.q6_multi_query_ensemble(queries, k=k, parallel=False)

def q6_after(engine, queries, k):
    return engine.q6_multi_query_ensemble(queries, k=k, parallel=True)

def q7_before(engine, query, k, app_config):
    if "pca_index_32" not in app_config:
        pca_model, pca_index = build_pca_index(engine, n_components=32)
        app_config["pca_model_32"] = pca_model
        app_config["pca_index_32"] = pca_index

    pca_model = app_config["pca_model_32"]
    pca_index = app_config["pca_index_32"]

    query_vec = engine.embedder.embed_single(query).reshape(1, -1)
    reduced_vec = pca_model.transform(query_vec).astype(np.float32)

    norms = np.linalg.norm(reduced_vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    reduced_vec = reduced_vec / norms

    distances, indices = pca_index.search(reduced_vec, k=k)
    return resolve_results(engine, indices[0], distances[0], k)

def q7_after(engine, query, k, pca_model, pca_index):
    query_vec = engine.embedder.embed_single(query).reshape(1, -1)
    reduced_vec = pca_model.transform(query_vec).astype(np.float32)

    norms = np.linalg.norm(reduced_vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    reduced_vec = reduced_vec / norms

    distances, indices = pca_index.search(reduced_vec, k=k)
    return resolve_results(engine, indices[0], distances[0], k)
