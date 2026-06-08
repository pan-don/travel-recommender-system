from flask import Blueprint, request, jsonify, current_app, render_template, Response
import time
import csv
import io
from src.utils import (
    preprocess_query, measure_time, q2_before, q2_after, q3_before, q3_after,
    q4_before, q4_after, q5_before, q5_after, q6_before, q6_after, q7_before, q7_after
)

bp = Blueprint('api', __name__)

def get_engine():
    if not current_app.query_engine:
        raise RuntimeError("QueryEngine is not initialized.")
    return current_app.query_engine

def format_response(results, start_time):
    query_time_ms = int((time.time() - start_time) * 1000)
    return jsonify({
        "results": results,
        "query_time_ms": query_time_ms,
        "total_found": len(results)
    })

def handle_error(e, code=500):
    return jsonify({"error": str(e), "code": code}), code

# --- HTML Routes ---
@bp.route('/')
def index():
    engine = get_engine()
    categories = sorted(engine.metadata['Category'].dropna().unique().tolist())
    cities = ["Jakarta", "Yogyakarta", "Bandung", "Surabaya", "Semarang", "Bali"]
    return render_template('index.html', categories=categories, cities=cities)

@bp.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('index.html')

@bp.route('/performance', methods=['GET'])
def performance():
    return render_template('performance.html')

# --- New HTML Search Routes for Benchmarking & Render ---

def _render_result(results, method_name, query_text, params, before_ms, after_ms):
    return render_template("result.html",
        results=results,
        query_info={
            "method": method_name,
            "query_text": query_text,
            "params": params,
            "query_time_ms": after_ms,
            "total_found": len(results)
        },
        before_time_ms=before_ms,
        after_time_ms=after_ms,
        improvement_pct=round((before_ms - after_ms) / before_ms * 100, 1) if before_ms else 0.0
    )

@bp.route('/search/q1', methods=['POST'])
def search_q1_html():
    engine = get_engine()
    query = request.form.get("query", "")
    k = int(request.form.get("k", 5))

    _, before_ms = measure_time(engine.q1_semantic_search, query, k)

    results, after_ms = measure_time(engine.q1_semantic_search, preprocess_query(query), k)

    return _render_result(results, "Q1 · Semantik", query, {"k": k}, before_ms, after_ms)

@bp.route('/search/q2', methods=['POST'])
def search_q2_html():
    engine = get_engine()
    query = request.form.get("query", "")
    category = request.form.get("category", "")
    k = int(request.form.get("k", 5))

    _, before_ms = measure_time(q2_before, engine, query, category, k)

    cat_idx = current_app.config["optimized"]["category_indexes"]
    results, after_ms = measure_time(q2_after, engine, query, category, k, cat_idx)

    return _render_result(results, "Q2 · Kategori", query, {"category": category, "k": k}, before_ms, after_ms)

@bp.route('/search/q3', methods=['POST'])
def search_q3_html():
    engine = get_engine()
    query = request.form.get("query", "")
    cities = request.form.getlist("cities")
    k = int(request.form.get("k", 5))

    if not cities:
        # Fallback if empty
        cities = ["Jakarta", "Yogyakarta", "Bandung", "Surabaya", "Semarang", "Bali"]

    _, before_ms = measure_time(q3_before, engine, query, cities, k)

    results, after_ms = measure_time(q3_after, engine, query, cities, k)

    return _render_result(results, "Q3 · Kota", query, {"cities": cities, "k": k}, before_ms, after_ms)

@bp.route('/search/q4', methods=['POST'])
def search_q4_html():
    engine = get_engine()
    query = request.form.get("query", "")
    k = int(request.form.get("k", 20))

    _, before_ms = measure_time(q4_before, engine, query, k)

    city_idx = current_app.config["optimized"]["city_indexes"]
    results, after_ms = measure_time(q4_after, engine, query, k, city_idx)

    return _render_result(results, "Q4 · Luas", query, {"k": k}, before_ms, after_ms)

@bp.route('/search/q5', methods=['POST'])
def search_q5_html():
    engine = get_engine()
    dest_id_name = request.form.get("destination_id", "")
    k = int(request.form.get("k", 5))

    # Try finding the destination by name to get the ID if text was passed
    dest_id = dest_id_name
    match = engine.metadata[engine.metadata['Place_Name'].str.lower() == dest_id_name.lower()]
    if not match.empty:
        dest_id = str(match.iloc[0]['Place_Id'])
    elif not dest_id in engine.id_map:
        # Fallback search if ID doesn't exist
        dest_id = list(engine.id_map.keys())[0] if engine.id_map else ""

    if not dest_id:
        return _render_result([], "Q5 · Mirip Destinasi", dest_id_name, {"k": k}, 0.0, 0.0)

    try:
        _, before_ms = measure_time(q5_before, engine, dest_id, k)
        results, after_ms = measure_time(q5_after, engine, dest_id, k)
    except Exception as e:
        results = []
        before_ms = 0.0
        after_ms = 0.0

    return _render_result(results, "Q5 · Mirip Destinasi", dest_id_name, {"k": k, "id": dest_id}, before_ms, after_ms)

@bp.route('/search/q6', methods=['POST'])
def search_q6_html():
    engine = get_engine()
    q1 = request.form.get("query1", "")
    q2 = request.form.get("query2", "")
    q3 = request.form.get("query3", "")
    k = int(request.form.get("k", 5))

    queries = [q for q in [q1, q2, q3] if q]
    while len(queries) < 3:
        queries.append(queries[-1] if queries else "wisata")

    _, before_ms = measure_time(q6_before, engine, queries, k)
    results, after_ms = measure_time(q6_after, engine, queries, k)

    return _render_result(results, "Q6 · Multi-Query", " | ".join(queries), {"k": k}, before_ms, after_ms)

@bp.route('/search/q7', methods=['POST'])
def search_q7_html():
    engine = get_engine()
    query = request.form.get("query", "")
    k = int(request.form.get("k", 5))

    _, before_ms = measure_time(q7_before, engine, query, k, current_app.config["optimized"])

    pca_model = current_app.config["optimized"]["pca_model"]
    pca_index = current_app.config["optimized"]["pca_index"]
    results, after_ms = measure_time(q7_after, engine, query, k, pca_model, pca_index)

    return _render_result(results, "Q7 · PCA", query, {"k": k}, before_ms, after_ms)

# --- Benchmark Routes ---

TEST_QUERIES = {
    "q1": {"query": "pantai yang tenang dan cocok untuk snorkeling bersama keluarga", "k": 5},
    "q2": {"query": "warisan budaya dan seni tradisional", "category": "Budaya", "k": 5},
    "q3": {"query": "taman bermain modern untuk anak", "allowed_cities": ["Jakarta", "Surabaya"], "k": 5},
    "q4": {"query": "destinasi wisata populer dan terkenal di Indonesia", "k": 20},
    "q5": {"destination_id": "Candi Borobudur", "k": 5},
    "q6": {"queries": ["wisata alam pegunungan", "hiking dan trekking", "view sunrise indah"], "k": 5},
    "q7": {"query": "air terjun yang indah dan tersembunyi", "k": 5}
}

def _calc_precision(res_base, res_test):
    base_ids = [r['id'] for r in res_base]
    test_ids = [r['id'] for r in res_test]
    if not base_ids:
        return 0.0
    overlap = len(set(base_ids).intersection(set(test_ids)))
    return overlap / len(base_ids)

@bp.route('/api/benchmark', methods=['GET'])
def run_benchmark():
    engine = get_engine()
    n_runs = 20

    cat_idx = current_app.config["optimized"]["category_indexes"]
    city_idx = current_app.config["optimized"]["city_indexes"]
    pca_model = current_app.config["optimized"]["pca_model"]
    pca_index = current_app.config["optimized"]["pca_index"]

    benchmark_results = {}
    total_saved_ms = 0.0

    # helper for precision id lookup
    def get_dest_id(name):
        match = engine.metadata[engine.metadata['Place_Name'].str.lower() == name.lower()]
        if not match.empty:
            return str(match.iloc[0]['Place_Id'])
        return list(engine.id_map.keys())[0] if engine.id_map else ""

    # Q1
    q1_params = TEST_QUERIES["q1"]
    b_ms_list, a_ms_list = [], []
    for _ in range(n_runs):
        res_b, ms = measure_time(engine.q1_semantic_search, q1_params["query"], q1_params["k"])
        b_ms_list.append(ms)
        res_a, ms = measure_time(engine.q1_semantic_search, preprocess_query(q1_params["query"]), q1_params["k"])
        a_ms_list.append(ms)
    b_avg = round(sum(b_ms_list) / n_runs, 2)
    a_avg = round(sum(a_ms_list) / n_runs, 2)
    benchmark_results["q1"] = {
        "method": "Q1 · Semantik",
        "before_ms": b_avg,
        "after_ms": a_avg,
        "improvement_pct": round((b_avg - a_avg) / b_avg * 100, 1),
        "precision_overlap": _calc_precision(res_b, res_a)
    }

    # Q2
    q2_params = TEST_QUERIES["q2"]
    b_ms_list, a_ms_list = [], []
    for _ in range(n_runs):
        res_b, ms = measure_time(q2_before, engine, q2_params["query"], q2_params["category"], q2_params["k"])
        b_ms_list.append(ms)
        res_a, ms = measure_time(q2_after, engine, q2_params["query"], q2_params["category"], q2_params["k"], cat_idx)
        a_ms_list.append(ms)
    b_avg = round(sum(b_ms_list) / n_runs, 2)
    a_avg = round(sum(a_ms_list) / n_runs, 2)
    benchmark_results["q2"] = {
        "method": "Q2 · Kategori",
        "before_ms": b_avg,
        "after_ms": a_avg,
        "improvement_pct": round((b_avg - a_avg) / b_avg * 100, 1) if b_avg > 0 else 0,
        "precision_overlap": _calc_precision(res_b, res_a)
    }

    # Q3
    q3_params = TEST_QUERIES["q3"]
    b_ms_list, a_ms_list = [], []
    for _ in range(n_runs):
        res_b, ms = measure_time(q3_before, engine, q3_params["query"], q3_params["allowed_cities"], q3_params["k"])
        b_ms_list.append(ms)
        res_a, ms = measure_time(q3_after, engine, q3_params["query"], q3_params["allowed_cities"], q3_params["k"])
        a_ms_list.append(ms)
    b_avg = round(sum(b_ms_list) / n_runs, 2)
    a_avg = round(sum(a_ms_list) / n_runs, 2)
    benchmark_results["q3"] = {
        "method": "Q3 · Kota",
        "before_ms": b_avg,
        "after_ms": a_avg,
        "improvement_pct": round((b_avg - a_avg) / b_avg * 100, 1) if b_avg > 0 else 0,
        "precision_overlap": _calc_precision(res_b, res_a)
    }

    # Q4
    q4_params = TEST_QUERIES["q4"]
    b_ms_list, a_ms_list = [], []
    for _ in range(n_runs):
        res_b, ms = measure_time(q4_before, engine, q4_params["query"], q4_params["k"])
        b_ms_list.append(ms)
        res_a, ms = measure_time(q4_after, engine, q4_params["query"], q4_params["k"], city_idx)
        a_ms_list.append(ms)
    b_avg = round(sum(b_ms_list) / n_runs, 2)
    a_avg = round(sum(a_ms_list) / n_runs, 2)
    benchmark_results["q4"] = {
        "method": "Q4 · Luas",
        "before_ms": b_avg,
        "after_ms": a_avg,
        "improvement_pct": round((b_avg - a_avg) / b_avg * 100, 1) if b_avg > 0 else 0,
        "precision_overlap": _calc_precision(res_b, res_a)
    }

    # Q5
    q5_params = TEST_QUERIES["q5"]
    b_ms_list, a_ms_list = [], []
    dest_id = get_dest_id(q5_params["destination_id"])
    for _ in range(n_runs):
        res_b, ms = measure_time(q5_before, engine, dest_id, q5_params["k"])
        b_ms_list.append(ms)
        res_a, ms = measure_time(q5_after, engine, dest_id, q5_params["k"])
        a_ms_list.append(ms)
    b_avg = round(sum(b_ms_list) / n_runs, 2)
    a_avg = round(sum(a_ms_list) / n_runs, 2)
    benchmark_results["q5"] = {
        "method": "Q5 · Mirip",
        "before_ms": b_avg,
        "after_ms": a_avg,
        "improvement_pct": round((b_avg - a_avg) / b_avg * 100, 1) if b_avg > 0 else 0,
        "precision_overlap": _calc_precision(res_b, res_a)
    }

    # Q6
    q6_params = TEST_QUERIES["q6"]
    b_ms_list, a_ms_list = [], []
    for _ in range(n_runs):
        res_b, ms = measure_time(q6_before, engine, q6_params["queries"], q6_params["k"])
        b_ms_list.append(ms)
        res_a, ms = measure_time(q6_after, engine, q6_params["queries"], q6_params["k"])
        a_ms_list.append(ms)
    b_avg = round(sum(b_ms_list) / n_runs, 2)
    a_avg = round(sum(a_ms_list) / n_runs, 2)
    benchmark_results["q6"] = {
        "method": "Q6 · Multi-Query",
        "before_ms": b_avg,
        "after_ms": a_avg,
        "improvement_pct": round((b_avg - a_avg) / b_avg * 100, 1) if b_avg > 0 else 0,
        "precision_overlap": _calc_precision(res_b, res_a)
    }

    # Q7
    q7_params = TEST_QUERIES["q7"]
    b_ms_list, a_ms_list = [], []
    for _ in range(n_runs):
        res_b, ms = measure_time(q7_before, engine, q7_params["query"], q7_params["k"], current_app.config["optimized"])
        b_ms_list.append(ms)
        res_a, ms = measure_time(q7_after, engine, q7_params["query"], q7_params["k"], pca_model, pca_index)
        a_ms_list.append(ms)
    b_avg = round(sum(b_ms_list) / n_runs, 2)
    a_avg = round(sum(a_ms_list) / n_runs, 2)
    benchmark_results["q7"] = {
        "method": "Q7 · PCA",
        "before_ms": b_avg,
        "after_ms": a_avg,
        "improvement_pct": round((b_avg - a_avg) / b_avg * 100, 1) if b_avg > 0 else 0,
        "precision_overlap": _calc_precision(res_b, res_a)
    }

    improvements = []
    total_saved = 0.0
    fastest = ("None", float('inf'))
    biggest_gain = ("None", float('-inf'))

    for q_key, q_data in benchmark_results.items():
        improvements.append(q_data["improvement_pct"])
        saved = q_data["before_ms"] - q_data["after_ms"]
        total_saved += saved

        if q_data["after_ms"] < fastest[1]:
            fastest = (q_key, q_data["after_ms"])

        if q_data["improvement_pct"] > biggest_gain[1]:
            biggest_gain = (q_key, q_data["improvement_pct"])

    avg_imp = sum(improvements) / len(improvements) if improvements else 0.0

    benchmark_results["summary"] = {
        "avg_improvement_pct": round(avg_imp, 1),
        "fastest_query": f"{fastest[0].upper()} ({fastest[1]} ms)",
        "biggest_gain": f"{biggest_gain[0].upper()} ({biggest_gain[1]}%)",
        "total_saved_ms": round(total_saved, 2)
    }

    current_app.config["last_benchmark"] = benchmark_results
    return jsonify(benchmark_results)

@bp.route('/api/benchmark/csv', methods=['GET'])
def download_benchmark_csv():
    if "last_benchmark" not in current_app.config:
        return handle_error("Run benchmark first.", 400)

    bench_data = current_app.config["last_benchmark"]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["query_method", "before_ms", "after_ms", "improvement_pct", "precision_overlap"])

    for k in ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]:
        row = bench_data[k]
        writer.writerow([row["method"], row["before_ms"], row["after_ms"], row["improvement_pct"], row["precision_overlap"]])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=benchmark.csv"}
    )

# --- API Routes ---
@bp.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "engine_loaded": current_app.query_engine is not None})

@bp.route('/api/destinations', methods=['GET'])
def get_destinations():
    try:
        engine = get_engine()
        # Extract unique cities and categories for the frontend
        cities = sorted(engine.metadata['City'].dropna().unique().tolist())
        categories = sorted(engine.metadata['Category'].dropna().unique().tolist())

        # We can also return a list of items for Q5 (Item-based search dropdown)
        items = engine.metadata[['Place_Id', 'Place_Name']].to_dict('records')

        return jsonify({
            "cities": cities,
            "categories": categories,
            "items": items
        })
    except Exception as e:
        return handle_error(e)

@bp.route('/api/search', methods=['POST'])
def search_q1():
    start_time = time.time()
    try:
        data = request.json or {}
        query = data.get('query')
        k = data.get('k', 5)

        if not query:
            return handle_error("Missing 'query' parameter", 400)

        results = get_engine().q1_semantic_search(query, k=int(k))
        return format_response(results, start_time)
    except Exception as e:
        return handle_error(e)

@bp.route('/api/search/category', methods=['POST'])
def search_q2():
    start_time = time.time()
    try:
        data = request.json or {}
        query = data.get('query')
        category = data.get('category')
        k = data.get('k', 5)

        if not query or not category:
            return handle_error("Missing 'query' or 'category' parameter", 400)

        results = get_engine().q2_category_filter(query, category, k=int(k))
        return format_response(results, start_time)
    except Exception as e:
        return handle_error(e)

@bp.route('/api/search/city', methods=['POST'])
def search_q3():
    start_time = time.time()
    try:
        data = request.json or {}
        query = data.get('query')
        cities = data.get('cities', [])
        k = data.get('k', 5)

        if not query or not cities:
            return handle_error("Missing 'query' or 'cities' parameter", 400)

        if isinstance(cities, str):
            cities = [cities]

        results = get_engine().q3_city_semantic_filter(query, cities, k=int(k))
        return format_response(results, start_time)
    except Exception as e:
        return handle_error(e)

@bp.route('/api/search/rating', methods=['POST'])
def search_q4():
    # As discussed, rating is ignored due to dataset constraint. Acting as broad search.
    start_time = time.time()
    try:
        data = request.json or {}
        query = data.get('query')
        k = data.get('k', 50)

        if not query:
            return handle_error("Missing 'query' parameter", 400)

        results = get_engine().q4_broad_semantic_search(query, k=int(k))
        return format_response(results, start_time)
    except Exception as e:
        return handle_error(e)

@bp.route('/api/similar/<destination_id>', methods=['GET'])
def search_q5(destination_id):
    start_time = time.time()
    try:
        k = request.args.get('k', 5)
        results = get_engine().q5_item_based_search(str(destination_id), k=int(k))
        return format_response(results, start_time)
    except ValueError as e:
        return handle_error(e, 404)
    except Exception as e:
        return handle_error(e)

@bp.route('/api/search/ensemble', methods=['POST'])
def search_q6():
    start_time = time.time()
    try:
        data = request.json or {}
        queries = data.get('queries', [])
        k = data.get('k', 5)

        if not queries or not isinstance(queries, list) or len(queries) != 3:
            return handle_error("Must provide an array of exactly 3 'queries'", 400)

        results = get_engine().q6_multi_query_ensemble(queries, k=int(k))
        return format_response(results, start_time)
    except Exception as e:
        return handle_error(e)

@bp.route('/api/search/pca', methods=['POST'])
def search_q7():
    start_time = time.time()
    try:
        data = request.json or {}
        query = data.get('query')
        k = data.get('k', 5)

        if not query:
            return handle_error("Missing 'query' parameter", 400)

        results = get_engine().q7_pca_reduced_query(query, k=int(k))
        return format_response(results, start_time)
    except Exception as e:
        return handle_error(e)
