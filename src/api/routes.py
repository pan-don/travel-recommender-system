from flask import Blueprint, request, jsonify, current_app, render_template
import time

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
    return render_template('index.html')

@bp.route('/result', methods=['GET', 'POST'])
def result():
    # Deprecated html route, fallback for old form if needed
    return render_template('index.html')

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
