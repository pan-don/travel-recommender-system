from flask import Flask
from src.query_engine import QueryEngine
import time

def create_app():
    app = Flask(__name__, static_folder='../web/static', template_folder='../web/templates')

    # Initialize singleton engine at startup
    print("Initializing QueryEngine... this might take a few seconds.")
    start_time = time.time()
    try:
        app.query_engine = QueryEngine()
        print(f"QueryEngine initialized in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Failed to initialize QueryEngine: {e}")
        app.query_engine = None

    app.config["optimized"] = {}

    if app.query_engine:
        print("Pre-building optimized sub-indexes...")
        from src.utils import build_category_indexes, build_city_indexes, build_pca_index

        t0 = time.time()
        app.config["optimized"]["category_indexes"] = build_category_indexes(app.query_engine)
        app.config["optimized"]["city_indexes"] = build_city_indexes(app.query_engine)

        # Build 64-component PCA index for Q7
        pca_model, pca_index = build_pca_index(app.query_engine, n_components=64)
        app.config["optimized"]["pca_model"] = pca_model
        app.config["optimized"]["pca_index"] = pca_index
        print(f"Sub-indexes built in {time.time() - t0:.2f} seconds.")

    with app.app_context():
        from . import routes
        app.register_blueprint(routes.bp)

    return app
