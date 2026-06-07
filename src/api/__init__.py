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

    with app.app_context():
        from . import routes
        app.register_blueprint(routes.bp)

    return app
