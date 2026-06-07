import sys
import os

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.query_engine import QueryEngine

def print_results(title: str, results: list):
    """
    Helper function to cleanly print out the results of a query.
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")

    if not results:
        print("No results found.")
        return

    for res in results:
        print(f"Rank: {res['rank']} | ID: {res['id']} | Score: {res['score']:.4f}")
        print(f"Name: {res['name']}")
        print(f"City: {res['city']} | Category: {res['category']}")
        print("-" * 30)

def main():
    print("Loading QueryEngine... (this may take a few seconds as the model loads)")
    engine = QueryEngine()

    # --- Q1: SEMANTIC SEARCH ---
    q1_text = "tempat bersejarah dengan nilai budaya di jakarta"
    res1 = engine.q1_semantic_search(q1_text, k=3)
    print_results("Q1 - SEMANTIC SEARCH\nQuery: " + q1_text, res1)

    # --- Q2: CATEGORY FILTER ---
    q2_text = "tempat yang bagus untuk anak-anak"
    q2_cat = "Taman Hiburan"
    res2 = engine.q2_category_filter(q2_text, category=q2_cat, k=3)
    print_results(f"Q2 - CATEGORY FILTER\nQuery: '{q2_text}' | Category: '{q2_cat}'", res2)

    # --- Q3: CITY + SEMANTIC FILTER ---
    q3_text = "pantai yang indah untuk bersantai"
    q3_cities = ["Bandung", "Jakarta"] # Using available cities since it's an indonesian dataset, let's see what's in these cities
    res3 = engine.q3_city_semantic_filter(q3_text, allowed_cities=q3_cities, k=3)
    print_results(f"Q3 - CITY + SEMANTIC FILTER\nQuery: '{q3_text}' | Cities: {q3_cities}", res3)

    # --- Q4: BROAD SEMANTIC SEARCH ---
    # As discussed, rating filter removed. Broad semantic search.
    q4_text = "taman nasional alam"
    res4 = engine.q4_broad_semantic_search(q4_text, k=5)  # We will just print top 5 of the 50 returned to keep output clean
    print_results(f"Q4 - BROAD SEMANTIC SEARCH (Showing top 5 of 50)\nQuery: '{q4_text}'", res4[:5])

    # --- Q5: ITEM-BASED SEARCH ---
    # Let's find something similar to place ID "1" (usually Monas)
    q5_id = "1"
    res5 = engine.q5_item_based_search(q5_id, k=3)
    print_results(f"Q5 - ITEM-BASED SEARCH\nFinding similar to ID: {q5_id}", res5)

    # --- Q6: MULTI-QUERY ENSEMBLE ---
    q6_queries = ["pemandangan gunung", "udara sejuk", "hutan pinus"]
    res6 = engine.q6_multi_query_ensemble(q6_queries, k=3)
    print_results(f"Q6 - MULTI-QUERY ENSEMBLE\nQueries: {q6_queries}", res6)

    # --- Q7: PCA-REDUCED QUERY ---
    q7_text = "tempat bersejarah dengan nilai budaya di jakarta" # Same as Q1 for comparison
    res7 = engine.q7_pca_reduced_query(q7_text, k=3)
    print_results("Q7 - PCA-REDUCED QUERY (64 dim)\nQuery: " + q7_text, res7)

    print("\nAll queries executed successfully.")

if __name__ == "__main__":
    main()
