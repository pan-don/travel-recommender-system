# TravelLens: Tourist Destination Recommendation System

## Overview
TravelLens is an ML-powered backend system that recommends Indonesian tourist destinations. It relies on vector similarity using a FAISS vector database to match user queries with the most relevant destinations.

## Tech Stack
- **Framework:** Flask (Python 3.11+)
- **Vector Database:** FAISS (faiss-cpu)
- **Embedding Model:** sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Data Handling:** Pandas, NumPy

## Project Structure
```text
TravelLens/
├── data/
│   ├── index/                     # Stores the built FAISS index file (e.g., .index)
│   └── raw/                       # Original raw CSV data for destinations
│       └── tourism_with_image_clean.csv
├── scripts/                       # Offline scripts for preprocessing and building indexes
│   ├── build_index.py             # Script to build the FAISS index from data
│   └── ingest_data.py             # Script to read and preprocess the CSV dataset
├── src/                           # Main source code directory
│   ├── __init__.py
│   ├── api/                       # Contains web API routing logic
│   │   ├── __init__.py
│   │   └── app.py                 # Flask web application definition and routes
│   ├── core/                      # Contains core ML and data processing logic
│   │   ├── __init__.py
│   │   ├── embedder.py            # SentenceTransformers embedding generation logic
│   │   ├── query_engine.py        # Logic to match incoming requests against the FAISS index
│   │   └── vector_db.py           # FAISS index initialization, storage, and retrieval
│   └── web/                       # Frontend assets
│       ├── static/                # CSS and image assets
│       │   ├── images/
│       │   └── style.css
│       └── templates/             # HTML templates for Flask
│           ├── index.html
│           └── result.html
├── README.md                      # Project documentation
├── requirements.txt               # Pinned package dependencies
└── run.py                         # Entry point script to run the Flask application
```

### Module Descriptions
- **data**: Central repository for raw, ingested datasets and compiled index structures used across models.
- **scripts**: Utility scripts executed offline to set up the data ingestion and indexing processes prior to the server start.
- **src/api**: Flask routing logic, binding the backend ML algorithms to web interface outputs.
- **src/core**: The ML engine grouping embedding generation, querying, and the fast similarity search via FAISS.
- **src/web**: Houses the presentation layer with HTML templates, static images, and CSS.

## Setup
1. Clone the repository and navigate into the folder:
```bash
git clone https://github.com/pan-don/TravelLens.Tourist_Destination_Recommendation_System.git
cd TravelLens.Tourist_Destination_Recommendation_System
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Generate the FAISS index offline by executing the scripts:
```bash
python scripts/ingest_data.py
python scripts/build_index.py
```
2. Run the application:
```bash
python run.py
```
3. Access the web interface at `http://127.0.0.1:5000/`. Enter a city, category, and description to receive customized tourist destination recommendations.

## Query Optimization Results
The system now uses `paraphrase-multilingual-MiniLM-L12-v2` embeddings stored inside a local `FAISS` index. This architectural shift from previous basic vector searches leads to more accurate semantic context understanding and significantly reduces query latency.
