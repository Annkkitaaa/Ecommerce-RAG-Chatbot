# E-Commerce RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot system for e-commerce product search and order management. The system provides semantic search capabilities for products and handles order queries through a conversational interface.

## Project Overview

This project implements a chatbot that can:
- Search products using semantic similarity
- Handle customer order inquiries
- Process high-priority order queries
- Provide product recommendations
- Interact through a command-line interface

## Project Structure
```
ecommerce_rag/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── .gitignore             # Git ignore file
│
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   │   ├── Product_Information_Dataset.csv
│   │   └── Order_Data_Dataset.csv
│   └── processed/        # Processed data files
│       ├── processed_products.csv
│       ├── processed_orders.csv
│       ├── product_embeddings.pkl
│       └── preprocessing_info.txt
│
├── src/                  # Source code
│   ├── __init__.py
│   ├── config.py        # Configuration settings
│   │
│   ├── api/             # API implementation
│   │   ├── __init__.py
│   │   ├── main.py     # FastAPI main application
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       ├── orders.py
│   │       └── products.py
│   │
│   └── rag/             # RAG implementation
│       ├── __init__.py
│       ├── assistant.py # Main RAG assistant
│       └── utils.py     # Utility functions
│
├── scripts/             # Utility scripts
│   ├── run.py          # Main execution script
│   └── preprocess_data.py  # Data preprocessing script
|   ├── chat.py         
│   └── debug_data.py 
│
└── tests/              # Test files
    ├── __init__.py
    ├── test_api.py    # API tests
    └── test_rag.py    # RAG system tests
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ecommerce_rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Data Setup

1. Place your raw data files in the `data/raw/` directory:
   - `Product_Information_Dataset.csv`
   - `Order_Data_Dataset.csv`

2. Run data preprocessing:
```bash
python scripts/preprocess_data.py
```

## Running the Application

### Starting the API Server
```bash
python scripts/run.py api
```
The API will be available at http://localhost:8000

### Using the Chat Interface
```bash
python scripts/run.py chat
```

Example chat commands:
```
# Set customer ID
set customer 37077

# Product queries
What are the top 5 highly-rated guitar products?
Show me microphones under $200

# Order queries
What are the details of my last order?
Fetch 5 most recent high-priority orders
```

## API Endpoints

### Product Endpoints
- `GET /products/search`: Search products by query
- `GET /products/category/{category}`: Get products by category
- `GET /products/top-rated`: Get top-rated products

### Order Endpoints
- `GET /orders/customer/{customer_id}`: Get customer orders
- `GET /orders/priority/{priority}`: Get orders by priority

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## Components

### RAG Assistant
- Located in `src/rag/assistant.py`
- Handles semantic search and query processing
- Uses sentence transformers for embeddings
- Processes both product and order queries

### API Service
- Located in `src/api/`
- Implements RESTful endpoints
- Handles data validation and error handling
- Provides documentation via Swagger UI

### Data Processing
- Located in `scripts/preprocess_data.py`
- Cleans and preprocesses raw data
- Creates embeddings for semantic search
- Saves processed data for quick access

## Environment Variables

Required environment variables in `.env`:
```
HOST=0.0.0.0
PORT=8000
API_BASE_URL=http://localhost:8000
DATA_DIR=./data
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Dependencies

Key dependencies:
- FastAPI: Web framework
- sentence-transformers: Semantic search
- pandas: Data processing
- pytest: Testing
- uvicorn: ASGI server