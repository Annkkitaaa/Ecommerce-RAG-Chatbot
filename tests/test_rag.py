import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
from sentence_transformers import SentenceTransformer

from src.rag.assistant import ECommerceRAG
from src.rag.utils import (
    preprocess_text,
    calculate_semantic_similarity,
    format_price,
    parse_date_range,
    filter_dataframe_by_date
)
from src.config import Settings

@pytest.fixture(scope="session")
def sample_data():
    """Create sample data for testing"""
    # Create temporary directory
    tmp_path = Path('tests/temp')
    tmp_path.mkdir(parents=True, exist_ok=True)
    
    # Sample product data
    product_data = {
        'Product_ID': range(1, 6),
        'Product_Title': [
            'Electric Guitar Pro',
            'Bass Guitar Beginner',
            'Guitar Strings Pack',
            'Professional Microphone',
            'Drum Set Complete'
        ],
        'Description': [
            'Professional electric guitar with premium features',
            'Perfect bass guitar for beginners',
            'High-quality guitar strings set',
            'Studio-quality condenser microphone',
            'Complete drum set for professionals'
        ],
        'Category': [
            'Guitars',
            'Guitars',
            'Accessories',
            'Microphones',
            'Drums'
        ],
        'Price': [999.99, 599.99, 29.99, 299.99, 1499.99],
        'Rating': [4.8, 4.5, 4.9, 4.7, 4.6]
    }
    
    # Sample order data
    order_data = {
        'Order_ID': range(1, 6),
        'Customer_Id': [1001, 1001, 1002, 1002, 1003],
        'Product': ['Electric Guitar Pro', 'Guitar Strings Pack', 'Bass Guitar Beginner', 'Professional Microphone', 'Drum Set Complete'],
        'Product_Category': ['Guitars', 'Accessories', 'Guitars', 'Microphones', 'Drums'],
        'Order_Date': [
            '2024-01-01',
            '2024-01-15',
            '2024-01-20',
            '2024-02-01',
            '2024-02-15'
        ],
        'Time': [
            '10:30:00',
            '14:45:00',
            '09:15:00',
            '16:20:00',
            '11:00:00'
        ],
        'Sales': [999.99, 29.99, 599.99, 299.99, 1499.99],
        'Shipping_Cost': [25.00, 5.00, 25.00, 15.00, 50.00],
        'Order_Priority': ['High', 'Low', 'Medium', 'High', 'Medium']
    }
    
    # Create DataFrames
    product_df = pd.DataFrame(product_data)
    order_df = pd.DataFrame(order_data)
    
    # Save to CSV files
    product_path = tmp_path / 'test_products.csv'
    order_path = tmp_path / 'test_orders.csv'
    
    product_df.to_csv(product_path, index=False)
    order_df.to_csv(order_path, index=False)
    
    return {
        'product_path': product_path,
        'order_path': order_path,
        'product_df': product_df,
        'order_df': order_df
    }

@pytest.fixture
def rag_assistant(sample_data):
    """Create RAG assistant instance for testing"""
    assistant = ECommerceRAG(
        product_dataset_path=sample_data['product_path'],
        order_dataset_path=sample_data['order_path'],
        model_name="all-MiniLM-L6-v2"
    )
    return assistant

class TestDataLoading:
    def test_data_loading(self, rag_assistant, sample_data):
        """Test if data is loaded correctly"""
        assert len(rag_assistant.product_df) == len(sample_data['product_df'])
        assert len(rag_assistant.order_df) == len(sample_data['order_df'])
        assert all(col in rag_assistant.product_df.columns 
                  for col in ['Product_ID', 'Product_Title', 'Price'])
        assert all(col in rag_assistant.order_df.columns 
                  for col in ['Order_ID', 'Customer_Id', 'Sales'])

    def test_data_preprocessing(self, rag_assistant):
        """Test data preprocessing steps"""
        # Check if NaN values are handled
        assert not rag_assistant.product_df.isna().any().any()
        assert not rag_assistant.order_df.isna().any().any()
        
        # Check if embeddings are created
        assert hasattr(rag_assistant, 'product_embeddings')
        assert len(rag_assistant.product_embeddings) == len(rag_assistant.product_df)
        assert isinstance(rag_assistant.product_embeddings, np.ndarray)

class TestProductSearch:
    def test_semantic_search(self, rag_assistant):
        """Test semantic search functionality"""
        # Test basic search
        results = rag_assistant.semantic_search("guitar")
        assert len(results) > 0
        assert any("Guitar" in result["Product_Title"] for result in results)
        
        # Test search with specific terms
        results = rag_assistant.semantic_search("professional electric guitar")
        assert results[0]["Product_Title"] == "Electric Guitar Pro"
        
        # Test search with related terms
        results = rag_assistant.semantic_search("beginner bass")
        assert "Bass Guitar Beginner" in [r["Product_Title"] for r in results]
        
        # Test search with non-existent terms
        results = rag_assistant.semantic_search("xyzabc123")
        assert len(results) > 0  # Should return something, but likely low relevance

    def test_product_response_generation(self, rag_assistant):
        """Test product response generation"""
        # Test basic product query
        response = rag_assistant.generate_product_response("guitar strings")
        assert "Guitar Strings Pack" in response
        assert "$29.99" in response
        assert "4.9" in response
        
        # Test query with no results
        response = rag_assistant.generate_product_response("xyzabc123")
        assert "couldn't find any products" in response.lower()
        
        # Test query with multiple matches
        response = rag_assistant.generate_product_response("guitar")
        assert "Electric Guitar" in response
        assert "Bass Guitar" in response

class TestOrderHistory:
    def test_order_retrieval(self, rag_assistant):
        """Test order history retrieval"""
        # Test valid customer ID
        response = rag_assistant.generate_order_response(1001)
        assert "Order Details:" in response
        assert "$999.99" in response
        assert "$29.99" in response
        assert "High" in response
        assert "Low" in response
        
        # Test invalid customer ID
        response = rag_assistant.generate_order_response(9999)
        assert "No order history found" in response
        
        # Test customer with multiple orders
        response = rag_assistant.generate_order_response(1002)
        assert "$599.99" in response
        assert "$299.99" in response

    def test_order_response_formatting(self, rag_assistant):
        """Test order response formatting"""
        response = rag_assistant.generate_order_response(1001)
        assert "Date:" in response
        assert "Total:" in response
        assert "Shipping:" in response
        assert "Priority:" in response

class TestQueryProcessing:
    def test_product_queries(self, rag_assistant):
        """Test different types of product queries"""
        queries = [
            "show me guitars",
            "I want to buy a microphone",
            "looking for drum sets",
            "what guitar strings do you have",
            "search for professional equipment"
        ]
        
        for query in queries:
            response = rag_assistant.process_query(query)
            assert response != ""
            assert "Price:" in response
            assert "Rating:" in response

    def test_order_queries(self, rag_assistant):
        """Test different types of order queries"""
        queries = [
            "show my orders",
            "order history",
            "what have I bought",
            "recent purchases",
            "past orders"
        ]
        
        for query in queries:
            response = rag_assistant.process_query(query, customer_id=1001)
            assert response != ""
            assert "Order Details:" in response

    def test_edge_cases(self, rag_assistant):
        """Test edge cases in query processing"""
        # Empty query
        response = rag_assistant.process_query("")
        assert response != ""
        assert "help you with" in response.lower()
        
        # Very short query
        response = rag_assistant.process_query("a")
        assert "more specific" in response.lower()
        
        # Very long query
        long_query = "guitar " * 50
        response = rag_assistant.process_query(long_query)
        assert len(response) > 0
        
        # Special characters
        response = rag_assistant.process_query("guitar!!!???")
        assert "Guitar" in response
        
        # Mixed case
        response = rag_assistant.process_query("GuItAr StRiNgS")
        assert "Guitar Strings" in response

class TestUtilityFunctions:
    def test_preprocess_text(self):
        """Test text preprocessing utility"""
        assert preprocess_text("  HELLO  world  ") == "hello world"
        assert preprocess_text("") == ""
        assert preprocess_text(None) == ""
        assert preprocess_text("Guitar!!! Strings???") == "guitar!!! strings???"

    def test_format_price(self):
        """Test price formatting utility"""
        assert format_price(999.99) == "$999.99"
        assert format_price(1000) == "$1,000.00"
        assert format_price(0) == "$0.00"
        assert format_price(1234567.89) == "$1,234,567.89"

    def test_date_parsing(self):
        """Test date parsing utility"""
        # Valid dates
        start, end = parse_date_range("2024-01-01", "2024-12-31")
        assert start.year == 2024
        assert end.year == 2024
        
        # Invalid dates
        with pytest.raises(ValueError):
            parse_date_range("invalid", "2024-12-31")
        
        # Reversed dates
        with pytest.raises(ValueError):
            parse_date_range("2024-12-31", "2024-01-01")

def test_cleanup(sample_data):
    """Clean up temporary test files"""
    shutil.rmtree(sample_data['product_path'].parent)