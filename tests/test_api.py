import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.config import Settings
import pandas as pd
from pathlib import Path

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def sample_data():
    """Create sample data for API testing"""
    # Sample product data
    product_data = {
        'Product_ID': [1, 2, 3],
        'Product_Title': ['Electric Guitar', 'Bass Guitar', 'Guitar Strings'],
        'Description': ['Professional electric guitar', 'Bass guitar for beginners', 'High quality strings'],
        'Category': ['Guitars', 'Guitars', 'Accessories'],
        'Price': [999.99, 599.99, 29.99],
        'Rating': [4.8, 4.5, 4.9]
    }
    
    # Sample order data
    order_data = {
        'Order_ID': [1, 2, 3],
        'Customer_Id': [1001, 1001, 1002],
        'Product_Category': ['Guitars', 'Accessories', 'Guitars'],
        'Order_Date': ['2024-01-01', '2024-01-15', '2024-01-20'],
        'Sales': [999.99, 29.99, 599.99],
        'Shipping_Cost': [25.00, 5.00, 25.00],
        'Order_Priority': ['High', 'Low', 'Medium']
    }
    
    # Create temporary CSV files
    tmp_path = Path('tests/temp')
    tmp_path.mkdir(exist_ok=True)
    
    product_df = pd.DataFrame(product_data)
    order_df = pd.DataFrame(order_data)
    
    product_path = tmp_path / 'test_products.csv'
    order_path = tmp_path / 'test_orders.csv'
    
    product_df.to_csv(product_path, index=False)
    order_df.to_csv(order_path, index=False)
    
    # Update settings to use test data
    settings = Settings()
    settings.PRODUCT_DATA_PATH = product_path
    settings.ORDER_DATA_PATH = order_path
    
    return {
        'product_path': product_path,
        'order_path': order_path,
        'settings': settings
    }

def test_root(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "version" in response.json()

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_customer_orders(client):
    """Test customer orders endpoint"""
    # Test valid customer ID
    response = client.get("/orders/customer/1001")
    assert response.status_code == 200
    orders = response.json()
    assert len(orders) == 2
    assert all(order["Customer_Id"] == 1001 for order in orders)
    
    # Test invalid customer ID
    response = client.get("/orders/customer/9999")
    assert response.status_code == 404

def test_get_orders_by_priority(client):
    """Test orders by priority endpoint"""
    # Test valid priority
    response = client.get("/orders/priority/high")
    assert response.status_code == 200
    orders = response.json()
    assert len(orders) > 0
    assert all(order["Order_Priority"].lower() == "high" for order in orders)
    
    # Test invalid priority
    response = client.get("/orders/priority/invalid")
    assert response.status_code == 404

def test_search_products(client):
    """Test product search endpoint"""
    # Test basic search
    response = client.get("/products/search", params={"query": "guitar"})
    assert response.status_code == 200
    products = response.json()
    assert len(products) > 0
    assert all("Guitar" in product["Product_Title"] for product in products)
    
    # Test search with filters
    response = client.get("/products/search", 
                         params={
                             "query": "guitar",
                             "min_rating": 4.5,
                             "max_price": 1000.0
                         })
    assert response.status_code == 200
    products = response.json()
    assert all(product["Rating"] >= 4.5 for product in products)
    assert all(product["Price"] <= 1000.0 for product in products)
    
    # Test no results
    response = client.get("/products/search", params={"query": "piano"})
    assert response.status_code == 404

def test_get_top_rated_products(client):
    """Test top-rated products endpoint"""
    # Test basic request
    response = client.get("/products/top-rated")
    assert response.status_code == 200
    products = response.json()
    assert len(products) > 0
    assert all(product["Rating"] >= 4.0 for product in products)
    
    # Test with minimum rating
    response = client.get("/products/top-rated", params={"min_rating": 4.8})
    assert response.status_code == 200
    products = response.json()
    assert all(product["Rating"] >= 4.8 for product in products)
    
    # Test with category filter
    response = client.get("/products/top-rated", 
                         params={
                             "category": "Guitars",
                             "min_rating": 4.0
                         })
    assert response.status_code == 200
    products = response.json()
    assert all(product["Category"] == "Guitars" for product in products)

def test_get_product_recommendations(client):
    """Test product recommendations endpoint"""
    # Test valid product ID
    response = client.get("/products/recommendations/1")
    assert response.status_code == 200
    products = response.json()
    assert len(products) > 0
    
    # Test invalid product ID
    response = client.get("/products/recommendations/999")
    assert response.status_code == 404

def test_api_error_handling(client):
    """Test API error handling"""
    # Test invalid customer ID format
    response = client.get("/orders/customer/invalid")
    assert response.status_code == 422
    
    # Test invalid date format
    response = client.get("/orders/summary/daily", params={"date": "invalid-date"})
    assert response.status_code == 400
    
    # Test missing required parameters
    response = client.get("/products/search")
    assert response.status_code == 422

def test_cleanup(sample_data):
    """Clean up temporary test files"""
    sample_data['product_path'].unlink()
    sample_data['order_path'].unlink()
    sample_data['product_path'].parent.rmdir()