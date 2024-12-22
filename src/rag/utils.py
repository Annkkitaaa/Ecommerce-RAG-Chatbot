import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import util

def preprocess_text(text: str) -> str:
    """
    Preprocess text for better matching
    
    Args:
        text: Input text to preprocess
    
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text

def calculate_semantic_similarity(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine similarity between query and documents
    
    Args:
        query_embedding: Query embedding vector
        document_embeddings: Matrix of document embeddings
    
    Returns:
        Array of similarity scores
    """
    return util.cos_sim(query_embedding, document_embeddings)[0]

def format_price(price: float) -> str:
    """
    Format price with proper currency symbol and decimals
    
    Args:
        price: Price value to format
    
    Returns:
        Formatted price string
    """
    return f"${price:,.2f}"

def parse_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse date range strings into datetime objects
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        Tuple of parsed start and end dates
    """
    parsed_start = None
    parsed_end = None
    
    if start_date:
        try:
            parsed_start = datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid start_date format. Use YYYY-MM-DD")
    
    if end_date:
        try:
            parsed_end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid end_date format. Use YYYY-MM-DD")
    
    if parsed_start and parsed_end and parsed_start > parsed_end:
        raise ValueError("start_date cannot be later than end_date")
    
    return parsed_start, parsed_end

def filter_dataframe_by_date(
    df: pd.DataFrame,
    date_column: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Filter DataFrame by date range
    
    Args:
        df: Input DataFrame
        date_column: Name of date column
        start_date: Start date for filtering
        end_date: End date for filtering
    
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if start_date:
        filtered_df = filtered_df[
            pd.to_datetime(filtered_df[date_column]) >= start_date
        ]
    
    if end_date:
        filtered_df = filtered_df[
            pd.to_datetime(filtered_df[date_column]) <= end_date
        ]
    
    return filtered_df

def calculate_order_statistics(orders_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics for orders
    
    Args:
        orders_df: DataFrame containing order data
    
    Returns:
        Dictionary of summary statistics
    """
    stats = {
        "total_orders": len(orders_df),
        "total_sales": float(orders_df['Sales'].sum()),
        "average_order_value": float(orders_df['Sales'].mean()),
        "total_shipping_cost": float(orders_df['Shipping_Cost'].sum()),
        "orders_by_priority": orders_df['Order_Priority'].value_counts().to_dict(),
        "orders_by_category": orders_df['Product_Category'].value_counts().to_dict()
    }
    
    return stats

def format_product_response(product: Dict[str, Any], include_score: bool = False) -> str:
    """
    Format product information into a readable string
    
    Args:
        product: Dictionary containing product information
        include_score: Whether to include similarity score
    
    Returns:
        Formatted product string
    """
    response = []
    response.append(f"- {product['Product_Title']}")
    response.append(f"  Price: {format_price(product['Price'])}")
    response.append(f"  Rating: {product['Rating']:.1f} stars")
    
    if 'Description' in product and product['Description']:
        description = product['Description'][:100]
        response.append(f"  Description: {description}...")
    
    if include_score and 'similarity_score' in product:
        response.append(f"  Relevance: {product['similarity_score']:.2f}")
    
    return "\n".join(response)

def format_order_response(order: Dict[str, Any]) -> str:
    """
    Format order information into a readable string
    
    Args:
        order: Dictionary containing order information
    
    Returns:
        Formatted order string
    """
    response = []
    response.append("Order Details:")
    response.append(f"- Date: {order.get('Order_Date', 'Unknown Date')}")
    response.append(f"- Product: {order.get('Product_Category', 'Unknown Product')}")
    response.append(f"- Total: {format_price(order.get('Sales', 0))}")
    response.append(f"- Shipping: {format_price(order.get('Shipping_Cost', 0))}")
    response.append(f"- Priority: {order.get('Order_Priority', 'N/A')}")
    
    return "\n".join(response)