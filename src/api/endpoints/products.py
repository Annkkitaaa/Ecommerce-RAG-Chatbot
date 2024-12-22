from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import pandas as pd
from ...config import Settings

router = APIRouter()
settings = Settings()

# Load product data
PRODUCT_DF = pd.read_csv(settings.PRODUCT_DATA_PATH)
PRODUCT_DF.fillna('', inplace=True)

@router.get("/search", response_model=List[Dict[str, Any]])
async def search_products(
    query: str = Query(..., min_length=2),
    category: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None,
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Search products with various filters
    """
    # Start with all products
    filtered_products = PRODUCT_DF.copy()
    
    # Apply search query across multiple fields
    if query:
        search_mask = (
            filtered_products['Product_Title'].str.contains(query, case=False, na=False) |
            filtered_products['Description'].str.contains(query, case=False, na=False) |
            filtered_products['Category'].str.contains(query, case=False, na=False)
        )
        filtered_products = filtered_products[search_mask]
    
    # Apply category filter
    if category:
        filtered_products = filtered_products[
            filtered_products['Category'].str.contains(category, case=False, na=False)
        ]
    
    # Apply rating filter
    if min_rating is not None:
        filtered_products = filtered_products[
            filtered_products['Rating'] >= min_rating
        ]
    
    # Apply price filter
    if max_price is not None:
        filtered_products = filtered_products[
            filtered_products['Price'] <= max_price
        ]
    
    if filtered_products.empty:
        raise HTTPException(
            status_code=404,
            detail="No products found matching the criteria"
        )
    
    # Sort by relevance (currently using rating as a proxy)
    filtered_products = filtered_products.sort_values('Rating', ascending=False)
    
    # Limit results
    filtered_products = filtered_products.head(limit)
    
    return filtered_products.to_dict('records')

@router.get("/category/{category}", response_model=List[Dict[str, Any]])
async def get_products_by_category(
    category: str,
    limit: int = Query(default=10, ge=1, le=50),
    min_rating: Optional[float] = None
):
    """
    Retrieve products in a specific category
    """
    # Filter by category
    category_products = PRODUCT_DF[
        PRODUCT_DF['Category'].str.contains(category, case=False, na=False)
    ].copy()
    
    if category_products.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No products found in category '{category}'"
        )
    
    # Apply rating filter if specified
    if min_rating is not None:
        category_products = category_products[
            category_products['Rating'] >= min_rating
        ]
    
    # Sort by rating and limit results
    category_products = category_products.sort_values('Rating', ascending=False)
    category_products = category_products.head(limit)
    
    return category_products.to_dict('records')

@router.get("/top-rated", response_model=List[Dict[str, Any]])
async def get_top_rated_products(
    min_rating: float = Query(4.0, ge=0, le=5),
    category: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Get top-rated products with optional category filter
    """
    # Filter by rating
    top_products = PRODUCT_DF[PRODUCT_DF['Rating'] >= min_rating].copy()
    
    # Apply category filter if specified
    if category:
        top_products = top_products[
            top_products['Category'].str.contains(category, case=False, na=False)
        ]
    
    if top_products.empty:
        raise HTTPException(
            status_code=404,
            detail="No products found matching the criteria"
        )
    
    # Sort by rating and limit results
    top_products = top_products.sort_values('Rating', ascending=False)
    top_products = top_products.head(limit)
    
    return top_products.to_dict('records')

@router.get("/recommendations/{product_id}", response_model=List[Dict[str, Any]])
async def get_product_recommendations(
    product_id: int,
    limit: int = Query(default=5, ge=1, le=20)
):
    """
    Get product recommendations based on category and rating
    """
    # Get the target product
    try:
        target_product = PRODUCT_DF[PRODUCT_DF['Product_ID'] == product_id].iloc[0]
    except (IndexError, KeyError):
        raise HTTPException(
            status_code=404,
            detail=f"Product with ID {product_id} not found"
        )
    
    # Find similar products in the same category
    similar_products = PRODUCT_DF[
        (PRODUCT_DF['Category'] == target_product['Category']) &
        (PRODUCT_DF['Product_ID'] != product_id)
    ].copy()
    
    if similar_products.empty:
        raise HTTPException(
            status_code=404,
            detail="No similar products found"
        )
    
    # Sort by rating and limit results
    similar_products = similar_products.sort_values('Rating', ascending=False)
    similar_products = similar_products.head(limit)
    
    return similar_products.to_dict('records')