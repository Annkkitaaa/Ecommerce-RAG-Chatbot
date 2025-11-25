from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from ...config import Settings

router = APIRouter()
settings = Settings()

# Load order data
try:
    ORDER_DF = pd.read_csv(settings.ORDER_DATA_PATH)
    # Fill NaN values appropriately by dtype (avoid chained assignment warning)
    for col in ORDER_DF.columns:
        if ORDER_DF[col].dtype == 'object':
            ORDER_DF[col] = ORDER_DF[col].fillna('')
        else:
            ORDER_DF[col] = ORDER_DF[col].fillna(0)
    print(f"Successfully loaded orders data from {settings.ORDER_DATA_PATH}")
except Exception as e:
    print(f"Error loading orders data: {str(e)}")
    ORDER_DF = None

@router.get("/customer/{customer_id}", response_model=List[Dict[str, Any]])
async def get_customer_orders(
    customer_id: int,
    limit: int = Query(default=10, ge=1, le=100)
):
    """Retrieve orders for a specific customer"""
    if ORDER_DF is None:
        raise HTTPException(status_code=500, detail="Order data not loaded")
    
    customer_orders = ORDER_DF[ORDER_DF['Customer_Id'] == customer_id].copy()
    
    if customer_orders.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"No orders found for customer {customer_id}"
        )
    
    # Sort by date descending and limit results
    if 'Order_DateTime' in customer_orders.columns:
        customer_orders = customer_orders.sort_values('Order_DateTime', ascending=False)
    elif 'Order_Date' in customer_orders.columns:
        customer_orders = customer_orders.sort_values('Order_Date', ascending=False)
    
    customer_orders = customer_orders.head(limit)
    
    return customer_orders.to_dict('records')

@router.get("/priority/{priority}", response_model=List[Dict[str, Any]])
async def get_orders_by_priority(
    priority: str,
    limit: int = Query(default=10, ge=1, le=100)
):
    """Retrieve orders with specific priority level"""
    if ORDER_DF is None:
        raise HTTPException(status_code=500, detail="Order data not loaded")
    
    priority_orders = ORDER_DF[
        ORDER_DF['Order_Priority'].str.lower() == priority.lower()
    ].copy()
    
    if priority_orders.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No orders found with priority '{priority}'"
        )
    
    # Sort by date descending and limit results
    if 'Order_DateTime' in priority_orders.columns:
        priority_orders = priority_orders.sort_values('Order_DateTime', ascending=False)
    elif 'Order_Date' in priority_orders.columns:
        priority_orders = priority_orders.sort_values('Order_Date', ascending=False)
    
    priority_orders = priority_orders.head(limit)
    
    return priority_orders.to_dict('records')