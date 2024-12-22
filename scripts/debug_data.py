from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_order_query():
    """Test order query functionality"""
    # Load order data directly
    processed_dir = project_root / 'data' / 'processed'
    orders_file = processed_dir / 'processed_orders.csv'
    
    print("\nTesting order data access:")
    print(f"Looking for file: {orders_file}")
    
    if not orders_file.exists():
        print("Error: Processed orders file not found!")
        return
        
    # Load and examine orders
    orders_df = pd.read_csv(orders_file)
    orders_df['Order_DateTime'] = pd.to_datetime(orders_df['Order_DateTime'])
    
    # Test specific customer order
    customer_id = 81792
    customer_orders = orders_df[orders_df['Customer_Id'] == customer_id]
    print(f"\nOrders for customer {customer_id}:")
    if customer_orders.empty:
        print("No orders found!")
    else:
        print(customer_orders.sort_values('Order_DateTime', ascending=False)
              [['Order_DateTime', 'Product', 'Sales', 'Order_Priority']].head())
    
    # Test high priority orders
    high_priority = orders_df[
        orders_df['Order_Priority'].str.lower() == 'high'
    ].sort_values('Order_DateTime', ascending=False)
    
    print("\nRecent high priority orders:")
    print(high_priority[['Order_DateTime', 'Product', 'Sales', 'Customer_Id']].head())

if __name__ == "__main__":
    test_order_query()