import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECommerceRAG:
    def __init__(self, 
                 product_dataset_path: str, 
                 order_dataset_path: str,
                 model_name: str = "all-MiniLM-L6-v2"):
        """Initialize RAG system"""
        # Load datasets
        self.product_df = pd.read_csv(product_dataset_path)
        self.order_df = pd.read_csv(order_dataset_path)
        
        # Initialize model
        self.model = SentenceTransformer(model_name)
        
        # Preprocess data
        self._preprocess_data()
        self._create_product_embeddings()

    def _preprocess_data(self):
        """Preprocess the datasets"""
        # Clean up orders
        self.order_df['Order_DateTime'] = pd.to_datetime(self.order_df['Order_DateTime'])
        self.order_df = self.order_df.sort_values('Order_DateTime', ascending=False)
        self.order_df.fillna('', inplace=True)

        # Clean up products
        self.product_df.fillna('', inplace=True)
        
        logger.info(f"Preprocessed {len(self.product_df)} products and {len(self.order_df)} orders")

    def _create_product_embeddings(self):
        texts = self.product_df.apply(
            lambda x: f"{x['Product_Title']} {x['Description']}", 
            axis=1
        ).tolist()
        self.product_embeddings = self.model.encode(texts)

    def get_customer_orders(self, customer_id: int) -> List[Dict[str, Any]]:
        """Get orders for a specific customer"""
        customer_orders = self.order_df[self.order_df['Customer_Id'] == customer_id]
        if customer_orders.empty:
            return []
        return customer_orders.sort_values('Order_DateTime', ascending=False).to_dict('records')

    def get_high_priority_orders(self) -> List[Dict[str, Any]]:
        """Get recent high priority orders"""
        high_priority = self.order_df[
            self.order_df['Order_Priority'].str.lower() == 'high'
        ].sort_values('Order_DateTime', ascending=False).head(5)
        return high_priority.to_dict('records')

    def process_query(self, query: str, customer_id: Optional[int] = None) -> str:
        """Process user query and return response"""
        query = query.lower()
        
        # Check for order-related queries
        is_order_query = any(word in query for word in ['order', 'orders', 'purchase'])
        is_high_priority_query = 'high priority' in query or 'priority' in query
        
        # Handle order queries
        if is_order_query or is_high_priority_query:
            if is_high_priority_query:
                orders = self.get_high_priority_orders()
                if orders:
                    return self._format_high_priority_orders(orders)
                return "No high priority orders found."
            
            if not customer_id:
                return "Could you please provide your Customer ID?"
            
            orders = self.get_customer_orders(customer_id)
            if not orders:
                return f"No orders found for customer {customer_id}"
            
            # Return most recent order
            return self._format_customer_order(orders[0])
        
        # Handle product queries
        products = self.semantic_search(query)
        return self._format_product_results(products)

    def semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """Search for products"""
        query_embedding = self.model.encode(query)
        similarities = np.dot(self.product_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-5:][::-1]
        return [self.product_df.iloc[idx].to_dict() for idx in top_indices]

    def _format_customer_order(self, order: Dict[str, Any]) -> str:
        """Format single order details"""
        date_str = pd.Timestamp(order['Order_DateTime']).strftime('%Y-%m-%d %H:%M:%S')
        return (f"Your order was placed on {date_str} for '{order['Product']}'. "
                f"The total amount was ${float(order['Sales']):.2f}, "
                f"with a shipping cost of ${float(order['Shipping_Cost']):.2f}. "
                f"The order priority is marked as '{order['Order_Priority']}'.")

    def _format_high_priority_orders(self, orders: List[Dict[str, Any]]) -> str:
        """Format high priority orders list"""
        response = "Here are the 5 most recent high-priority orders:\n\n"
        for i, order in enumerate(orders, 1):
            date_str = pd.Timestamp(order['Order_DateTime']).strftime('%Y-%m-%d %H:%M:%S')
            response += (f"{i}. On {date_str}, {order['Product']} was ordered "
                       f"for ${float(order['Sales']):.2f} "
                       f"with a shipping cost of ${float(order['Shipping_Cost']):.2f}. "
                       f"(Customer ID: {order['Customer_Id']})\n")
        return response

    def _format_product_results(self, products: List[Dict[str, Any]]) -> str:
        """Format product search results"""
        if not products:
            return "No products found matching your criteria."
        
        response = "Here are some relevant products:\n\n"
        for product in products:
            response += (f"● {product['Product_Title']}\n"
                       f"  - Rating: {float(product['Rating']):.1f} stars\n"
                       f"  - Price: ${float(product['Price']):.2f}\n")
            if product.get('Description'):
                response += f"  - Description: {product['Description'][:100]}...\n"
            response += "\n"
        
        return response.strip() + "\nLet me know if you'd like more details!"