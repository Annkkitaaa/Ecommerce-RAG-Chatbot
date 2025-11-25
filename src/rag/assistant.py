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
        self.product_df = pd.read_csv(product_dataset_path)
        self.order_df = pd.read_csv(order_dataset_path)
        self.model = SentenceTransformer(model_name)
        self._preprocess_data()
        self._create_product_embeddings()
    
    def _preprocess_data(self):
        """Preprocess datasets"""
        self.product_df.fillna('', inplace=True)
        self.order_df.fillna('', inplace=True)

        # Handle both raw and processed order data formats
        if 'Order_DateTime' not in self.order_df.columns:
            # Raw format: combine Order_Date and Time
            if 'Order_Date' in self.order_df.columns and 'Time' in self.order_df.columns:
                self.order_df['Order_DateTime'] = pd.to_datetime(
                    self.order_df['Order_Date'].astype(str) + ' ' +
                    self.order_df['Time'].astype(str)
                )
            else:
                logger.warning("Order data missing datetime information")
                return
        else:
            # Processed format: just convert to datetime
            self.order_df['Order_DateTime'] = pd.to_datetime(self.order_df['Order_DateTime'])

        self.order_df = self.order_df.sort_values('Order_DateTime', ascending=False)
    
    def _create_product_embeddings(self):
        """Create product embeddings"""
        texts = self.product_df.apply(
            lambda x: f"{x['Product_Title']} {x['Description']}", 
            axis=1
        ).tolist()
        self.product_embeddings = self.model.encode(texts)
    
    def get_customer_orders(self, customer_id: int) -> List[Dict[str, Any]]:
        """Get orders for a specific customer"""
        customer_orders = self.order_df[self.order_df['Customer_Id'] == customer_id]
        return customer_orders.sort_values('Order_DateTime', ascending=False).to_dict('records')
    
    def get_high_priority_orders(self) -> List[Dict[str, Any]]:
        """Get high priority orders"""
        high_priority = self.order_df[
            self.order_df['Order_Priority'].str.lower() == 'high'
        ]
        return high_priority.sort_values('Order_DateTime', ascending=False).head(5).to_dict('records')
    
    def format_single_order(self, order: Dict[str, Any]) -> str:
        """Format single order details"""
        return (f"Your order was placed on {pd.Timestamp(order['Order_DateTime']).strftime('%Y-%m-%d %H:%M:%S')} "
                f"for '{order['Product']}'. The total amount was ${float(order['Sales']):.2f}, "
                f"with a shipping cost of ${float(order['Shipping_Cost']):.2f}. "
                f"The order priority is marked as '{order['Order_Priority']}'.")
    
    def format_high_priority_orders(self, orders: List[Dict[str, Any]]) -> str:
        """Format high priority orders list"""
        if not orders:
            return "No high priority orders found."
        
        response = "Here are the 5 most recent high-priority orders:\n\n"
        for i, order in enumerate(orders, 1):
            response += (
                f"{i}. On {pd.Timestamp(order['Order_DateTime']).strftime('%Y-%m-%d %H:%M:%S')}, "
                f"{order['Product']} was ordered for ${float(order['Sales']):.2f} "
                f"with a shipping cost of ${float(order['Shipping_Cost']):.2f}. "
                f"(Customer ID: {order['Customer_Id']})\n"
            )
        return response
    
    def format_product_results(self, products: List[Dict[str, Any]]) -> str:
        """Format product results"""
        if not products:
            return "No products found matching your criteria."
        
        response = "Here are some products that might interest you:\n\n"
        for product in products:
            response += (f"● {product['Product_Title']}\n"
                       f"  - Rating: {float(product['Rating']):.1f} stars\n"
                       f"  - Price: ${float(product['Price']):.2f}\n")
            if product.get('Description'):
                response += f"  - Description: {product['Description'][:100]}...\n"
            response += "\n"
        
        return response.strip() + "\nLet me know if you'd like more details!"
    
    def semantic_search(self, query: str, min_rating: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search with rating filter
        """
        query_embedding = self.model.encode(query)
        similarities = np.dot(self.product_embeddings, query_embedding)
        
        # Create DataFrame with similarities
        results_df = self.product_df.copy()
        results_df['similarity'] = similarities
        
        # Apply rating filter if specified
        if min_rating is not None:
            results_df = results_df[results_df['Rating'] >= min_rating]
        
        # Sort by similarity and get top results
        results_df = results_df.sort_values('similarity', ascending=False).head(5)
        
        return results_df.to_dict('records')

    def process_query(self, query: str, customer_id: Optional[int] = None) -> str:
        """Process user query"""
        query = query.lower()
        
        # Extract rating requirement if present
        min_rating = None
        if 'above' in query and any(char.isdigit() for char in query):
            try:
                # Find rating value in query
                rating_idx = query.find('above') + 5
                rating_str = ''.join(c for c in query[rating_idx:] if c.isdigit() or c == '.')
                min_rating = float(rating_str)
            except ValueError:
                pass
        
        # Handle high priority orders query
        if 'high priority' in query or ('recent' in query and 'priority' in query):
            orders = self.get_high_priority_orders()
            return self.format_high_priority_orders(orders)
        
        # Handle regular order queries
        if any(keyword in query for keyword in ['order', 'orders', 'purchase', 'bought']):
            if not customer_id:
                return "Could you please provide your Customer ID?"
            
            orders = self.get_customer_orders(customer_id)
            if not orders:
                return f"No orders found for customer {customer_id}"
            return self.format_single_order(orders[0])
        
        # Handle product queries
        products = self.semantic_search(query, min_rating=min_rating)
        
        # If no products found with rating filter, explain why
        if not products and min_rating is not None:
            return f"No products found with rating above {min_rating}. Try adjusting the rating threshold."
            
        return self.format_product_results(products)
