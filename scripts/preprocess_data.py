#!/usr/bin/env python3
"""
Data preprocessing script for E-commerce RAG Chatbot
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle
from typing import Tuple, Dict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_datasets(product_path: Path, order_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and perform initial cleaning of datasets
    """
    logger.info("Loading datasets...")
    
    # Load datasets
    product_df = pd.read_csv(product_path)
    order_df = pd.read_csv(order_path)
    
    # Print the columns for verification
    logger.info(f"Product CSV columns: {product_df.columns.tolist()}")
    logger.info(f"Order CSV columns: {order_df.columns.tolist()}")
    
    # Basic cleaning
    product_df.fillna('', inplace=True)
    order_df.fillna('', inplace=True)
    
    return product_df, order_df

def preprocess_product_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess product dataset
    """
    logger.info("Preprocessing product data...")
    
    # Create combined text field for semantic search using actual column names
    df['combined_text'] = (
        df['title'].str.strip() + ' ' +
        df['description'].str.strip() + ' ' +
        df['main_category'].str.strip() + ' ' +
        df['categories'].str.strip()  # Added categories for better search
    )
    
    # Clean numeric fields
    df['Price'] = pd.to_numeric(df['price'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
    df['Rating_Count'] = pd.to_numeric(df['rating_number'], errors='coerce')
    
    # Remove products with invalid prices or ratings
    df = df[df['Price'].notna()]
    df = df[df['Rating'].notna()]
    df = df[df['Price'] > 0]
    df = df[df['Rating'].between(0, 5)]
    
    # Create a unique product ID if not present
    if 'Product_ID' not in df.columns:
        df['Product_ID'] = range(1, len(df) + 1)
    
    # Extract features as a list
    df['feature_list'] = df['features'].apply(lambda x: str(x).split('|') if pd.notnull(x) else [])
    
    # Rename columns to match expected schema
    df = df.rename(columns={
        'title': 'Product_Title',
        'main_category': 'Category',
        'description': 'Description',
        'price': 'Price',
        'average_rating': 'Rating',
        'rating_number': 'Rating_Count',
        'store': 'Store',
        'parent_asin': 'Product_ID'
    })
    
    # Select and reorder columns
    columns_to_keep = [
        'Product_ID', 'Product_Title', 'Description', 'Category',
        'Price', 'Rating', 'Rating_Count', 'Store', 'feature_list',
        'combined_text'
    ]
    
    df = df[columns_to_keep]
    
    return df

def preprocess_order_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess order dataset
    """
    logger.info("Preprocessing order data...")
    
    # Convert date and time fields
    df['Order_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Time'])
    
    # Clean numeric fields
    numeric_columns = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove orders with invalid amounts
    df = df[df['Sales'].notna() & df['Shipping_Cost'].notna()]
    df = df[df['Sales'] > 0]
    
    # Create a unique order ID if not present
    if 'Order_ID' not in df.columns:
        df['Order_ID'] = range(1, len(df) + 1)
    
    # Standardize categorical fields
    df['Order_Priority'] = df['Order_Priority'].str.strip().str.title()
    df['Payment_method'] = df['Payment_method'].str.strip().str.title()
    df['Customer_Login_type'] = df['Customer_Login_type'].str.strip().str.title()
    df['Gender'] = df['Gender'].str.strip().str.title()
    df['Device_Type'] = df['Device_Type'].str.strip().str.title()
    
    # Calculate additional metrics
    df['Total_Amount'] = df['Sales'] * df['Quantity']
    df['Net_Profit'] = df['Profit'] - df['Shipping_Cost']
    
    # Keep only needed columns and reorder
    columns_to_keep = [
        'Order_ID', 'Order_DateTime', 'Customer_Id', 'Gender',
        'Device_Type', 'Customer_Login_type', 'Product_Category',
        'Product', 'Quantity', 'Sales', 'Total_Amount', 'Discount',
        'Profit', 'Net_Profit', 'Shipping_Cost', 'Order_Priority',
        'Payment_method'
    ]
    
    df = df[columns_to_keep]
    
    return df

def create_embeddings(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Create embeddings for product descriptions
    """
    logger.info(f"Creating embeddings using {model_name}...")
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df['combined_text'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    
    return embeddings

def save_processed_data(
    product_df: pd.DataFrame,
    order_df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path
):
    """
    Save processed datasets and embeddings
    """
    logger.info("Saving processed data...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed CSVs
    product_df.to_csv(output_dir / 'processed_products.csv', index=False)
    order_df.to_csv(output_dir / 'processed_orders.csv', index=False)
    
    # Save embeddings
    with open(output_dir / 'product_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save preprocessing info
    info = {
        'timestamp': datetime.now().isoformat(),
        'product_count': len(product_df),
        'order_count': len(order_df),
        'embedding_shape': embeddings.shape,
        'product_columns': product_df.columns.tolist(),
        'order_columns': order_df.columns.tolist()
    }
    
    with open(output_dir / 'preprocessing_info.txt', 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Saved {len(product_df)} products and {len(order_df)} orders")

def main():
    """
    Main preprocessing pipeline
    """
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    # Update paths to point to raw directory
    product_path = raw_dir / 'Product_Information_Dataset.csv'
    order_path = raw_dir / 'Order_Data_Dataset.csv'
    
    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        product_df, order_df = load_datasets(product_path, order_path)
        
        # Preprocess data
        product_df = preprocess_product_data(product_df)
        order_df = preprocess_order_data(order_df)
        
        # Create embeddings
        embeddings = create_embeddings(product_df)
        
        # Save processed data
        save_processed_data(product_df, order_df, embeddings, processed_dir)
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()