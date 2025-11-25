from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_BASE_URL: str = "http://localhost:8000"

    # Data Paths
    DATA_DIR: Path = Path(__file__).parent.parent / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # Use processed data if available, otherwise use raw data
    PRODUCT_DATA_PATH: Path = (
        PROCESSED_DATA_DIR / "processed_products.csv"
        if (PROCESSED_DATA_DIR / "processed_products.csv").exists()
        else RAW_DATA_DIR / "Product_Information_Dataset.csv"
    )
    ORDER_DATA_PATH: Path = (
        PROCESSED_DATA_DIR / "processed_orders.csv"
        if (PROCESSED_DATA_DIR / "processed_orders.csv").exists()
        else RAW_DATA_DIR / "Order_Data_Dataset.csv"
    )

    # Model Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Development Settings
    DEBUG: bool = True
    RELOAD: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}