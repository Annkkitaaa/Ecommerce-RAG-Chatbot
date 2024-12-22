from setuptools import setup, find_packages

setup(
    name="ecommerce-rag",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.95.1",
        "uvicorn==0.22.0",
        "pandas==2.0.1",
        "requests==2.30.0",
        "python-dotenv==1.0.0",
        "transformers==4.29.2",
        "torch==2.0.1",
        "sentence-transformers==2.2.2"
    ],
)