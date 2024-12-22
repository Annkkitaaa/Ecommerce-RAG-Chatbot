#!/usr/bin/env python3
import sys
from pathlib import Path
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.rag.assistant import ECommerceRAG
from src.config import Settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatSession:
    def __init__(self):
        """Initialize chat session"""
        settings = Settings()
        self.assistant = ECommerceRAG(
            product_dataset_path=settings.PRODUCT_DATA_PATH,
            order_dataset_path=settings.ORDER_DATA_PATH
        )
        self.customer_id = None

    def process_input(self, user_input: str) -> str:
        """Process user input and return response"""
        user_input = user_input.strip().lower()

        # Handle exit command
        if user_input in ['quit', 'exit', 'bye']:
            return "Goodbye!"

        # Handle set customer command
        if user_input.startswith('set customer'):
            try:
                self.customer_id = int(user_input.split()[-1])
                return f"Customer ID set to: {self.customer_id}"
            except ValueError:
                return "Invalid customer ID. Please use a number."
            except IndexError:
                return "Please provide a customer ID (e.g., 'set customer 12345')"

        # Handle empty input
        if not user_input:
            return "Please type something!"

        # Process query with current customer ID
        try:
            response = self.assistant.process_query(user_input, self.customer_id)
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

def main():
    """Main chat loop"""
    print("Initializing E-commerce Chatbot...")
    
    session = ChatSession()
    
    print("\nChatbot ready! You can:")
    print("1. Ask about products (e.g., 'Show me guitars', 'What microphones do you have?')")
    print("2. Check orders (e.g., 'Show my orders' - requires customer ID)")
    print("3. Get high priority orders (e.g., 'Show high priority orders')")
    print("\nCommands:")
    print("- 'set customer <id>' to set customer ID")
    print("- 'quit' to exit")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            response = session.process_input(user_input)
            print(f"\nChatbot: {response}")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again!")

if __name__ == "__main__":
    main()