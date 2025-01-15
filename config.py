from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
OAI_API_KEY = os.getenv('OAI_API_KEY') 