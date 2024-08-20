import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up your OpenAI API key securely
openai.api_key = os.getenv('OPENAI_API_KEY')

# Test if the API key is loaded correctly
print(f"API Key Loaded: {openai.api_key[:4]}...")

# Test making an API request
try:
    response = openai.Model.list()
    print("API request successful.")
    print("Available models:", response["data"])
except Exception as e:
    print(f"API request failed: {e}")
