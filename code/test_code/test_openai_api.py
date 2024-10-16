# test_openai_connection.py
import openai
import os
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv('/root/.ipython/WegnerThesis/.env')
api_key = os.getenv("OPENAI_API_KEY")

# Set up the API key
openai.api_key = api_key

try:
    # Make a simple API call to test the connection
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, OpenAI!"}]
    )
    print("API connection successful!")
    print("Response:", response.choices[0].message['content'])
except openai.error.AuthenticationError:
    print("Authentication error: Check your API key.")
except openai.error.RateLimitError:
    print("Rate limit error: You may have exceeded your quota.")
except Exception as e:
    print(f"An error occurred: {e}")
