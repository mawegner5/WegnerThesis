#!/usr/bin/env python3

import os
from dotenv import load_dotenv

def main():
    # Path to your .env file
    env_path = "/remote_home/WegnerThesis/.env"
    
    # Check if the .env file exists
    if not os.path.exists(env_path):
        print(f"Error: .env file not found at {env_path}")
        return
    
    # Load the .env file
    load_dotenv(env_path)
    
    # Retrieve the OPENAI_API_KEY
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        print("OPENAI_API_KEY successfully loaded:")
        print(api_key)
    else:
        print("Error: OPENAI_API_KEY not found in the .env file.")

if __name__ == "__main__":
    main()
