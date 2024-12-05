import openai

# Set up your OpenAI API key securely test
openai.api_key = os.getenv('OPENAI_API_KEY')

def test_openai_connection():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, can you respond with a simple 'Hello World'?"}
        ]
    )
    
    reply = response.choices[0].message['content'].strip()
    return reply

# Test the connection
result = test_openai_connection()
print("OpenAI Response:", result)
