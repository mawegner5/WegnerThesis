import openai
import pandas as pd
import os
from dotenv import load_dotenv
import time
import re

# Load your API key from the environment file
load_dotenv('/root/.ipython/WegnerThesis/.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Paths to data files
attributes_file = '/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/class_attributes.csv'

# Load the attributes DataFrame
df = pd.read_csv(attributes_file)

# Remove the first column if it's unnamed (from CSV indexing)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Get the list of bird species
bird_species = df.iloc[:, 0].tolist()

# Get the list of attributes (excluding the bird species column)
attributes = df.columns.tolist()[1:]

# Initialize a new DataFrame to store LLM responses
llm_df = pd.DataFrame(columns=['bird_species'] + attributes)

# Function to parse LLM response
def parse_llm_response(response_text, attributes):
    # Initialize a dictionary to store attribute values
    attribute_values = {}
    # Split the response into lines
    lines = response_text.strip().split('\n')
    for line in lines:
        # Try to extract attribute and answer
        match = re.match(r'\s*-\s*(.*?)\s*:\s*(yes|no)', line, re.IGNORECASE)
        if match:
            attr = match.group(1).strip()
            answer = match.group(2).strip().lower()
            if attr in attributes:
                attribute_values[attr] = 1 if answer == 'yes' else 0
    return attribute_values

# Function to interact with OpenAI API to ask about all attributes for a bird
def ask_llm_about_attributes(bird_name, attributes):
    prompt = f"For the bird species '{bird_name}', indicate 'yes' or 'no' for each of the following attributes:\n"
    for attribute in attributes:
        prompt += f"- {attribute}\n"
    prompt += "Provide your answers in the format: - attribute: yes or no."

    max_retries = 5
    retries = 0

    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use "gpt-3.5-turbo" if you don't have access to GPT-4
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an ornithologist with expert knowledge of bird species and their attributes. "
                            "When asked about a specific bird and its attributes, you respond with 'yes' or 'no' for each attribute."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                n=1,
                stop=None,
                temperature=0,  # Deterministic output
            )
            answer = response['choices'][0]['message']['content'].strip()
            attribute_values = parse_llm_response(answer, attributes)
            # Check if we got all attributes
            if len(attribute_values) == len(attributes):
                return attribute_values
            else:
                print(f"Incomplete response for '{bird_name}'. Retrying...")
                retries += 1
                time.sleep(2)
        except openai.error.RateLimitError as e:
            print(f"Rate limit error encountered: {e}. Retrying in 10 seconds...")
            time.sleep(10)
            retries += 1
        except openai.error.APIError as e:
            print(f"APIError encountered: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            retries += 1
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            retries += 1
    print(f"Max retries reached for '{bird_name}'. Setting all attribute values to -1.")
    return {attribute: -1 for attribute in attributes}

# Iterate over each bird species
for bird in bird_species:
    print(f"Processing bird species: {bird}")
    bird_row = {'bird_species': bird}
    attribute_values = ask_llm_about_attributes(bird, attributes)
    bird_row.update(attribute_values)
    llm_df = llm_df.append(bird_row, ignore_index=True)

# Save the DataFrame to CSV
output_file = '/root/.ipython/WegnerThesis/data/generated_data/LLM_classified_bird_attributes.csv'
llm_df.to_csv(output_file, index=False)
