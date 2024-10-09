import openai
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import re
import time

# Load your API key from the environment file
load_dotenv('/root/.ipython/WegnerThesis/.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Paths to data files
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/'

classes_file = os.path.join(data_dir, 'classes.txt')
attributes_file = os.path.join(data_dir, 'predicates.txt')
predicate_matrix_file = os.path.join(data_dir, 'predicate-matrix-binary.txt')

# Read classes
classes_df = pd.read_csv(classes_file, sep='\t', header=None, names=['class_id', 'class_name'])
classes_df['class_name'] = classes_df['class_name'].str.replace('+', ' ').str.lower()

# Read attributes
attributes_df = pd.read_csv(attributes_file, sep='\t', header=None, names=['attribute_id', 'attribute_name'])
attributes_df['attribute_name'] = attributes_df['attribute_name'].str.lower()

# Read predicate matrix
predicate_matrix = pd.read_csv(predicate_matrix_file, sep=' ', header=None)
predicate_matrix.columns = attributes_df['attribute_name']
predicate_matrix.index = classes_df['class_name']

# Function to interact with the OpenAI API to predict the animal species
def predict_animal(descriptors):
    prompt = (
        f"Based on the following characteristics: {descriptors}\n"
        "Please provide the most likely animal species.\n"
        "Respond ONLY with the animal name."
    )
    
    max_retries = 5
    retries = 0
    
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # or "gpt-3.5-turbo" if you don't have access to GPT-4
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a zoologist with deep expertise in animal species. "
                            "When given animal characteristics, you respond ONLY with the animal name, "
                            "nothing else."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0,  # Deterministic output
            )
            predicted_animal = response['choices'][0]['message']['content'].strip().lower()
            return predicted_animal
        except openai.error.RateLimitError as e:
            print(f"Rate limit error encountered: {e}. Retrying in 20 seconds...")
            time.sleep(20)
            retries += 1
        except openai.error.APIError as e:
            print(f"APIError encountered: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            retries += 1
        except Exception as e:
            print(f"An unexpected error occurred: {e}.")
            return "Error"
    print("Max retries reached. Skipping this descriptor.")
    return "Error"

# Initialize results lists
results = []
descriptors_list = []

# Iterate over each animal
for index, row in classes_df.iterrows():
    animal = row['class_name']
    # Get the attribute values for this animal
    attributes = predicate_matrix.loc[animal]
    
    # Get attributes that are 1 and 0
    positive_attributes = attributes[attributes == 1].index.tolist()
    negative_attributes = attributes[attributes == 0].index.tolist()
    
    # Generate descriptors
    descriptors = f"The animal is {', '.join(positive_attributes)}. It is not {', '.join(negative_attributes)}."
    
    # Save the descriptors along with the animal name
    descriptors_list.append([animal, descriptors])
    
    # Now, use the LLM to predict the animal
    predicted_animal = predict_animal(descriptors)
    print(f"True Animal: {animal}, Predicted Animal: {predicted_animal}")
    
    # Append to results
    results.append([animal, predicted_animal])

# Create DataFrame for descriptors
descriptors_df = pd.DataFrame(descriptors_list, columns=['Animal', 'Descriptors'])

# Save the descriptors to a CSV file
descriptors_csv_path = os.path.join(data_dir, 'descriptors.csv')
descriptors_df.to_csv(descriptors_csv_path, index=False)

# Create a DataFrame with results
results_df = pd.DataFrame(results, columns=['True Animal', 'Predicted Animal'])

# Calculate accuracy
results_df['Correct'] = results_df.apply(lambda row: row['True Animal'] == row['Predicted Animal'], axis=1)
accuracy = results_df['Correct'].mean() * 100
print(f"Accuracy: {accuracy:.2f}%")

# Save the results to a CSV file
results_csv_path = os.path.join(data_dir, 'predictions.csv')
results_df.to_csv(results_csv_path, index=False)
