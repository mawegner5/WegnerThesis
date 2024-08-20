import openai
import pandas as pd
import os
from dotenv import load_dotenv
import re
import time

# Load your API key from the environment file
load_dotenv('/root/.ipython/WegnerThesis/.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the threshold hyperparameter
threshold = 10  # Example threshold value; adjust this as needed

# Load the dataset
input_csv_path = f'/root/.ipython/WegnerThesis/data/generated_data/predicted_classes_wordyarray_{threshold}.csv'
df = pd.read_csv(input_csv_path)

# Function to interact with the OpenAI API to predict the bird species
def predict_species(descriptors):
    prompt = (
        f"Consider the following descriptors: {descriptors}. "
        "Please thoroughly analyze all possible bird species and "
        "provide the best guess. Do not rush and consider a wide range "
        "of bird species before making your decision. Respond only with "
        "the species name."
    )
    
    for attempt in range(3):  # Retry up to 3 times if there's an API error or invalid response
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert ornithologist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,  # Increase to allow for more detailed responses
                temperature=0.7,  # Slightly increase temperature for more varied responses
            )
            predicted_species = response['choices'][0]['message']['content'].strip()
            
            # Clean the response to extract the bird species name
            species_name = re.search(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', predicted_species)
            if species_name and len(species_name.group(0).strip()) > 2:
                return species_name.group(0).strip()
            else:
                print(f"Invalid response: {predicted_species}. Retrying...")
                continue
        except openai.error.APIError as e:
            print(f"APIError encountered: {e}. Retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying
        except Exception as e:
            print(f"An error occurred: {e}. Skipping this descriptor.")
            return "Error"

    return "API Error"

# Iterate through the dataset and get predictions
results = []
for index, row in df.iterrows():
    true_species = row['bird_species']
    descriptors = row['sentences']
    predicted_species = predict_species(descriptors)
    results.append([true_species, predicted_species])

# Save the results to a new CSV file
output_csv_path = f'/root/.ipython/WegnerThesis/data/generated_data/threshold_{threshold}_LLM_predictions.csv'
results_df = pd.DataFrame(results, columns=['True Species', 'Predicted Species'])
results_df.to_csv(output_csv_path, index=False)

# Calculate and print accuracy
correct_predictions = results_df[results_df['True Species'] == results_df['Predicted Species']].shape[0]
accuracy = correct_predictions / len(results_df) * 100
print(f"Accuracy: {accuracy:.2f}%")
