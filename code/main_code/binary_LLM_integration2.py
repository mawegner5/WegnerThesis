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
threshold = 75.0  # Example threshold value; adjust this as needed

# Load the dataset
input_csv_path = f'/root/.ipython/WegnerThesis/data/generated_data/predicted_classes_wordyarray_{threshold}.csv'
df = pd.read_csv(input_csv_path)

# Function to clean the LLM response and extract only species names
def clean_llm_response(response_text):
    try:
        # Split by commas and strip spaces
        species_list = [species.strip() for species in response_text.split(',')]
        
        # Clean each species name
        cleaned_species_list = []
        for species in species_list:
            # Remove unwanted characters but allow hyphens and apostrophes
            species_clean = re.sub(r"[^A-Za-z\s\-\']", "", species)
            species_clean = species_clean.strip()
            if species_clean:
                cleaned_species_list.append(species_clean)
        
        # Return only the first 3 species
        return cleaned_species_list[:3]
    except Exception as e:
        print(f"Error cleaning response: {e}")
        return []

# Function to interact with the OpenAI API to predict the bird species
def predict_species(descriptors):
    prompt = (
        f"Consider these bird characteristics: {descriptors}. "
        "Please provide the top 3 most likely bird species, separated by commas."
    )
    
    max_retries = 5
    retries = 0
    
    while retries < max_retries:  # Retry up to max_retries times
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an ornithologist with deep expertise in bird species."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7,  # Encourage diverse but reasonable guesses
            )
            raw_response = response['choices'][0]['message']['content'].strip()
            cleaned_response = clean_llm_response(raw_response)
            
            if len(cleaned_response) >= 1:
                return cleaned_response
            else:
                print(f"Unexpected response format: '{raw_response}'. Retrying...")
                retries += 1
                time.sleep(5)
        except openai.error.APIError as e:
            retries += 1
            print(f"APIError encountered: {e}. Retrying ({retries}/{max_retries})...")
            time.sleep(5)
        except Exception as e:
            retries += 1
            print(f"An error occurred: {e}. Retrying ({retries}/{max_retries})...")
            time.sleep(5)
    
    print("Max retries reached. Skipping this descriptor.")
    return ["Error"]

# Function to check if the true species is in the predicted species list
def is_correct_prediction(true_species, predicted_species_list):
    true_species_clean = true_species.lower().strip()
    predicted_species_clean = [species.lower().strip() for species in predicted_species_list]
    return true_species_clean in predicted_species_clean

# Iterate through the dataset and get predictions
results = []
for index, row in df.iterrows():
    true_species = row['bird_species']
    descriptors = row['sentences']
    predicted_species_list = predict_species(descriptors)
    
    results.append([true_species, ", ".join(predicted_species_list)])

# Save the results to a new CSV file
output_csv_path = f'/root/.ipython/WegnerThesis/data/generated_data/threshold_{threshold}_LLM_predictions.csv'
results_df = pd.DataFrame(results, columns=['True Species', 'Predicted Species'])
results_df.to_csv(output_csv_path, index=False)

# Calculate and print accuracy (considering any correct prediction as a win)
def is_prediction_correct(row):
    true_species = row['True Species']
    predicted_species = row['Predicted Species']
    predicted_species_list = [species.strip() for species in predicted_species.split(',')]
    return is_correct_prediction(true_species, predicted_species_list)

correct_predictions = results_df.apply(is_prediction_correct, axis=1).sum()
accuracy = correct_predictions / len(results_df) * 100
print(f"Accuracy: {accuracy:.2f}%")
