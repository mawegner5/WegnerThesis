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
threshold = 1.0  # Adjust this as needed to match the binned data

# Load the binned dataset
input_csv_path = f'/root/.ipython/WegnerThesis/data/generated_data/binned_classes_array_{threshold}.csv'
df = pd.read_csv(input_csv_path)

# Function to interact with the OpenAI API to predict the bird species
def predict_species(descriptors):
    prompt = (
        f"Consider these bird characteristics: {descriptors}. "
        "Please analyze and suggest the top 3 most likely bird species. "
        "Respond only with the species names separated by commas."
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
            predicted_species = response['choices'][0]['message']['content'].strip()
            
            # Split the response into a list of species
            species_list = [species.strip() for species in predicted_species.split(",")]
            
            # Validate each species name in the list
            valid_species = []
            for species in species_list:
                # Remove unwanted characters but allow hyphens and apostrophes
                species_clean = re.sub(r"[^A-Za-z\s\-\']", "", species)
                species_clean = species_clean.strip()
                if species_clean and len(species_clean) > 2:
                    valid_species.append(species_clean)
            
            if len(valid_species) >= 1:
                return valid_species
            else:
                print(f"Invalid response: '{predicted_species}'. Retrying...")
                retries += 1
                time.sleep(5)
        except openai.error.APIError as e:
            retries += 1
            print(f"APIError encountered: {e}. Retrying ({retries}/{max_retries})...")
            time.sleep(5)
        except Exception as e:
            retries += 1
            print(f"An error occurred: {e}. Skipping this descriptor.")
            return ["Error"]
    
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
    
    # Check if any of the top predictions match the true species
    if is_correct_prediction(true_species, predicted_species_list):
        results.append([true_species, "Correct"])
    else:
        results.append([true_species, ", ".join(predicted_species_list)])

# Save the results to a new CSV file
output_csv_path = f'/root/.ipython/WegnerThesis/data/generated_data/threshold_{threshold}_LLM_predictions.csv'
results_df = pd.DataFrame(results, columns=['True Species', 'Predicted Species'])
results_df.to_csv(output_csv_path, index=False)

# Calculate and print accuracy (considering any correct prediction as a win)
correct_predictions = results_df[results_df['Predicted Species'] == 'Correct'].shape[0]
accuracy = correct_predictions / len(results_df) * 100
print(f"Accuracy: {accuracy:.2f}%")
