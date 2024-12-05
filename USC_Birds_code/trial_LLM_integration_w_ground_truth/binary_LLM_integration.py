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
        # Extract the part after "are:" or similar phrases
        match = re.search(r"(?<=are[:\s]).*", response_text, re.IGNORECASE)
        if match:
            response_text = match.group(0)

        # Split by commas or newlines, strip spaces, and remove non-species text
        species_list = re.findall(r'[A-Za-z\s\-]+', response_text)
        species_list = [species.strip() for species in species_list if len(species.strip()) > 0]

        # Return only the first 3 species
        return species_list[:3]
    except Exception as e:
        print(f"Error cleaning response: {e}")
        return []

# Function to interact with the OpenAI API to predict the bird species
def predict_species(descriptors):
    prompt = (
        f"Consider these bird characteristics: {descriptors}. "
        "Please provide the top 3 most likely bird species, separated by commas."
    )
    
    while True:  # Continue retrying until a valid response is obtained
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an ornithologist with deep expertise in bird species."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7,  # Adjusting to encourage diverse but reasonable guesses
            )
            raw_response = response['choices'][0]['message']['content'].strip()
            cleaned_response = clean_llm_response(raw_response)
            
            if len(cleaned_response) == 3:
                return cleaned_response
            else:
                print(f"Unexpected response format: '{raw_response}'. Cleaning response...")
                return ["Error", "Error", "Error"]
        except openai.error.APIError as e:
            print(f"APIError encountered: {e}. Retrying...")
            time.sleep(5)  # Wait before retrying
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(5)

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
correct_predictions = results_df[results_df['Predicted Species'].apply(lambda x: x.split(", ")[0]) == results_df['True Species']].shape[0]
accuracy = correct_predictions / len(results_df) * 100
print(f"Accuracy: {accuracy:.2f}%")
