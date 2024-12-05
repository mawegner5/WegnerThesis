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
threshold = 0.25  # Adjust this as needed to match the binned data

# Load the binned dataset
input_csv_path = f'/root/.ipython/WegnerThesis/data/generated_data/binned_classes_array_{threshold}.csv'
df = pd.read_csv(input_csv_path)

# Function to interact with the OpenAI API to predict the bird species
def predict_species(descriptors):
    prompt = (
        f"Consider these bird characteristics: {descriptors}\n"
        "Based on these characteristics, provide a list of the top 3 most likely bird species.\n"
        "Respond ONLY with the species names separated by commas, in the format:\n"
        "Species1, Species2, Species3\n"
        "Do not include any extra text, explanations, or punctuation."
    )

    max_retries = 5
    retries = 0

    while retries < max_retries:  # Retry up to max_retries times
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an ornithologist with deep expertise in bird species. "
                            "When given bird characteristics, you respond ONLY with the species names, "
                            "nothing else."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0,  # Set to 0 for deterministic output
            )
            predicted_species = response['choices'][0]['message']['content'].strip()

            # Use regex to extract species names, ensuring we only get words and spaces
            species_list = re.findall(r'[A-Za-z][A-Za-z\s\-]*[A-Za-z]', predicted_species)
            species_list = [species.strip() for species in species_list if len(species.strip()) > 2]

            if len(species_list) >= 1:
                return species_list[:3]  # Return up to top 3 species
            else:
                print(f"Invalid response: '{predicted_species}'. Retrying...")
                retries += 1
                time.sleep(2)
        except openai.error.RateLimitError as e:
            print(f"Rate limit error encountered: {e}")
            time.sleep(20)  # Wait before retrying
            retries += 1
        except openai.error.APIError as e:
            print(f"APIError encountered: {e}. Retrying...")
            time.sleep(5)  # Wait before retrying
            retries += 1
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Skipping this descriptor.")
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

    # Optional: Print progress every 10 iterations
    if (index + 1) % 10 == 0:
        print(f"Processed {index + 1}/{len(df)} entries.")

# Save the results to a new CSV file
output_csv_path = f'/root/.ipython/WegnerThesis/data/generated_data/threshold_{threshold}_LLM_predictions_binned.csv'
results_df = pd.DataFrame(results, columns=['True Species', 'Predicted Species'])
results_df.to_csv(output_csv_path, index=False)

# Calculate and print accuracy (considering any correct prediction as a win)
correct_predictions = results_df[results_df['Predicted Species'] == 'Correct'].shape[0]
accuracy = correct_predictions / len(results_df) * 100
print(f"Accuracy: {accuracy:.2f}%")
