import openai
import pandas as pd
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os
import time

# Set the threshold value
threshold = 50  # Change this value as needed

# Load the .env file
load_dotenv()

# Set up your OpenAI API key securely
openai.api_key = os.getenv('OPENAI_API_KEY')

def guess_bird_class(attributes, bird_species_list):
    # Convert bird species list to strings
    bird_species_list_str = [str(bird) for bird in bird_species_list]
    
    # Formulate the prompt for the LLM
    prompt = (
        f"Given these attributes: {attributes}, guess the bird species "
        f"from this list: {', '.join(bird_species_list_str)}. Only return "
        "the exact species name from the list."
    )

    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that guesses bird species based on attributes."},
                    {"role": "user", "content": prompt}
                ]
            )
            guess = response.choices[0]['message']['content'].strip()
            
            # Ensure the response is one of the bird species from the list
            for species in bird_species_list_str:
                if species.lower() in guess.lower():
                    return species
            
            # If no match, select a random bird species from the list as a fallback
            return "Unknown"
        except openai.error.RateLimitError:
            print("Rate limit reached. Waiting before retrying...")
            time.sleep(10)  # Wait for 10 seconds before retrying
        except openai.error.OpenAIError as e:
            print(f"An error occurred: {e}")
            break

# File paths
input_file = f"/root/.ipython/WegnerThesis/data/generated_data/binary_class_attributes_{threshold}.csv"
output_file = f"/root/.ipython/WegnerThesis/data/generated_data/predicted_classes_{threshold}.csv"

# Load the dataset
df = pd.read_csv(input_file)

# Extract the list of bird species
bird_species_list = df.iloc[:, 0].tolist()

# Drop the first column (which contains the species name) and the threshold column
attributes_df = df.drop(['Threshold'], axis=1)

# Initialize lists to store the actual and predicted classesr
actual_labels = df.iloc[:, 0].tolist()  # List of actual bird species
predicted_labels = []

# Iterate through each row and make predictions
for index, row in attributes_df.iterrows():
    attributes = row.to_dict()
    attribute_list = ', '.join([f"{key}: {('Yes' if value == 1 else 'No')}" for key, value in attributes.items()])
    
    # Call the function to guess the bird species
    predicted_class = guess_bird_class(attribute_list, bird_species_list)
    
    predicted_labels.append(predicted_class)

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Actual Class': actual_labels,
    'Predicted Class': predicted_labels
})

# Save the results to a CSV file
results_df.to_csv(output_file, index=False)

# Calculate and print accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
