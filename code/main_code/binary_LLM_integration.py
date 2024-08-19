import openai
import pandas as pd
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os
import time

# Set the threshold value
threshold = 0.5  # Change this value as needed

# Load the .env file
load_dotenv()

# Set up your OpenAI API key securely
openai.api_key = os.getenv('OPENAI_API_KEY')

def convert_attribute_to_sentence(attribute, value, threshold):
    """ Convert each attribute to a natural language sentence. """
    numeric_value = float(value)

    # Determine if the attribute is present or not
    presence = "has" if numeric_value >= threshold else "does not have"

    # Define a mapping for attributes to more descriptive sentences
    attribute_mapping = {
        "has_bill_shape": "a {} shaped bill",
        "has_wing_color": "{} colored wings",
        "has_upperparts_color": "upperparts that are {}",
        "has_underparts_color": "underparts that are {}",
        "has_breast_pattern": "a breast that is {}",
        "has_back_color": "a back that is {}",
        "has_tail_shape": "a {} shaped tail",
        "has_upper_tail_color": "an upper tail that is {}",
        "has_head_pattern": "a {} head pattern",
        "has_breast_color": "a breast that is {}",
        "has_throat_color": "a throat that is {}",
        "has_eye_color": "eyes that are {}",
        "has_bill_length": "a bill that is {} than its head",
        "has_forehead_color": "a forehead that is {}",
        "has_under_tail_color": "an under tail that is {}",
        "has_nape_color": "a nape that is {}",
        "has_belly_color": "a belly that is {}",
        "has_wing_shape": "{} wings",
        "has_size": "a size that is {}",
        "has_shape": "a shape that is {}",
        "has_back_pattern": "a back that is {}",
        "has_tail_pattern": "a tail that is {}",
        "has_belly_pattern": "a belly that is {}",
        "has_primary_color": "a primary color that is {}",
        "has_leg_color": "legs that are {}",
        "has_bill_color": "a bill that is {}",
        "has_crown_color": "a crown that is {}",
        "has_wing_pattern": "wings with a {} pattern"
    }

    # Split the attribute to extract the main part and the detail (if any)
    parts = attribute.split("::")
    attribute_main = parts[0]
    detail = parts[1] if len(parts) > 1 else ""

    # Generate the sentence
    if attribute_main in attribute_mapping:
        sentence_structure = attribute_mapping[attribute_main]
        if detail:
            return f"It {presence} {sentence_structure.format(detail)}."
        else:
            return f"It {presence} {sentence_structure}."
    else:
        # Fallback for any unmapped attributes
        return f"It {presence} {attribute.replace('_', ' ')}."

def guess_bird_class(attributes, bird_species_list):
    """ Guess the bird species based on attributes """
    bird_species_list_str = [str(bird) for bird in bird_species_list]
    
    # Formulate the prompt for the LLM
    prompt = (
        f"Pretend you are an amateur bird enthusiast. Given these attributes: {attributes}, guess the bird species "
        f"from this list: {', '.join(bird_species_list_str)}. Only return the exact species name from the list."
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
input_file = "/root/.ipython/data/class_attributes.csv"
output_file = f"/root/.ipython/WegnerThesis/data/generated_data/predicted_classes_{threshold}.csv"
attributes_file = f"/root/.ipython/WegnerThesis/data/generated_data/attributes_fed_to_llm_{threshold}.csv"

# Load the dataset
df = pd.read_csv(input_file)

# Extract the list of bird species
bird_species_list = df.iloc[:, 0].tolist()

# Drop the first column (which contains the species name)
attributes_df = df.drop(df.columns[0], axis=1)

# Initialize lists to store the actual and predicted classes
actual_labels = df.iloc[:, 0].tolist()  # List of actual bird species
predicted_labels = []
attributes_fed = []

# Iterate through each row and make predictions
for index, row in attributes_df.iterrows():
    attributes = row.to_dict()
    
    # Convert each attribute to a natural language sentence
    sentence_list = [convert_attribute_to_sentence(key, value, threshold) for key, value in attributes.items()]
    attribute_list = ' '.join(sentence_list)
    
    # Append the attributes fed to the LLM to the list
    attributes_fed.append(attribute_list)
    
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

# Create a DataFrame for the attributes fed to the LLM
attributes_df = pd.DataFrame({
    'Attributes Fed to LLM': attributes_fed
})

# Save the attributes fed to the LLM to a CSV file
attributes_df.to_csv(attributes_file, index=False)

# Calculate and print accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
