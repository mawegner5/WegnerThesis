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

def convert_attribute_to_sentence(attribute, value):
    # Define a mapping for attributes to more descriptive sentences
    attribute_mapping = {
        "has_bill_shape": "It has a {} shaped bill.",
        "has_wing_color": "It has {} colored wings.",
        "has_upperparts_color": "Its upperparts are {}.",
        "has_underparts_color": "Its underparts are {}.",
        "has_breast_pattern": "Its breast is {}.",
        "has_back_color": "Its back is {}.",
        "has_tail_shape": "It has a {} shaped tail.",
        "has_upper_tail_color": "Its upper tail is {}.",
        "has_head_pattern": "It has a {} head pattern.",
        "has_breast_color": "Its breast is {}.",
        "has_throat_color": "Its throat is {}.",
        "has_eye_color": "Its eyes are {}.",
        "has_bill_length": "Its bill is {} than its head.",
        "has_forehead_color": "Its forehead is {}.",
        "has_under_tail_color": "Its under tail is {}.",
        "has_nape_color": "Its nape is {}.",
        "has_belly_color": "Its belly is {}.",
        "has_wing_shape": "It has {} wings.",
        "has_size": "It is {} in size.",
        "has_shape": "It is {} in shape.",
        "has_back_pattern": "Its back is {}.",
        "has_tail_pattern": "Its tail is {}.",
        "has_belly_pattern": "Its belly is {}.",
        "has_primary_color": "Its primary color is {}.",
        "has_leg_color": "Its legs are {}.",
        "has_bill_color": "Its bill is {}.",
        "has_crown_color": "Its crown is {}.",
        "has_wing_pattern": "Its wings have a {} pattern."
    }

    # Split the attribute to extract the main part and the detail (if any)
    parts = attribute.split("::")
    attribute_main = parts[0]
    detail = parts[1] if len(parts) > 1 else ""

    # Determine if the attribute is present or not
    presence = "does" if value == "Yes" else "doesn't"

    # Generate the sentence
    if attribute_main in attribute_mapping:
        if "{}" in attribute_mapping[attribute_main]:
            return attribute_mapping[attribute_main].format(detail)
        else:
            return attribute_mapping[attribute_main].replace("{}", presence)
    else:
        # Fallback for any unmapped attributes
        return f"It {presence} have {attribute.replace('_', ' ')}."

def guess_bird_class(attributes, bird_species_list):
    """ Guess the bird species based on attributes """
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
attributes_file = f"/root/.ipython/WegnerThesis/data/generated_data/attributes_fed_to_llm_{threshold}.csv"

# Load the dataset
df = pd.read_csv(input_file)

# Extract the list of bird species
bird_species_list = df.iloc[:, 0].tolist()

# Drop the first column (which contains the species name) and the threshold column
attributes_df = df.drop(['Threshold'], axis=1)

# Initialize lists to store the actual and predicted classes
actual_labels = df.iloc[:, 0].tolist()  # List of actual bird species
predicted_labels = []
attributes_fed = []

# Iterate through each row and make predictions
for index, row in attributes_df.iterrows():
    attributes = row.to_dict()
    
    # Convert each attribute to a natural language sentence
    sentence_list = [convert_attribute_to_sentence(key, 'Yes' if value == 1 else 'No') for key, value in attributes.items()]
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
    'Attributes Fed to LLM': attributes_fed,
    'Predicted Class': predicted_labels
})

# Save the attributes and predictions to a CSV file
attributes_df.to_csv(attributes_file, index=False)

# Calculate and print accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
