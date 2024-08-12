import openai
import pandas as pd
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Set up your OpenAI API key securely
openai.api_key = os.getenv('OPENAI_API_KEY')

def guess_bird_class(attributes):
    # Formulate the prompt for the LLM
    prompt = f"Given these attributes, guess the bird species: {attributes}. Only return the guessed species name."
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that guesses bird species based on attributes."},
            {"role": "user", "content": prompt}
        ]
    )
    
    guess = response.choices[0]['message']['content'].strip()
    return guess

# File paths
input_file = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/class_attributes.xlsx"
output_file = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/predicted_classes.csv"

# Load the dataset
df = pd.read_excel(input_file)

# Drop the first column (which contains the actual bird species name) to avoid cheating
actual_classes = df.iloc[:, 0]
attributes_df = df.drop(df.columns[0], axis=1)

# Initialize lists to store the actual and predicted classes
actual_labels = []
predicted_labels = []

# Iterate through each row and make predictions
for index, row in attributes_df.iterrows():
    attributes = row.to_dict()
    attribute_list = ', '.join([f"{key}: {value}%" for key, value in attributes.items()])
    
    # Call the function to guess the bird species
    predicted_class = guess_bird_class(attribute_list)
    
    actual_labels.append(actual_classes.iloc[index])
    predicted_labels.append(predicted_class)

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Actual Class': actual_labels,
    'Predicted Class': predicted_labelså
})

# Save the results to a CSV file
results_df.to_csv(output_file, index=False)

# Calculate and print accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
