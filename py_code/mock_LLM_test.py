import pandas as pd
from sklearn.metrics import accuracy_score

# Mock function to simulate the OpenAI API response
def mock_guess_bird_class(attributes):
    # A simple mock that always returns "Mock Bird" for testing
    return "Mock Bird"

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
    
    # Call the mock function to simulate guessing the bird species
    predicted_class = mock_guess_bird_class(attribute_list)
    
    actual_labels.append(actual_classes.iloc[index])
    predicted_labels.append(predicted_class)

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Actual Class': actual_labels,
    'Predicted Class': predicted_labels
})

# Save the results to a CSV file
results_df.to_csv(output_file, index=False)

# Calculate and print accuracy (will always be 0% with the mock function)
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}% (using mock data)")
