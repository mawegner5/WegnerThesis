import os
import pandas as pd
import numpy as np

# Define paths
data_dir = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data"
predicate_matrix_file = os.path.join(data_dir, "predicate-matrix-binary.txt")
predicates_file = os.path.join(data_dir, "predicates.txt")
classes_file = os.path.join(data_dir, "classes.txt")
output_file = os.path.join(data_dir, "predicate_matrix_with_labels.csv")

def main():
    # Load the predicate matrix
    predicate_matrix = np.loadtxt(predicate_matrix_file, dtype=int)
    
    # Load predicates (column headers)
    with open(predicates_file, "r") as f:
        predicates = [line.strip().split("\t")[1] for line in f.readlines()]
    
    # Load classes (row titles)
    with open(classes_file, "r") as f:
        classes = [line.strip().split("\t")[1] for line in f.readlines()]
    
    # Create DataFrame
    df = pd.DataFrame(predicate_matrix, index=classes, columns=predicates)
    
    # Save to CSV
    df.to_csv(output_file)
    print(f"Predicate matrix with labels saved to {output_file}")

if __name__ == "__main__":
    main()
