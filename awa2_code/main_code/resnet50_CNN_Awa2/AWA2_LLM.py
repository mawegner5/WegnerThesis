#!/usr/bin/env python3
"""
AWA2_LLM.py

This script does the "grand finale" step:
1. It loads the predicted attributes CSV from the final ResNet50 model's test or validation run.
2. Reads each row (actual species, predicted attributes).
3. Converts the 0/1 attributes into a descriptive sentence about the animal.
4. Calls ChatGPT (GPT-4 or GPT-3.5) with a carefully crafted prompt asking for:
   - The single best guess (top-1).
   - The top 3 best guesses.
5. We do partial-credit matching to see if ChatGPT guessed the actual species. 
   e.g., "red fox" vs. "fox" are considered the same.
6. Outputs two CSVs:
   - top1_chatgpt_predictions.csv
   - top3_chatgpt_predictions.csv
7. Creates a figure (bar plot) of how frequently ChatGPT guessed each species (top-1 or top-3) 
   and whether correct or not. Also can do a species-level accuracy figure.
"""

import os
import csv
import re
import openai
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from collections import defaultdict

########################################
# 1. Configuration
########################################

# Where your predicted attributes CSV is stored (from test_resnet50.py).
# If you're testing on validation, just point to that CSV. If on test, point to that CSV.
predicted_csv_path = "/remote_home/WegnerThesis/test_outputs/predicted_attributes.csv"  

# Output folder for the CSVs & plots:
output_folder = "/remote_home/WegnerThesis/test_outputs/LLM_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Path to your .env storing OPENAI_API_KEY
env_path = "/remote_home/WegnerThesis/.env"

# Whether to read attributes from file or hard-code them. 
# Let's assume we read them from the known file (predicates.txt).
attributes_txt_path = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data/predicates.txt"

# And read species from file, or we can store them. Let's do file-based as well (classes.txt).
classes_txt_path = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data/classes.txt"

# Model for ChatGPT
openai_model_name = "gpt-4"  # or "gpt-3.5-turbo"

# Additional ChatGPT config
max_tokens = 200
temperature = 0.2  # fairly deterministic
top_n_guesses = 3  # We'll ask ChatGPT to guess up to 3 species

# Rate-limit or sleep between calls if needed
sleep_between_calls = 1.0  # seconds

########################################
# 2. Load environment & set openai key
########################################

load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment.")

########################################
# 3. Read Attributes & Species
########################################

def read_txt_list(txt_path):
    """
    Reads lines from a txt file, ignoring blank lines. 
    We'll assume they're enumerated: 
         1 attribute1
         2 attribute2
    We'll parse out just the attribute name (last column).
    """
    items = []
    with open(txt_path, "r") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            # e.g., "1\tblack"
            # Or "1 black"
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts)==2:
                # e.g., parts[0]="1", parts[1]="black"
                items.append(parts[1].strip())
            else:
                # If line is just "black"
                items.append(line.strip())
    return items

all_attributes = read_txt_list(attributes_txt_path)  # we expect 85
all_species = read_txt_list(classes_txt_path)        # we expect 50

# If they have plus signs, we might want to unify them: "red+fox" -> "red fox"
def unify_species_name(species_name):
    return species_name.replace("+"," ")

all_species = [unify_species_name(s) for s in all_species]

########################################
# 4. Utility: Convert 0/1 row -> sentence
########################################

def attributes_to_sentence(attr_bools, attr_names):
    """
    attr_bools is a list (or np array) of 85 ints (0 or 1).
    attr_names is the 85 attribute strings in the correct order.

    We'll produce a statement like:
      "This animal: 
        has black, 
        does not have white, 
        does not have blue, 
        has brown, etc..."
    Or we can produce a simpler approach. 
    Let's just mention the "1" ones in a positive sense and "0" ones in a negative sense 
    but keep it short so we don't kill tokens.
    """
    # Optional approach #1: List everything:
    # positives = []
    # negatives = []
    # for (flag, attr) in zip(attr_bools, attr_names):
    #     if flag==1:
    #         positives.append(f"has {attr}")
    #     else:
    #         negatives.append(f"does not have {attr}")
    # statement = "It " + ", ".join(positives + negatives) + "."

    # But that might be very long. Maybe let's only mention positives. 
    # And also mention some form of summary about negatives but not all details to reduce token usage
    positives = [attr for (flag, attr) in zip(attr_bools, attr_names) if flag==1]
    positives_part = ""
    if positives:
        positives_part = "It has: " + ", ".join(positives) + ". "
    # If you want to mention count of negatives:
    num_negatives = sum(1 for x in attr_bools if x==0)
    negatives_part = f"It does not have {num_negatives} other attributes. "
    # Combined
    statement = positives_part + negatives_part
    return statement.strip()

########################################
# 5. Partial-credit matching function
########################################

def species_match(ground_truth, guess):
    """
    ground_truth and guess are strings like "red fox" or "fox".
    We'll do a basic substring check in a normalized manner:
      - remove punctuation
      - lower-case
      - split on spaces
      - if any of guess is in ground_truth or vice versa
    A more advanced approach might use a measure of similarity.
    We'll do a simpler approach for now:
      if any word in guess is also in the ground_truth (like 'fox' in 'red fox'), we consider it correct.
    """
    # unify them
    def normalize(s):
        s = s.lower()
        s = re.sub(r"[^\w\s]+", "", s)  # remove punctuation
        return s.strip()

    ground_norm = normalize(ground_truth)
    guess_norm = normalize(guess)

    # if guess_norm in ground_norm or ground_norm in guess_norm:
    #    return True

    # Let's do word sets:
    ground_words = set(ground_norm.split())
    guess_words = set(guess_norm.split())
    # if there's any intersection > 0, let's call it correct
    # e.g., "red" and "fox" in ground vs "silver fox" -> intersection = {"fox"} -> correct
    if ground_words.intersection(guess_words):
        return True
    return False

########################################
# 6. ChatGPT calls for top-1 and top-3
########################################

def chatgpt_guess_species(attribute_sentence):
    """
    We'll prompt ChatGPT with a short instruction to guess the single best guess, 
    then top 3 guesses in a separate call or a single call that returns a structured answer.
    We'll do it in a single call for efficiency: we ask ChatGPT to give 
    - best guess
    - top 3 guesses 
    in a structured JSON format or so.
    """
    system_message = "You are an expert zoologist. You identify animals based on described attributes."
    user_message = f"""Given this description of an animal's attributes:
{attribute_sentence}

1) Provide your single best guess for the animal's species. 
2) Provide your top 3 guesses in descending order of likelihood.
Please respond in JSON with two fields: "top1" and "top3", each a list. 
For example:
{{
  "top1": ["tiger"],
  "top3": ["tiger", "lion", "leopard"]
}}
No additional commentary, strictly this JSON format.
"""

    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        content = response["choices"][0]["message"]["content"].strip()
        # Attempt to parse JSON from content
        # If it fails, fallback
        # We'll do a naive approach with a regex or eval
        # or we can do a quick check for JSON
        guess_top1 = []
        guess_top3 = []

        import json
        try:
            parsed = json.loads(content)
            # Expecting keys "top1" and "top3" which are lists of strings
            guess_top1 = parsed.get("top1", [])
            guess_top3 = parsed.get("top3", [])
            # Ensure they are lists of strings
            if not isinstance(guess_top1, list):
                guess_top1 = [str(guess_top1)]
            if not isinstance(guess_top3, list):
                guess_top3 = [str(guess_top3)]
        except:
            # If we fail to parse, let's set them empty or parse manually
            guess_top1 = []
            guess_top3 = []
        return guess_top1, guess_top3
    except Exception as e:
        print("OpenAI error:", e)
        return [], []

########################################
# 7. Main logic: read predicted CSV, call LLM, store results
########################################

def main():
    # read predicted CSV
    df = pd.read_csv(predicted_csv_path)
    # We expect columns: image_name, actual_species, attribute1, attribute2, ... or some structure
    # Possibly you used a structure with columns=[image_name, attribute1, attribute2, ..., actual_species]? 
    # We'll assume the CSV has 'image_name' in col 0, 'species' in col 1, then the 85 attributes in the rest.
    # If your CSV is different, just adapt the code below.
    
    # Let's do a check for how many columns
    # we expect 1 for image_name, 1 for actual species (?), 85 for attributes = 87 total
    # but your CSV might not have actual species if you're only storing predictions. 
    # We'll assume your CSV does have actual species in a column named 'actual_species'.
    
    # If you do not have a direct 'actual_species' column, adapt the code.
    # Example:
    # columns: [image_name, black, white, blue, ..., vegetation, insects, ... oldworld, ... domestic, actual_species]
    
    # We'll assume the last column is the actual species, but let's confirm:
    columns = df.columns.tolist()
    # e.g. columns[-1] = 'actual_species' (?)
    # Or if you have a named column, let's search for it:
    if "actual_species" in columns:
        species_col = "actual_species"
    else:
        # if none found, fallback to last column:
        species_col = columns[-1]
        print(f"Warning: using {species_col} as species column.")
    
    # Let's gather attribute columns (the ones that are 0/1)
    # We'll skip 'image_name' and 'actual_species'
    attribute_cols = [c for c in columns if c not in ("image_name", species_col)]
    
    # We'll create lists to store final top1 & top3 results
    rows_top1 = []  # each row: [image_name, actual_species, guess_top1, correct_bool]
    rows_top3 = []  # each row: [image_name, actual_species, guess_top3_1, guess_top3_2, guess_top3_3, correct_bool_1, correct_bool_2, correct_bool_3]
    
    # We'll also track overall stats for a bar chart
    # We'll store the counts: species -> {top1_correct: x, top1_total: y, top3_correct: a, top3_total: b}
    species_stats = defaultdict(lambda: {"top1_correct":0, "top1_total":0, "top3_correct":0, "top3_total":0})
    
    for idx, row in df.iterrows():
        image_name = row["image_name"]
        actual_species = row[species_col]
        # unify actual species if needed
        actual_species = unify_species_name(str(actual_species))
        
        # get the 0/1 attributes
        attr_values = [int(row[c]) for c in attribute_cols]
        # build a sentence
        sentence = attributes_to_sentence(attr_values, all_attributes)
        
        # Call ChatGPT
        time.sleep(sleep_between_calls)
        guess_top1, guess_top3 = chatgpt_guess_species(sentence)
        
        # unify them
        guess_top1_unified = [unify_species_name(g) for g in guess_top1]
        guess_top3_unified = [unify_species_name(g) for g in guess_top3]
        
        # Evaluate top1 correctness
        if len(guess_top1_unified) == 0:
            guess_top1_unified = [""]  # fallback if no guess
        top1_guess = guess_top1_unified[0]
        top1_correct = species_match(actual_species, top1_guess)
        
        # Evaluate top3 correctness
        # if fewer than 3 returned, pad
        while len(guess_top3_unified)<3:
            guess_top3_unified.append("")
        top3_correct_flags = [species_match(actual_species, g) if g else False for g in guess_top3_unified]
        # We can consider top3 correct if any are correct
        top3_correct = any(top3_correct_flags)
        
        rows_top1.append([
            image_name,
            actual_species,
            top1_guess,
            top1_correct
        ])
        rows_top3.append([
            image_name,
            actual_species,
            guess_top3_unified[0],
            guess_top3_unified[1],
            guess_top3_unified[2],
            top3_correct_flags[0],
            top3_correct_flags[1],
            top3_correct_flags[2]
        ])
        
        # stats
        species_stats[actual_species]["top1_total"] += 1
        if top1_correct:
            species_stats[actual_species]["top1_correct"] += 1
        
        species_stats[actual_species]["top3_total"] += 1
        if top3_correct:
            species_stats[actual_species]["top3_correct"] += 1
    
    # Write top1 CSV
    top1_csv_path = os.path.join(output_folder, "top1_chatgpt_predictions.csv")
    with open(top1_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name","actual_species","chatgpt_top1_guess","top1_correct"])
        for rowdata in rows_top1:
            writer.writerow(rowdata)
    print(f"Saved top-1 ChatGPT predictions to {top1_csv_path}")
    
    # Write top3 CSV
    top3_csv_path = os.path.join(output_folder, "top3_chatgpt_predictions.csv")
    with open(top3_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name","actual_species","guess1","guess2","guess3","correct1","correct2","correct3"])
        for rowdata in rows_top3:
            writer.writerow(rowdata)
    print(f"Saved top-3 ChatGPT predictions to {top3_csv_path}")
    
    ########################################
    # 8. Build a species-level accuracy figure
    ########################################
    species_list_sorted = sorted(species_stats.keys())
    top1_acc = []
    top3_acc = []
    
    for sp in species_list_sorted:
        stats = species_stats[sp]
        if stats["top1_total"]>0:
            acc1 = stats["top1_correct"]/stats["top1_total"]
        else:
            acc1 = 0.0
        if stats["top3_total"]>0:
            acc3 = stats["top3_correct"]/stats["top3_total"]
        else:
            acc3 = 0.0
        top1_acc.append(acc1)
        top3_acc.append(acc3)
    
    # Let's do a bar plot with two bars for each species: top1 and top3
    x_positions = np.arange(len(species_list_sorted))
    width=0.4
    
    plt.figure(figsize=(15,8))
    plt.bar(x_positions - width/2, top1_acc, width=width, label="Top-1 Acc")
    plt.bar(x_positions + width/2, top3_acc, width=width, label="Top-3 Acc")
    plt.xticks(x_positions, species_list_sorted, rotation=90)
    plt.ylim([0,1])
    plt.xlabel("Species")
    plt.ylabel("Accuracy")
    plt.title("ChatGPT Accuracy by Species (Top-1 vs Top-3)")
    plt.legend()
    plt.tight_layout()
    
    species_acc_fig_path = os.path.join(output_folder, "chatgpt_species_accuracy.png")
    plt.savefig(species_acc_fig_path, dpi=150)
    plt.close()
    print(f"Saved species-level accuracy bar chart to {species_acc_fig_path}")

    print("All done with AWA2_LLM script. Enjoy your results!")

########################################

if __name__ == "__main__":
    main()
