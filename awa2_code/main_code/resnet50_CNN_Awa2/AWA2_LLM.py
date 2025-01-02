#!/usr/bin/env python3
"""
AWA2_LLM.py

This script takes the per-image predicted attributes CSV from test_resnet50.py
(which includes actual_species plus the 0/1 predicted attributes) and feeds
each row into ChatGPT, requesting a top-1 guess and a top-3 guess.

Steps:
1. Read the CSV with columns like:
   [image_name, actual_species, Actual_<attr1>, ..., Predicted_<attr1>, ...].
   - We only need the "Predicted_<attr>" columns for building the attribute sentence.
2. Convert predicted 0/1 attribute vector into a short descriptive sentence.
3. Provide a coaching preamble, then query ChatGPT with that description.
4. Parse ChatGPT's JSON response to get top1 and top3 guesses.
5. Compare guesses to actual_species with partial-credit substring matching.
6. Save top1 results, top3 results, and produce a species-level accuracy chart.
7. Additionally, store the attribute-sentences in a separate file for debugging.
"""

import os
import re
import csv
import time
import json
import openai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

########################################
# 1. Configuration
########################################

# CHOOSE: "validate" or "test"
PHASE_TO_EVAL = "validate"  # or "test"

# This is the CSV from test_resnet50.py output, e.g.:
#   /remote_home/WegnerThesis/test_outputs/resnet50_validate_predictions.csv
# or
#   /remote_home/WegnerThesis/test_outputs/resnet50_test_predictions.csv
# We'll build it dynamically below based on PHASE_TO_EVAL:
CSV_BASENAME = f"resnet50_{PHASE_TO_EVAL}_predictions.csv"
PREDICTED_CSV_PATH = os.path.join("/remote_home/WegnerThesis/test_outputs", CSV_BASENAME)

# Folder to place LLM results (CSV outputs, figures, etc.)
OUTPUT_FOLDER = "/remote_home/WegnerThesis/test_outputs/LLM_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# We'll store the attribute sentences for debugging
SENTENCE_DEBUG_PATH = os.path.join(OUTPUT_FOLDER, f"attribute_sentences_{PHASE_TO_EVAL}.txt")

# Environment-based key
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables!")
else:
    print("OPENAI_API_KEY successfully loaded.")

# Preamble used to instruct the LLM
PREAMBLE = (
    "You are a zoology expert. I will describe an animal's attributes "
    "in short statements, and you will guess its species. "
    "Here is the format:\n"
    "1) Single best guess (in a JSON array named 'top1').\n"
    "2) Top 3 guesses (in a JSON array named 'top3').\n"
    "Example:\n"
    "{\n"
    '  "top1": ["lion"],\n'
    '  "top3": ["lion", "tiger", "leopard"]\n'
    "}\n"
    "No extra commentary. Provide strictly that JSON."
)

# ChatGPT model and parameters
OPENAI_MODEL_NAME = "gpt-4"      # or "gpt-3.5-turbo"
MAX_TOKENS = 200
TEMPERATURE = 0.2
SLEEP_BETWEEN_CALLS = 1.0  # in seconds

# We have text files listing all attributes / classes if needed
# but for this code, we won't re-load them if we only rely on the CSV columns.
# We'll do minimal referencing.

########################################
# 2. Function to unify species name
########################################

def unify_species_name(name:str) -> str:
    """Convert 'red+fox' -> 'red fox' for partial-credit checks."""
    return name.replace("+", " ")

########################################
# 3. Convert 0/1 -> short sentence
########################################

def attributes_to_sentence(attr_bools, attr_names):
    """
    Takes predicted 0/1 for each attribute, returns a short sentence:
      "It has: black, brown, etc. It does not have X other attributes."
    """
    positives = [n for (flag, n) in zip(attr_bools, attr_names) if flag == 1]
    pos_str = ""
    if positives:
        pos_str = "It has: " + ", ".join(positives) + ". "
    num_neg = len(attr_bools) - len(positives)
    neg_str = f"It does not have {num_neg} other attributes."
    return pos_str + neg_str

########################################
# 4. Partial-credit matching
########################################

def species_match(ground_truth, guess):
    """
    Return True if there's an overlap in any word between ground_truth and guess.
    E.g. "red fox" vs "fox" -> True
    """
    def normalize(s):
        s = s.lower()
        s = re.sub(r"[^\w\s]+", "", s)  # remove punctuation
        return s.strip()
    g_norm = normalize(ground_truth)
    p_norm = normalize(guess)
    g_set = set(g_norm.split())
    p_set = set(p_norm.split())
    return len(g_set.intersection(p_set)) > 0

########################################
# 5. ChatGPT call
########################################

def chatgpt_guess_species(attribute_sentence):
    """
    We'll send PREAMBLE as the system message,
    then user content with the attribute_sentence + instructions.
    We expect JSON in the response with top1 / top3.
    """
    user_msg = (
        f"Here is the animal's attributes:\n"
        f"{attribute_sentence}\n\n"
        "Return JSON with 'top1' and 'top3' arrays. "
        "No extra commentary."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": PREAMBLE},
                {"role": "user", "content": user_msg}
            ]
        )
        content = resp["choices"][0]["message"]["content"].strip()
        # Attempt to parse JSON
        top1, top3 = [], []
        try:
            data = json.loads(content)
            top1 = data.get("top1", [])
            top3 = data.get("top3", [])
            if not isinstance(top1, list):
                top1 = [str(top1)]
            if not isinstance(top3, list):
                top3 = [str(top3)]
        except:
            # If parsing fails, fallback to empty
            top1, top3 = [], []
        return top1, top3
    except Exception as e:
        print("OpenAI error:", e)
        return [], []

########################################
# 6. Main logic
########################################

def main():
    # 6.1: Read the CSV from test_resnet50
    if not os.path.exists(PREDICTED_CSV_PATH):
        raise FileNotFoundError(f"Could not find predicted CSV: {PREDICTED_CSV_PATH}")
    
    df = pd.read_csv(PREDICTED_CSV_PATH)
    columns = df.columns.tolist()
    print(f"Loaded CSV with columns: {columns}")
    
    # Expecting columns like: 
    # ["image_name","actual_species","Actual_black","Actual_white",...,"Predicted_black","Predicted_white",...]
    # We'll separate actual vs. predicted columns. We only need predicted to feed LLM.
    # But we do need actual_species for correctness check.
    if "actual_species" in columns:
        species_col = "actual_species"
    else:
        # fallback
        species_col = columns[1]
        print(f"Warning: using {species_col} as species column.")
    
    # Identify predicted columns
    predicted_cols = [c for c in columns if c.startswith("Predicted_")]
    # We might keep the actual attribute names from "Predicted_something"
    attr_names = [c.replace("Predicted_","") for c in predicted_cols]

    # We'll store a text file with the sentences for debugging
    sentence_f = open(SENTENCE_DEBUG_PATH, "w", encoding="utf-8")
    sentence_f.write("Debug of attribute sentences:\n\n")

    # We'll store top-1 results, top-3 results
    top1_rows = []
    top3_rows = []

    # We'll track species-level stats for the bar chart
    species_stats = defaultdict(lambda: {"top1_correct":0,"top1_total":0,"top3_correct":0,"top3_total":0})

    # 6.2: Iterate over each row
    for idx, row in df.iterrows():
        image_name = row["image_name"]
        actual_sp = unify_species_name(str(row[species_col]))
        
        # Gather the predicted attributes
        predicted_vals = []
        for pc in predicted_cols:
            val = row[pc]  # should be 0 or 1
            if isinstance(val, str) and "[" in val:
                # this indicates the row might be a string like "[1.0, 0.0, ...]"
                # means we didn't properly format the CSV in test_resnet50
                # but let's handle it gracefully
                # parse the string as a python list
                # safe parse
                arr = re.findall(r"[-+]?\d*\.\d+|\d+", val) # match floats/ints
                val = float(arr[0]) if arr else 0.0
            val_i = int(float(val))  # convert to int
            predicted_vals.append(val_i)
        
        # Build attribute sentence
        sentence = attributes_to_sentence(predicted_vals, attr_names)
        
        # For debugging, store it
        sentence_f.write(f"{idx+1}) image={image_name}, species={actual_sp}\n")
        sentence_f.write(f"    => {sentence}\n\n")

        # 6.3: LLM call
        time.sleep(SLEEP_BETWEEN_CALLS)
        guess_top1, guess_top3 = chatgpt_guess_species(sentence)
        guess_top1 = [unify_species_name(g) for g in guess_top1]
        guess_top3 = [unify_species_name(g) for g in guess_top3]
        
        # Evaluate top1 correctness
        if len(guess_top1)==0:
            guess_top1 = [""]
        top1_guess = guess_top1[0]
        top1_correct = species_match(actual_sp, top1_guess)

        # Evaluate top3 correctness
        while len(guess_top3)<3:
            guess_top3.append("")
        top3_correct_flags = [species_match(actual_sp, g) for g in guess_top3]
        top3_correct_any = any(top3_correct_flags)

        # Add to row storage
        top1_rows.append([image_name, actual_sp, top1_guess, top1_correct])
        top3_rows.append([
            image_name, actual_sp,
            guess_top3[0], guess_top3[1], guess_top3[2],
            top3_correct_flags[0], top3_correct_flags[1], top3_correct_flags[2]
        ])

        # Update stats
        species_stats[actual_sp]["top1_total"] += 1
        if top1_correct:
            species_stats[actual_sp]["top1_correct"] += 1
        species_stats[actual_sp]["top3_total"] += 1
        if top3_correct_any:
            species_stats[actual_sp]["top3_correct"] += 1

    # Close the debug file
    sentence_f.close()
    print(f"Saved attribute sentences to {SENTENCE_DEBUG_PATH}")

    # 6.4: Write out top-1 CSV
    top1_csv = os.path.join(OUTPUT_FOLDER, f"top1_chatgpt_predictions_{PHASE_TO_EVAL}.csv")
    with open(top1_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name","actual_species","chatgpt_top1","top1_correct"])
        for r in top1_rows:
            w.writerow(r)
    print(f"Saved top-1 results to {top1_csv}")

    # 6.5: Write out top-3 CSV
    top3_csv = os.path.join(OUTPUT_FOLDER, f"top3_chatgpt_predictions_{PHASE_TO_EVAL}.csv")
    with open(top3_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name","actual_species","guess1","guess2","guess3","correct1","correct2","correct3"])
        for r in top3_rows:
            w.writerow(r)
    print(f"Saved top-3 results to {top3_csv}")

    # 6.6: Build a species-level accuracy bar chart
    species_list_sorted = sorted(species_stats.keys())
    top1_acc = []
    top3_acc = []

    for sp in species_list_sorted:
        stats = species_stats[sp]
        t1_acc = 0.0
        t3_acc = 0.0
        if stats["top1_total"]>0:
            t1_acc = stats["top1_correct"] / stats["top1_total"]
        if stats["top3_total"]>0:
            t3_acc = stats["top3_correct"] / stats["top3_total"]
        top1_acc.append(t1_acc)
        top3_acc.append(t3_acc)

    import numpy as np
    x_positions = np.arange(len(species_list_sorted))
    width=0.4

    plt.figure(figsize=(15,8))
    plt.bar(x_positions - width/2, top1_acc, width=width, label="Top-1 Acc")
    plt.bar(x_positions + width/2, top3_acc, width=width, label="Top-3 Acc")
    plt.xticks(x_positions, species_list_sorted, rotation=90)
    plt.ylim([0,1])
    plt.xlabel("Species")
    plt.ylabel("Accuracy")
    plt.title(f"ChatGPT Accuracy by Species (Top-1 vs Top-3) - {PHASE_TO_EVAL}")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_FOLDER, f"chatgpt_species_accuracy_{PHASE_TO_EVAL}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved species-level accuracy bar chart to {fig_path}")

    print(f"All done with AWA2_LLM script on PHASE='{PHASE_TO_EVAL}'. Enjoy your results!")

########################################

if __name__ == "__main__":
    main()
