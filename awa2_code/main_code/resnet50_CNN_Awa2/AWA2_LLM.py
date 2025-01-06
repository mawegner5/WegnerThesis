#!/usr/bin/env python3
"""
AWA2_LLM.py

This script:
1) Loads environment from .env for OPENAI_API_KEY (needs 'python-dotenv').
2) Reads the CSV from test_resnet50.py (e.g. "resnet50_validate_predictions.csv" or "resnet50_test_predictions.csv").
3) Builds a comprehensive prompt for ChatGPT, listing attributes each animal 'has' and 'does not have'.
4) Calls ChatGPT model="gpt-4" for a single best guess + top-3 guesses (strict JSON).
5) Parses the JSON from the raw text. 
6) Outputs:
   - top1_chatgpt_predictions_<PHASE>.csv
   - top3_chatgpt_predictions_<PHASE>.csv
   - LLM_debug_sentences_<PHASE>.csv (stores the full prompt, raw JSON, final guesses, etc.)
   - A bar chart of species-level top-1 vs top-3 accuracy.
"""

import os
import re
import csv
import json
import time
import openai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from dotenv import load_dotenv

################################################################################
# 1. Configuration
################################################################################

# Which dataset phase do you want to feed the LLM? "validate" or "test"
PHASE_TO_EVAL = "test" 

# The CSV produced by test_resnet50.py: 
PREDICTED_CSV_PATH = f"/remote_home/WegnerThesis/test_outputs/resnet50_{PHASE_TO_EVAL}_predictions.csv"

# Output folder for LLM results
OUTPUT_FOLDER = "/remote_home/WegnerThesis/test_outputs/LLM_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Path to .env that holds OPENAI_API_KEY
ENV_PATH = "/remote_home/WegnerThesis/.env"

# Text files that list the 85 attributes & 50 species
PREDICATES_TXT = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data/predicates.txt"
CLASSES_TXT    = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data/classes.txt"

# Use the up-to-date GPT-4 model or a stable snapshot if available
OPENAI_MODEL_NAME = "gpt-4o-mini"

# ChatGPT params
MAX_TOKENS = 500
TEMPERATURE = 0.2
SLEEP_BETWEEN_CALLS = 1.0  # seconds to wait between calls to reduce rate-limit issues

# A system message to give context. We'll feed the user prompt with the attribute sentence.
SYSTEM_PREAMBLE = (
    "You are a helpful assistant. We will play a game:\n"
    "I will describe an animal with a list of attributes it has and does not have, "
    "and you must guess the single most likely species plus your top-3 guesses.\n"
    "Do not provide any commentary or explanation beyond the JSON answer.\n"
)

# We'll fill in the user prompt with the actual attribute sentence.
USER_PROMPT_TEMPLATE = """Pretend you are a young adult playing a guess-the-animal game.
Below is a list of attributes for one single animal:
{attr_sentence}

1) Provide your single best guess for the animal's species.
2) Provide your top 3 guesses in descending order of likelihood.

Respond in strict JSON with keys: "top1" and "top3", each a list of strings.

For example:
{{
  "top1": ["lion"],
  "top3": ["lion", "tiger", "leopard"]
}}
No extra text outside the JSON.
"""

################################################################################
# 2. Load environment & read attributes
################################################################################

load_dotenv(ENV_PATH)
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file!")
openai.api_key = api_key
print("[INFO] Loaded OpenAI API key successfully.")

def read_txt_list(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts) == 2:
                items.append(parts[1])
            else:
                items.append(line)
    return items

all_attributes = read_txt_list(PREDICATES_TXT)
all_species    = read_txt_list(CLASSES_TXT)

def unify_species_name(name: str) -> str:
    """Removes plus signs, lowercases, etc."""
    return name.replace("+", " ").strip().lower()

all_species = [unify_species_name(s) for s in all_species]

################################################################################
# 3. Convert 0/1 -> "It has: X... It does not have: Y..."
################################################################################

def build_attribute_sentence(attr_bools, attr_names):
    """
    Instead of summarizing "It doesn't have 30 attributes," 
    let's explicitly list them all, to feed a more explicit prompt.
    """
    has_list = []
    has_not_list = []
    for flag, aname in zip(attr_bools, attr_names):
        if flag == 1:
            has_list.append(aname)
        else:
            has_not_list.append(aname)
    # "It has: X, Y, Z. It does NOT have: A, B, C..."
    s1 = ""
    s2 = ""
    if has_list:
        s1 = "It has: " + ", ".join(has_list) + ". "
    if has_not_list:
        s2 = "It does not have: " + ", ".join(has_not_list) + "."
    return (s1 + s2).strip()

################################################################################
# 4. Partial overlap for correctness
################################################################################

def species_match(ground_truth: str, guess: str) -> bool:
    """Any shared word => correct. E.g. 'red fox' vs 'fox' => correct."""
    def norm(s):
        s = s.lower()
        s = re.sub(r"[^\w\s]+", "", s)
        return s.strip()
    gt = norm(ground_truth)
    gu = norm(guess)
    set_gt = set(gt.split())
    set_gu = set(gu.split())
    return len(set_gt.intersection(set_gu)) > 0

################################################################################
# 5. ChatGPT call with fallback to parse JSON
################################################################################

def best_effort_parse_json(raw_text):
    """
    If ChatGPT returns text with paragraphs, try to find JSON with a regex.
    Return a dict or empty if not found.
    """
    match = re.search(r"(\{.*\})", raw_text, flags=re.DOTALL)
    if not match:
        return {}
    possible_json = match.group(1).strip()
    try:
        parsed = json.loads(possible_json)
        return parsed
    except Exception as e:
        print(f"[WARN] JSON parse error: {e}")
        return {}

def ask_chatgpt(attr_sentence: str):
    """
    Sends the attribute sentence to ChatGPT with system+user messages.
    Returns (raw_response, top1_list, top3_list).
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(attr_sentence=attr_sentence)
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PREAMBLE},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw_msg = resp["choices"][0]["message"]["content"].strip()
        # parse JSON
        try:
            data = json.loads(raw_msg)
        except:
            data = best_effort_parse_json(raw_msg)

        top1 = data.get("top1", [])
        top3 = data.get("top3", [])
        # ensure they are lists
        if not isinstance(top1, list):
            top1 = [str(top1)]
        if not isinstance(top3, list):
            top3 = [str(top3)]
        return raw_msg, top1, top3

    except Exception as e:
        print("[OpenAI error]", e)
        # fallback
        return f"[OpenAI error: {e}]", [], []

################################################################################
# 6. Main logic
################################################################################

def main():
    if not os.path.exists(PREDICTED_CSV_PATH):
        raise FileNotFoundError(f"Could not find {PREDICTED_CSV_PATH}")

    df = pd.read_csv(PREDICTED_CSV_PATH)
    print(f"[INFO] Loaded predicted CSV: {PREDICTED_CSV_PATH}, shape={df.shape}")

    # This CSV presumably has columns: 
    #   "image_name", "actual_species",
    #   "Actual_black", "Predicted_black", "Actual_white", "Predicted_white", ...
    # or some variant. We only want columns that start with "Predicted_"
    all_cols = df.columns.tolist()
    if "actual_species" in all_cols:
        species_col = "actual_species"
    else:
        # fallback: assume the 2nd column is species
        species_col = all_cols[1]

    # gather predicted columns
    pred_cols = [c for c in all_cols if c.startswith("Predicted_")]
    if not pred_cols:
        raise ValueError("No 'Predicted_' columns found in your CSV. Check your test_resnet50 output format.")
    
    # We'll store debug data
    debug_path = os.path.join(OUTPUT_FOLDER, f"LLM_debug_sentences_{PHASE_TO_EVAL}.csv")
    debug_rows = []
    # We'll store top1, top3 CSV data
    top1_rows = []
    top3_rows = []
    # We'll track species-level stats
    species_stats = defaultdict(lambda: {"top1_correct":0,"top1_total":0,"top3_correct":0,"top3_total":0})

    total = len(df)
    for i, row in df.iterrows():
        if i % 20 == 0:
            print(f"[INFO] Processing row {i}/{total} ...")

        image_name = row["image_name"]
        actual_sp  = unify_species_name(str(row[species_col]))

        # build attribute vector
        pred_flags = [int(row[c]) for c in pred_cols]
        # build a comprehensive sentence with positives & negatives
        attr_sentence = build_attribute_sentence(pred_flags, all_attributes)

        # Query ChatGPT
        time.sleep(SLEEP_BETWEEN_CALLS)  # throttle
        raw_resp, guess_top1, guess_top3 = ask_chatgpt(attr_sentence)

        # unify guess strings
        guess_top1 = [unify_species_name(g) for g in guess_top1]
        guess_top3 = [unify_species_name(g) for g in guess_top3]

        if not guess_top1:
            guess_top1 = [""]
        g1 = guess_top1[0]
        correct1 = species_match(actual_sp, g1)

        while len(guess_top3) < 3:
            guess_top3.append("")
        c3 = [species_match(actual_sp, g) for g in guess_top3]
        any3 = any(c3)

        top1_rows.append([image_name, actual_sp, g1, correct1])
        top3_rows.append([
            image_name, actual_sp,
            guess_top3[0], guess_top3[1], guess_top3[2],
            c3[0], c3[1], c3[2]
        ])

        species_stats[actual_sp]["top1_total"] += 1
        if correct1:
            species_stats[actual_sp]["top1_correct"] += 1
        species_stats[actual_sp]["top3_total"] += 1
        if any3:
            species_stats[actual_sp]["top3_correct"] += 1

        # debug row
        debug_rows.append([
            image_name,
            actual_sp,
            attr_sentence,
            raw_resp,
            json.dumps(guess_top1),
            json.dumps(guess_top3)
        ])

    # Save debug CSV
    with open(debug_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name","actual_species","constructed_sentence","raw_chatgpt_output",
                    "final_top1_list","final_top3_list"])
        w.writerows(debug_rows)
    print(f"[INFO] Wrote debug CSV -> {debug_path}")

    # Save top1
    top1_csv = os.path.join(OUTPUT_FOLDER, f"top1_chatgpt_predictions_{PHASE_TO_EVAL}.csv")
    with open(top1_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name","actual_species","chatgpt_top1","top1_correct"])
        for rowdata in top1_rows:
            w.writerow(rowdata)
    print(f"[INFO] Wrote top-1 predictions -> {top1_csv}")

    # Save top3
    top3_csv = os.path.join(OUTPUT_FOLDER, f"top3_chatgpt_predictions_{PHASE_TO_EVAL}.csv")
    with open(top3_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name","actual_species","guess1","guess2","guess3","correct1","correct2","correct3"])
        for rowdata in top3_rows:
            w.writerow(rowdata)
    print(f"[INFO] Wrote top-3 predictions -> {top3_csv}")

    # Make a bar chart
    sorted_species = sorted(species_stats.keys())
    x_positions = np.arange(len(sorted_species))
    top1_accs = []
    top3_accs = []

    for sp in sorted_species:
        info = species_stats[sp]
        a1 = info["top1_correct"]/info["top1_total"] if info["top1_total"]>0 else 0
        a3 = info["top3_correct"]/info["top3_total"] if info["top3_total"]>0 else 0
        top1_accs.append(a1)
        top3_accs.append(a3)

    plt.figure(figsize=(15,8))
    width = 0.4
    plt.bar(x_positions - width/2, top1_accs, width=width, label="Top-1 Acc")
    plt.bar(x_positions + width/2, top3_accs, width=width, label="Top-3 Acc")
    plt.xticks(x_positions, sorted_species, rotation=90)
    plt.ylim([0,1])
    plt.xlabel("Species")
    plt.ylabel("Accuracy")
    plt.title(f"ChatGPT Accuracy by Species (Top-1 vs Top-3) [{PHASE_TO_EVAL}]")
    plt.legend()
    plt.tight_layout()
    figpath = os.path.join(OUTPUT_FOLDER, f"chatgpt_species_accuracy_{PHASE_TO_EVAL}.png")
    plt.savefig(figpath, dpi=150)
    plt.close()
    print(f"[INFO] Bar chart saved -> {figpath}")

    print("[DONE] Completed AWA2_LLM script using model='gpt-4'.")


if __name__ == "__main__":
    main()
