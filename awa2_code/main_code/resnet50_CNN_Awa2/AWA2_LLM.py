#!/usr/bin/env python3
"""
AWA2_LLM.py - A more robust version

Steps:
1) Loads environment from .env for OPENAI_API_KEY (requires 'python-dotenv').
2) Reads the CSV from test_resnet50.py, e.g. "resnet50_validate_predictions.csv".
3) Converts predicted attribute columns into a short descriptive sentence for ChatGPT.
4) Calls ChatGPT with a system + user prompt, instructing JSON output:
   { "top1": [...], "top3": [...] }
5) Attempts to parse the JSON from the raw text, or best-effort if the text is verbose.
6) Saves multiple CSV outputs:
   - A debug CSV with raw ChatGPT text, final parsed guesses, etc.
   - top1 and top3 predictions with correctness flags
   - a bar chart of top1/top3 accuracy per species
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
from dotenv import load_dotenv

########################################
# Configuration
########################################

PHASE_TO_EVAL = "validate"  # or "test"
PREDICTED_CSV_PATH = f"/remote_home/WegnerThesis/test_outputs/resnet50_{PHASE_TO_EVAL}_predictions.csv"
OUTPUT_FOLDER = "/remote_home/WegnerThesis/test_outputs/LLM_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ENV_PATH = "/remote_home/WegnerThesis/.env"

PREDICATES_TXT = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data/predicates.txt"
CLASSES_TXT    = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data/classes.txt"

OPENAI_MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4"
MAX_TOKENS = 200
TEMPERATURE = 0.2
SLEEP_BETWEEN_CALLS = 1.0

SYSTEM_PREAMBLE = (
    "You are a helpful zoology assistant. I will provide an animal's attributes. "
    "Return the best guess + top 3 guesses in strict JSON. No extra commentary."
)

USER_PROMPT_TEMPLATE = """Given this animal description:

{attr_sentence}

1) Provide your single best guess for the animal's species.
2) Provide your top 3 guesses in descending order of likelihood.

Respond strictly in JSON with keys "top1" and "top3", each a list of strings.

Example:
{{
  "top1": ["lion"],
  "top3": ["lion", "tiger", "leopard"]
}}"""

########################################
# Environment load
########################################

load_dotenv(ENV_PATH)
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env or environment variables!")
openai.api_key = openai_api_key
print("OPENAI_API_KEY loaded successfully.")

########################################
# Read attribute + species lists
########################################

def read_txt_list(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts) == 2:
                items.append(parts[1].strip())
            else:
                items.append(line.strip())
    return items

all_attributes = read_txt_list(PREDICATES_TXT)
all_species    = read_txt_list(CLASSES_TXT)

def unify_species_name(name: str) -> str:
    return name.replace("+", " ")

all_species = [unify_species_name(s) for s in all_species]

########################################
# Convert 0/1 -> sentence
########################################

def attributes_to_sentence(attr_bools, attr_names):
    pos_attrs = [an for (flag, an) in zip(attr_bools, attr_names) if flag == 1]
    neg_count = len(attr_bools) - len(pos_attrs)
    s_pos = ""
    if pos_attrs:
        s_pos = "It has: " + ", ".join(pos_attrs) + ". "
    return f"{s_pos}It does not have {neg_count} other attributes."

########################################
# Partial credit matching
########################################

def species_match(ground_truth, guess):
    """Any word overlap => correct."""
    def norm(s):
        s = s.lower()
        s = re.sub(r"[^\w\s]+", "", s)
        return s.strip()
    gt = norm(ground_truth)
    gu = norm(guess)
    set_gt = set(gt.split())
    set_gu = set(gu.split())
    return len(set_gt.intersection(set_gu)) > 0

########################################
# ChatGPT call
########################################

def best_effort_parse_json(raw_text):
    """
    If the user sees ChatGPT returning paragraphs, 
    we try to find a JSON substring with regex. 
    Return parsed dict or empty.
    """
    # simplest approach: search for '{' ... '}'.
    # might do a more advanced approach if needed.
    json_match = re.search(r"(\{.*\})", raw_text, flags=re.DOTALL)
    if not json_match:
        return {}
    possible_json = json_match.group(1).strip()
    try:
        data = json.loads(possible_json)
        return data
    except Exception as e:
        print(f"[WARN] best_effort_parse_json error: {e}")
        return {}

def chatgpt_guess_species(attr_sentence):
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
        raw_content = resp["choices"][0]["message"]["content"].strip()
        # attempt direct JSON parse
        try:
            data = json.loads(raw_content)
        except:
            # fallback to best_effort
            data = best_effort_parse_json(raw_content)

        top1 = data.get("top1", [])
        top3 = data.get("top3", [])
        # Make sure we have lists
        if not isinstance(top1, list):
            top1 = [str(top1)]
        if not isinstance(top3, list):
            top3 = [str(top3)]
        return raw_content, top1, top3

    except Exception as e:
        print("OpenAI error:\n", e)
        return f"[OpenAI error: {e}]", [], []

########################################
# Main
########################################

def main():
    # read CSV
    if not os.path.exists(PREDICTED_CSV_PATH):
        raise FileNotFoundError(f"File not found: {PREDICTED_CSV_PATH}")
    df = pd.read_csv(PREDICTED_CSV_PATH)
    print(f"[INFO] Loaded CSV {PREDICTED_CSV_PATH}, shape={df.shape}")

    # test_resnet50 has columns like: 
    #   image_name, actual_species,
    #   Actual_black,Predicted_black, Actual_white,Predicted_white,...
    # we gather only "Predicted_..." columns
    all_cols = df.columns.tolist()
    if "actual_species" in all_cols:
        species_col = "actual_species"
    else:
        species_col = all_cols[1]  # fallback
    predicted_cols = [c for c in all_cols if c.startswith("Predicted_")]

    # debug CSV for raw gpt text
    debug_csv = os.path.join(OUTPUT_FOLDER, f"LLM_debug_sentences_{PHASE_TO_EVAL}.csv")
    debug_rows = []

    top1_rows = []
    top3_rows = []
    species_stats = defaultdict(lambda: {"top1_correct":0, "top1_total":0,
                                        "top3_correct":0, "top3_total":0})

    total = len(df)
    for i, row in df.iterrows():
        if (i%50)==0:
            print(f"[INFO] Processing row {i}/{total}...")

        img_name = row["image_name"]
        act_sp   = unify_species_name(str(row[species_col]))

        # gather predicted flags
        pred_flags = [int(row[col]) for col in predicted_cols]
        # build sentence
        sentence = attributes_to_sentence(pred_flags, all_attributes)

        # call ChatGPT
        time.sleep(SLEEP_BETWEEN_CALLS)
        raw_gpt, guess1, guess3 = chatgpt_guess_species(sentence)

        # unify guess strings
        guess1 = [unify_species_name(g) for g in guess1]
        guess3 = [unify_species_name(g) for g in guess3]

        if not guess1:
            guess1=[""]
        top1_guess = guess1[0]
        top1_correct = species_match(act_sp, top1_guess)

        while len(guess3)<3:
            guess3.append("")
        flags3 = [species_match(act_sp, g) for g in guess3]
        any3 = any(flags3)

        top1_rows.append([img_name, act_sp, top1_guess, top1_correct])
        top3_rows.append([
            img_name, act_sp,
            guess3[0], guess3[1], guess3[2],
            flags3[0], flags3[1], flags3[2]
        ])

        species_stats[act_sp]["top1_total"] += 1
        if top1_correct:
            species_stats[act_sp]["top1_correct"] += 1
        species_stats[act_sp]["top3_total"] += 1
        if any3:
            species_stats[act_sp]["top3_correct"] += 1

        # store debug row
        debug_rows.append([
            img_name,
            act_sp,
            sentence,
            raw_gpt,
            json.dumps(guess1),
            json.dumps(guess3)
        ])

    # save debug
    debug_out = os.path.join(OUTPUT_FOLDER, f"LLM_debug_sentences_{PHASE_TO_EVAL}.csv")
    with open(debug_out, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["image_name","actual_species","constructed_sentence","raw_chatgpt_output","final_top1_list","final_top3_list"])
        w.writerows(debug_rows)
    print(f"[INFO] Saved debug CSV: {debug_out}")

    # top1
    top1_csv = os.path.join(OUTPUT_FOLDER, f"top1_chatgpt_predictions_{PHASE_TO_EVAL}.csv")
    with open(top1_csv, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["image_name","actual_species","chatgpt_top1","top1_correct"])
        for rowdata in top1_rows:
            w.writerow(rowdata)
    print(f"[INFO] Saved top1 predictions: {top1_csv}")

    # top3
    top3_csv = os.path.join(OUTPUT_FOLDER, f"top3_chatgpt_predictions_{PHASE_TO_EVAL}.csv")
    with open(top3_csv, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["image_name","actual_species","guess1","guess2","guess3","correct1","correct2","correct3"])
        for rowdata in top3_rows:
            w.writerow(rowdata)
    print(f"[INFO] Saved top3 predictions: {top3_csv}")

    # species bar chart
    sorted_species = sorted(species_stats.keys())
    x_positions = np.arange(len(sorted_species))
    top1_accs, top3_accs = [], []
    for sp in sorted_species:
        info = species_stats[sp]
        a1 = info["top1_correct"]/info["top1_total"] if info["top1_total"]>0 else 0
        a3 = info["top3_correct"]/info["top3_total"] if info["top3_total"]>0 else 0
        top1_accs.append(a1)
        top3_accs.append(a3)

    plt.figure(figsize=(15,8))
    width=0.4
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
    print(f"[INFO] Bar chart saved: {figpath}")

    print("[DONE] All done with updated AWA2_LLM script.")

if __name__ == "__main__":
    main()
