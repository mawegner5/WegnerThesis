#!/usr/bin/env python3
"""
AWA2_LLM.py

Given the ground-truth attributes (predicate_matrix_with_labels.csv) for each
species in AWA2, we will feed those attributes to an LLM (ChatGPT) to see
if it can guess the species. We gather both:
 - single best guess
 - top 3 guesses
We do partial matching to decide correctness (e.g. "bear" in "grizzly+bear").

Outputs:
 - llm_single_guess_results.csv
 - llm_top3_guesses_results.csv
 - llm_accuracy_summary.txt (with overall and species-level metrics)
 - llm_accuracy_by_species.png (bar chart)

Dependencies:
 pip install openai python-dotenv pandas matplotlib

Ensure you have an .env with OPENAI_API_KEY=sk-... in /remote_home/WegnerThesis/.env
"""

import os
import re
import time
import csv
import openai
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import List, Tuple, Optional

# --------------------------------------------------------------------------
# User settings
# --------------------------------------------------------------------------

PREDICATES_CSV = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data/predicate_matrix_with_labels.csv"
OUTPUT_DIR = "/remote_home/WegnerThesis/trial_API_run"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Single-guess CSV output
SINGLE_GUESS_CSV = os.path.join(OUTPUT_DIR, "llm_single_guess_results.csv")
# Top-3 guesses CSV output
TOP3_GUESS_CSV = os.path.join(OUTPUT_DIR, "llm_top3_guesses_results.csv")

# Accuracy summary text
ACCURACY_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "llm_accuracy_summary.txt")

# Maximum number of attempts if we get rate-limited
MAX_RETRIES = 5

# LLM model and temperature
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.0  # zero means more deterministic

# We'll sleep a bit after each API call to be polite
SLEEP_BETWEEN_CALLS = 0.5


# --------------------------------------------------------------------------
# 1. Setup: load .env, set openai.api_key
# --------------------------------------------------------------------------
def load_openai_key():
    env_file = "/remote_home/WegnerThesis/.env"
    if not os.path.exists(env_file):
        raise FileNotFoundError(f"Missing .env file at {env_file}")
    load_dotenv(env_file)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not found in .env")
    openai.api_key = key


# --------------------------------------------------------------------------
# 2. Read the attributes CSV
# --------------------------------------------------------------------------
def read_attributes(csv_path: str) -> pd.DataFrame:
    """
    Reads the AWA2 attribute matrix from CSV.
    Rows = species (like 'antelope'), columns = 85 attribute names,
    each cell = 0 or 1.
    """
    df = pd.read_csv(csv_path, index_col=0)
    # df index are species (like 'antelope', 'grizzly+bear')
    # columns are attributes (like 'black', 'white', etc.)
    return df


# --------------------------------------------------------------------------
# 3. Convert row of 0/1 attributes into a textual description
#    "This animal has black, does not have white, does not have blue, ..."
# --------------------------------------------------------------------------
def attributes_to_sentence(species: str, attributes: pd.Series) -> str:
    """
    Takes a row of 0/1 attributes and builds a descriptive sentence.
    Example:
      "This animal has black, brown, furry, claws. It does not have white, flippers, horns..."
    """
    has_list = []
    not_list = []
    for attr_name, val in attributes.items():
        if val == 1:
            has_list.append(attr_name)
        else:
            not_list.append(attr_name)
    
    # Construct text
    text = "This animal has " + ", ".join(has_list) + ". "
    text += "It does not have " + ", ".join(not_list) + "."
    return text


# --------------------------------------------------------------------------
# 4. Fuzzy partial matching function
#    "bear" in "polar+bear" => True
#    case-insensitive, also treat '+' as space
# --------------------------------------------------------------------------
def partial_species_match(true_species: str, predicted_species: str) -> bool:
    """
    Return True if predicted_species is "close enough" to the true species:
      - ignoring plus signs vs spaces
      - ignoring case/punctuation
      - if either is substring of the other
    """
    def normalize(s: str) -> str:
        s = s.replace("+", " ")
        s = s.lower().strip()
        # remove punctuation except spaces
        s = re.sub(r"[^a-z0-9\s]", "", s)
        return s
    
    norm_true = normalize(true_species)
    norm_pred = normalize(predicted_species)

    if not norm_pred:
        return False
    
    # If either is substring of the other
    if norm_pred in norm_true or norm_true in norm_pred:
        return True
    return False


# --------------------------------------------------------------------------
# 5. Query the LLM for a single guess
# --------------------------------------------------------------------------
def get_single_guess(description: str) -> str:
    """
    Ask the LLM: "Based on the following attributes, guess the species
    with no extra words. If it fails or rate-limit, we retry up to MAX_RETRIES.
    """
    system_msg = ("You are a helpful assistant. Given a description of an animal's attributes, "
                  "you will respond ONLY with the single best guess for the species. "
                  "Do not provide explanations or extra words.")
    user_prompt = (f"{description} "
                   f"\nWhat is the single best guess for the species? "
                   f"Only respond with the species name, nothing else.")

    for attempt in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                temperature=OPENAI_TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt}
                ]
            )
            guess = response.choices[0].message.content.strip()
            return guess

        except openai.error.RateLimitError as e:
            wait_time = 2 ** attempt
            print(f"RateLimitError: {e}. Waiting {wait_time} sec then retrying...")
            time.sleep(wait_time)
        except openai.error.OpenAIError as e:
            wait_time = 2 ** attempt
            print(f"OpenAI API Error: {e}. Waiting {wait_time} sec then retrying...")
            time.sleep(wait_time)

    # If we reach here, we failed
    return ""


# --------------------------------------------------------------------------
# 6. Query the LLM for top 3 guesses
# --------------------------------------------------------------------------
def get_top3_guesses(description: str) -> List[str]:
    """
    Ask the LLM for top 3 guesses, parse them from the response.
    We'll instruct the LLM to just give 3 lines, each line is a species.
    Return them as a list of strings. If no valid parse, return fewer or empty.
    """
    system_msg = ("You are a helpful assistant. Given a description of an animal's attributes, "
                  "respond ONLY with your top 3 best guesses for the species. "
                  "Output each guess on its own line, with no extra words.")
    user_prompt = (f"{description} "
                   f"\nWhat are the top 3 best guesses for the species? "
                   f"Respond with each guess on its own line, no extra text.")

    for attempt in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                temperature=OPENAI_TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt}
                ]
            )
            raw_answer = response.choices[0].message.content.strip()
            # Attempt to parse lines:
            lines = raw_answer.split("\n")
            # Clean up any blank lines or numbering like "1) Bear"
            guesses = []
            for line in lines:
                line = line.strip()
                # remove leading digits or symbols
                line = re.sub(r"^[0-9\.\)\-]+", "", line).strip()
                if line:
                    guesses.append(line)
            return guesses

        except openai.error.RateLimitError as e:
            wait_time = 2 ** attempt
            print(f"RateLimitError: {e}. Wait {wait_time} sec then retry...")
            time.sleep(wait_time)
        except openai.error.OpenAIError as e:
            wait_time = 2 ** attempt
            print(f"OpenAI API Error: {e}. Wait {wait_time} sec then retry...")
            time.sleep(wait_time)

    # Fallback
    return []


# --------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------
def main():
    load_openai_key()
    df = read_attributes(PREDICATES_CSV)
    species_list = list(df.index)  # e.g. ['antelope','grizzly+bear', ...]
    attribute_cols = list(df.columns)

    single_results = []  # store (species, llm_guess, correct)
    top3_results = []    # store (species, [guess1, guess2, guess3], correct_any)

    # We'll keep track of stats
    correct_single_count = 0
    correct_top3_count = 0

    # For species-level breakdown
    species_correct_counts = {sp: 0 for sp in species_list}
    species_counts = {sp: 1 for sp in species_list}  # each species = exactly 1 row in your CSV

    # For each row in df
    for i, species in enumerate(species_list):
        row_attrs = df.loc[species]
        description = attributes_to_sentence(species, row_attrs)
        print(f"\n=== {i+1}/{len(species_list)}. Querying LLM for species: {species} ===")

        # 1) Single guess
        guess_single = get_single_guess(description)
        print(f" Single guess: '{guess_single}'")
        single_correct = partial_species_match(species, guess_single)
        if single_correct:
            correct_single_count += 1
            species_correct_counts[species] += 1
        single_results.append((species, guess_single, "yes" if single_correct else "no"))

        # short sleep to avoid rate-limit
        time.sleep(SLEEP_BETWEEN_CALLS)

        # 2) Top-3 guess
        guess_top3 = get_top3_guesses(description)
        # We'll measure correct_any = True if ANY guess in top3 is partial-match
        correct_any = False
        for guess in guess_top3:
            if partial_species_match(species, guess):
                correct_any = True
                break
        if correct_any:
            correct_top3_count += 1
        print(f" Top 3 guesses: {guess_top3} => correct_any={correct_any}")
        top3_results.append((species, guess_top3, "yes" if correct_any else "no"))

        time.sleep(SLEEP_BETWEEN_CALLS)

    # 3) Save results to CSV
    with open(SINGLE_GUESS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "llm_guess", "correct"])
        writer.writerows(single_results)
    print(f"Saved single-guess results to {SINGLE_GUESS_CSV}")

    with open(TOP3_GUESS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "llm_guesses", "correct_any"])
        for row in top3_results:
            sp, guesses, corr = row
            # join guesses with a pipe or semicolon
            guesses_str = "|".join(guesses)
            writer.writerow([sp, guesses_str, corr])
    print(f"Saved top-3 guess results to {TOP3_GUESS_CSV}")

    # 4) Compute overall accuracy
    total_species = len(species_list)
    single_accuracy = correct_single_count / total_species
    top3_accuracy = correct_top3_count / total_species

    # 5) Species-level accuracy. In your dataset, each species is just once,
    #    so species_correct_counts[species] is either 1 or 0 for single guess
    #    We'll do a bar chart of that.
    species_acc = {}
    for sp in species_list:
        species_acc[sp] = species_correct_counts[sp]  # either 0 or 1

    # 6) Save summary to text
    with open(ACCURACY_SUMMARY_FILE, "w") as f:
        f.write(f"AWA2 LLM results:\n")
        f.write(f"Total species tested: {total_species}\n")
        f.write(f"Single guess correct: {correct_single_count} => {single_accuracy*100:.2f}%\n")
        f.write(f"Top-3 guess correct: {correct_top3_count} => {top3_accuracy*100:.2f}%\n\n")

        f.write("Species-level single-guess correctness:\n")
        for sp in species_list:
            val = species_acc[sp]
            f.write(f"  {sp}: {val}\n")  # 1 or 0

    print(f"Final single-guess accuracy: {single_accuracy*100:.2f}%")
    print(f"Final top-3 accuracy: {top3_accuracy*100:.2f}%")
    print(f"Accuracy summary saved to {ACCURACY_SUMMARY_FILE}")

    # 7) Make bar chart
    # We'll show species on x-axis, 0 or 1 on y-axis => might be basically yes/no.
    # You can do a nicer chart with sorting, etc.
    species_names = list(species_acc.keys())
    correctness_vals = [species_acc[sp] for sp in species_names]

    plt.figure(figsize=(16, 6))
    plt.bar(species_names, correctness_vals, color="blue")
    plt.xticks(rotation=90)
    plt.ylim([0, 1.2])
    plt.title("LLM Single-Guess Correctness by Species (AWA2 dataset)")
    plt.xlabel("Species")
    plt.ylabel("Correctness (1=correct, 0=incorrect)")

    chart_path = os.path.join(OUTPUT_DIR, "llm_accuracy_by_species.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Bar chart of species-level correctness saved to {chart_path}")


# --------------------------------------------------------------------------
# Helper: convert 0/1 row to 'has/does not have' text
# --------------------------------------------------------------------------
def attributes_to_sentence(species: str, attributes: pd.Series) -> str:
    """
    Takes a row of 0/1 attributes and builds a descriptive text:
    'This animal has black, brown, ... It does not have white, flippers...'
    """
    has_list = []
    not_list = []
    for attr_name, val in attributes.items():
        if val == 1:
            has_list.append(attr_name)
        else:
            not_list.append(attr_name)
    text = (
        "This animal has " + ", ".join(has_list) + ". "
        "It does not have " + ", ".join(not_list) + "."
    )
    return text


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
