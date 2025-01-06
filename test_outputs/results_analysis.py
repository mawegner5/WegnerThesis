#!/usr/bin/env python3
"""
results_analysis.py

Analyzes the final ResNet50 model performance (validation/test),
plus the LLM top-1/top-3 predictions. Summarizes key findings and
answers typical questions about the results.

Outputs a textual report to:
    /remote_home/WegnerThesis/test_outputs/results_analysis.txt

Usage:
    python results_analysis.py
"""

import os
import csv
import pandas as pd

OUTPUT_DIR = "/remote_home/WegnerThesis/test_outputs"
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "results_analysis.txt")

###############################################################################
# 1. Input Files
###############################################################################
# ResNet50 attribute predictions
RESNET_VAL_PRED     = os.path.join(OUTPUT_DIR, "resnet50_validate_predictions.csv")
RESNET_VAL_CONF     = os.path.join(OUTPUT_DIR, "resnet50_validate_attribute_confusion.csv")
RESNET_VAL_CLASSREP = os.path.join(OUTPUT_DIR, "resnet50_validate_classification_report.csv")

RESNET_TEST_PRED     = os.path.join(OUTPUT_DIR, "resnet50_test_predictions.csv")
RESNET_TEST_CONF     = os.path.join(OUTPUT_DIR, "resnet50_test_attribute_confusion.csv")
RESNET_TEST_CLASSREP = os.path.join(OUTPUT_DIR, "resnet50_test_classification_report.csv")

# LLM results
LLM_TOP1_VAL = os.path.join(OUTPUT_DIR, "LLM_results", "top1_chatgpt_predictions_validate.csv")
LLM_TOP3_VAL = os.path.join(OUTPUT_DIR, "LLM_results", "top3_chatgpt_predictions_validate.csv")
LLM_DEBUG_VAL= os.path.join(OUTPUT_DIR, "LLM_results", "LLM_debug_sentences_validate.csv") 
# (the latter might store the attribute sentences for debugging)

# Potential synonyms to unify in LLM species guesses
# You can keep adding pairs as needed:
SYNONYM_MAP = {
    "buffalo": ["bison"],       # so "bison" ~ "buffalo" => correct
    "bison":   ["buffalo"],
    "orca":    ["killer whale", "killer+whale"],
    "hippo":   ["hippopotamus"],
    # etc...
}

###############################################################################
# 2. Helper: unify synonyms or plus-sign species
###############################################################################
import re

def unify_species_name(name):
    """
    Clean up species guess:
     - Lowercase
     - Replace plus sign with space
     - Remove punctuation except for internal plus if you want
    """
    s = name.lower()
    s = s.replace("+", " ")
    # remove trailing punctuation
    # or do a normalizing approach:
    s = re.sub(r"[^\w\s]+", "", s)
    return s.strip()

def is_synonym_or_match(ground_truth, guess):
    """
    Returns True if guess is ground_truth, OR guess is in the synonyms of ground_truth,
    OR ground_truth is in synonyms of guess. 
    """
    gt_clean = unify_species_name(ground_truth)
    guess_clean = unify_species_name(guess)

    # same string?
    if gt_clean == guess_clean:
        return True
    
    # check synonyms
    # e.g. if gt_clean = "buffalo" and guess_clean = "bison", or vice versa
    # We'll see if guess_clean in SYNONYM_MAP[gt_clean] or etc.
    if gt_clean in SYNONYM_MAP:
        if guess_clean in SYNONYM_MAP[gt_clean]:
            return True
    if guess_clean in SYNONYM_MAP:
        if gt_clean in SYNONYM_MAP[guess_clean]:
            return True

    return False

###############################################################################
# 3. Main analysis
###############################################################################
def main():
    sections = []
    
    # We'll open a few CSVs and gather some interesting data.

    # --- 3.1. ResNet50 Validate Summaries ---
    if os.path.exists(RESNET_VAL_CLASSREP):
        df_val_classrep = pd.read_csv(RESNET_VAL_CLASSREP, index_col=0)
        # We'll note the 'accuracy' row or 'weighted avg' in classification report
        # The classification report has a row named "accuracy" or "micro avg".
        # Let’s see if we find a row named "accuracy":
        if "accuracy" in df_val_classrep.index:
            val_accuracy = df_val_classrep.loc["accuracy"]["precision"]  # 'precision' col is typically the raw accuracy
            sections.append(f"ResNet50 Validation Accuracy (multi-label 'accuracy' from classification_report): {val_accuracy:.4f}")
        # or do micro avg
        if "micro avg" in df_val_classrep.index:
            val_micro_f1 = df_val_classrep.loc["micro avg"]["f1-score"]
            sections.append(f"ResNet50 Validation micro-F1: {val_micro_f1:.4f}")
    else:
        sections.append("No classification report found for validation data.")
    
    # --- 3.2. ResNet50 Test Summaries ---
    if os.path.exists(RESNET_TEST_CLASSREP):
        df_test_classrep = pd.read_csv(RESNET_TEST_CLASSREP, index_col=0)
        if "accuracy" in df_test_classrep.index:
            test_accuracy = df_test_classrep.loc["accuracy"]["precision"]
            sections.append(f"ResNet50 Test Accuracy (multi-label 'accuracy'): {test_accuracy:.4f}")
        if "micro avg" in df_test_classrep.index:
            test_micro_f1 = df_test_classrep.loc["micro avg"]["f1-score"]
            sections.append(f"ResNet50 Test micro-F1: {test_micro_f1:.4f}")
    else:
        sections.append("No classification report found for test data.")

    # --- 3.3. LLM Validate Analysis (top-1) ---
    # We'll check how many were correct, plus how many become correct if synonyms are considered
    if os.path.exists(LLM_TOP1_VAL):
        df_llm_top1 = pd.read_csv(LLM_TOP1_VAL)
        # columns: [image_name, actual_species, chatgpt_top1, top1_correct]
        # top1_correct is True/False ignoring synonyms. 
        # We can re-check correctness with synonyms:
        recheck_correct = []
        for idx, row in df_llm_top1.iterrows():
            actual_sp = row["actual_species"]
            guess_sp = row["chatgpt_top1"]
            # original correctness:
            orig_correct = bool(row["top1_correct"])
            # let's see if it becomes correct if synonyms are allowed
            if orig_correct:
                recheck_correct.append(True)
            else:
                # check synonyms
                if is_synonym_or_match(actual_sp, guess_sp):
                    recheck_correct.append(True)
                else:
                    recheck_correct.append(False)
        df_llm_top1["top1_correct_syn"] = recheck_correct

        # Summaries
        orig_correct_count = sum(df_llm_top1["top1_correct"])
        syn_correct_count  = sum(df_llm_top1["top1_correct_syn"])
        total_llm_top1     = len(df_llm_top1)
        
        orig_acc = orig_correct_count / total_llm_top1 if total_llm_top1>0 else 0
        syn_acc  = syn_correct_count  / total_llm_top1 if total_llm_top1>0 else 0

        sections.append(f"LLM Validate Top-1 Accuracy (original): {orig_acc:.3f}  (synonym-savvy): {syn_acc:.3f}")

        # We'll optionally save the re-check
        recheck_path = os.path.join(OUTPUT_DIR, "LLM_results", "top1_chatgpt_predictions_validate_synonyms.csv")
        df_llm_top1.to_csv(recheck_path, index=False)
        sections.append(f"  (Wrote synonym re-check to: {recheck_path})")

    else:
        sections.append("No top1_chatgpt_predictions_validate.csv found.")
    
    # --- 3.4. LLM Validate Analysis (top-3) ---
    if os.path.exists(LLM_TOP3_VAL):
        df_llm_top3 = pd.read_csv(LLM_TOP3_VAL)
        # columns: [image_name, actual_species, guess1, guess2, guess3, correct1, correct2, correct3]
        # We'll create new columns correct1_syn, correct2_syn, correct3_syn
        c1_syn, c2_syn, c3_syn = [], [], []
        for idx, row in df_llm_top3.iterrows():
            actual_sp = row["actual_species"]
            g1, g2, g3 = row["guess1"], row["guess2"], row["guess3"]
            oc1 = bool(row["correct1"])  # original correctness ignoring synonyms
            oc2 = bool(row["correct2"])
            oc3 = bool(row["correct3"])
            # check synonyms
            if oc1:
                c1_syn.append(True)
            else:
                c1_syn.append(is_synonym_or_match(actual_sp, g1))
            if oc2:
                c2_syn.append(True)
            else:
                c2_syn.append(is_synonym_or_match(actual_sp, g2))
            if oc3:
                c3_syn.append(True)
            else:
                c3_syn.append(is_synonym_or_match(actual_sp, g3))
        df_llm_top3["correct1_syn"] = c1_syn
        df_llm_top3["correct2_syn"] = c2_syn
        df_llm_top3["correct3_syn"] = c3_syn

        # The overall top-3 is correct if any of correct1_syn/correct2_syn/correct3_syn is True
        new_any_syn = []
        old_any = []
        for idx, row in df_llm_top3.iterrows():
            old_any_val = (row["correct1"] or row["correct2"] or row["correct3"])
            new_any_val = (row["correct1_syn"] or row["correct2_syn"] or row["correct3_syn"])
            old_any.append(old_any_val)
            new_any_syn.append(new_any_val)
        df_llm_top3["top3_correct_original"] = old_any
        df_llm_top3["top3_correct_syn"]      = new_any_syn

        total_rows = len(df_llm_top3)
        old_correct_ct = sum(df_llm_top3["top3_correct_original"])
        new_correct_ct = sum(df_llm_top3["top3_correct_syn"])
        old_acc = old_correct_ct / total_rows if total_rows>0 else 0
        new_acc = new_correct_ct / total_rows if total_rows>0 else 0
        sections.append(f"LLM Validate Top-3 Accuracy (original): {old_acc:.3f}  (synonym-savvy): {new_acc:.3f}")

        # Save re-check
        recheck_path3 = os.path.join(OUTPUT_DIR, "LLM_results", "top3_chatgpt_predictions_validate_synonyms.csv")
        df_llm_top3.to_csv(recheck_path3, index=False)
        sections.append(f"  (Wrote synonym re-check to: {recheck_path3})")

    else:
        sections.append("No top3_chatgpt_predictions_validate.csv found.")

    # --- 3.5. Possibly mention interesting questions ---
    sections.append("")
    sections.append("=== Additional Observations / Potential Q&A ===")
    sections.append("* Did the model or LLM frequently confuse certain species? Check the recheck CSV for patterns.")
    sections.append("* Are there attributes with particularly low precision/recall? See the attribute_confusion CSV.")
    sections.append("* If 'bison' was guessed instead of 'buffalo' (and synonyms accounted), we see the synergy.")
    sections.append("* For more advanced analysis, consider partial word overlaps or GPT-based synonyms expansions.")
    sections.append("* Summaries show that synonyms can meaningfully improve LLM top-1 and top-3 metrics.")

    # 4) Write the final summary to results_analysis.txt
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("====== RESULTS ANALYSIS REPORT ======\n\n")
        for line in sections:
            f.write(line + "\n")
    print(f"[DONE] Wrote analysis summary to {OUTPUT_TXT}")

###############################################################################
# 4. Entry Point
###############################################################################
if __name__ == "__main__":
    main()
