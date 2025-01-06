#!/usr/bin/env python3
"""
results_analysis.py

A super comprehensive analysis script for your Animals with Attributes project.

It attempts to answer (almost) every question your thesis advisor might ask, by:
1. Checking older trial results in /remote_home/WegnerThesis/charts_figures_etc:
   - model_performance_summary.csv
   - classification reports for various models & trials
   - training curves (loss, jaccard) for visual or numeric comparison

2. Examining final ResNet50 results on both validation and test sets:
   - LLM results (top-1 vs top-3 CSVs from /LLM_results)
   - Summaries of partial-credit matching (including synonyms)
   - Confusion pairs: e.g. "bison" guess for "buffalo" and so on
   - Trends in LLM_debug_sentences_xxx.csv, e.g. which attributes repeated, etc.

3. Additional synonyms logic: 
   If ChatGPT says "water buffalo" but the actual species is "buffalo," 
   we can count that as correct. Similarly "cougar" ~ "mountain lion," etc.

4. Produce or update results_analysis.txt (plus extra CSV or figure if we like),
   highlighting all key findings.

Usage:
  python /remote_home/WegnerThesis/test_outputs/results_analysis.py

Outputs:
  /remote_home/WegnerThesis/test_outputs/results_analysis.txt 
  plus any additional CSV/plots to help you answer thorough questions.
"""

import os
import re
import csv
import json
import glob
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from dotenv import load_dotenv   # in case we want environment usage for any reason

################################################################################
# 1. CONFIG
################################################################################

TEST_OUTPUTS_DIR   = "/remote_home/WegnerThesis/test_outputs"
CHARTS_FIGS_DIR    = "/remote_home/WegnerThesis/charts_figures_etc"

RESULTS_TXT_PATH   = os.path.join(TEST_OUTPUTS_DIR, "results_analysis.txt")

# We can store some intermediate analysis in LLM_results or in the root of test_outputs
LLM_RESULTS_DIR    = os.path.join(TEST_OUTPUTS_DIR, "LLM_results")

# Some relevant CSVs from your final runs of test_resnet50.py:
RESNET50_VAL_PRED  = os.path.join(TEST_OUTPUTS_DIR, "resnet50_validate_predictions.csv")
RESNET50_VAL_CONF  = os.path.join(TEST_OUTPUTS_DIR, "resnet50_validate_attribute_confusion.csv")
RESNET50_VAL_CREP  = os.path.join(TEST_OUTPUTS_DIR, "resnet50_validate_classification_report.csv")

RESNET50_TEST_PRED = os.path.join(TEST_OUTPUTS_DIR, "resnet50_test_predictions.csv")
RESNET50_TEST_CONF = os.path.join(TEST_OUTPUTS_DIR, "resnet50_test_attribute_confusion.csv")
RESNET50_TEST_CREP = os.path.join(TEST_OUTPUTS_DIR, "resnet50_test_classification_report.csv")

# LLM final results (top1/top3) for validate/test
TOP1_VAL_CSV       = os.path.join(LLM_RESULTS_DIR, "top1_chatgpt_predictions_validate.csv")
TOP3_VAL_CSV       = os.path.join(LLM_RESULTS_DIR, "top3_chatgpt_predictions_validate.csv")
TOP1_TEST_CSV      = os.path.join(LLM_RESULTS_DIR, "top1_chatgpt_predictions_test.csv")
TOP3_TEST_CSV      = os.path.join(LLM_RESULTS_DIR, "top3_chatgpt_predictions_test.csv")

DEBUG_VAL_CSV      = os.path.join(LLM_RESULTS_DIR, "LLM_debug_sentences_validate.csv")
DEBUG_TEST_CSV     = os.path.join(LLM_RESULTS_DIR, "LLM_debug_sentences_test.csv")

# The summary file from older trials:
PERF_SUMMARY_CSV   = os.path.join(CHARTS_FIGS_DIR, "model_performance_summary.csv")

# We can scan classification_report_*.csv in CHARTS_FIGS_DIR for older trials.
CLASS_REPORT_GLOB  = os.path.join(CHARTS_FIGS_DIR, "classification_report_*.csv")

# Synonyms: E.g. "bison" <-> "buffalo", "orca" <-> "killer whale", "puma" <-> "cougar" ...
# Extend as desired. The approach: if we see "bison" but ground truth is "buffalo", we treat as correct.
SPECIES_SYNONYMS = {
    # each key is the standard name, the value is a set of synonyms
    "buffalo": {"bison", "water buffalo"},
    "bison":   {"buffalo"}, 
    "killer whale": {"orca"},
    "orca": {"killer whale"},
    "puma": {"mountain lion", "cougar", "panther"},
    "cougar": {"puma", "mountain lion", "panther"},
    "mountain lion": {"puma", "cougar", "panther"},
    "panther": {"puma", "cougar", "mountain lion"},
    # Add more as you see fit
}

################################################################################
# 2. HELPER FUNCTIONS
################################################################################

def unify_species_name(s: str) -> str:
    """
    Lowercases, removes punctuation except spaces, etc.
    """
    s = s.lower()
    s = re.sub(r"[^\w\s]+", "", s)  # remove punctuation
    return s.strip()

def synonyms_match(ground_truth: str, guess: str, synonyms_dict=None):
    """
    1) If ground_truth == guess, return True
    2) If either is in synonyms of the other, return True
    3) If partial word overlap is enough, return True
    """
    if synonyms_dict is None:
        synonyms_dict = SPECIES_SYNONYMS

    gt = unify_species_name(ground_truth)
    gu = unify_species_name(guess)

    # direct match
    if gt == gu:
        return True

    # check synonyms
    # e.g. if gt="buffalo", synonyms_dict["buffalo"] = {"bison", "water buffalo"}
    if gt in synonyms_dict:
        if gu in synonyms_dict[gt]:
            return True
    if gu in synonyms_dict:
        if gt in synonyms_dict[gu]:
            return True

    # partial overlap approach
    set_gt = set(gt.split())
    set_gu = set(gu.split())
    if len(set_gt.intersection(set_gu)) > 0:
        return True

    return False

def compute_accuracy_top1(df_top1, synonyms_dict=None):
    """
    df_top1 has columns: [image_name, actual_species, chatgpt_top1, top1_correct].
    We'll ignore 'top1_correct' from the CSV, and re-check correctness with synonyms.
    """
    correct_count = 0
    total = len(df_top1)
    for idx, row in df_top1.iterrows():
        actual = row["actual_species"]
        guess  = row["chatgpt_top1"]
        if synonyms_match(actual, guess, synonyms_dict):
            correct_count += 1
    acc = correct_count / total if total>0 else 0.0
    return acc

def compute_accuracy_top3(df_top3, synonyms_dict=None):
    """
    df_top3 has columns:
      [image_name, actual_species, guess1, guess2, guess3, correct1, correct2, correct3]
    We'll do our own synonyms check ignoring 'correctX' from CSV.
    """
    correct_count = 0
    total = len(df_top3)
    for idx, row in df_top3.iterrows():
        actual = row["actual_species"]
        guesses = [row["guess1"], row["guess2"], row["guess3"]]
        any_correct = False
        for g in guesses:
            if synonyms_match(actual, g, synonyms_dict):
                any_correct = True
                break
        if any_correct:
            correct_count += 1
    acc = correct_count / total if total>0 else 0.0
    return acc

################################################################################
# 3. MAIN LOGIC
################################################################################

def main():
    # We'll accumulate a big text block for results_analysis.txt
    analysis_lines = []
    analysis_lines.append("===== COMPREHENSIVE RESULTS ANALYSIS =====\n\n")

    # --------------------------------------------------------------------------
    # A) Compare older trials from 'model_performance_summary.csv'
    # --------------------------------------------------------------------------
    if os.path.exists(PERF_SUMMARY_CSV):
        df_perf = pd.read_csv(PERF_SUMMARY_CSV)
        # e.g. columns: [Trial,Model,Best Validation Jaccard,Best Validation Loss,Training Time (s),Timestamp,...]
        analysis_lines.append("[A] OLDER TRIALS SUMMARY\n")
        analysis_lines.append(f"Found {len(df_perf)} total entries in {PERF_SUMMARY_CSV}.\n")

        # Let's break it down by model
        all_models = df_perf["Model"].unique()
        for m in all_models:
            df_m = df_perf[df_perf["Model"]==m]
            best_jacc = df_m["Best Validation Jaccard"].max()
            row_best   = df_m.loc[df_m["Best Validation Jaccard"].idxmax()]
            best_trial = row_best["Trial"]
            analysis_lines.append(
              f"  Model={m} has {len(df_m)} trials. Best Jacc={best_jacc:.3f} from trial '{best_trial}'.\n"
            )
        analysis_lines.append("\n")
    else:
        analysis_lines.append("[A] No older trial summary found (model_performance_summary.csv missing).\n\n")

    # --------------------------------------------------------------------------
    # B) Classification reports in charts_figures_etc (older & final)
    # --------------------------------------------------------------------------
    # Let's parse them quickly to see if we can gather a summary of average f1 or so
    analysis_lines.append("[B] CLASSIFICATION REPORTS (older & final)\n")
    reports = glob.glob(CLASS_REPORT_GLOB)  # e.g. classification_report_resnet50_trial0.csv
    if len(reports)==0:
        analysis_lines.append("No classification_report_*.csv found.\n\n")
    else:
        analysis_lines.append(f"Found {len(reports)} classification reports to parse.\n")
        # We'll parse each, find 'accuracy' or the weighted avg f1
        # typically the classification report in CSV has a row 'accuracy' or 'weighted avg'
        # your code may differ
        for cr_path in reports:
            df_rep = pd.read_csv(cr_path, index_col=0)
            # Might contain rows: [black,white,...,accuracy,macro avg,weighted avg]
            # columns: [precision,recall,f1-score,support]
            row_list = df_rep.index.tolist()
            # We'll try to see if 'accuracy' row is there
            maybe_acc = None
            if "accuracy" in row_list:
                maybe_acc = df_rep.loc["accuracy"]["precision"]  # sometimes 'precision' col is used to store the accuracy number
            # or if there's a 'weighted avg' row
            maybe_weighted = None
            if "weighted avg" in row_list:
                maybe_weighted = df_rep.loc["weighted avg"]["f1-score"]
            analysis_lines.append(f"  {os.path.basename(cr_path)} -> ")
            if maybe_acc is not None:
                analysis_lines.append(f" accuracy={maybe_acc:.3f},")
            if maybe_weighted is not None:
                analysis_lines.append(f" weighted_f1={maybe_weighted:.3f},")
            analysis_lines.append("\n")
        analysis_lines.append("\n")

    # --------------------------------------------------------------------------
    # C) Final model: Evaluate LLM top1/top3 accuracy on validate & test with synonyms
    # --------------------------------------------------------------------------
    analysis_lines.append("[C] LLM TOP-1 / TOP-3 ACCURACY (with synonyms) ON VALIDATE & TEST\n")

    # Validate set
    val_top1_acc_syn = None
    val_top3_acc_syn = None
    if os.path.exists(TOP1_VAL_CSV) and os.path.exists(TOP3_VAL_CSV):
        df_t1v = pd.read_csv(TOP1_VAL_CSV)
        df_t3v = pd.read_csv(TOP3_VAL_CSV)
        val_top1_acc_syn = compute_accuracy_top1(df_t1v)
        val_top3_acc_syn = compute_accuracy_top3(df_t3v)
        analysis_lines.append(
            f"  Validate top-1 accuracy (w/ synonyms) = {val_top1_acc_syn:.3f}, top-3 = {val_top3_acc_syn:.3f}\n"
        )
    else:
        analysis_lines.append("  Validate top1/top3 LLM CSV missing.\n")

    # Test set
    test_top1_acc_syn = None
    test_top3_acc_syn = None
    if os.path.exists(TOP1_TEST_CSV) and os.path.exists(TOP3_TEST_CSV):
        df_t1t = pd.read_csv(TOP1_TEST_CSV)
        df_t3t = pd.read_csv(TOP3_TEST_CSV)
        test_top1_acc_syn = compute_accuracy_top1(df_t1t)
        test_top3_acc_syn = compute_accuracy_top3(df_t3t)
        analysis_lines.append(
            f"  Test top-1 accuracy (w/ synonyms) = {test_top1_acc_syn:.3f}, top-3 = {test_top3_acc_syn:.3f}\n"
        )
    else:
        analysis_lines.append("  Test top1/top3 LLM CSV missing.\n")

    analysis_lines.append("\n")

    # --------------------------------------------------------------------------
    # D) Detailed confusion for LLM guesses: e.g. how often actual='chimpanzee' guess='gorilla'
    #    We can do a quick pair frequency for top-1 on test, show top confusions
    # --------------------------------------------------------------------------
    analysis_lines.append("[D] LLM TOP-1 CONFUSION PAIRS ON TEST SET\n")
    if os.path.exists(TOP1_TEST_CSV):
        df_test1 = pd.read_csv(TOP1_TEST_CSV)
        pair_counts = defaultdict(int)
        for idx, row in df_test1.iterrows():
            actual = unify_species_name(row["actual_species"])
            guess  = unify_species_name(row["chatgpt_top1"])
            pair_counts[(actual, guess)] += 1

        # Sort by freq desc
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
        analysis_lines.append("    Most frequent actual->guess pairs (top 10):\n")
        for (act, gus), cnt in sorted_pairs[:10]:
            # also check if synonyms
            is_syn = synonyms_match(act, gus)
            analysis_lines.append(f"      actual='{act}' guess='{gus}' freq={cnt}, synonyms_match={is_syn}\n")
        analysis_lines.append("\n")
    else:
        analysis_lines.append("  Missing top1_chatgpt_predictions_test.csv\n\n")

    # --------------------------------------------------------------------------
    # E) Possibly read / analyze resnet50_test_predictions.csv for attribute-level stats
    #    This is a multi-attribute file with shape [image_name, actual_species, actual_attr_vector, predicted_attr_vector]
    # --------------------------------------------------------------------------
    analysis_lines.append("[E] MULTI-ATTRIBUTE ANALYSIS (RESNET50 TEST)\n")
    if os.path.exists(RESNET50_TEST_PRED):
        df_test_pred = pd.read_csv(RESNET50_TEST_PRED, header=0)
        # e.g. columns: ["image_name","actual_species","actual_attributes","predicted_attributes"]
        # actual_attributes/predicted_attributes might be stored as strings "[1.0,0.0, ...]"
        # We can parse them, count how many attributes are correct, etc.
        # We'll do a quick jaccard or so:
        total_imgs = len(df_test_pred)
        total_jaccard = 0.0
        for idx, row in df_test_pred.iterrows():
            # parse actual
            actual_str = row["actual_attributes"]
            # actual_str might be like "[1.0, 0.0, 1.0, ...]"
            actual_list = json.loads(actual_str) if isinstance(actual_str,str) else actual_str

            predicted_str = row["predicted_attributes"]
            pred_list = json.loads(predicted_str) if isinstance(predicted_str,str) else predicted_str

            # each is a list of 85 floats or ints
            # compute jaccard
            a_np = np.array(actual_list).astype(int)
            p_np = np.array(pred_list).astype(int)

            intersection = np.sum((a_np==1) & (p_np==1))
            union        = np.sum(((a_np==1)|(p_np==1)))
            if union>0:
                local_j = intersection/union
            else:
                local_j = 0.0
            total_jaccard += local_j
        mean_jaccard = total_jaccard/total_imgs if total_imgs>0 else 0
        analysis_lines.append(
            f"  Based on {total_imgs} test images, mean attribute Jaccard: {mean_jaccard:.3f}\n\n"
        )
    else:
        analysis_lines.append("  Missing resnet50_test_predictions.csv => no attribute-level test analysis.\n\n")

    # --------------------------------------------------------------------------
    # F) Look for synonyms usage in LLM_debug_sentences_test.csv
    #    For instance, how often does ChatGPT guess 'water buffalo' if actual is 'buffalo'?
    # --------------------------------------------------------------------------
    analysis_lines.append("[F] LLM DEBUG: Checking synonyms usage in test debug CSV\n")
    if os.path.exists(DEBUG_TEST_CSV):
        df_debug_test = pd.read_csv(DEBUG_TEST_CSV)
        # columns: [image_name, actual_species, constructed_sentence, raw_chatgpt_output, final_top1_list, final_top3_list]
        # We'll see how often the top1 guess is a recognized synonym
        syn_correct_count = 0
        total_rows = len(df_debug_test)
        for idx, row in df_debug_test.iterrows():
            actual = row["actual_species"]
            # final_top1_list is a JSON array string e.g. '["bison"]'
            top1_list_str = row["final_top1_list"]
            top1_list = json.loads(top1_list_str) if isinstance(top1_list_str,str) else []
            if len(top1_list)>0:
                guess1 = top1_list[0]
                if synonyms_match(actual, guess1):
                    syn_correct_count += 1
        test_debug_syn_acc = syn_correct_count/total_rows if total_rows>0 else 0
        analysis_lines.append(
            f"  Among {total_rows} test debug rows, top1 guess is a synonym in {100*test_debug_syn_acc:.2f}%.\n\n"
        )
    else:
        analysis_lines.append("  Missing LLM_debug_sentences_test.csv => cannot analyze synonyms usage.\n\n")

    # --------------------------------------------------------------------------
    # G) Additional possible expansions
    #    - Generating more CSVs or bar charts with synonyms
    #    - Checking the distribution of actual species vs LLM guesses, etc.
    # --------------------------------------------------------------------------

    # We'll just mention that we could do more if needed:
    analysis_lines.append("[G] Additional expansions possible...\n")
    analysis_lines.append("   - E.g. build confusion matrix across all species for top-1 guesses.\n")
    analysis_lines.append("   - E.g. track average 'confidence' if ChatGPT returned probability.\n")
    analysis_lines.append("   - E.g. parse each attribute_something...\n\n")

    # --------------------------------------------------------------------------
    # H) Write everything to results_analysis.txt
    # --------------------------------------------------------------------------
    with open(RESULTS_TXT_PATH, "w", encoding="utf-8") as f:
        f.writelines(analysis_lines)

    print(f"[INFO] Comprehensive analysis completed. See {RESULTS_TXT_PATH}")

if __name__=="__main__":
    main()
