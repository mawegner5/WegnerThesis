#!/usr/bin/env python3
"""
data_describe.py

Scans the AwA2 dataset folder, enumerates:
  - # of images per species in (train, validate, test)
  - total # of images
  - approximate attribute distribution
  - image dimension stats (width/height min, max, mean from sampling)

Saves a text report to:
  /remote_home/WegnerThesis/test_outputs/data_description.txt
"""

import os
import random
from PIL import Image
import numpy as np
import pandas as pd

# Root data directory
DATA_DIR = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data"

# Where to save the textual report
OUTPUT_TXT_PATH = "/remote_home/WegnerThesis/test_outputs/data_description.txt"

# Name of the attribute matrix CSV
ATTR_MATRIX_CSV = os.path.join(DATA_DIR, "predicate_matrix_with_labels.csv")

# Subfolders for the dataset splits
SPLITS = ["train", "validate", "test"]

# Limit how many images we sample for dimension stats to avoid huge overhead
N_SAMPLES = 200

def main():
    # 1) Read attribute matrix
    #    Rows: species; Columns: attribute. Values: 0/1
    attr_df = pd.read_csv(ATTR_MATRIX_CSV, index_col=0)
    # Example of species index: "grizzly+bear"
    # Convert " " to "+" if needed, but usually they've been replaced in the dataset already.

    # We'll unify species naming: "white+rat" -> "white+rat" is consistent with folder naming
    species_list = attr_df.index.tolist()  # e.g. 50 species
    attribute_list = attr_df.columns.tolist()  # e.g. 85 attributes

    # 2) Gather species -> image counts for each split
    #    Also track total images
    split_counts = {split: {} for split in SPLITS}  # e.g. split_counts["train"]["grizzly+bear"] = 72
    total_images_per_split = {split: 0 for split in SPLITS}
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Missing {split_dir}, skipping.")
            continue
        for species_name in os.listdir(split_dir):
            species_path = os.path.join(split_dir, species_name)
            if not os.path.isdir(species_path):
                # skip e.g. files, or misnamed items
                continue
            # Count images
            images = os.listdir(species_path)
            n_imgs = sum([1 for x in images if not x.startswith(".")])  # ignore hidden
            split_counts[split][species_name] = n_imgs
            total_images_per_split[split] += n_imgs

    # 3) Total images across entire dataset
    grand_total_images = sum(total_images_per_split.values())

    # 4) Compute attribute distribution *approx*:
    #    If a species has attribute=1, we assume *all images of that species* have it.
    #    Summation over species for each attribute.
    attribute_counts = {attr: 0 for attr in attribute_list}
    # We consider the entire dataset (train/validate/test)
    # For each species, get the total # images across all splits
    # then if species has 1 for that attribute, add that image count
    species_image_count_all = {}
    for s in species_list:
        # sum across splits
        n_s = 0
        for sp in SPLITS:
            n_s += split_counts[sp].get(s, 0)
        species_image_count_all[s] = n_s

    # Now for each attribute, see which species have it = 1
    for attr in attribute_list:
        col_data = attr_df[attr]  # Series with index= species
        # e.g. col_data["grizzly+bear"] = 1.0 or 0.0
        # sum up the images
        total_for_attr = 0
        for s in species_list:
            if s in species_image_count_all:
                if col_data[s] == 1:
                    total_for_attr += species_image_count_all[s]
        attribute_counts[attr] = total_for_attr

    # 5) Sample up to N_SAMPLES images for dimension stats
    #    We'll gather (width, height)
    wh_list = []
    # gather all image paths quickly
    all_image_paths = []
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        for species_name in os.listdir(split_dir):
            species_path = os.path.join(split_dir, species_name)
            if not os.path.isdir(species_path):
                continue
            images = [os.path.join(species_path, x)
                      for x in os.listdir(species_path)
                      if not x.startswith(".")]
            all_image_paths.extend(images)

    random.shuffle(all_image_paths)
    sample_paths = all_image_paths[:N_SAMPLES]  # up to N_SAMPLES
    for p in sample_paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
                wh_list.append((w, h))
        except:
            # if any error reading, skip
            pass

    # compute dimension stats
    wh_list_np = np.array(wh_list, dtype=np.float32)
    if len(wh_list_np)>0:
        min_w, min_h = wh_list_np.min(axis=0)
        max_w, max_h = wh_list_np.max(axis=0)
        mean_w, mean_h = wh_list_np.mean(axis=0)
    else:
        min_w = min_h = max_w = max_h = mean_w = mean_h = 0

    # 6) Write results to a text file
    with open(OUTPUT_TXT_PATH, "w") as f:
        f.write("===== AwA2 Dataset Description =====\n\n")

        f.write("=== 1) IMAGES PER SPLIT ===\n")
        for sp in SPLITS:
            f.write(f" {sp} total images: {total_images_per_split[sp]}\n")
        f.write(f" Grand total images across all splits: {grand_total_images}\n\n")

        f.write("=== 2) IMAGES PER SPECIES (across splits) ===\n")
        for s in sorted(species_list):
            # sum across splits
            n_s = sum(split_counts[sp].get(s, 0) for sp in SPLITS)
            f.write(f"  {s}: {n_s}\n")
        f.write("\n")

        f.write("=== 3) APPROX. ATTRIBUTE DISTRIBUTION ===\n")
        f.write("(If species declared an attribute=1, all its images are counted.)\n")
        # sort attributes by descending freq
        attr_sorted = sorted(attribute_counts.keys(), key=lambda a: attribute_counts[a], reverse=True)
        for attr in attr_sorted:
            f.write(f"  {attr}: {attribute_counts[attr]} images\n")
        f.write("\n")

        f.write("=== 4) IMAGE DIMENSION STATS (sample-based) ===\n")
        f.write(f"Used up to {N_SAMPLES} random images to measure dimension.\n")
        f.write(f"  #sampled: {len(wh_list_np)}\n")
        f.write(f"  Width range:  {min_w} - {max_w} ; mean={mean_w:.2f}\n")
        f.write(f"  Height range: {min_h} - {max_h} ; mean={mean_h:.2f}\n")

    print(f"[DONE] Wrote dataset description to: {OUTPUT_TXT_PATH}")

if __name__ == "__main__":
    main()
