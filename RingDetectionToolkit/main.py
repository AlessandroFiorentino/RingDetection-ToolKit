# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
# For license terms see LICENSE file.
#
# SPDX-License-Identifier: GPL-2.0-or-later
##
# This program is free software under GPLv2+
# See https://www.gnu.org/licenses/gpl-2.0.html

# ============================ IMPORTS ============================ #
# Standard library imports

"""
Ring Detection and Analysis Toolkit — Main Entrypoint

This script drives the end-to-end workflow for ring detection and evaluation:
- Repeatedly runs the adaptive clustering + circle-fitting pipeline over many random seeds
- Offers both linear and multiprocessing modes for speed vs. simplicity trade-offs
- Computes detection efficiency and normalized deviation metrics across runs
- Reports timing and performance statistics
- Generates histograms of the normalized error ratios for visual analysis

Usage flags:
    MULTIPROCESSING = True   → parallel execution with multiprocessing.Pool
    MULTIPROCESSING = False  → serial execution with a single loop

    VERBOSE = True  → detailed per-seed logs (only recommended in normal mode with small N_PR)
    VERBOSE = False → minimal console output (recommended for large N_PR or multiprocessing)

Notes:
  • In multiprocessing mode, keep VERBOSE=False to avoid interleaved log messages
  • For step-by-step debugging, run (MULTIPROCESSING=False) with VERBOSE=True and a small N_PR
"""

import time
from multiprocessing import Pool, cpu_count

# Third-party imports
import numpy as np
from tqdm import tqdm

# Local imports
from .ringdetection import (main_procedure_adaptive, analyze_ratii_efficiency,
                            plot_ratii_histograms, calculate_and_print_statistics)
                            #,main_procedure) decomment if you want to run the normal main procedure

# ============================ CONSTANTS ============================ #
DEBUG = False           # Global debug flag for additional output
VERBOSE = True # Controls verbose output; not recommended for large numbers of rings

# -------------------- RING GENERATION PARAMETERS -------------------- #
NUM_RINGS = 3                # Default number of rings to generate
X_MIN, X_MAX = 0.2, 0.8      # X-coordinate bounds for circle centers
Y_MIN, Y_MAX = 0.2, 0.8      # Y-coordinate bounds for circle centers
R_MIN, R_MAX = 0.165, 0.643  # Radius bounds for generated circles
POINTS_PER_RING = 500        # Points per generated ring
RADIUS_SCATTER = 0.01        # Scatter level for generated points

# -------------------- CLUSTERING PARAMETERS -------------------- #
# Local constants
MIN_DBSCAN_EPS = 1e-3   # Minimum DBSCAN eps
MAX_DBSCAN_EPS = 1      # Maximum DBSCAN eps

# Min max clusters per ring
MIN_CLUSTERS_PER_RING = 3  # Minimum number of clusters per ring
MAX_CLUSTERS_PER_RING = 4  # Maximum number of clusters per ring

# Compute total minimum and maximum clusters across all rings
MIN_CLUSTERS = MIN_CLUSTERS_PER_RING * NUM_RINGS  # Total minimum clusters
MAX_CLUSTERS = MAX_CLUSTERS_PER_RING * NUM_RINGS  # Total maximum clusters
MIN_SAMPLES = 7  # Minimum number of samples for clustering (DBSCAN parameter)

# -------------------- THRESHOLD Parameters -------------------- #
SIGMA_THRESHOLD_RM = 2.0     # Sigma threshold for merging similar rings
SIGMA_THRESHOLD = 2.0        # Sigma threshold for extracting the best ring
S_SCALE = 1.8                # Scaling factor for sigma threshold relaxation
FITTING_PAIR_TRESHOLD = 20   # Threshold for matching found-to-original rings

# -------------------- PARAMETERS FOR THE MAIN --------------------#
# Number of repetitions for the main procedure
N_PR = 5

# Global constant to set if you want to save
SAVE_RESULTS = False

# Number of bins for the histograms
N_BINS = 200

# Enable the possibility to go multiprocessing
MULTIPROCESSING = False


# Worker function for multiprocessing
def worker(seeed: int) -> np.ndarray:
    """
    Execute the main adaptive procedure for a given seed.

    Args:
        seed (int): The random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (n, 3) with [r_x, r_y, r_r] for valid rings.
    """
    np.random.seed(seeed + 1)
    return main_procedure_adaptive(verbose=VERBOSE, seed=seeed)

# ======================== HERE IS THE MAIN ========================#

if __name__ == "__main__":
    print("\nRunning in multiprocessing mode..."
          if MULTIPROCESSING else "\nRunning in normal mode...")

    # Initialize array for storing all valid ring ratii [r_x, r_y, r_r]
    combined_good_ratii = np.empty((0, 3))

    # Start global timer
    start_time = time.time()

    if MULTIPROCESSING:
        # Determine number of CPU cores to use (can subtract 1 if desired)
        num_cores = max(1, cpu_count())# - 1)
        print(f"Using {num_cores} CPU cores for parallel processing")

        # Run main_procedure_adaptive in parallel using multiprocessing
        with Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap(worker, range(N_PR)),
                                total=N_PR, desc="Processing seeds in parallel"))

        # Combine valid results (exclude empty arrays)
        combined_good_ratii = np.vstack([res for res in results if res.size > 0])

    else:
        # Serial loop over all seeds with progress bar
        for seed in tqdm(range(N_PR), total=N_PR, desc="Processing seeds"):
            np.random.seed(seed + 1)

            # Run adaptive procedure
            good_ratii = main_procedure_adaptive(verbose=VERBOSE, seed=seed)

            # Store valid results
            if good_ratii.size > 0:
                combined_good_ratii = np.vstack((combined_good_ratii, good_ratii))

    # Evaluate total efficiency
    total_eff = analyze_ratii_efficiency(combined_good_ratii, NUM_RINGS, N_PR)

    # Print timing and performance metrics
    elapsed_time = time.time() - start_time
    print(f"\nTotal Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Average Time per Seed: {elapsed_time / N_PR:.2f} seconds")
    print(f"Processing Rate: {N_PR / elapsed_time:.2f} seeds/second")

    # Final result analysis and visualization
    if combined_good_ratii.size > 0:
        ratii_x, ratii_y, ratii_r = combined_good_ratii.T

        # Uncomment to filter out NaN or infinite values for the main procedure
        # mask = np.isfinite(ratii_x) & np.isfinite(ratii_y) & np.isfinite(ratii_r)
        # ratii_x, ratii_y, ratii_r = ratii_x[mask], ratii_y[mask], ratii_r[mask]

        print("\nFinal Statistics (Good Fits Only):")
        # Print the main statistics
        calculate_and_print_statistics(ratii_x, ratii_y, ratii_r)
        # Finally plot the ratii histograms
        plot_ratii_histograms(ratii_x, ratii_y, ratii_r, bins=N_BINS)
    else:
        print("\nNo good fits found for visualization")
