# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
# For license terms see LICENSE file.
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software under GPLv2+
# See https://www.gnu.org/licenses/gpl-2.0.html

"""
Complete Ring Detection and Analysis System

This module provides the core implementation for adaptive ring detection including:
- Synthetic ring data generation
- Iterative clustering and fitting procedures
- Geometric validation and refinement
- Performance evaluation against ground truth
"""

# ============================ IMPORTS ============================ #
# Standard library imports
import time

# Third-party imports
import numpy as np

# ============================ CONSTANTS ============================ #

# -------------------- RING GENERATION PARAMETERS -------------------- #
NUM_RINGS = 3                # Default number of rings to generate
X_MIN, X_MAX = 0.2, 0.8      # X-coordinate bounds for circle centers
Y_MIN, Y_MAX = 0.2, 0.8      # Y-coordinate bounds for circle centers
R_MIN, R_MAX = 0.165, 0.643  # Radius bounds for generated circles
POINTS_PER_RING = 500        # Points per generated ring
RADIUS_SCATTER = 0.01        # Scatter level for generated points

# -------------------- CLUSTERING PARAMETERS -------------------- #
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


__all__ = [
    # Main procedure adaptive
    'main_procedure_adaptive',
]

# ============================ MAIN PROCEDURE ADAPTIVE ============================ #

def main_procedure_adaptive(verbose: bool = False,
                           seed: int = 1
                           ) -> np.ndarray:

    """
    Adaptive procedure for ring detection and circle fitting in 2D point rings.

    This function implements a multi-step adaptive approach to:
    1. Generate synthetic ring data
    2. Perform iterative clustering and circle fitting
    3. Validate and refine detected rings
    4. Evaluate results against ground truth

    The algorithm features:
    - Automatic parameter adjustment
    - Outlier rejection
    - Cluster merging
    - Progressive threshold relaxation

    Parameters:
        verbose (bool): If True, enables detailed progress reporting and visualization
        seed (int): Random seed for reproducible results

    Returns:
        np.ndarray: Array containing the fitting quality metrics (ratii) with shape (n, 3) where:
            - ratii[:, 0]: x-coordinate deviations
            - ratii[:, 1]: y-coordinate deviations
            - ratii[:, 2]: radius deviations
            Returns empty array if no valid rings found

    Notes:
        - Uses adaptive clustering with progressive relaxation
        - Implements multiple validation checks
        - Supports iterative refinement
    """

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Record the start time for execution timing
    start_time = time.time()

    # --- Step 1: Setup ---
    # Generate sample circles and print them
    sample_circles = generate_circles(num_circles=NUM_RINGS, x_min=X_MIN,
                                      x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX,
                                      r_min=R_MIN, r_max=R_MAX)
    original_circles = sample_circles.copy()

    # Generate sample points from circles
    sample_points = generate_rings_complete_vectorized(circles=sample_circles,
                                   points_per_ring=POINTS_PER_RING,
                                   radius_scatter=RADIUS_SCATTER)
    original_points = sample_points.copy()

    if verbose:
        # Plot the original circles and points
        plot_circles(sample_circles, title="Original Circles", label="Original Circle")
        plot_points_base(sample_points, title="Sample Data", label="Sample Data", hold=False)

    found_rings = []  # To store refined ring parameters
    iteration = 1  # Initialize iteration counter

    # Sigma threshold for ring extraction
    sigma_threshold = SIGMA_THRESHOLD

    # Flag to control breaking out of both loops
    stop_outer_loop = False

    # Initial clustering constraints
    min_clusters = MIN_CLUSTERS  # Minimum number of clusters to find
    max_clusters = MAX_CLUSTERS  # Maximum number of clusters to find
    min_samples = MIN_SAMPLES    # Minimum samples for each cluster

    # Main loop to fit circles to the points
    while len(sample_points) > MIN_SAMPLES * MIN_CLUSTERS_PER_RING * 1.5 and not stop_outer_loop:

        if verbose:
            print(f"\nIteration {iteration}...")

        if verbose and iteration != 1:
            print(f"Sample points remaining: {len(sample_points)}\n")
            plot_points(sample_points, title="Remaining Points",
                        label="Remaining sample points", hold=False)

        # --- Step 2: Try Direct Fit ---
        fit_result = fit_circle_to_points_fast(sample_points)
        circle, errors, rmse = fit_result

        # Check if the circle is good
        if is_a_good_circle(
            circle=circle,
            errors=errors,
            r_max=R_MAX,
            r_min=R_MIN,
            rmse=rmse,
            alpha=3,
            avg_rmse=RADIUS_SCATTER # Use RADIUS_SCATTER as the reference for avg_rmse
        ):
            # Refine fit by excluding outliers
            best_ring, _, best_fit, outliers = exclude_outliers(
                best_ring=sample_points,
                best_fit=fit_result,
                beta=3,
                verbose=verbose
            )

            #Unpack and store the results
            circle, errors, rmse = best_fit
            found_rings.append([
                circle[0], circle[1], circle[2],  # x, y, r
                errors[0], errors[1], errors[2],  # err_x, err_y, err_r
                rmse
            ])

            if verbose:
                print("All points fitted. Stopping.\n")

                #Step 1: Print the fitted circle
                print_circle(circle, title="Fitted circle", errors=errors, rmse=rmse,
                            label="Direct Fit Circle")

                # Step 2: Plot the best ring points in red
                plot_points_base(best_ring, color='red', label="Best Ring", hold=True)

                # Step 3: Plot outliers in blue if they exist
                if len(outliers) > 0:
                    plot_points_base(outliers, color='blue', label="Outliers", hold=True)

                # Step 4: Plot the fitted circle in green
                plot_circle(circle, color="green", linestyle="solid",
                            label="Fitted Circle", center_as_o=True, hold=False)

            break  # Exit loop if good direct fit found

        if verbose:
            # Provide detailed feedback on why the circle is not good
            print("\nNot all points fitted. Continuing.")
            print(f"  - RMSE: {rmse}, while the threshold is "
            f"{3 * RADIUS_SCATTER} (alpha * RADIUS_SCATTER)\n")

        # --- Step 3: Adaptive Clustering ---
        retry_count = 0
        max_retries = 6  # Maximum number of retries

        while retry_count < max_retries and not stop_outer_loop:
            retry_count += 1

            # Perform adaptive clustering
            cluster_labels, cluster_count = adaptive_clustering(
                sample_points,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                min_samples=min_samples,
                verbose=verbose
            )

            # If no clusters found, break inner loop
            if cluster_count == 0:
                if verbose:
                    print("No clusters found in adaptive clustering\n")
                stop_outer_loop = True  # Set flag to stop outer loop
                break

            # Create and process cluster dictionary
            cluster_dict, _ = create_cluster_dict(sample_points, cluster_labels, verbose=verbose)

            # --- Step 4: Fit Circles to Clusters ---
            fit_circles_to_clusters(cluster_dict, verbose=verbose)

            # --- Step 5: Filter Clusters ---
            cluster_dict = filter_fitted_clusters(cluster_dict, verbose=verbose)

            # Check if any valid clusters remain
            valid_clusters = [c for c in cluster_dict.values() if (
                c['valid'] and c['circle'] is not None)]

            if not valid_clusters:
                if verbose:
                    print(f"\n\n!!!!No good clusters found.Retrying with adjusted parameters"
                          f"(Retry {retry_count}/{max_retries})...!!!!")
                    print(f"Entering {retry_count}nd level adaptive clustering\n")

                # Adjust clustering parameters
                min_samples = MIN_SAMPLES + retry_count  # Increase min_samples
                min_clusters = max_clusters
                max_clusters += MAX_CLUSTERS - MIN_CLUSTERS
                continue

            # --- Step 6: Merge Similar Clusters ---
            cluster_dict = compare_and_merge_clusters(cluster_dict,
                                                      sigma=SIGMA_THRESHOLD_RM, verbose=verbose)

            if verbose:
                print(f"Sigma threshold for ring merging: {SIGMA_THRESHOLD_RM}\n")

                # Show the dictionary
                show_dictionary(
                    cluster_dict,
                    cluster_dict.keys(),
                    title=f"Merged Clusters (Retry {retry_count} of the current iteration)",
                    plt_points=True,
                    plt_circles=True,
                    prt_circles=True,
                    hold=False
                )

            if verbose:
                print(f"Sigma threshold for best ring extraction: {sigma_threshold}\n")

            # --- Step 7: Extract Best Ring from the Fitted Circles ---
            best_ring, other_points, best_fit, _ = extract_best_ring(
                cluster_dict,
                sample_points,
                sigma_threshold=sigma_threshold,
                verbose=verbose
            )

            if best_ring.size == 0:
                if verbose:
                    print("!!!!No valid ring extracted!!!\n")
                continue

            # Unpack the best fit
            circle, errors, rmse = best_fit

            # Check if the initial fit is good before refining
            if is_a_good_circle(
                circle=circle,
                errors=errors,
                r_max=R_MAX,
                r_min=R_MIN,
                rmse=rmse,
                alpha=5,
                avg_rmse=RADIUS_SCATTER
            ):
                # Refine the best ring fit
                best_ring, other_points, best_fit, outliers = exclude_outliers(
                    best_ring=best_ring,
                    other_points=other_points,
                    best_fit=best_fit,
                    beta=3,
                    verbose=verbose
                )

                #Unpack the new best fit results
                circle, errors, rmse = best_fit

                # Store the results
                found_rings.append([
                    circle[0], circle[1], circle[2],  # x, y, r
                    errors[0], errors[1], errors[2],   # err_x, err_y, err_r
                    rmse
                ])

                if verbose:
                    # Step 1: Print the best fit parameters
                    print_circle(circle, errors=errors, rmse=rmse,
                                label="Refined Fit Circle", title="Good Ring Found")

                    # Step 2: Plot the fitted circle in green
                    plot_circle(circle, title="Fitted circle", color="green", linestyle="solid",
                                label="Fitted Circle", center_as_o=True, hold=True)

                    # Step 3: Plot all 'other_points' in gray
                    plot_points(other_points, color='gray', label="Other Points", hold=True)

                    # Step 4: Plot outliers in blue if they exist
                    if len(outliers) > 0:
                        plot_points(outliers, color='blue', label="Outliers", hold=True)

                    # Step 5: Plot the best ring points in red
                    plot_points(best_ring, color='red', label="Best Ring", hold=False)

                # Update sample points and CONTINUE processing
                sample_points = other_points

                # Reduce the number of clusters for subsequent iterations
                min_clusters = max(MIN_CLUSTERS_PER_RING, min_clusters - MIN_CLUSTERS_PER_RING)
                max_clusters = max(MAX_CLUSTERS_PER_RING, max_clusters - MAX_CLUSTERS_PER_RING)

                sigma_threshold *= S_SCALE  # Scale the sigma threshold for the next iteration

                break  # Exit retry loop but continue outer loop

            if verbose:
                print("\n!!!Initial fit rejected - trying with relaxed parameters!!!\n")

            # Adjust parameters for next retry
            min_samples = MIN_SAMPLES + retry_count  # Increase min_samples
            min_clusters = max_clusters
            max_clusters += MAX_CLUSTERS - MIN_CLUSTERS

         # Only break outer loop if we've exhausted all retries
        if retry_count >= max_retries:
            if verbose:
                print("\n!!!!Max retries reached - continuing with remaining points!!!!\n")
            break

        # Prepare for next iteration
        iteration += 1  # Increment the iteration counter

    # --- End of Loop ---
    end_time = time.time()
    if verbose:
        print(f"\nExecution Time: {end_time - start_time:.2f} seconds\n")

    # --- Evaluate Results ---
    if found_rings:
        # Convert found rings to numpy array and split into:
        # - fitted_circles: [x, y, r] parameters
        # - fitted_errors: corresponding uncertainties
        found_rings = np.array(found_rings)
        fitted_circles = found_rings[:, :3]
        fitted_errors = found_rings[:, 3:6]

        # Match found circles against original ground truth
        fitted_pairs, ratii, messages = find_fitting_pairs(
            original_circles=original_circles,
            new_best_fit=(fitted_circles, fitted_errors, None),
            threshold=FITTING_PAIR_TRESHOLD,
            verbose=verbose
        )

        if verbose:
            # Print matching diagnostics
            for msg in messages:
                print(msg)

            # Visualize results:
            # 1. Plot original points as background
            plot_points(original_points, color='gray', label="Original Points", hold=True)

            if len(fitted_pairs) > 0:
                # 2. If successful matches exist, first print, then plot them
                print(f"Number of fitted pairs: {len(fitted_pairs)}")
                fit_circles = np.array([fitted_circles[int(f_idx)]
                                        for _, _, f_idx, _ in fitted_pairs if f_idx is not None])
                plot_circles(fit_circles, title="Fitted Circles vs Sample Points",
                            label="Fitted Circle", center_as_o=True, hold=False)

    else:
        # No rings found - return empty array
        ratii = np.array([])

    return ratii
