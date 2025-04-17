# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
#
# For license terms see LICENSE file.
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software under GPLv2+
# See https://www.gnu.org/licenses/gpl-2.0.html

"""
Core Functions for Ring Detection System

This module contains the primary algorithms for:
- Synthetic circle and ring generation
- Adaptive clustering of point data
- Ring extraction utilities
- Evaluation and comparison of the results

Key Components:
1. Data Generation - Create synthetic test cases
2. Clustering - Adaptive DBSCAN implementation
3. Utility Functions - Supporting calculations
"""

__all__ = [
    # Synthetic Data Generation
    'generate_circles',
    'generate_rings_complete',
    'generate_rings_complete_vectorized',

    # Clustering Algorithm
    'filter_labels',
    'adaptive_clustering',
    'create_cluster_dict',
    'filter_fitted_clusters',
    'compatible_clusters',
    'compare_and_merge_clusters',

    # Point Extraction
    'extract_points',
    'extract_best_ring',
    'exclude_outliers',

    # Evaluation
    'calculate_ratii',
    'find_fitting_pairs'
]


# ============================== IMPORTS ==================================== #

# Standard library imports
from collections import Counter
from typing import Union, List, Tuple, Dict, Any, Optional, Set
import warnings

# Third-party imports
import numpy as np
from sklearn.cluster import DBSCAN

# Local constants
MIN_DBSCAN_EPS = 1e-3   # Minimum DBSCAN eps
MAX_DBSCAN_EPS = 1      # Maximum DBSCAN eps

# ====================== SYNTHETIC DATA GENERATION ========================== #

def generate_circles(num_circles: int,
                     x_min: float = 0.2, x_max: float = 0.8,
                     y_min: float = 0.2, y_max: float = 0.8,
                     r_min: float = 0.15, r_max: float = 0.8) -> np.ndarray:
    """
    Generates a numpy array of circles of the form [x, y, r], where:
    - (x, y) is the center of the circle.
    - r is the radius of the circle.

    Args:
        num_circles (int): Number of circles to generate.
        x_min (float): Minimum value for the x-coordinate of the circle centers. Defaults to 0.2.
        x_max (float): Maximum value for the x-coordinate of the circle centers. Defaults to 0.8.
        y_min (float): Minimum value for the y-coordinate of the circle centers. Defaults to 0.2.
        y_max (float): Maximum value for the y-coordinate of the circle centers. Defaults to 0.8.
        r_min (float): Minimum value for the radius of the circles. Defaults to 0.2.
        r_max (float): Maximum value for the radius of the circles. Defaults to 0.8.

    Returns:
        -np.ndarray: A numpy array of shape (num_circles, 3), where each row is [x, y, r].

    Notes:
        - Issues warnings (does not raise exceptions) for:
          * Negative radii (r_min < 0)
          * Radii potentially exceeding unit bounds (r_max > 1)
        - These warnings indicate potential visualization issues but don't prevent execution
    """
    # Check if the minimum radius is negative
    if r_min < 0:
        warnings.warn("The minimum radius is negative."
        "If you don't see a ring, it's because of this.")

    # Check if the maximum radius is greater than 1
    if r_max > 1:
        warnings.warn("The maximum radius is too big. "
        "If you don't see a ring, it's because of this.")

    circle_list = []

    for _ in range(num_circles):
        x_circle = np.random.uniform(x_min, x_max)
        y_circle = np.random.uniform(y_min, y_max)
        r_circle = np.random.uniform(r_min, r_max)
        circle_list.append([x_circle, y_circle, r_circle])

    return np.array(circle_list)

def generate_rings_complete(circles: np.ndarray,
                            points_per_ring: int = 500,
                            radius_scatter: float = 0.01) -> np.ndarray:
    """
    Generates point clouds representing circular rings with controlled
        radial and positional scatter.

    Creates a set of points for each input circle with:
    1. Radial scatter: Points vary in distance from center
    2. Positional scatter: Points are randomly displaced in x-y plane

    Args:
        circles (np.ndarray): Array of shape (N,3) [x_center, y_center, radius]
        points_per_ring (int): Number of points per circle
        radius_scatter (float): Maximum scatter distance in any direction

    Returns:
        np.ndarray: Array of (x, y) coordinates with shape (N*points_per_ring, 2)
    """

    def is_a_good_point(point: np.ndarray) -> bool:
        """Filter function to ensure points are within the bounds [0, 1] for both x and y."""
        return point[0] >= 0 and point[0] <= 1 and point[1] >= 0 and point[1] <= 1

    # Input validation
    if circles.shape[1] != 3:
        raise ValueError("Circles array must have shape (N,3)")
    if points_per_ring < 0:
        raise ValueError("Points per ring must be non-negative")
    if radius_scatter < 0:
        raise ValueError("Radius scatter must be non-negative")

    x_coords_all = []
    y_coords_all = []

    for center_x, center_y, ring_radius in circles:
        # Generate base points with radial scatter
        angles = np.random.uniform(0, 2*np.pi, points_per_ring)
        radial_offsets = np.random.uniform(-radius_scatter, radius_scatter, points_per_ring)
        radii = ring_radius + radial_offsets

        # Convert to Cartesian coordinates
        x_coords = radii * np.cos(angles) + center_x
        y_coords = radii * np.sin(angles) + center_y

        # Add positional scatter (square with side length = radius_scatter)
        x_coords += np.random.uniform(-radius_scatter/2, radius_scatter/2, points_per_ring)
        y_coords += np.random.uniform(-radius_scatter/2, radius_scatter/2, points_per_ring)

        # Collect coordinates for this ring
        x_coords_all.append(x_coords)
        y_coords_all.append(y_coords)

    # Concatenate all coordinates into single numpy arrays
    x_coords_all = np.concatenate(x_coords_all)
    y_coords_all = np.concatenate(y_coords_all)
    all_points = np.column_stack((x_coords_all, y_coords_all))

    # Filter points if a filter function is provided
    return np.array(list(filter(is_a_good_point, all_points)))

def generate_rings_complete_vectorized(circles: np.ndarray,
                                      points_per_ring: int = 500,
                                      radius_scatter: float = 0.01,
                                      bounds: Tuple[float, float, float, float] = (0, 1, 0, 1),
                                      verbose: bool = False) -> np.ndarray:
    """
    Vectorized generation of rings with both radial and positional scattering.

    Creates points with:
    1. Radial variation: Points vary in distance from center (ring width)
    2. Positional jitter: Points get random XY displacements (positional noise)

    Args:
        circles: Array of shape (N,3) [x_center, y_center, radius]
        points_per_ring: Points to generate per circle
        radius_scatter: Maximum scatter distance (both radial and positional)
        bounds: (x_min, x_max, y_min, y_max) boundaries for filtering
        verbose: Print generation statistics

    Returns:
        Filtered (x,y) points as numpy array of shape (M,2)

    Raises:
        ValueError: For invalid inputs
    """
    # Input validation
    if circles.shape[1] != 3:
        raise ValueError("Circles array must have shape (N,3) with columns [x,y,r]")
    if points_per_ring < 0:
        raise ValueError("Points per ring must be non-negative")
    if radius_scatter < 0:
        raise ValueError("Radius scatter must be non-negative")
    if len(bounds) != 4 or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("Bounds must be (x_min, x_max, y_min, y_max) with min < max")

    x_min, x_max, y_min, y_max = bounds
    n_circles = len(circles)
    total_points = n_circles * points_per_ring

    # Base generation (vectorized)
    angles = np.random.uniform(0, 2*np.pi, (n_circles, points_per_ring))

    # Radial scattering
    radii = circles[:, 2][:, np.newaxis] + np.random.uniform(
        -radius_scatter, radius_scatter, (n_circles, points_per_ring))

    # Convert to Cartesian with centers
    x_base = radii * np.cos(angles) + circles[:, 0][:, np.newaxis]
    y_base = radii * np.sin(angles) + circles[:, 1][:, np.newaxis]

    # Positional scattering (jitter in both directions)
    x_jitter = np.random.uniform(-radius_scatter/2, radius_scatter/2,
                                (n_circles, points_per_ring))
    y_jitter = np.random.uniform(-radius_scatter/2, radius_scatter/2,
                                (n_circles, points_per_ring))

    # Apply jitter
    x_coords = x_base + x_jitter
    y_coords = y_base + y_jitter

    # Combine and filter
    all_points = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    # Vectorized boundary check
    in_bounds = (
        (all_points[:, 0] >= x_min) &
        (all_points[:, 0] <= x_max) &
        (all_points[:, 1] >= y_min) &
        (all_points[:, 1] <= y_max))
    filtered_points = all_points[in_bounds]

    if verbose:
        kept_ratio = len(filtered_points) / total_points
        print(f"Generated {n_circles * points_per_ring:,} points")
        print(f"Kept {len(filtered_points):,} points ({kept_ratio:.1%})")
        print(f"Positional scatter: ±{radius_scatter/2:.4f} units")
        print(f"Radial scatter: ±{radius_scatter:.4f} units\n")

    return filtered_points


# ======================== CLUSTERING ALGORITHM =========================== #

def filter_labels(labels: Union[List[int], np.ndarray],
                 min_points: int,
                 verbose: bool = False) -> Tuple[np.ndarray, int]:

    """
    Filters a list of labels based on a minimum points threshold.

    Args:
    labels: A list of integer labels.
    min_points: The minimum number of points required for a label to be considered valid.

    Returns:
    A tuple containing:
        - The filtered list of labels with invalid labels set to -1.
        - The number of valid (surviving) labels.
    Raises:
        TypeError: If labels is not a list or numpy array
        ValueError: If min_points is negative
    """

    # Input validation
    if not isinstance(labels, (list, np.ndarray)):
        raise TypeError("Labels must be a list or numpy array")
    if min_points < 0:
        raise ValueError("min_points must be non-negative")

    # Count occurrences of each label
    label_counts = Counter(labels)

    # Number of initial clusters
    num_initial_clusters = len(label_counts) - (1 if -1 in labels else 0)

    if verbose:
        print("Initial clusters:", num_initial_clusters)

    # Filter labels based on min_points
    filtered_labels = np.array([
        label if label_counts[label] >= min_points else -1
        for label in labels
    ])

    # Count valid (surviving) labels
    num_valid_labels = len(set(filtered_labels)) - (1 if -1 in labels else 0)

    if verbose:
        print("Final clusters:  ", num_valid_labels)

    return filtered_labels, num_valid_labels

def adaptive_clustering(points: np.ndarray,
                        min_clusters: int = 3,
                        max_clusters: int = 4,
                        initial_eps: float = 1.0,
                        min_samples: int = 5,
                        max_iter: int = 30,
                        initial_zoom: float = 2.0,
                        verbose: bool = False) -> Tuple[np.ndarray, int]:
    """
    Adjusts the eps parameter for DBSCAN until the number of clusters is within the desired range.

    Parameters:
        points (np.ndarray): The points to cluster. Must be a 2D array of shape (N, 2).
        min_clusters (int): Minimum acceptable number of clusters. Must be greater than 0.
        max_clusters (int): Maximum acceptable number of clusters.
            Must be greater than min_clusters.
        initial_eps (float): Starting value for eps. Must be positive.
        min_samples (int): DBSCAN min_samples parameter. Must be greater than 3.
        max_iter (int): Maximum number of iterations. Must be greater than 0.
        initial_zoom (float): Initial zoom factor for adjusting eps. Must be greater than 1.
        verbose (bool): If True, print detailed progress and results.

    Returns:
        Tuple[np.ndarray, int]:
            - labels: Cluster labels for each point.
            - cluster_count: Number of clusters found (excluding noise).

    Raises:
        ValueError: If any parameter constraints are violated
    """
    # Input validation
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must be a 2D NumPy array of shape (N, 2).")
    if min_clusters <= 0 or max_clusters < min_clusters:
        raise ValueError("min_clusters must be > 0 and <= than max_clusters.")
    if initial_eps <= 0:
        raise ValueError("initial_eps must be positive.")
    if min_samples < 3:
        raise ValueError("min_samples must be at least 3.")
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0.")
    if initial_zoom <= 1:
        raise ValueError("initial_zoom must be > 1")

    assert MIN_DBSCAN_EPS > 0, "MIN_DBSCAN_EPS must be positive"
    assert MAX_DBSCAN_EPS > MIN_DBSCAN_EPS, "MAX_DBSCAN_EPS must be greater than MIN_DBSCAN_EPS"

    if verbose:
        print(f"Target clusters: {min_clusters} to {max_clusters}\n")

    # Handle zoom direction constants and value
    zoom_dir = 0  # Initial zoom direction (neutral)
    zoom_up = 1  # Zoom direction for increasing eps (merging clusters)
    zoom_down = -1  # Zoom direction for decreasing eps (splitting clusters)
    zoom = initial_zoom  # Initial zoom factor for adjusting eps

    current_eps = initial_eps  # Start with the initial eps value
    for iteration in range(max_iter):
        # Perform DBSCAN clustering with the current eps value
        clustering = DBSCAN(eps=current_eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_  # Get cluster labels for each point

        # cleanup labels and count (real) clusters
        labels, cluster_count = filter_labels(labels, min_samples, verbose=verbose)

        # Print current iteration details if verbose mode is enabled
        if verbose:
            print(f"AC: Iteration {iteration+1:2d}: current_eps = {current_eps:.4f}, "
                  f"clusters = {cluster_count:2d}, dir = {zoom_dir:2d}, zoom = {zoom:.4f}")

        if cluster_count < min_clusters:
            # Too few clusters -> decrease current_eps to split clusters.
            if zoom_dir == zoom_up:
                zoom = 2 * zoom / (zoom + 1)  # Smoothly adjust zoom factor if direction changes
            current_eps /= zoom
            zoom_dir = zoom_down

        elif cluster_count > max_clusters:
            # Too many clusters -> increase current_eps to merge clusters.
            if zoom_dir == zoom_down:
                zoom = (zoom + 1) / 2  # Smoothly adjust zoom factor if direction changes
            current_eps *= zoom
            zoom_dir = zoom_up
        else:
            # Clusters within the desired range, break out of the loop
            break

        if current_eps < MIN_DBSCAN_EPS or current_eps > MAX_DBSCAN_EPS:
            if verbose:
                print("AC: DBSCAN eps out of range")
            break

    if verbose:
        # Compute number of unclustered points (noise)
        num_unclustered = np.sum(labels == -1)

        # Print clustering results
        print(f"\nFound {cluster_count} clusters\n")
        if num_unclustered > 0:
            print(f'There are {num_unclustered} unclustered points\n')
        else:
            print('There are NO unclustered points\n')  # No noise points
        print(f"Final eps: {current_eps:.4f}")  # Final eps value used for clustering

        # Print the number of points in each cluster
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points
            cluster_points = points[labels == label]
            print(f"Cluster {label}: {len(cluster_points)} points")

    return labels, cluster_count

def create_cluster_dict(points: np.ndarray,
                       labels: np.ndarray,
                       verbose: bool = False) -> Tuple[dict, set]:
    """
    Creates a structured dictionary representation of clustered point data.

    Organizes points into labeled clusters with metadata for subsequent processing.
    Handles noise points (label = -1) separately from valid clusters.

    Parameters:
        points (np.ndarray): 2D array of shape (N, 2) containing point coordinates.
        labels (np.ndarray): 1D array of cluster labels for each point.
        verbose (bool): If True, prints cluster statistics.

    Returns:
        Tuple[dict, set]:
            - cluster_dict: Dictionary with cluster metadata
            - unique_labels: Set of all unique labels found

    Cluster Dictionary Structure:
            label: {
                'labels': [label],        # Cluster identifier(s)
                'points': np.ndarray,    # Array of point coordinates
                'size': int,              # Number of points in cluster
                'circle': None,           # Will store fitted circle params
                'errors': None,          # Will store fitting errors
                'rmse': None,             # Will store fitting RMSE
                'valid': bool            # False for noise (label=-1)
            },

    """

    # Input validation
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must be a 2D numpy array of shape (N, 2)")

    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        raise ValueError("Labels must be a 1D numpy array")

    if len(points) != len(labels):
        raise ValueError("Points and labels must have the same length")

    # Get all unique labels
    unique_labels = set(labels)

    # Build cluster dictionary
    cluster_dict = {}

    # scan all labels
    for label in unique_labels:

        # Select the points in the cluster
        cluster_points = points[labels == label]

        # Number of points in the cluster
        num_points = len(cluster_points)

        if verbose:
            print(f"CCD: Cluster {label:2d}: {num_points:4d} points")

        cluster_dict[label] = {
            'labels': [label],
            'points': cluster_points,
            'size'  : len(cluster_points),
            'circle': None,
            'errors': None,
            'rmse'  : None,
            'valid' : label >= 0
        }

    if verbose:
        print("\n")
        # Calculate the number of valid clusters (excluding noise) and noise points
        valid_clusters = [label for label in unique_labels if label >= 0]
        num_valid_clusters = len(valid_clusters)
        num_noise_points = len(points[labels == -1]) if -1 in unique_labels else 0
        print(f"CCD: Found {num_valid_clusters} clusters and {num_noise_points} noise points")

    return cluster_dict, unique_labels

def filter_fitted_clusters(cluster_dict: Dict[int, Dict[str, Any]],
                         verbose: bool = False) -> Dict[int, Dict[str, Any]]:
    """
    Updates the cluster_dict by filtering fitted circles based on quality criteria.
    The validity flag in each cluster is updated based on the result of is_a_good_circle.

    Args:
        cluster_dict (dict): Cluster dictionary created by create_cluster_dict.
        verbose (bool): If True, prints details during filtering.

    Returns:
        dict: The updated cluster dictionary with the 'valid' field set according to the filtering.

    Raises:
        TypeError: If cluster_dict is not a dictionary or contains invalid types.
    """
    # Input validation
    if not isinstance(cluster_dict, dict):
        raise TypeError("cluster_dict must be a dictionary")

    # Calculate average RMSE across clusters that are already valid and have a finite rmse.
    avg_rmse = np.mean(np.array([
        c['rmse'] for c in cluster_dict.values()
        if c['valid'] and c['rmse'] is not None and np.isfinite(c['rmse'])
        ]))

    for key, cluster in cluster_dict.items():
        # Skip clusters already marked as invalid (e.g. noise) from previous steps.
        if not cluster['valid']:
            if verbose:
                print(f"\nCluster {key}: Pre-marked invalid. Skipping filtering.")
            continue

        # Retrieve fitted data from the cluster.
        circle = cluster['circle']
        errors = cluster['errors']
        rmse = cluster['rmse']

        # Check circle quality using average RMSE.
        is_valid = is_a_good_circle(circle, errors, rmse, avg_rmse=avg_rmse)
        cluster['valid'] = is_valid

    return cluster_dict

def compatible_clusters(cdict: Dict[int, Dict],
                       i: int,
                       j: int,
                       sigma: float = 3.0,
                       verbose: bool = False) -> bool:
    """
    Determines if two clusters are compatible for merging based on their circle fits.

    Args:
        cdict (dict): Cluster dictionary containing circle fit information
        i (int): First cluster ID
        j (int): Second cluster ID
        sigma (float): Multiplier for error-based compatibility threshold (default: 3.0)
        verbose (bool): Whether to print comparison details (default: False)

    Returns:
        bool: True if clusters are compatible for merging, False otherwise

    Raises:
        KeyError: If either cluster ID is not found in the dictionary
        ValueError: If sigma is negative
    """
    # Basic input validation
    if i not in cdict or j not in cdict:
        raise KeyError(f"Cluster IDs {i} or {j} not found in dictionary")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    # Calculate absolute differences between circle parameters
    c_diff = np.abs(cdict[i]['circle'] - cdict[j]['circle'])
    # Sum of errors from both clusters
    e_sum = cdict[i]['errors'] + cdict[j]['errors']

    # Center compatibility check
    center_distance = np.linalg.norm(c_diff[:2])
    center_threshold = sigma * np.linalg.norm(e_sum[:2])

    # Radius compatibility check
    radius_difference = c_diff[2]
    radius_threshold = sigma * e_sum[2]

    # Determine compatibility
    is_compatible = center_distance < center_threshold and radius_difference < radius_threshold

    if verbose:
        print(f"\nComparing clusters {i} and {j}:")
        print(f"Center distance: {center_distance:.4f} (threshold: {center_threshold:.4f})")
        print(f"Radius difference: {radius_difference:.4f} (threshold: {radius_threshold:.4f})")

        if not is_compatible:
            reasons = []
            if center_distance >= center_threshold:
                reasons.append(f"center distance {center_distance:.4f}"
                               f">= threshold {center_threshold:.4f}")
            if radius_difference >= radius_threshold:
                reasons.append(f"radius difference {radius_difference:.4f}"
                               f">= threshold {radius_threshold:.4f}")
            print(f"Clusters not merged because: {' and '.join(reasons)}")
        else:
            print("Clusters are compatible for merging")

    return is_compatible

def compare_and_merge_clusters(cluster_dict: Dict[int, Dict],
                               sigma: float = 3.0,
                               verbose: bool = False
                               ) -> Dict[int, Dict]:
    """
    Merges compatible clusters based on spatial and radial proximity within error bounds.
    Maintains the cluster with the lowest label as valid and marks others as merged.

    Args:
        cluster_dict (Dict[int, Dict]): Dictionary of clusters with keys as cluster
            IDs and values as cluster parameters
        sigma (float): Multiplier for error-based compatibility threshold (must be >= 0)
        verbose (bool): Enable detailed logging

    Returns:
        Updated cluster dictionary with merged clusters

    Raises:
        ValueError: If sigma is negative
    """

    # Validate input parameters
    if sigma < 0:
        raise ValueError("Sigma must be a non-negative value")

    # Create working copy to avoid modifying during iteration
    merged_dict = cluster_dict.copy()

    # keep track of all merged keys
    processed_keys: Set[int] = set()

    # Get sorted list of valid cluster IDs
    sorted_keys = sorted(cluster_dict.keys())

    # scan all keys in sorted order
    for current_key in sorted_keys:
        current_cluster = merged_dict[current_key]
        if not current_cluster['valid'] or current_key in processed_keys:
            continue

        # keep track of (current) merged keys
        merged_keys = [current_key]

        # Compare with other clusters
        for other_key in sorted_keys:
            other_cluster = merged_dict[other_key]
            if not other_cluster['valid'] or other_key\
                <= current_key or other_key in processed_keys:
                continue

            # Check compatibility
            are_compatible = compatible_clusters(merged_dict,
                                                 current_key, other_key, sigma, verbose)
            if verbose:
                print(f"Clusters {current_key} and {other_key} are compatible? {are_compatible}")

            # Merge if compatible
            if are_compatible:
                merged_points = np.vstack([
                    current_cluster['points'],
                    other_cluster['points']
                ])

                # Fit new circle to merged points
                fit_result = fit_circle_to_points(merged_points)
                if fit_result is None:
                    continue

                new_circle, new_errors, new_rmse = fit_result

                # Only accept merge if RMSE doesn't increase
                if new_rmse <= max(current_cluster['rmse'], other_cluster['rmse']):
                    current_cluster.update({
                        'points': merged_points,
                        'circle': new_circle,
                        'errors': new_errors,
                        'rmse': new_rmse
                    })

                    if verbose:
                        print(f"Merging clusters: [{current_key},"
                              f"{other_key}] into: cluster {current_key}")
                        print(f"New RMSE: {new_rmse:.4f}\n")

                    # append other_key to merged_keys
                    merged_keys.append(other_key)

        if len(merged_keys) == 1:
            current_cluster['merged_from'] = None
            # PAPO non serve c'è la copia globale all'inizio
            #merged_dict[current_key] = current_cluster.copy()
        else:
            current_cluster['merged_from'] = merged_keys

            # Mark non-primary members as merged
            for member_key in merged_keys[1:]:
                merged_dict[member_key]['valid'] = False
                merged_dict[member_key]['merged_into'] = current_key

        # add merged_keys to processed set
        processed_keys.update(merged_keys)

    return merged_dict

# ======================== POINT EXTRACTION UTILITIES ====================== #

def extract_points(sample_points: np.ndarray,
                   circle: np.ndarray,
                   errors: np.ndarray,
                   sigma_threshold: float = 3.0,
                   radius_scatter: float = 0.01,
                   context: str = "cluster",
                   identifier: Optional[Union[int, str]] = None,
                   verbose: bool = False) -> np.ndarray:
    """
    Selects points within dynamic boundaries of a fitted circle based on context.

    Implements a dual-mode filtering algorithm that:
    - For 'cluster' context: Uses error-bounded selection with fallback scatter
    - For 'outliers' context: Uses statistical deviation-based thresholds

    Args:
        sample_points (np.ndarray): Input points to filter
        circle (np.ndarray): Circle parameters [cx, cy, r]
        errors (np.ndarray): Circle errors [err_x, err_y, err_r]
        sigma_threshold (float): Multiplier for error boundaries
        radius_scatter (float): Fallback radial scatter
        context (str): Usage context ('cluster' or 'outliers')
        identifier (Optional[Union[int, str]]): Cluster ID or other identifier for verbose
        verbose (bool): Show processing details

    Returns:
        Boolean mask of selected points

    Raises:
        ValueError: For invalid input shapes or values
    """

    # Input Validation
    # Check parameter ranges
    if sigma_threshold <= 0:
        raise ValueError("sigma_threshold must be positive")
    if radius_scatter < 0:
        raise ValueError("radius_scatter cannot be negative")
    if context not in ["cluster", "outliers"]:
        raise ValueError("context must be either 'cluster' or 'outliers'")

     # Parameter Extraction
    c_x, c_y, radius = circle
    err_x, err_y, err_r = errors

    # Boundary Calculations

    # This is another way I am doing it, more efficiency but less precision
    # # Combined center error (Euclidean distance)
    # center_error = sigma_threshold * np.sqrt(err_x**2 + err_y**2)
    # # Radius boundary is either: sigma-scaled radius error OR
    # # minimum radius scatter (whichever is larger)
    # radius_boundary = max(sigma_threshold * err_r, radius_scatter)

    # # Total boundary combines center position uncertainty and radius uncertainty
    # total_boundary = radius_boundary + center_error
    # # Final selection bounds
    # lower_bound = radius - total_boundary
    # upper_bound = radius + total_boundary

    # This is the old way, less efficiency but more precise
    if context == "cluster":

        # Combined center error (Euclidean distance)
        center_error = sigma_threshold * np.sqrt(err_x**2 + err_y**2)

        # Radius boundary is either: sigma-scaled radius error OR
        # minimum radius scatter (whichever is larger)
        radius_boundary = max(sigma_threshold * err_r, radius_scatter)

        # Total boundary combines center position uncertainty and radius uncertainty
        total_boundary = radius_boundary + center_error

        # Final selection bounds
        lower_bound = radius - total_boundary
        upper_bound = radius + total_boundary

    else:

        # Calculate distances from the best_ring center
        # (uncomment to enable outlier-specific thresholds)
        std_dev = np.std(np.linalg.norm(sample_points - np.array([c_x, c_y]), axis=1))
        lower_bound = radius - sigma_threshold * std_dev  # New lower bound for outlier detection
        upper_bound = radius + sigma_threshold * std_dev  # New upper bound for outlier detection


    if verbose:
        # Context-aware header
        if context == "cluster" and identifier is not None:
            print(f"\nEvaluating Cluster {identifier}:")
        elif context == "outliers":
            print("\nExcluding outliers:")

        # Common metrics
        print(f"  - Center: ({round(c_x, 3):.3f} ± {round(err_x, 3):.3f},"
              f"{round(c_y, 3):.3f} ± {round(err_y, 3):.3f})")
        print(f"  - Radius: {round(radius, 3):.3f} ± {round(err_r, 3):.3f}")

        # Conditional prints
        if context == "cluster":
            print(f"  - Center error contribution: ±{round(center_error, 3):.3f}")

        print(f"  - Radius error bound (σ×err_r):"
              f"{round(sigma_threshold * err_r, 3):.3f} (σ = {sigma_threshold})")
        print(f"  - Radius scatter: {round(radius_scatter, 3):.3f}")

        if context == "cluster":
            print(f"  - Using boundary: {round(total_boundary, 3):.3f}"
                  f"(combined center + radius error)")

        print(f"  - Selection range: [{round(lower_bound, 3):.3f}, {round(upper_bound, 3):.3f}]")

     # Calculate distances from center
    distances = np.linalg.norm(sample_points - [c_x, c_y], axis=1)

    # Create selection mask
    mask = (distances >= lower_bound) & (distances <= upper_bound)

    if verbose:
        print(f"  - Points in boundary: {np.sum(mask)}")
        if context == "outliers":
            print(f"  - Points remaining: {np.sum(~mask)}\n")

    return mask

def extract_best_ring(cluster_dict: Dict[int, Dict[str, Any]],
                      sample_points: np.ndarray,
                      sigma_threshold: float = 3.0,
                      radius_scatter: float = 0.01,
                      verbose: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple], Optional[int]]:
    """
    Selects the best-fitting ring using combined center and radius uncertainties.

    The function:
    1. Filters valid clusters from the input dictionary
    2. Extracts points within error boundaries for each cluster
    3. Refits circles to the boundary-selected points
    4. Selects the best candidate based on refined RMSE

    Args:
        cluster_dict (Dict[int, Dict[str, Any]]): Dictionary of clusters with keys as cluster IDs
        sample_points (np.ndarray): All available points (shape [N, 2])
        sigma_threshold (float): Multiplier for error boundaries
        radius_scatter (float): Minimum radial scatter allowance
        verbose (bool): Enable detailed progress output

    Returns:
        Tuple containing:
        - Points belonging to the best ring (shape [M, 2])
        - All other points (shape [K, 2])
        - Tuple of (refined_circle, refined_errors, refined_rmse) for best ring
        - Cluster ID of the winning ring (or None if no valid rings found)
    """

    # Input validation with assertions
    assert sigma_threshold >= 0, "sigma_threshold must be non-negative"
    assert radius_scatter >= 0, "radius_scatter must be non-negative"
    assert isinstance(cluster_dict, dict), "cluster_dict must be a dictionary"
    assert sample_points.ndim == 2 and sample_points.shape[1] == 2, \
           "sample_points must be 2D array of shape [N, 2]"

    candidates = []

    # Process each cluster in the dictionary
    for cluster_id, cluster in cluster_dict.items():
        # Skip invalid clusters or those missing required data
        if not cluster['valid'] or cluster['circle'] is None or \
        cluster['errors'] is None or cluster['rmse'] is None:
            continue

        # Extract cluster parameters
        circle = cluster['circle']
        errors = cluster['errors']
        rmse = cluster['rmse']

        # Get points within error boundaries using unified extractor
        mask = extract_points(
            sample_points=sample_points,
            circle=circle,
            errors=errors,
            sigma_threshold=sigma_threshold,
            radius_scatter=radius_scatter,
            context="cluster",
            identifier=cluster_id,
            verbose=verbose
        )

        # Get candidate points within boundaries
        candidate_points = sample_points[mask]

        # Skip candidates with too few points
        if len(candidate_points) < 3:
            continue

        # Refit circle using only the boundary-selected points
        fit_result = fit_circle_to_points(candidate_points) #Remember to change
        #fit_result = fit_circle_to_points_fast(candidate_points)
        if fit_result is None: # Skip if refitting fails
            continue

        # Unpack refit results
        new_circle, new_errors, new_rmse = fit_result

        # Store candidate information
        candidates.append({
            'cluster_id': cluster_id,
            'mask': mask,
            'original_circle': circle,
            'original_errors': errors,
            'original_rmse': rmse,
            'refined_circle': new_circle,
            'refined_errors': new_errors,
            'refined_rmse': new_rmse
        })

    # Handle case where no valid candidates were found
    if not candidates:
        if verbose:
            print("No valid candidates found after boundary filtering")
        return np.empty((0, 2)), sample_points.copy(), None, None

    # Select best candidate with lowest refined RMSE
    best_candidate = min(candidates, key=lambda x: x['refined_rmse'])
    best_mask = best_candidate['mask']

    # Split points into ring members and others
    best_points = sample_points[best_mask]
    other_points = sample_points[~best_mask]

    # Package refined fit parameters
    best_fit = (
        best_candidate['refined_circle'],
        best_candidate['refined_errors'],
        best_candidate['refined_rmse']
    )

    # Verbose output showing comparison between original and refined fits
    if verbose:
        print(f"Selected cluster {best_candidate['cluster_id']}:")
        err_x, err_y, err_r = best_candidate['original_errors']
        rmse = best_candidate['original_rmse']

        print_circle(
            circle=np.array(best_candidate['original_circle']),
            errors=np.array([err_x, err_y, err_r]),
            rmse=best_candidate['original_rmse'],
            title="\nOriginal cluster parameters")

        print_circle(
            best_fit[0],
            errors=best_fit[1],
            rmse=best_fit[2],
            title="\nRefined boundary fit")

        print(f"\nPoints in ring: {len(best_points)}")
        print(f"Points excluded: {len(other_points)}\n")

    return best_points, other_points, best_fit, best_candidate['cluster_id']

def exclude_outliers(best_ring: np.ndarray,
                     best_fit: Tuple[np.ndarray, np.ndarray, float],
                     other_points: Optional[np.ndarray] = None,
                     beta: float = 3,
                     verbose: bool = False
                     ) -> Union[
                         Tuple[np.ndarray, np.ndarray, Tuple, np.ndarray],
                         Tuple[np.ndarray, Optional[np.ndarray], Tuple, np.ndarray]
                     ]:
    """
    Enhanced outlier exclusion using unified extraction methodology with full compatibility.

    Handles both initial processing (without existing other_points) and iterative refinement.
    Maintains consistent return types for easy integration in processing pipelines.

    Args:
        best_ring (np.ndarray): Points belonging to current ring (shape [N, 2])
        best_fit (Tuple[np.ndarray, np.ndarray, float]): Tuple containing (circle, errors, rmse)
        other_points (Optional[np.ndarray]): Optional array of previously excluded points
            (shape [M, 2])
        beta (float): Number of standard deviations for outlier threshold (>0)
        verbose (bool): Print refinement details

    Returns:
        If other_points is None:
            (new_ring, None, new_fit, outliers)
        If other_points is provided:
            (new_ring, updated_other_points, new_fit, outliers)
    """
    # Input validation
    assert best_ring.ndim == 2 and best_ring.shape[1] == 2, "Invalid ring shape"
    assert len(best_fit) == 3, "Invalid fit format"
    assert beta > 0, "Beta must be positive"

    # Unpack fit parameters
    circle, errors, rmse = best_fit

    # Get inlier/outlier mask using unified extractor
    mask = extract_points(
        sample_points=best_ring,
        circle=circle,
        errors=errors,
        sigma_threshold=beta,
        context="outliers",
        verbose=verbose
    )

    # Split points
    inliers = best_ring[mask]
    outliers = best_ring[~mask]

    # Update other points if provided
    updated_other = np.vstack([other_points, outliers]) if other_points is not None else None

    # Refit circle only if sufficient points remain
    if len(inliers) >= 3:
        new_circle, new_errors, new_rmse = fit_circle_to_points(inliers) #Change to speed up
        #new_circle, new_errors, new_rmse = fit_circle_to_points_fast(inliers)
        new_fit = (new_circle, new_errors, new_rmse)
        if verbose:
            print(f"\nRefinement improved RMSE: {rmse:.4f} → {new_rmse:.4f}")
    else:
        new_fit = best_fit
        if verbose:
            print("Insufficient points for refitting - using original parameters")

    # Return format depends on other_points presence
    return (inliers, updated_other, new_fit, outliers) if other_points is not None \
           else (inliers, new_fit, outliers)

# ======================== EVALUATION AND COMPARISON ====================== #

def calculate_ratii(pairs: List[Tuple[int, float, int, float]],
                   original_circles: np.ndarray,
                   fitted_circles: np.ndarray,
                   fitted_errors: np.ndarray,
                   threshold: float,
                   verbose: bool = False
                   ) -> Tuple[List[List[float]], List[str]]:

    """
    Compute error ratios for a given list of fitted pairs, generating detailed messages.

    Args:
        pairs (list of tuples): Each tuple is (f_idx, tot_error, o_idx, dist).
        original_circles (np.ndarray): Array of original circles [x, y, r].
        fitted_circles (np.ndarray): Array of fitted circles [x, y, r].
        fitted_errors (np.ndarray): Array of errors [err_x, err_y, err_r].
        threshold (float): The error threshold multiplier.
        verbose (bool): If True, print messages as they are constructed.

    Returns:
        Tuple containing:
        - List of error ratios [ratio_x, ratio_y, ratio_r] for each pair
        - List of formatted message strings for each comparison
    """

    # Input validation
    assert isinstance(pairs, list), "pairs must be a list"
    assert (len(fitted_circles) == len(fitted_errors)
            ), "Input arrays must have same length"
    assert threshold > 0, "threshold must be positive"

    def format_ratio(ratio):
        return f"{ratio:.1f}" if not (np.isinf(ratio) or ratio is None) else "∞"

    ratii = []
    messages = []

    # Loop over the provided pairs
    for (o_idx, dist, f_idx, _) in pairs:
        # Check if disk is undetected
        if dist == np.inf:
            continue

        # Get data
        orig = original_circles[o_idx]
        fit = fitted_circles[f_idx]
        err = fitted_errors[f_idx]

        # Unpack the values
        orig_x, orig_y, orig_r = orig
        fit_x, fit_y, fit_r = fit
        x_err, y_err, r_err = err

        # Calculate error ratios (guard against division by zero)
        ratio_x = abs(fit_x - orig_x) / x_err if x_err > 0 else np.inf
        ratio_y = abs(fit_y - orig_y) / y_err if y_err > 0 else np.inf
        ratio_r = abs(fit_r - orig_r) / r_err if r_err > 0 else np.inf

        # Append the computed quantities
        ratii.append([ratio_x, ratio_y, ratio_r])

        # Determine overall status using the maximum ratio
        max_ratio = max(ratio_x, ratio_y, ratio_r)
        status = "COMPARABLE" if max_ratio <= threshold else "NOT COMPARABLE"
        status_color = ("\033[92m" if status == "COMPARABLE" else "\033[91m"
                        )  # Green for comparable, red for not

        msg = (
            f"\nComparison Result (Fitted {f_idx} vs Original {o_idx}):\n"
            f"  Original: Center=({orig_x:.4f}, {orig_y:.4f}), Radius={orig_r:.4f}\n"
            f"  Fitted:   Center=({fit_x:.4f}±{x_err:.4f}, {fit_y:.4f}±{y_err:.4f}), "
            f"Radius={fit_r:.4f}±{r_err:.4f}\n"
            f"  Error Ratios (σ): X={format_ratio(ratio_x)}, Y={format_ratio(ratio_y)},"
                f"R={format_ratio(ratio_r)}\n"
            f"  {status_color}{status}\033[0m (Threshold: {threshold}σ, Max ratio:"
                f"{format_ratio(max_ratio)}σ)"
        )
        messages.append(msg)
        if verbose:
            print(msg)

    return ratii, messages


def find_fitting_pairs(original_circles: np.ndarray,
                                   new_best_fit: tuple,
                                   threshold: float = 3,
                                   verbose: bool = False
                                   ) -> tuple:
    """
    Matches original circles with fitted circles based on geometric proximity
    and relative error thresholds. Computes diagnostic ratios comparing fitted
    parameters with their uncertainties.

    The function assigns each original circle to the closest fitted circle
    whose Euclidean distance is within a given threshold times the total
    fitting uncertainty.

    Args:
        original_circles (np.ndarray): Array of original circles with shape (N, 3),
            where each row is [x, y, r].
        new_best_fit (tuple): Tuple of (fitted_circles, fitted_errors, _) where:
            - fitted_circles (np.ndarray): Array of circles with shape (N, 3), [x, y, r].
            - fitted_errors (np.ndarray): Array of uncertainties with shape (N, 3),
                [err_x, err_y, err_r].
            - _: Placeholder for optional additional data (e.g., RMSEs), unused here.
        threshold (float): Maximum allowed distance-to-error ratio for a valid match (default: 3).
        verbose (bool): If True, print diagnostic messages during the matching process.

    Returns:
        tuple:
            - fitted_pairs (np.ndarray): Array of tuples (original_idx,
                distance, fitted_idx, total_error),
              one for each original circle. If no match was found,
                distance = np.inf and fitted_idx = None.
            - good_ratii (np.ndarray): Array of shape (M, 3) containing error ratios
                [ratio_x, ratio_y, ratio_r]
              for each successfully matched pair.
            - messages (list of str): Formatted diagnostic strings describing each match.
    """
    # Validate threshold
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    # Unpack the fitted circles and errors
    fitted_circles, fitted_errors, _ = new_best_fit

    if not (isinstance(fitted_circles, np.ndarray) and isinstance(fitted_errors, np.ndarray)):
        raise TypeError("fitted_circles and fitted_errors must be numpy arrays")

    if fitted_circles.shape != fitted_errors.shape:
        raise ValueError("fitted_circles and fitted_errors must have the same shape")

    if verbose:
        print("=== Original Circles ===")
        print_circles(original_circles, title="Original Circles", label="Original Circle")
        print("\n=== Fitted Circles ===")
        print_circles(fitted_circles, errors=fitted_errors,
                      title="Fitted Circles with Errors", label="Fitted Circle")
        print("\n")

    # Compute the total error for each fitted circle
    tot_errors = np.linalg.norm(fitted_errors, axis=1)

    # Containers for accepted and rejected pairs
    fitted_pairs = []

    # Loop over each original circle #...and assign it by
    # comparing the center distance against threshold * tot_error.
    for o_idx, o_circle in enumerate(original_circles):

        # Set initial distance
        min_distance = np.inf
        min_f_idx = None
        min_tot_err = None

        # Find nearest fitted circle for this original circle
        for f_idx, f_circle in enumerate(fitted_circles):

            # Compute the distance
            dist = np.linalg.norm(o_circle - f_circle)

            # Compute total error for f_circle
            tot_err = tot_errors[f_idx]

            if verbose:
                print(f"Original Circle {o_idx} matching with Fitted {f_idx}"
                        f"with distance {dist:.4f} and total error {tot_err:.4f}")
                print(f"Distance/Error = {dist/tot_err:.2f} (Threshold: {threshold})")
                print("-"*80)

            # Check if o_circle and f_circle are compatibles
            if dist > threshold * tot_err:
                if verbose:
                    print(f"Fitted circle {f_idx}-> REJECTED (Exceeds error threshold)")
                    print("_"*80 + "\n\n")
                continue

            if verbose:
                print(f"Fitted circle {f_idx} -> VALID (Within error bounds)")
                print("_"*80 + "\n\n")

            # If the distance is the minimum found so far
            if dist < min_distance:
                # remember the minimum
                min_distance = dist
                min_f_idx = f_idx
                min_tot_err = tot_err

        # if min_f_idx != None
        if min_f_idx is not None:
            fitted_pairs.append((o_idx, min_distance, min_f_idx, min_tot_err))

            if verbose:
                print(f"Fitted circle {min_f_idx} -> SELECTED")
                print("_"*80 + "\n\n")
        else:
            fitted_pairs.append((o_idx, np.inf, None, None))

            if verbose:
                print(f"Original circle {o_idx} -> NOT DETECTED (No matching circle)")
                print("_"*80 + "\n\n")

    # Calculate ratii for accepted (good) circles
    #good_closest, good_ratii_x, good_ratii_y, good_ratii_r, good_msgs = calculate_ratii(
    good_ratii, good_msgs = calculate_ratii(
        fitted_pairs, original_circles, fitted_circles, fitted_errors, threshold, verbose=False
    )

    # Build overall messages list – you can interleave or separate as needed.
    messages = []
    messages.append("="*25 + " ACCEPTED (Good) Fits "+ "="*25)
    messages.extend(good_msgs)

    # Convert lists to numpy arrays where appropriate
    return np.array(fitted_pairs), np.array(good_ratii), messages
