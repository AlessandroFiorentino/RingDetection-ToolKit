# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
#
# For license terms see LICENSE file.
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software under GPLv2+
# See https://www.gnu.org/licenses/gpl-2.0.html

"""
Circle Fitting Algorithms

Provides multiple implementations for fitting circles to 2D point data:
1. Iterative Least Squares - Robust but slower (fit_circle_to_points)
2. Crawford's Algebraic Method - Fast approximation (fit_circle_to_points_fast)
3. Cluster-level wrappers - For processing grouped points
"""

# ============================== IMPORTS ==================================== #

# Standard library imports
from typing import Dict, Optional, Tuple

# Third-party imports
import numpy as np
from scipy.optimize import least_squares

# ============================== CONSTANTS ================================== #

MIN_SAMPLES = 5  # Minimum points required for circle fitting
DEBUG = False    # Global debug flag for additional output

# ============================== MODULE EXPORTS ============================= #

__all__ = [
    'fit_circle_to_points',
    'fit_circle_to_points_fast',
    'fit_circles_to_clusters',
    'fit_circles_to_clusters_fast'
]

# ======================== CORE FITTING FUNCTIONS =========================== #

def fit_circle_to_points(points: np.ndarray,
                         initial_center: Optional[Tuple[float, float]] = None,
                         initial_radius: Optional[float] = None,
                         verbose: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Fits a circle to a set of 2D points using least squares optimization.

    Args:
        points (np.ndarray): A 2D array of points with shape (N, 2),
            where N is the number of points.
        initial_center (Optional[Tuple[float, float]]): Precomputed initial center (c_x, c_y).
            If provided, it will be used directly.
        initial_radius (Optional[float]): Precomputed initial radius.
            If provided, it will be used directly.
        verbose (bool): If True, prints additional information about the fitting process.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, float]]: A tuple containing:
            - [fitted_cx, fitted_cy, fitted_r]: The fitted circle's center coordinates and radius.
            - [cx_error, cy_error, radius_error]: The estimated errors in c_x, c_y, and radius.
            - rmse: The root mean squared error of the fit, calculated as the square root of
              the residual sum of squares divided by the degrees of freedom.

        Returns None if the fit fails to converge or if there are not enough points to fit a circle.
    """
    def residuals(params: np.ndarray) -> np.ndarray:
        """
        Computes the residuals (differences between observed and predicted radii)
            for the circle fit.

        Args:
            params (np.ndarray): A 1D array containing the circle parameters [xc, yc, radius].

        Returns:
            np.ndarray: The residuals (observed radii - predicted radius).
        """
        x_center, y_center, radius = params  # Renamed variables to snake_case
        radii = np.sqrt((points[:, 0] - x_center) ** 2 + (points[:, 1] - y_center) ** 2)
        return radii - radius

    # Check for minimum number of points to fit a circle
    num_points = len(points)

    if num_points < MIN_SAMPLES:
        raise ValueError(f"FIT: Not enough points to fit a circle: {num_points}.")


    # Compute initial guess for center as the mean of the points
    center_guess = np.mean(points, axis=0) \
                   if initial_center is None else initial_center
    # Compute initial guess for radius as the mean distance from the center
    radius_guess = np.mean(np.sqrt(np.sum((points - center_guess)**2, axis=1))) \
                   if initial_radius is None else initial_radius
    initial_guess = np.append(center_guess, radius_guess)

    # Perform least squares optimization
    result = least_squares(residuals, initial_guess)

    # Return None if fails to converge
    if result.status <= 0:
        if verbose:
            print(f"FIT: Fit failed: {result.status}")
        return None, None, None
    if verbose:
        print(f"FIT: Status = {result.status}, Cost = {result.cost/num_points:.4e}, ", end='')

    # Extract results
    residuals = result.fun  # Residuals from the fit
    residual_sum_squares = np.sum(residuals ** 2)  # Renamed variable to snake_case
    degrees_freedom = len(points) - len(initial_guess)  # Renamed variable to snake_case
    rmse = np.sqrt(residual_sum_squares / degrees_freedom)  # Root mean squared error

    if verbose:
        print(f"RMSE = {rmse:.4f}\n")

    # Compute covariance matrix and parameter errors
    try:
        jacobian = result.jac  # Renamed variable to snake_case
        jacobian_transpose_jacobian = jacobian.T @ jacobian  # Renamed variable to snake_case
        if np.linalg.matrix_rank(jacobian_transpose_jacobian
                                 ) < jacobian_transpose_jacobian.shape[0]:
            raise np.linalg.LinAlgError("Singular matrix encountered in covariance calculation.")

        # Compute covariance matrix using pseudo-inverse to handle potential singularities
        covariance_matrix = np.linalg.pinv(jacobian_transpose_jacobian
                                           ) * residual_sum_squares / degrees_freedom
        param_errors = np.sqrt(np.diag(covariance_matrix))  # Extract parameter errors

    except np.linalg.LinAlgError:
        if verbose:
            # Handle cases where covariance calculation fails (e.g., singular matrix)
            print("FIT: Covariance calculation failed due to a singular matrix. Errors set to NaN.")
        param_errors = np.array([np.nan, np.nan, np.nan])

    if DEBUG:
        print(f"FIT: Covariance_matrix =\n{np.diag(covariance_matrix)}")
        print(f"FIT: Points: {len(points)}, Cost: {result.cost:.4f}, RMSE: {rmse:.4f}")

    return result.x, param_errors, rmse

def fit_circle_to_points_fast(points: np.ndarray,
                             verbose: bool = False
                             ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:

    """
    Fast circle fitting using a non-iterative method inspired by Crawford's algorithm.
    Maintains the same interface as fit_circle_to_points but uses a different algorithm.

    Implements the algebraic circle fit method from:
    jac.F. Crawford, "A non-iterative method for fitting circular arcs to measured points",
    Nuclear Instruments and Methods 211 (1983) 223-225:
    https://www.sciencedirect.com/science/article/pii/0167508783905756?via%3Dihub

    Args:
        points (np.ndarray): 2D array of shape (N, 2) containing points to fit
        initial_center: Ignored (exists for API compatibility)
        initial_radius: Ignored (exists for API compatibility)
        verbose (bool): If True, prints debugging information

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, float]]: A tuple containing:
            - [fitted_cx, fitted_cy, fitted_r]: The fitted circle's center coordinates and radius.
            - [cx_error, cy_error, radius_error]: The estimated errors in c_x, c_y, and radius.
            - rmse: The root mean squared error of the fit, calculated as the square root of
              the residual sum of squares divided by the degrees of freedom.

        Returns None if the fit fails to converge or if there are not enough points to fit a circle.
    """
    # Input Validation
    if len(points) < MIN_SAMPLES:
        raise ValueError(f"FIT: Not enough points to fit circle: {len(points)}")

    x_coord = points[:, 0]
    y_coord = points[:, 1]
    n_points = len(points)

    # --- Core fitting algorithm ---
    # Center coordinates relative to centroid
    x_m = np.mean(x_coord)
    y_m = np.mean(y_coord)
    u_mat = x_coord - x_m
    v_mat = y_coord - y_m

    # Build linear system
    sum_uu = u_mat.T @ u_mat  # Sum of u_mat*u_mat
    sum_vv = v_mat.T @ v_mat  # Sum of v_mat*v_mat
    sum_uv = u_mat.T @ v_mat  # Sum of u_mat*v_mat
    sum_uuu = u_mat.T @ u_mat**2  # Sum of u_mat^3
    sum_vvv = v_mat.T @ v_mat**2  # Sum of v_mat^3
    sum_uvv = u_mat.T @ v_mat**2  # Sum of u_mat*v_mat^2
    sum_vuu = v_mat.T @ u_mat**2  # Sum of v_mat*u_mat^2

    # Construct linear system
    mat_a = np.array([[sum_uu, sum_uv], [sum_uv, sum_vv]])
    vec_b = np.array([sum_uuu + sum_uvv, sum_vvv + sum_vuu]) / 2

    try:
        # Solve for center coordinates (relative to centroid)
        c_u, c_v = np.linalg.solve(mat_a, vec_b)

        # Transform back to original coordinate system
        c_x = c_u + x_m
        c_y = c_v + y_m

        # Calculate radius using completed squares formula
        radius = np.sqrt(c_u**2 + c_v**2 + (sum_uu + sum_vv)/len(x_coord))

    except np.linalg.LinAlgError as error:
        if verbose:
            print(f"FIT: Matrix solve failed: {str(error)}")
        return None, None, None

    # Calculate residuals (distances from points to circle)
    residuals = np.hypot(x_coord - c_x, y_coord - c_y) - radius
    rss = residuals.T @ residuals
    rmse = np.sqrt(rss / (n_points - 3)) # RMSE with 3 DOF lost

    if verbose:
        print(f"FIT: Cost = {rss/len(points):.4e}, RMSE = {rmse:.4f}")

    # Build Jacobian matrix for covariance estimation
    d_x = c_x - x_coord
    d_y = c_y - y_coord
    dist = np.hypot(d_x, d_y)
    valid = dist > 1e-8  # Avoid division by zero

    jac = np.empty((n_points, 3))
    jac[valid, 0] = d_x[valid]/dist[valid]  # ∂r/∂cx
    jac[valid, 1] = d_y[valid]/dist[valid]  # ∂r/∂cy
    jac[valid, 2] = -1                     # ∂r/∂r
    jac[~valid] = [0, 0, -1]  # Handle degenerate cases

    # Calculate parameter covariance matrix
    try:
        cov = np.linalg.pinv(jac.T @ jac) * rss/(len(points)-3)
        errors = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        if verbose:
            print("FIT: Covariance calculation failed (singular matrix)")
        errors = np.array([np.nan, np.nan, np.nan])

    if DEBUG:
        print(f"FIT: Covariance matrix diagonal: {np.diag(cov)}")
        print(f"FIT: Points: {n_points}, RSS: {rss:.4f}, RMSE: {rmse:.4f}")

    # Package results to match original format
    return np.array([c_x, c_y, radius]), errors, rmse

# ====================== CLUSTER-LEVEL FITTING ============================== #

def fit_circles_to_clusters(cluster_dict: Dict[int, Dict],
                            verbose: bool = False
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits circles to all valid clusters in the dictionary using nonlinear least squares.

    Processes each valid cluster through the following pipeline:
    1. Validates cluster point distribution
    2. Performs circle fitting via fit_circle_to_points()
    3. Updates cluster metadata with fit results
    4. Marks clusters with failed fits as invalid

    Parameters:
        cluster_dict (Dict[int, Dict]): A dictionary of clusters (from create_cluster_dict)
        verbose (bool): Whether to print verbose output

    Returns:
        tuple: (fitted_circles, fitted_errors, fitted_rmses, fitted_labels)

    Raises:
        ValueError: If cluster_dict has invalid structure

    """

    # Validate input dictionary structure
    if not isinstance(cluster_dict, dict):
        raise ValueError("cluster_dict must be a dictionary")

    for key, cluster in cluster_dict.items():
        # Skip invalid clusters (noise points)
        if not cluster['valid']:
            if verbose:
                print(f"FCC: Skipping noise cluster {key}")
            continue

        if verbose:
            print(f"FCC: Fitting circle to cluster {key}")

        # Fit circle to cluster points
        fit_results = fit_circle_to_points(cluster['points'])

        if fit_results is None:
            if verbose:
                print(f"\nFCC: Failed to fit circle to cluster {key}")
            cluster['valid'] = False
            continue

        # Unpack results
        circle, errors, rmse = fit_results

        # Update cluster dictionary
        cluster.update({
            'circle': circle,
            'errors': errors,
            'rmse': rmse,
            'valid': circle is not None
        })

def fit_circles_to_clusters_fast(cluster_dict: Dict[int, Dict],
                            verbose: bool = False
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rapidly fits circles to clusters using Crawford's non-iterative algebraic method.

    Optimized version of fit_circles_to_clusters() that uses direct matrix solving
    instead of iterative least squares. Provides 5-10x speed improvement for large
    clusters, but the accuracy is lower.

    Parameters:
        cluster_dict (Dict[int, Dict]): A dictionary of clusters (from create_cluster_dict)
        verbose (bool): Whether to print verbose output

    Returns:
        tuple: (fitted_circles, fitted_errors, fitted_rmses, fitted_labels)

    Raises:
        ValueError: If cluster_dict has invalid structure

    """

    # Validate input dictionary structure
    if not isinstance(cluster_dict, dict):
        raise ValueError("cluster_dict must be a dictionary")

    for key, cluster in cluster_dict.items():
        # Skip invalid clusters (noise points)
        if not cluster['valid']:
            if verbose:
                print(f"FCC: Skipping noise cluster {key}")
            continue

        if verbose:
            print(f"FCC: Fitting circle to cluster {key}")

        # Fit circle to cluster points
        fit_results = fit_circle_to_points_fast(cluster['points'])

        if fit_results is None:
            if verbose:
                print(f"\nFCC: Failed to fit circle to cluster {key}")
            cluster['valid'] = False
            continue

        # Unpack results
        circle, errors, rmse = fit_results

        # Update cluster dictionary
        cluster.update({
            'circle': circle,
            'errors': errors,
            'rmse': rmse,
            'valid': circle is not None
        })
