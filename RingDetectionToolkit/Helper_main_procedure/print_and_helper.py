# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
#
# For license terms see LICENSE file.
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software under GPLv2+
# See https://www.gnu.org/licenses/gpl-2.0.html

"""
Visualization and Analysis Utilities

This module provides functions for:
- Circle parameter visualization and reporting
- Statistical analysis of fitting results
- Quality assessment of detected rings
- Dictionary inspection and plotting
"""

# Standard library imports
#from collections import Counter
from typing import Union, Optional, Dict, Tuple, Any, Iterable


# Third-party imports
import numpy as np

__all__ = [
    # Core Printing Functions
    'print_circle',
    'print_circles',

    # Statistical Reporting
    'calculate_and_print_statistics',

    # Quality Assessment
    'is_a_good_circle',

    # Dictionary Visualization
    'show_dictionary',

    # Efficiency Analysis
    'analyze_ratii_efficiency'
]

# ========================== Core Printing Functions ========================== #

def print_circle(circle: Union[np.ndarray, list[float]],
                 errors: Optional[np.ndarray] = None,
                 title: Optional[str] = None,
                 label: Optional[str] = None,
                 rmse: Optional[float] = None) -> None:
    """
    Prints the details of a circle, including its center, radius, errors (if provided),
    and RMSE (if provided).

    Args:
        circle (np.ndarray): A NumPy array representing the circle as [x, y, r], where:
                             - x, y are the coordinates of the center.
                             - r is the radius.
        errors (np.ndarray, optional): A NumPy array representing the errors as:
            [err_x, err_y, err_r].
        title (str, optional): A title to print before the circle details.
        label (str, optional): A label to prepend to the circle details (e.g., "Cluster 1").
        rmse (float, optional): The Root Mean Square Error (RMSE).

    Raises:
        ValueError: If the length of the errors array does not match the length of the circle array.
    """
    # Input validation
    if not isinstance(circle, np.ndarray) or circle.shape != (3,):
        raise ValueError("Circle must be a NumPy array of shape (3,): [x, y, r].")
    if errors is not None and not (isinstance(errors, np.ndarray) and errors.shape == (3,)):
        raise ValueError("Errors must be a NumPy array of shape (3,): [err_x, err_y, err_r].")

    # Print the title if provided
    if title:
        print(title)

    # Print the label if provided, using efficient formatting
    print(f"{f'{label}: ' if label else ''}", end='')

    # Print the circle details with errors (if provided)
    if errors is not None:
        print(
            f"Center = ({circle[0]:.4f} ± {errors[0]:.4f}, "
            f"{circle[1]:.4f} ± {errors[1]:.4f}), "
            f"Radius = {circle[2]:.4f} ± {errors[2]:.4f}"
            f"{f', RMSE: {rmse:.4f}' if rmse is not None else ''}"
        )
    else:
        print(
            f"Center = ({circle[0]:.4f}, {circle[1]:.4f}), "
            f"Radius = {circle[2]:.4f}"
            f"{f', RMSE: {rmse:.4f}' if rmse is not None else ''}"
        )

def print_circles(circles: np.ndarray,
                  errors: Optional[np.ndarray] = None,
                  title: Optional[str] = None,
                  label: Optional[str] = None,
                  enum: Optional[np.ndarray] = None,
                  rmse: Optional[np.ndarray] = None) -> None:
    """
    Print details for multiple circles, including their centers, radii,
    and optional errors and RMSE.

    This function generalizes 'print_circle' to handle multiple circles.
    The 'enum' argument can be used to provide custom enumeration labels
    for each circle. If 'enum' is not provided, circles are enumerated
    from 0 to n-1.

    Args:
        circles (np.ndarray): A NumPy array of circles,
            where each circle is represented as [x, y, r].
        errors (Optional[np.ndarray]): A NumPy array of errors, where each row is:
            [err_x, err_y, err_r]. Defaults to None.
        title (Optional[str]): A title to print before the circle details. Defaults to None.
        label (Optional[str]): A label to prepend to each circle's details (e.g., "Cluster").
            Defaults to None.
        enum (Optional[np.ndarray]): A NumPy array of custom enumeration labels for each circle.
            Defaults to None.
        rmse (Optional[np.ndarray]): A NumPy array of RMSE values for each circle. Defaults to None.

    Raises:
        ValueError: If the input arrays have incorrect shapes or lengths. Then,
        if the lengths of 'errors', 'rmse', or 'enum' do not match the number of circles.
    """

    if circles.shape[1] != 3:
        raise ValueError("Circles must be a NumPy array of shape (3,) or (N, 3): [x, y, r].")

    # If 'errors' is provided, ensure it has the same shape as 'circles'
    if errors is not None:
        if errors.shape[1] != 3:
            raise ValueError("Errors must be a NumPy array of the same shape as circles: (N, 3).")

    # Validate the length of rmse if provided
    if rmse is not None and len(rmse) != len(circles):
        raise ValueError("The length of the rmse array must match the number of circles.")

    # If enum is not provided, generate default enumeration
    if enum is None:
        enum = np.arange(len(circles))  # Use NumPy array for enumeration
    elif len(enum) != len(circles):
        raise ValueError("The length of the enum array must match the number of circles.")

    # Print the title if provided
    if title:
        print(title)

    # Iterate over the circles and print each one
    for i, circle in enumerate(circles):
        # Get the errors and RMSE for the current circle (if provided)
        err = errors[i] if errors is not None else None
        rmse_value = rmse[i] if rmse is not None else None

        # Determine the label for the current circle
        enum_label = enum[i]
        lbl = f"{label} {enum_label}" if label else enum_label

        # Print the circle using the print_circle function
        print_circle(circle, errors=err, label=lbl, rmse=rmse_value)

# ========================== Statistical Reporting ========================== #

def calculate_and_print_statistics(ratii_x_array: np.ndarray,
                                   ratii_y_array: np.ndarray,
                                   ratii_r_array: np.ndarray) -> None:
    """
    Calculates and prints the average (mean), standard deviation,
    and standard error of the mean (SEM) for ratii_x, ratii_y, and ratii_r.

    Args:
        ratii_x_array (np.ndarray): Array of ratio_x values.
        ratii_y_array (np.ndarray): Array of ratio_y values.
        ratii_r_array (np.ndarray): Array of ratio_r values.

    Raises:
        ValueError: If the lengths of the input arrays are not the same.
    """
    # Check if the lengths of the input arrays are the same
    if len(ratii_x_array) != len(ratii_y_array) or len(ratii_x_array) != len(ratii_r_array):
        raise ValueError("The lengths of ratii_x, ratii_y, and ratii_r must be the same.")

    # Calculate the number of samples
    num_samples = len(ratii_x_array)

    # Calculate mean and standard deviation for each array
    mean_x, std_x = np.mean(ratii_x_array), np.std(ratii_x_array, ddof=1)
    mean_y, std_y = np.mean(ratii_y_array), np.std(ratii_y_array, ddof=1)
    mean_r, std_r = np.mean(ratii_r_array), np.std(ratii_r_array, ddof=1)

    # Calculate standard error of the mean (SEM)
    sem_x = std_x / np.sqrt(num_samples)
    sem_y = std_y / np.sqrt(num_samples)
    sem_r = std_r / np.sqrt(num_samples)

    # Print the results
    print("\nStatistics for Ratii:")
    print(f"Ratio X: Mean = {mean_x:.6f}, Std Dev = {std_x:.6f}, SEM = {sem_x:.6f}")
    print(f"Ratio Y: Mean = {mean_y:.6f}, Std Dev = {std_y:.6f}, SEM = {sem_y:.6f}")
    print(f"Ratio R: Mean = {mean_r:.6f}, Std Dev = {std_r:.6f}, SEM = {sem_r:.6f}")

# ========================== Quality Assessment ========================== #

def is_a_good_circle(circle: Union[np.ndarray, Tuple[float, float, float]],
                     errors: Union[np.ndarray, Tuple[float, float, float]],
                     rmse: Optional[Union[np.ndarray, float]] = None,
                     r_max: float = 0.8,
                     r_min: float = 0.2,
                     alpha: float = 3,
                     avg_rmse: Optional[float] = None) -> bool:

    """
    Determines if a circle is "good" based on its parameters and errors.

    Args:
        circle: The circle parameters as [cx, cy, r] (NumPy array or tuple)
        errors: The errors as [err_x, err_y, err_r] (NumPy array or tuple)
        rmse: The RMSE of the circle fit (NumPy array or float, optional)
        r_max: Maximum expected radius (default: 0.8)
        r_min: Minimum expected radius (default: 0.2)
        alpha: Multiplier for avg_rmse in RMSE condition (default: 3)
        avg_rmse: Average RMSE across all clusters (required if rmse is provided)

    Returns:
        bool: True if the circle meets all quality conditions, False otherwise
    """

    # Input validation
    assert alpha > 0, "alpha must be positive"
    assert not (rmse is not None and avg_rmse is
                None), "avg_rmse must be provided when rmse is given"

    # Check if radius is None
    if circle is None or circle[0] is None or not(np.isfinite(circle[0])
                                                  and np.isfinite(errors[0])):
        return False

    # Extract parameters safely
    _, _, radius = circle #_arr
    _, _, err_r = errors #_arr

    # Original conditions
    cond1 = radius < 1.5 * r_max
    cond2 = radius > 0.75 * r_min
    cond3 = err_r / radius < 0.5
    conditions = [cond1, cond2, cond3]

    # RMSE validation
    if rmse is not None and avg_rmse is not None:
        try:
            rmse_val = float(rmse)
            conditions.append(rmse_val < alpha * avg_rmse)
        except (ValueError, TypeError):
            return False

    return all(conditions)

# ========================== Dictionary Visualization ========================== #

def show_dictionary(cluster_dict: Dict[int, Dict[str, Any]],
                   cluster_keys: Iterable[int],
                   title: Optional[str] = None,
                   plt_points: bool = True,
                   plt_circles: bool = True,
                   prt_circles: bool = False,
                   hold: bool = False) -> None:
    """
    Plots the clusters from a cluster dictionary with enhanced noise handling.

    Args:
        cluster_dict (dict): The cluster dictionary created by `create_cluster_dict`
        cluster_keys (iterable): Keys of clusters to plot
        title (str, optional): Title of the plot
        plot_points (bool): Whether to plot the points
        plot_circles (bool): Whether to plot fitted circles
        hold (bool): Whether to hold the plot

    Raises:
        TypeError: If input types are incorrect
        ValueError: If cluster data is malformed
    """

    # Input validation
    if not isinstance(cluster_dict, dict):
        raise TypeError("cluster_dict must be a dictionary")
    if not hasattr(cluster_keys, '__iter__'):
        raise TypeError("cluster_keys must be iterable")
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None")

    # sort cluster keys (noise plotted first)
    cluster_keys = sorted(cluster_keys)

    # print circles
    if prt_circles:
        print("\nFitted Circles")
        for key in cluster_keys:
            cluster = cluster_dict[key]
            if cluster['circle'] is None or not cluster['valid']:
                continue

            print_circle(
                cluster['circle'],
                errors=cluster['errors'],
                rmse=cluster['rmse'],
                label = f"Cluster {key}"
            )

    # plot points (noise first)
    if plt_points:
        for key in cluster_keys:
            cluster = cluster_dict[key]

            points = cluster['points']

            color = get_color(key) if cluster['valid'] else 'gray'
            label = f"Cluster {key}" if cluster['valid'] else 'Noise'
            plot_points_base(points, color=color, label=label)

    # plot circles
    if plt_circles:
        for key in cluster_keys:
            cluster = cluster_dict[key]

            if cluster['circle'] is None or not cluster['valid']:
                continue

            color = get_color(key)
            label = f"Cluster {key}"

            plot_circle(
                cluster['circle'],
                color=color,
                linestyle='dashed',
                center_as_o=True
            )

    plot_commons(title=title, hold=hold, legend=True)

# ============================ Efficiency Analysis ============================ #

def analyze_ratii_efficiency(combined_ratii: np.ndarray,
                             num_rings: int,
                             num_seeds: int) -> Tuple[float, float, float]:
    """
    Computes efficiency statistics based on the number of detected rings.

    Args:
        combined_ratii (np.ndarray): Array of accepted (good) ratii with shape (N, 3).
        num_rings (int): Number of expected rings per seed.
        num_seeds (int): Number of independent seeds/runs.

    Returns:
        Tuple containing:
            - total_efficiency (float): Percentage of good rings over total expected.
    """

    # Validate inputs
    if not isinstance(num_rings, int) or num_rings < 1:
        raise ValueError("num_rings must be a positive integer")
    if not isinstance(num_seeds, int) or num_seeds < 1:
        raise ValueError("num_seeds must be a positive integer")

    # Count detected good rings
    count_good = combined_ratii.shape[0]

    # Compute total number of expected rings
    total_expected = num_rings * num_seeds

    # Compute efficiency as percentage
    total_efficiency = (count_good / total_expected) * 100

    # Print diagnostic summary
    print(f"\nTotal expected rings: {total_expected}")
    print(f"Good rings found: {count_good}")
    print(f"Total Efficiency: {total_efficiency:.2f}%")

    return total_efficiency
