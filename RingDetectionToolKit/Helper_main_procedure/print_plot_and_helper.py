# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
#
# For license terms see LICENSE file.
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software under GPLv2+
# See https://www.gnu.org/licenses/gpl-2.0.html

"""
Comprehensive Visualization and Analysis Toolkit

This module provides integrated utilities for:
- Textual reporting of circle parameters and statistics
- Graphical visualization of points, circles and clusters
- Quality assessment of detected rings
- Statistical analysis and efficiency evaluation
- Dictionary-based cluster inspection and plotting
"""

# Standard library imports
from typing import Union, Optional, Dict, Tuple, Any, Iterable

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    # Text Reporting
    'print_circle',
    'print_circles',
    'calculate_and_print_statistics',

    # Quality Assessment
    'is_a_good_circle',

    # Color Management
    'get_color',

    # Core Plotting
    'plot_commons',

    # Point Visualization
    'plot_points_base', 
    'plot_points',

    # Circle Visualization
    'plot_circle',
    'plot_circles',

    # Statistical Visualization
    'plot_ratii_histograms',

    # Cluster Inspection
    'show_dictionary',

    # Efficiency Analysis
    'analyze_ratii_efficiency'
]

# ============================ Text Reporting ============================ #

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

# =========================== Color Management =========================== #

def get_color(index: int = -1, cmap_name: str = "jet",
              num_distinct_colors: int = 15) -> Tuple[float, float, float, float]:
    """
    Returns a color from a specified colormap based on an index. Ensures distinct colors
    for the first 'num_distinct_colors' indices and cycles through the colormap for larger indices.

    Args:
        index (int, optional): An index to determine the color. If negative,
            the function cycles through the colormap.
        cmap_name (str, optional): Name of the colormap to use (default: "jet").
        num_distinct_colors (int, optional): Number of distinct colors to ensure for
            small indices (default: 15).

    Returns:
        - tuple: An RGBA color tuple, where each value is in the range [0, 1].

    Raises:
        ValueError: If 'num_distinct_colors' is less than 1.
    """
    if num_distinct_colors < 1:
        raise ValueError("num_distinct_colors must be at least 1.")

    if not hasattr(get_color, "cmap") or get_color.cmap.name != cmap_name:
        # Initialize or update the colormap if it doesn't exist or if the name has changed
        get_color.cmap = plt.get_cmap(cmap_name)
        get_color.counter = 0  # Counter for cycling colors

    # If a specific index is given, reset the counter
    if index >= 0:
        get_color.counter = index

    # Normalize the index to the range [0, 1] based on num_distinct_colors
    normalized_index = (get_color.counter % num_distinct_colors
                        ) / (num_distinct_colors - 1) # to avoid cycling

    # Update counter for the next call
    get_color.counter += 1

    # Return the corresponding color
    return get_color.cmap(normalized_index)

# ========================== Core Plotting Utilities ========================== #

def plot_commons(title: Optional[str] = None, hold: bool = True,
                 legend: bool = False) -> None:
    """
    Common plotting settings for all plots.

    This function sets the aspect ratio, axis labels, tick marks,
    grid, bounds, and legend if specified.

    Args:
        title (str, optional): Title of the plot.
        bound (bool, optional): Whether to set axis limits to (0, 1) for both axes.
            Defaults to True.
        hold (bool, optional): Whether to hold the plot
            (if False, the plot is displayed immediately). Defaults to True.
        legend (bool, optional): Whether to show legend with a custom bounding box.
            Defaults to False.
    """
    # Set the aspect ratio of the plot to be equal (ensures circles are not distorted)
    plt.gca().set_aspect("equal")

    # Set the labels for the x and y axes with a specified font size
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)

    # Set the x and y axis tick marks at intervals of 0.1 from 0 to 1
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Add a title to the plot if provided, with a larger font size and bold weight
    if title:
        plt.title(title, fontsize=16, fontweight="bold")

    # Set the axis limits to (0, 1) for both x and y axes if 'bound' is True
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add a grid to the plot
    plt.grid(True)

    # Add a legend to the plot if 'legend' is True, with a custom bounding box
    if legend:
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left",
                   ncol=2, borderaxespad=0)

    # Display the plot immediately if 'hold' is False
    if not hold:
        plt.show()

# =========================== Point Visualization =========================== #

def plot_points_base(points: np.ndarray, title: Optional[str] = None,
                     color: Optional[str] = None, label: Optional[str] = None,
                     hold: bool = True) -> None:
    """
    Plots raw points with customizable options. Can also handle clustering
    by coloring points based on cluster labels.

    Args:
        points (np.ndarray): A 2D NumPy array of shape (N, 2) representing the points to plot.
        title (Optional[str]): Title of the plot. Defaults to None.
        color (Optional[str]): Default color of the points. Ignored if 'cluster_labels' is provided.
            Defaults to None.
        label (Optional[str]): Label for the points (used in the legend). Defaults to None.
        cluster_labels (Optional[np.ndarray]): Cluster labels for each point.
            If provided, colors are assigned per cluster.
        hold (bool): Whether to hold the plot (if False, the plot is displayed immediately).
            Defaults to True.
    """

    # Unpack the points into x and y coordinates
    x_coords, y_coords = points[:, 0], points[:, 1]

    # Plot points with default color
    default_color = color if color else get_color()  # Fallback color
    plt.scatter(
        x_coords, y_coords, label=label,
        color=default_color if color else get_color(),
        s=30, alpha=0.8, edgecolors='w', linewidths=1
    )

    # Apply common plot settings
    plot_commons(title=title, hold=hold,
                 legend=label is not None)

def plot_points(points: np.ndarray, title: Optional[str] = None,
                color: Optional[str] = None, label: Optional[str] = None,
                cluster_labels: Optional[np.ndarray] = None,
                hold: bool = True) -> None:
    """
    Plots raw points with customizable options. Can also handle clustering
    by coloring points based on cluster labels.

    Args:
        points (np.ndarray): A 2D NumPy array of shape (N, 2) representing the points to plot.
        title (Optional[str]): Title of the plot. Defaults to None.
        color (Optional[str]): Default color of the points. Ignored if 'cluster_labels' is provided.
            Defaults to None.
        label (Optional[str]): Label for the points (used in the legend). Defaults to None.
        cluster_labels (Optional[np.ndarray]): Cluster labels for each point.
            If provided, colors are assigned per cluster.
        hold (bool): Whether to hold the plot (if False, the plot is displayed immediately).
            Defaults to True.
    """

    # Input validation
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must be a 2D NumPy array of shape (N, 2).")
    if not (cluster_labels is None or len(cluster_labels) == len(points)):
        raise ValueError("Cluster labels must have the same length as points.")

    # If cluster labels are provided, plot points with cluster colors
    if cluster_labels is not None:
        unique_labels = np.unique(cluster_labels)
        for lbl in unique_labels:
            # Assign colors to clusters, gray for noise
            cluster_color = 'gray' if lbl == -1 else get_color(lbl)
            cluster_points = points[cluster_labels == lbl]

            plot_points_base(cluster_points,
                color=cluster_color,
                label=f"Cluster {lbl}" if lbl != -1 else "Noise",
            )
    else:
        # Plot points with default color
        default_color = color if color else get_color()  # Fallback color
        plot_points_base(points,
            color=default_color if color else get_color(),
            label=label
        )

    # Apply common plot settings
    plot_commons(title=title, hold=hold,
                 legend=(label is not None or cluster_labels is not None))

# =========================== Circle Visualization =========================== #

def plot_circle(circle: np.ndarray, title: Optional[str] = None,
                color: str = 'blue', linestyle: str = 'solid',
                label: Optional[str] = None, center_as_o: bool = False,
                hold: bool = True, linewidth: int = 3) -> None:
    """
    Plots a single circle given as a NumPy array [x_c, y_c, radius].

    Args:
        circle (np.ndarray): A NumPy array containing the circle's center
            (x_c, y_c) and radius (radius).
        title (Optional[str]): Title of the plot. Defaults to None.
        color (str): Color of the circle. Defaults to 'blue'.
        linestyle (str): Line style of the circle (e.g., 'solid', 'dashed'). Defaults to 'solid'.
        label (Optional[str]): Label for the circle (used in the legend). Defaults to None.
        center_as_o (bool): Whether to mark the center with 'o' instead of '+'. Defaults to False.
        hold (bool): Whether to hold the plot (if False, the plot is displayed immediately).
            Defaults to True.
        linewidth (int): Width of the circle's edge. Defaults to 3.
    """
    # Input validation
    if not isinstance(circle, np.ndarray) or circle.shape != (3,):
        raise ValueError("Circle must be a NumPy array of shape (3,): [x_c, y_c, radius].")

    x_c, y_c, radius = circle
    if radius <= 0:
        raise ValueError("Radius must be a positive number.")

    # Create a Circle object with the specified properties
    circ = plt.Circle((x_c, y_c), radius, color=color, linestyle=linestyle,
                      fill=False, linewidth=linewidth, label=label)

    # Add the Circle object to the current plot
    plt.gca().add_artist(circ)

    # Always plot the center
    center_marker = 'o' if center_as_o else '+'
    plt.scatter(x_c, y_c, color=color, marker=center_marker, s=100, label=None)

    # Apply common plot settings
    plot_commons(title=title, hold=hold, legend=label is not None)

def plot_circles(circles: np.ndarray, title: Optional[str] = None,
                 color: Optional[str] = None, linewidth: int = 3,
                 label: Optional[str] = None,
                 enum: Optional[np.ndarray] = None,
                 center_as_o: bool = False, hold: bool = True,
                 linestyle: Optional[str] = 'solid') -> None:
    """
    Plots multiple circles with customizable options.

    Args:
        circles (np.ndarray): A NumPy array of circles, where each row is [x_c, y_c, radius].
        title (Optional[str]): Title of the plot. Defaults to None.
        color (Optional[str]): Default color of the circles. If None, a unique color is
            assigned to each circle. Defaults to None.
        linewidth (int): Width of the circles' edges. Defaults to 3.
        label (Optional[str]): Base label for the circles (used in the legend). If 'enum'
            is provided, the label will be appended with the enumeration value.
            Defaults to None.
        enum (Optional[np.ndarray]): Custom enumeration labels for the circles.
            If provided, it must have the same length as 'circles'. Defaults to None.
        center_as_o (bool): Whether to mark the centers with 'o' instead of '+'. Defaults to False.
        hold (bool): Whether to hold the plot (if False, the plot is displayed immediately).
            Defaults to True.
        linestyle (Optional[str]): Line style of the circles (e.g., 'solid', 'dashed', 'dotted').
            Defaults to 'solid'.
    """
    # Input validation
    if not (isinstance(circles, np.ndarray) and circles[0].shape[0] == 3):
        raise ValueError("Circles must be a NumPy array of shape (N, 3): [x_c, y_c, radius].")
    if np.any(circles[:, 2] <= 0):
        raise ValueError("All radii must be positive numbers.")

    # Validate enum length if provided
    if enum is not None:
        if not isinstance(enum, np.ndarray):
            raise TypeError("The 'enum' argument must be a NumPy array of enumeration labels.")
        if len(enum) != len(circles):
            raise ValueError("The length of the enum array must match the number of circles.")

    # Plot all circles
    for i, circle in enumerate(circles):
        # Pick a color
        this_color = color if color else get_color(i)

        # Prepare label
        if enum is not None:
            # Use the custom enumeration value
            enum_label = enum[i]
            this_label = f"{label} {enum_label}" if label else str(enum_label)
        elif label is not None:
            # Use the base label with default enumeration (e.g., "Label 0", "Label 1")
            this_label = f"{label} {i}"
        else:
            # No label
            this_label = None

        # Plot single circle
        plot_circle(
            circle=circle, title=None, color=this_color, linestyle=linestyle,
            label=this_label, center_as_o=center_as_o, hold=True, linewidth=linewidth
        )

    # Apply common plot settings
    plot_commons(title=title, hold=hold, legend=(label is not None or enum is not None))

# ========================== Statistical Visualization ========================== #


def plot_ratii_histograms(ratii_x, ratii_y, ratii_r, bins=100):
    """
    Plots histograms for the ratii_x, ratii_y, and ratii_r values.

    Args:
        ratii_x (np.ndarray): NumPy array of ratio_x values.
        ratii_y (np.ndarray): NumPy array of ratio_y values.
        ratii_r (np.ndarray): NumPy array of ratio_r values.
        bins (int): Number of bins for the histograms.

    Raises:
        ValueError: If any input is not a NumPy array or if the arrays do not have the same length.
    """
    # Validate that all ratii are numpy arrays
    if not (isinstance(ratii_x, np.ndarray)
            and isinstance(ratii_y, np.ndarray)
            and isinstance(ratii_r, np.ndarray)):
        raise ValueError("All inputs must be NumPy arrays.")

    # Validate that all arrays have the same length
    if not len(ratii_x) == len(ratii_y) == len(ratii_r):
        raise ValueError("All ratii arrays must have the same length.")

    plt.figure(figsize=(12, 6))

    # Histogram for ratio_x
    plt.subplot(1, 3, 1)
    plt.hist(ratii_x, bins=bins, color='blue', alpha=0.7)
    plt.title("Ratio X")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Histogram for ratio_y
    plt.subplot(1, 3, 2)
    plt.hist(ratii_y, bins=bins, color='green', alpha=0.7)
    plt.title("Ratio Y")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Histogram for ratio_r
    plt.subplot(1, 3, 3)
    plt.hist(ratii_r, bins=bins, color='red', alpha=0.7)
    plt.title("Ratio R")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# ========================== Cluster Inspection ========================== #

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