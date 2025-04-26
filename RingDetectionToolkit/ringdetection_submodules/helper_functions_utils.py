# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
#
# For the license terms see the file LICENSE, distributed along with this
# software.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""General utilities.
"""

# Standard library imports
import time
import warnings
from collections import Counter
from typing import (
    Any, Dict, Iterable, List,
    Literal, Optional, Set, Tuple, Union
)

# Third-party imports
import matplotlib.pyplot as plt
#import multiprocessing as mp
import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN
#from tqdm import tqdm

# =========================== Constants ==================================== #

DEBUG = False   # Global debug flag controlling some verbose output in fit procedures

VERBOSE = False # Global verbose flag controlling verbose output

RADIUS_SCATTER = 0.01  # Variation in radius

# Minimum samples for clustering (DBSCAN parameter)
MIN_SAMPLES = 5


MIN_DBSCAN_EPS = 1e-3   # Minimum DBSCAN eps
MAX_DBSCAN_EPS = 1      # Maximum DBSCAN eps

# =========================== Printing Functions =========================== #

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


# =========================== Plotting Functions =========================== #

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

# Define valid line styles
LinestyleType = Literal['solid', 'dashed', 'dotted', 'dashdot', 'none']

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

#=================================== Helper functions ===============================


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


def find_nearest_circle(initial_circles: Union[np.ndarray, List[List[float]]],
                        circle: Union[np.ndarray, List[float]]) -> Tuple[np.ndarray, int, float]:
    """
    Finds the nearest circle in 'initial_circles' to the given 'circle'.

    Args:
        initial_circles (Union[np.ndarray, List[List[float]]]): A list or NumPy array of circles,
            where each circle is represented as [xc, yc, r].
        circle (Union[np.ndarray, List[float]]): The circle to find the nearest neighbor for,
            in the same format [xc, yc, r].

    Returns:
        Tuple[np.ndarray, int, float]: A tuple containing:
            - The nearest circle in 'initial_circles' as a NumPy array.
            - The index of the nearest circle in 'initial_circles'.
            - The Euclidean distance between the given circle and the nearest circle.

    Raises:
        ValueError: If 'initial_circles' is empty or if 'circle' does not have exactly 3 elements.
    """
    # Input validation
    if len(initial_circles) == 0:
        raise ValueError("The list of initial circles cannot be empty.")
    if len(circle) != 3:
        raise ValueError("The circle must be represented "
        "as a list or array of 3 elements: [xc, yc, r].")

    # Convert inputs to NumPy arrays for vectorized operations
    initial_circles_np = np.array(initial_circles)
    circle_np = np.array(circle)

    # Calculate Euclidean distances between the given circle and all circles in initial_circles
    # Only consider the (xc, yc) coordinates for distance calculation
    distances = np.linalg.norm(initial_circles_np[:, :2] - circle_np[:2], axis=1)

    # Find the index of the minimum distance
    nearest_circle_index = np.argmin(distances)

    return (
        initial_circles_np[nearest_circle_index],
        nearest_circle_index,
        distances[nearest_circle_index]
    )

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

#=================================== Generate rings ========================

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

def generate_rings(circles: np.ndarray,
                   points_per_ring: int = 500,
                   radius_scatter: float = 0.01) -> np.ndarray:
    """
    Generates point clouds representing circular rings with controlled scatter.

    Creates a set of points for each input circle, with points randomly distributed
    around each ring's circumference with controlled radial variation.

    Args:
        circles (np.ndarray): Array of shape (N,3) where each row contains:
            [x_center, y_center, radius] defining a circle
        points_per_ring (int): Number of points to generate per circle (default: 500)
        radius_scatter (float): Maximum radial variation from perfect circle (default: 0.01)
            - Points are generated with radii in [radius-scatter, radius+scatter]
            - Must be non-negative

    Returns:
        np.ndarray: A numpy array of (x, y) coordinates representing
            all generated points for the rings.

    Raises:
        ValueError: If input validation fails:
            - circles array has incorrect shape (not Nx3)
            - points_per_ring is negative
            - radius_scatter is negative
    """

    def is_a_good_point(point: np.ndarray) -> bool:
        """Filter function to ensure points are within the bounds [0, 1] for both x and y.
        """
        return point[0] >= 0 and point[0] <= 1 and point[1] >= 0 and point[1] <= 1

    # Input validation
    if circles.shape[1] != 3:
        raise ValueError("Circles array must have shape (N,3) with columns [x,y,r]")
    if points_per_ring < 0:
        raise ValueError("Points per ring must be non-negative")
    if radius_scatter < 0:
        raise ValueError("Radius scatter must be non-negative")

    x_coords_all = []
    y_coords_all = []

    for center_x, center_y, ring_radius in circles:
        # Generate random angles and radii for the ring
        angles = np.random.uniform(0, 2 * np.pi, points_per_ring)
        radii = ring_radius + np.random.uniform(-radius_scatter, radius_scatter, points_per_ring)

        # Convert polar coordinates to Cartesian coordinates
        x_coords = radii * np.cos(angles) + center_x
        y_coords = radii * np.sin(angles) + center_y

        x_coords_all.append(x_coords)
        y_coords_all.append(y_coords)

    # Concatenate all coordinates into single numpy arrays
    x_coords_all = np.concatenate(x_coords_all)
    y_coords_all = np.concatenate(y_coords_all)
    all_points = np.column_stack((x_coords_all, y_coords_all))

    # Filter points if a filter function is provided
    return np.array(list(filter(is_a_good_point, all_points)))

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

# ============================== Main Algorithm ================================

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
        labels, cluster_count = filter_labels(labels, min_samples, verbose=True)

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

# ================================= Fit procedures ===============================

def fit_circle_to_points(points: np.ndarray,
                         initial_center: Optional[Tuple[float, float]] = None,
                         initial_radius: Optional[float] = None,
                         verbose: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Fits a circle to a set of 2D points using least squares optimization.

    Args:
        points (np.ndarray): A 2D array of points with shape (N, 2),
            where N is the number of points.
        initial_center (Optional[Tuple[float, float]]): Precomputed initial center (c_x, cy).
            If provided, it will be used directly.
        initial_radius (Optional[float]): Precomputed initial radius.
            If provided, it will be used directly.
        verbose (bool): If True, prints additional information about the fitting process.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, float]]: A tuple containing:
            - [fitted_c_x, fitted_c_y, fitted_r]: The fitted circle's center coordinates and radius.
            - [c_x_error, c_y_error, radius_error]: The estimated errors in c_x, c_y, and radius.
            - rmse: The root mean squared error of the fit, calculated as the square root of
              the residual sum of squares divided by the degrees of freedom.

        Returns None if the fit fails to converge or if there are not enough points to fit a circle.
    """
    def residuals(params: np.ndarray) -> np.ndarray:
        """
        Computes the residuals (differences between observed and predicted radii)
            for the circle fit.

        Args:
            params (np.ndarray): A 1D array containing the circle parameters [xc, yc, r].

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
        # Handle cases where covariance calculation fails (e.g., singular matrix)
        print("FIT: Covariance calculation failed due to a singular matrix. Errors set to NaN.")
        param_errors = np.array([np.nan, np.nan, np.nan])

    if DEBUG:
        print(f"FIT: Covariance_matrix =\n{np.diag(covariance_matrix)}")
        print(f"FIT: Points: {len(points)}, Cost: {result.cost:.4f}, RMSE: {rmse:.4f}")

    return result.x, param_errors, rmse

def fit_circle_to_points_fast(points: np.ndarray,
                             initial_center: Optional[Tuple[float, float]] = None,
                             initial_radius: Optional[float] = None,
                             verbose: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:

    """
    Fast circle fitting using a non-iterative method inspired by Crawford's algorithm.
    Maintains the same interface as fit_circle_to_points but uses a different algorithm.

    Implements the algebraic circle fit method from:
    J.F. Crawford, "A non-iterative method for fitting circular arcs to measured points", 
    Nuclear Instruments and Methods 211 (1983) 223-225:
    https://www.sciencedirect.com/science/article/pii/0167508783905756?via%3Dihub

    Args:
        points: 2D array of shape (N, 2) containing points to fit
        initial_center: Ignored (exists for API compatibility)
        initial_radius: Ignored (exists for API compatibility)
        verbose: If True, prints debugging information

    Returns:
        Tuple containing (matches fit_circle_to_points format):
        - circle: Array [cx, cy, r] of fitted parameters
        - errors: Array [cx_err, cy_err, r_err] of parameter uncertainties
        - rmse: Root mean squared error of fit
    """
    # Input Validation
    if len(points) < MIN_SAMPLES:
        raise ValueError(f"FIT: Not enough points to fit circle: {len(points)}")

    x = points[:, 0]
    y = points[:, 1]
    n_points = len(points)

    # --- Core fitting algorithm ---
    # Center coordinates relative to centroid
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m

    # Build linear system
    Suu = u.T @ u  # Sum of u*u
    Svv = v.T @ v  # Sum of v*v
    Suv = u.T @ v  # Sum of u*v
    Suuu = u.T @ u**2  # Sum of u^3
    Svvv = v.T @ v**2  # Sum of v^3
    Suvv = u.T @ v**2  # Sum of u*v^2
    Svuu = v.T @ u**2  # Sum of v*u^2

    # Construct linear system: A * [cu; cv] = B
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Svuu]) / 2

    try:
        # Solve for center coordinates (relative to centroid)
        cu, cv = np.linalg.solve(A, B)

        # Transform back to original coordinate system
        cx = cu + x_m
        cy = cv + y_m

        # Calculate radius using completed squares formula
        r = np.sqrt(cu**2 + cv**2 + (Suu + Svv)/len(x))

    except np.linalg.LinAlgError as e:
        if verbose:
            print(f"FIT: Matrix solve failed: {str(e)}")
        return None, None, None

    # Calculate residuals (distances from points to circle)
    residuals = np.hypot(x - cx, y - cy) - r
    rss = residuals.T @ residuals
    rmse = np.sqrt(rss / (n_points - 3)) # RMSE with 3 DOF lost

    if verbose:
        print(f"FIT: Cost = {rss/len(points):.4e}, RMSE = {rmse:.4f}")

    # Build Jacobian matrix for covariance estimation
    dx = cx - x
    dy = cy - y
    dist = np.hypot(dx, dy)
    valid = dist > 1e-8  # Avoid division by zero

    J = np.empty((n_points, 3))
    J[valid, 0] = dx[valid]/dist[valid]  # ∂r/∂cx
    J[valid, 1] = dy[valid]/dist[valid]  # ∂r/∂cy
    J[valid, 2] = -1                     # ∂r/∂r
    J[~valid] = [0, 0, -1]  # Handle degenerate cases

    # Calculate parameter covariance matrix
    try:
        cov = np.linalg.pinv(J.T @ J) * rss/(len(points)-3)
        errors = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        if verbose:
            print("FIT: Covariance calculation failed (singular matrix)")
        errors = np.array([np.nan, np.nan, np.nan])

    if DEBUG:
        print(f"FIT: Covariance matrix diagonal: {np.diag(cov)}")
        print(f"FIT: Points: {n_points}, RSS: {rss:.4f}, RMSE: {rmse:.4f}")

    # Package results to match original format
    return np.array([cx, cy, r]), errors, rmse


def fit_circles_to_clusters(cluster_dict: Dict[int, Dict],
                            verbose: bool = False
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit circles to the given clusters using a dictionary-based approach.

    Parameters:
        cluster_dict (dict): A dictionary of clusters (from create_cluster_dict)
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
                print(f"FCC: Failed to fit circle to cluster {key}")
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

# ================================== Dictionary for clusters ======================

def create_cluster_dict(points, labels, verbose=False):
    """
    Create a dictionary of clusters.

    Parameters:
        - points (numpy.ndarray): Array of data points.
        - labels (numpy.ndarray): Array of cluster labels.

    Returns:
        - dict: Dictionary of clusters.
    """

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
        # Calculate the number of valid clusters (excluding noise) and noise points
        valid_clusters = [label for label in unique_labels if label >= 0]
        num_valid_clusters = len(valid_clusters)
        num_noise_points = len(points[labels == -1]) if -1 in unique_labels else 0
        print(f"CCD: Found {num_valid_clusters} clusters and {num_noise_points} noise points")

    return cluster_dict, unique_labels

# ================================= Filtering function ===========================

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
                print(f"Cluster {key}: Pre-marked invalid. Skipping filtering.")
            continue

        # Retrieve fitted data from the cluster.
        circle = cluster['circle']
        errors = cluster['errors']
        rmse = cluster['rmse']

        # Check circle quality using average RMSE.
        is_valid = is_a_good_circle(circle, errors, rmse, avg_rmse=avg_rmse)
        cluster['valid'] = is_valid

    return cluster_dict

def compare_and_merge_clusters(cluster_dict: Dict[int, Dict],
                               sigma: float = 3.0,
                               verbose: bool = False
                               ) -> Dict[int, Dict]:
    """
    Merges compatible clusters based on spatial and radial proximity within error bounds.
    Maintains the cluster with the lowest label as valid and marks others as merged.

    Args:
        cluster_dict: Dictionary of clusters with keys as cluster
            IDs and values as cluster parameters
        sigma: Multiplier for error-based compatibility threshold (must be >= 0)
        verbose: Enable detailed logging

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

# =========================== Extraction of the best ring ==========================

def extract_points(sample_points: np.ndarray,
                   circle: np.ndarray,
                   errors: np.ndarray,
                   sigma_threshold: float = 3.0,
                   radius_scatter: float = 0.01,
                   context: str = "cluster",
                   identifier: Optional[Union[int, str]] = None,
                   verbose: bool = False) -> np.ndarray:
    """
    Universal point extraction function with context-aware output.

    Args:
        sample_points: Input points to filter
        circle: Circle parameters [cx, cy, r]
        errors: Circle errors [err_x, err_y, err_r]
        sigma_threshold: Multiplier for error boundaries
        radius_scatter: Fallback radial scatter
        context: Usage context ('cluster' or 'outliers')
        identifier: Cluster ID or other identifier for verbose
        verbose: Show processing details

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

    if context == "outliers":
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
        print(f"  - Center error contribution: ±{round(center_error, 3):.3f}")
        print(f"  - Radius error bound (σ×err_r):"
              f"{round(sigma_threshold * err_r, 3):.3f} (σ = {sigma_threshold})")
        print(f"  - Radius scatter: {round(radius_scatter, 3):.3f}")
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
            print(f"  - Points remaining: {np.sum(~mask)}")

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
        cluster_dict: Dictionary of clusters with keys as cluster IDs
        sample_points: All available points (shape [N, 2])
        sigma_threshold: Multiplier for error boundaries
        radius_scatter: Minimum radial scatter allowance
        verbose: Enable detailed progress output

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
        #fit_result = fit_circle_to_points(candidate_points) Remember to change
        fit_result = fit_circle_to_points(candidate_points)
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
                     other_points: np.ndarray,
                     best_fit: Tuple[np.ndarray, np.ndarray, float],
                     beta: float = 3,
                     verbose: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple], np.ndarray]:

    """
    Refines the best ring by excluding outliers and returns updated results.

    Args:
        best_ring: Points belonging to the current best ring (N, 2)
        other_points: Other excluded points (M, 2)
        best_fit: Tuple containing (circle, errors, rmse)
                  where circle = [c_x, c_y, r], errors = [err_x, err_y, err_r]
        beta: Number of standard deviations for outlier threshold
        verbose: Print refinement details

    Returns:
        tuple: (new_best_ring, new_other_points, new_best_fit, outliers)
    """
    # Unpack current best fit parameters
    circle, _, _ = best_fit
    c_x, c_y, radius = circle

    # Calculate distances from current center
    distances = np.linalg.norm(best_ring - np.array([c_x, c_y]), axis=1)
    std_dev = np.std(distances)

    # Calculate outlier thresholds
    upper_threshold = radius + beta * std_dev
    lower_threshold = radius - beta * std_dev

    # Identify inliers and outliers
    inlier_mask = (distances >= lower_threshold) & (distances <= upper_threshold)
    new_best_ring = best_ring[inlier_mask]
    outliers = best_ring[~inlier_mask]

    # Update other points
    new_other_points = np.vstack([other_points, outliers]) if outliers.size > 0 else other_points

    # Refit circle to inliers if possible
    new_best_fit = None
    if new_best_ring.shape[0] >= 3:
        new_best_fit = fit_circle_to_points(new_best_ring)
        if verbose and new_best_fit:
            print(f"Refinement improved RMSE from {best_fit[2]:.4f} to {new_best_fit[2]:.4f}")
    else:
        if verbose:
            print("Not enough points for refinement - using original fit")
        new_best_fit = best_fit  # Fallback to original fit

    if verbose:
        print(f"Original points: {len(best_ring)}")
        print(f"New best points: {len(new_best_ring)}")
        print(f"Outliers removed: {len(outliers)}")
        print(f"Total other points: {len(new_other_points)}\n")

    return new_best_ring, new_other_points, new_best_fit, outliers

def evaluate_fit_comparability(original_circles: np.ndarray,
                               new_best_fit: Tuple[np.ndarray, np.ndarray, np.ndarray],
                               threshold: float = 3,
                               verbose: bool = False
                               ) -> Tuple[List[str],
                                          np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Evaluates fitted circles against original circles using error-bound ratios.
    Maintains 1:1 mapping between originals and their closest fits, with radius tiebreaker.

    Args:
        original_circles: Array of original circles [x, y, r]
        new_best_fit: Tuple containing fitted circles, errors, and RMSES
        threshold: Error bar threshold for compatibility
        verbose: Show comparison details

    Returns:
        Tuple containing messages, closest fits, and error ratios
    """
    # Unpack inputs
    fitted_circles, fitted_errors, _ = new_best_fit

    if verbose:
        print("=== Original Circles ===")
        print_circles(original_circles, title="Original Circles", label="Original Circle")
        print("\n=== Fitted Circles ===")
        print_circles(fitted_circles, errors=fitted_errors,
                     title="Fitted Circles with Errors", label="Fitted Circle")

    # Create mapping with enhanced tiebreaker logic
    mapping = []
    for fit_idx, fit_circle in enumerate(fitted_circles):
        orig_idx, orig_dist = find_nearest_circle(original_circles, fit_circle)[1:]
        orig_r = original_circles[orig_idx][2]
        radius_diff = abs(fit_circle[2] - orig_r)
        mapping.append((fit_idx, orig_idx, orig_dist, radius_diff))

    # Group by original circle with tiebreaker
    best_matches = {}
    for fit_idx, orig_idx, dist, radius_diff in mapping:
        current = best_matches.get(orig_idx, None)

        if not current:
            # First match for this original
            best_matches[orig_idx] = {
                'fit_idx': fit_idx,
                'distance': dist,
                'radius_diff': radius_diff
            }
        else:
            # Compare with existing match
            if (dist < current['distance']) or \
               (dist == current['distance'] and radius_diff < current['radius_diff']):
                best_matches[orig_idx] = {
                    'fit_idx': fit_idx,
                    'distance': dist,
                    'radius_diff': radius_diff
                }

    # Process matches
    messages = []
    closest_fitted = []
    ratii_x, ratii_y, ratii_r = [], [], []

    for orig_idx, match in best_matches.items():
        fit_idx = match['fit_idx']

        # Get parameters
        orig_x, orig_y, orig_r = original_circles[orig_idx]
        fit_x, fit_y, fit_r = fitted_circles[fit_idx]
        x_err, y_err, r_err = fitted_errors[fit_idx]

        # Calculate ratios
        ratio_x = abs(fit_x - orig_x) / x_err if x_err > 0 else np.inf
        ratio_y = abs(fit_y - orig_y) / y_err if y_err > 0 else np.inf
        ratio_r = abs(fit_r - orig_r) / r_err if r_err > 0 else np.inf

        # Store results
        ratii_x.append(ratio_x)
        ratii_y.append(ratio_y)
        ratii_r.append(ratio_r)
        closest_fitted.append(fitted_circles[fit_idx])

        # Generate message
        max_ratio = max(ratio_x, ratio_y, ratio_r)
        status = "COMPARABLE" if max_ratio <= threshold else "NOT COMPARABLE"

        # Formatting helper for infinite cases
        def format_ratio(ratio):
            return f"{ratio:.1f}" if not np.isinf(ratio) else "∞"

        # Create message
        msg = (f"\nFitted Circle {fit_idx} (closest to Original {orig_idx}):\n"
               f"  Radius difference: {format_ratio(ratio_r)}σ; "
               f"Center differences: x = {format_ratio(ratio_x)}σ, y = {format_ratio(ratio_y)}σ\n"
               f"  => {status} (Threshold: {threshold}σ, Max ratio: {format_ratio(max_ratio)}σ)\n")
        messages.append(msg)

    return (
        messages,
        np.array(closest_fitted),
        np.array(ratii_x),
        np.array(ratii_y),
        np.array(ratii_r)
    )


def main_procedure(verbose=False, seed=2):
    """
    Main procedure for generating rings, fitting circles, and evaluating results.

    Parameters:
        verbose (bool): If True, print detailed progress messages.
        seed (int): seed for reproducibility.

    Returns:
        ratii_x (np.ndarray): Array of x-axis ratii.
        ratii_y (np.ndarray): Array of y-axis ratii.
        ratii_r (np.ndarray): Array of radius ratii.
    """
    # Initialize lists to store ratii for this run (will be converted to numpy arrays)
    ratii_x = []
    ratii_y = []
    ratii_r = []

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Record the start time for execution timing
    start_time = time.time()

    # --- Step 1: Setup ---
    # Generate sample circles and print them
    sample_circles = generate_circles(NUM_RINGS)
    original_circles = sample_circles.copy()

    # Generate sample points from circles
    sample_points = generate_rings_complete(sample_circles)
    original_points = sample_points.copy()

    if verbose:
        # Plot the original circles and points
        plot_circles(sample_circles, title="Original Circles", label="Original Circle")
        plot_points(sample_points, title="Sample Data", label="Sample Data", hold=False)

    found_rings = []  # To store refined ring parameters
    iteration = 1  # Initialize iteration counter

    # Initial clustering constraints
    min_clusters = MIN_CLUSTERS  # Minimum number of clusters to find
    max_clusters = MAX_CLUSTERS  # Maximum number of clusters to find

    # Sigma threshold for ring extraction
    sigma_threshold = SIGMA_THRESHOLD

    # Main loop to fit circles to the points
    while len(sample_points) > MIN_SAMPLES * MIN_CLUSTERS_PER_RING * 1.5:
        if verbose:
            print(f"\nIteration {iteration}...")

        if verbose and iteration != 1:
            print(f"Sample points remaining: {len(sample_points)}")
            plot_points(sample_points, title="Remaining Points", hold=False)


        # Fit a circle to the points
        fitted_circle, param_errors, rmse = fit_circle_to_points(sample_points, verbose=True)

        # If the fit is good (RMSE is low), stop the loop
        if rmse < 3 * RADIUS_SCATTER:
            if verbose:
                print("All points fitted. Stopping.")

                # Plot the fitted circle in green
                plot_circle(fitted_circle, color="green", linestyle="solid",
                       label="Fitted Circle", center_as_o=True, hold=True)

                # Plot all the points in red
                plot_points(points=sample_points, color='red',
                            label="Fitted circle for best ring", hold=False)

            # Store the fitted circle parameters
            found_rings.append([
                fitted_circle[0], fitted_circle[1], fitted_circle[2],  # x, y, r
                param_errors[0], param_errors[1], param_errors[2]   # err_x, err_y, err_r
            ])

            break

        print("Not all points fitted. Continuing.")

        # --- Step 2: Adaptive Clustering ---
        cluster_labels, cluster_count = adaptive_clustering(
            sample_points,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_samples=MIN_SAMPLES,
            verbose=verbose
        )

        # If no clusters are found, stop the loop
        if cluster_count == 0:
            if verbose:
                print("No more clusters found. Stopping.")
            break

        # --- Step 3: Create Cluster Dictionary ---
        cluster_dict, _ = create_cluster_dict(sample_points,
                                              cluster_labels, verbose=verbose)

        # --- Step 4: Fit Circles to Clusters ---
        fit_circles_to_clusters(cluster_dict, verbose=verbose)

        # --- Step 5: Filter Clusters ---
        cluster_dict = filter_fitted_clusters(cluster_dict, verbose=verbose)

        # --- Step 6: Merge Similar Clusters ---
        cluster_dict = compare_and_merge_clusters(cluster_dict,
                                                  sigma=SIGMA_THRESHOLD_RM, verbose=verbose)

        if verbose:
            # Plot merged clusters
            show_dictionary(
                cluster_dict,
                cluster_dict.keys(),
                title="Merged Clusters",
                plt_points=True,
                plt_circles=True,
                prt_circles=True,
                hold=False
            )

        # --- Step 7: Extract Best Ring ---
        best_points, other_points, best_fit, _ = extract_best_ring(
            cluster_dict,
            sample_points,
            sigma_threshold=sigma_threshold,
            verbose=verbose
        )

        if verbose:
            print(f"Sigma threshold: {sigma_threshold}\n")

        # If no best ring is found, stop the loop
        if best_points.size == 0:
            if verbose:
                print("No more rings detected. Stopping.\n")
            break

        # --- Step 8: Refine Fit on Best Ring ---
        best_ring, other_points, best_fit, _ = exclude_outliers(
            best_points,
            other_points,
            best_fit,
            beta=3,
            verbose=verbose
        )

        # Store the refined ring parameters
        if best_fit is not None:
            circle, errors, rmse = best_fit
            found_rings.append([
                circle[0], circle[1], circle[2],  # x, y, r
                errors[0], errors[1], errors[2]   # err_x, err_y, err_r
            ])

        if verbose:
            # Plot the results
            plot_circle(circle, color="green", linestyle="solid",
                       label="Fitted Circle", center_as_o=True, hold=True)
            plot_points(other_points, color='gray', label="Other Points", hold=True)
            plot_points(best_ring, color='red', label="Best Ring",
                       title="Fitted Circle for Best Ring", hold=False)

        # Prepare for next iteration
        sample_points = other_points
        min_clusters = max(3, min_clusters - MIN_CLUSTERS_PER_RING)
        max_clusters = max(5, max_clusters - MAX_CLUSTERS_PER_RING)
        sigma_threshold *= S_SCALE
        iteration += 1

    # --- End of Loop ---
    end_time = time.time()
    if verbose:
        print(f"Execution Time: {end_time - start_time:.2f} seconds\n")

    # --- Evaluate Results ---
    if found_rings:
        found_rings = np.array(found_rings)
        fitted_circles = found_rings[:, :3]
        fitted_errors = found_rings[:, 3:6]

        # Evaluate comparability between fitted and original circles
        messages, closest_fitted, r_x, r_y, r_r = evaluate_fit_comparability(
            original_circles,
            (fitted_circles, fitted_errors, None),  # Pack as tuple for new function
            threshold=3,
            verbose=verbose
        )

        # Store the ratii as numpy arrays
        ratii_x = np.array(r_x)
        ratii_y = np.array(r_y)
        ratii_r = np.array(r_r)

        if verbose:
            for msg in messages:
                print(msg)

            # Plot final results
            plot_points(original_points, color='gray', label="Original Points", hold=True)
            plot_circles(closest_fitted, title="Best Fitted Circles vs Original Points",
                         center_as_o=True, hold=False)

    return ratii_x, ratii_y, ratii_r

def main_procedure_adaptive(verbose=False, seed=1):
    """
    Adaptive main procedure for generating rings, fitting circles, and evaluating results.

    Parameters:
        verbose (bool): If True, print detailed progress messages.
        seed (int): Seed for reproducibility.

    Returns:
        ratii_x (np.ndarray): Array of x-axis ratii.
        ratii_y (np.ndarray): Array of y-axis ratii.
        ratii_r (np.ndarray): Array of radius ratii.
    """
    # Initialize lists to store ratii (will be converted to numpy arrays)
    ratii_x = []
    ratii_y = []
    ratii_r = []

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
    sample_points = generate_rings_complete(circles=sample_circles,
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
            best_ring, best_fit, outliers = exclude_outliers(
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
                plot_circle(circle, color="green", linestyle="solid", label="Fitted Circle", center_as_o=True, hold=False)

            break  # Exit loop if good direct fit found

        if verbose:
            # Provide detailed feedback on why the circle is not good
            print("\nNot all points fitted. Continuing.")
            print(f"  - RMSE: {rmse}, while the threshold is "
            f"{3 * RADIUS_SCATTER} (alpha * RADIUS_SCATTER)\n")

        # --- Step 3: Adaptive Clustering ---
        # Initial clustering constraints
        if iteration == 1:
            min_clusters = MIN_CLUSTERS  # Minimum number of clusters to find
            max_clusters = MAX_CLUSTERS  # Maximum number of clusters to find
            min_samples = MIN_SAMPLES    # Minimum samples for each cluster

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
            valid_clusters = [c for c in cluster_dict.values() if c['valid'] and c['circle'] is not None]
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

            # Unpack initial fit
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
        #sample_points = other_points  # Update the remaining points
        iteration += 1  # Increment the iteration counter

    # --- End of Loop ---
    end_time = time.time()
    if verbose:
        print(f"\nExecution Time: {end_time - start_time:.2f} seconds\n")

    # --- Evaluate Results ---
    if found_rings:
        found_rings = np.array(found_rings)
        fitted_circles = found_rings[:, :3]
        fitted_errors = found_rings[:, 3:6]

        # Evaluate comparability
        messages, closest_fitted, r_x, r_y, r_r = evaluate_fit_comparability(
            original_circles,
            (fitted_circles, fitted_errors, None),
            threshold=3,
            verbose=verbose
        )

        # Convert ratii to numpy arrays
        ratii_x = np.array(r_x)
        ratii_y = np.array(r_y)
        ratii_r = np.array(r_r)

        if verbose:
            for msg in messages:
                print(msg)

            plot_points(original_points, color='gray', label="Original Points", hold=True)
            plot_circles(closest_fitted, title="Best Fitted Circles vs Original Points",
                         center_as_o=True, hold=False)

        # plot_points(original_points, color='gray', label="Original Points", hold=True)
        # plot_circles(closest_fitted, title="Best Fitted Circles vs Original Points",
        #                 center_as_o=True, hold=False)
    else:
        # Return empty numpy arrays if no rings found
        ratii_x = np.array([])
        ratii_y = np.array([])
        ratii_r = np.array([])

    return ratii_x, ratii_y, ratii_r
