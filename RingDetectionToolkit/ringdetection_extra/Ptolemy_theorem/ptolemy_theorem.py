# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
# For license terms see LICENSE file.
##
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software under GPLv2+
# See https://www.gnu.org/licenses/gpl-2.0.html

"""
Geometric Verification Toolkit using Ptolemy's Theorem

This module provides geometric verification methods for circle detection using:
- Ptolemy's theorem for cyclic quadrilateral verification
- Four-point circle fitting
- Point-circle compatibility analysis

Key Features:
1. Geometric Verification:
   - ptolemy_check(): Validates if four points are concyclic
   - fit_circle_to_four_points(): Fits circle to quadrilateral
   - circumcircle(): Computes circumcircle of points

2. Point Analysis:
   - count_points_on_circle(): Finds compatible points with a circle
   - is_a_good_circle_fast(): Quick circle validation

3. Visualization:
   - plot_quadrilateral_and_circle(): Visualizes quadrilateral and circumcircle
   - process_points(): Full verification pipeline with plotting
"""

# ============================ IMPORTS ============================ #

# Standard library imports
from typing import List, Optional, Tuple
import sys
import os

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# =========================== Find the correct path ========================
# Go up one directory to reach ringdetection.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Local imports
from ringdetection import (
    generate_circles, print_circles,
    print_circle, plot_points, plot_circle, plot_circles,
    get_color, find_nearest_circle, RADIUS_SCATTER
)

# ============================ CONSTANTS ============================ #

NR = 3 # number of rings
PPR = 10  # Points per ring
CLUSTER_CX_MIN, CLUSTER_CX_MAX = -1.0, 2.0 # Bounds for cluster center x-coordinate
CLUSTER_CY_MIN, CLUSTER_CY_MAX = -1.0, 2.0 # Bounds for cluster center y-coordinate
CLUSTER_CR_MIN, CLUSTER_CR_MAX = 0.1, 1.5 # Bounds for cluster radius
MAX_RMSE = 3 * RADIUS_SCATTER  # Maximum acceptable cluster RMSE


# ============================ EXPORT LIST ============================ #
__all__ = [
    # Geometric verification
    'ptolemy_check',
    'fit_circle_to_four_points',
    'circumcircle',

    # Point analysis
    'count_points_on_circle',
    'is_a_good_circle_fast',

    # Visualization
    'plot_quadrilateral_and_circle',
    'process_points',

    # Data generation
    'generate_rings'
]

# ============================ GEOMETRIC VERIFICATION ============================ #

def ptolemy_check(point_a: Tuple[float, float],
                  point_b: Tuple[float, float],
                  point_c: Tuple[float, float],
                  point_d: Tuple[float, float],
                  rtol: float = 1e-3) -> bool:
    """
    Check if four points in a plane satisfy Ptolemy's inequality as an equality.

    Ptolemy's theorem states that for four points A, B, C, D in a plane:
    AC * BD ≤ AB * CD + BC * DA, with equality if and only if the points are concyclic
    (lie on the same circle in order A-B-C-D or any cyclic permutation).

    Args:
        point_a, point_b, point_c, point_d: Tuples representing 2D points (x, y coordinates)
        rtol: Relative tolerance for floating point comparison

    Returns:
        bool: True if the points satisfy AC * BD ≈ AB * CD + BC * DA within tolerance
    """
    # Calculate all pairwise distances between points
    dist_ac = np.linalg.norm(point_a - point_c)
    dist_bd = np.linalg.norm(point_b - point_d)
    dist_ab = np.linalg.norm(point_a - point_b)
    dist_bc = np.linalg.norm(point_b - point_c)
    dist_cd = np.linalg.norm(point_c - point_d)
    dist_da = np.linalg.norm(point_d - point_a)

    # Check if the product of diagonals equals the sum of products of opposite sides
    return np.isclose(dist_ac * dist_bd, dist_ab * dist_cd + dist_bc * dist_da, rtol=rtol)


def fit_circle_to_four_points(point_a: Tuple[float, float],
                              point_b: Tuple[float, float],
                              point_c: Tuple[float, float],
                              point_d: Tuple[float, float],
                              tolerance: float = 1e-6
                              ) -> Optional[Tuple[List[float], float]]:
    """
    Fit the circumcircle of a quadrilateral (or any four points).

    Args:
        A, B, C, D (tuple): Coordinates of the four points as (x, y) tuples.
        tolerance (float): Tolerance to check for collinearity. Default is 1e-6.

    Returns:
        list: [center_x, center_y, radius] of the fitted circle.
        float: Root Mean Square Error (RMSE) of the fit.
        Returns None if the points are nearly collinear.
    """
    # Extract coordinates of the points
    x_1, y_1 = point_a
    x_2, y_2 = point_b
    x_3, y_3 = point_c
    x_4, y_4 = point_d

    # Calculate the determinant of the matrix (to check for collinearity)
    determinant = (x_1 - x_3) * (y_2 - y_3) - (y_1 - y_3) * (x_2 - x_3)

    # If the determinant is near zero, the points are nearly collinear
    if abs(determinant) < tolerance:
        return None  # Return None to indicate invalid input (collinear points)

    # Calculate the center of the circle using coordinate geometry formulas
    center_x = ((x_1**2 + y_1**2) * (y_2 - y_3) +
                (x_2**2 + y_2**2) * (y_3 - y_1) +
                (x_3**2 + y_3**2) * (y_1 - y_2)) / (2 * determinant)
    center_y = ((x_1**2 + y_1**2) * (x_3 - x_2) +
                (x_2**2 + y_2**2) * (x_1 - x_3) +
                (x_3**2 + y_3**2) * (x_2 - x_1)) / (2 * determinant)

    # Calculate the radius for each point
    radius1 = np.sqrt((center_x - x_1)**2 + (center_y - y_1)**2)
    radius2 = np.sqrt((center_x - x_2)**2 + (center_y - y_2)**2)
    radius3 = np.sqrt((center_x - x_3)**2 + (center_y - y_3)**2)
    radius4 = np.sqrt((center_x - x_4)**2 + (center_y - y_4)**2)

    # Calculate the average radius
    radius = np.mean([radius1, radius2, radius3, radius4])

    # Calculate the Root Mean Square Error (RMSE) of the fit
    rmse = np.sqrt(np.mean([(radius - radius1)**2, (radius - radius2)**2,
                            (radius - radius3)**2, (radius - radius4)**2]))

    # Return the circle parameters and RMSE
    return [center_x, center_y, radius], rmse

def circumcircle(points: List[np.ndarray]) -> Tuple[Tuple[float, float], float]:
    """
    Calculates the circumcircle (center and radius) of a triangle or quadrilateral.

    Note:
        If points are nearly collinear, a warning is issued and a fallback small
        determinant is used to avoid division by zero.

    Args:
        points (List[np.ndarray]): A list of 4 two-dimensional points.

    Returns:
        Tuple[Tuple[float, float], float]:
            - (center_x, center_y): Coordinates of the circumcircle center.
            - radius: Radius of the circumcircle.
    """
    # Extract coordinates of the first three points
    x_1, y_1 = points[0]
    x_2, y_2 = points[1]
    x_3, y_3 = points[2]

    # Compute determinant to check collinearity and prepare for the formula
    determinant = (x_1 - x_3) * (y_2 - y_3) - (y_1 - y_3) * (x_2 - x_3)
    if abs(determinant) < 1e-6:
        # If determinant is very small, points are almost collinear
        print("Warning: Points are nearly collinear. Circumcircle might not be accurate.")
        determinant = 1e-6

    # Apply the coordinate geometry formula to find the circumcenter
    center_x = ((x_1**2 + y_1**2) * (y_2 - y_3) +
                (x_2**2 + y_2**2) * (y_3 - y_1) +
                (x_3**2 + y_3**2) * (y_1 - y_2)) / (2 * determinant)

    center_y = ((x_1**2 + y_1**2) * (x_3 - x_2) +
                (x_2**2 + y_2**2) * (x_1 - x_3) +
                (x_3**2 + y_3**2) * (x_2 - x_1)) / (2 * determinant)

    # Compute radius using distance from center to one of the points
    radius = np.sqrt((center_x - x_1)**2 + (center_y - y_1)**2)

    return (center_x, center_y), radius

# ============================ POINT ANALYSIS ============================ #

def is_a_good_circle_fast(circle: Tuple[float, float, float],
                          rmse: float) -> bool:
    """
    Quickly validate if a circle meets quality criteria based on position, size and fit error.

    Checks if a circle:
    1. Has its center within allowed bounds (x and y coordinates)
    2. Has radius within allowed range
    3. Has RMSE (Root Mean Square Error) below threshold

    Args:
        circle: A tuple of (center_x, center_y, radius) representing the circle
        rmse: Root Mean Square Error of the circle fit

    Returns:
        bool: True if the circle meets all quality criteria, False otherwise
    """

    return circle[0] >= CLUSTER_CX_MIN and circle[0] <= CLUSTER_CX_MAX and \
        circle[1] >= CLUSTER_CY_MIN and circle[1] <= CLUSTER_CY_MAX and \
        circle[2] >= CLUSTER_CR_MIN and circle[2] <= CLUSTER_CR_MAX and \
        rmse <= MAX_RMSE

def count_points_on_circle(points: List[List[float]],
                           circle: List[float],
                           atol: float = 2 * RADIUS_SCATTER
                           ) -> Tuple[int, float]:
    """
    Count points that lie approximately on a given circle and compute their fitting error.

    For each point, calculates its distance from the circle's center and checks if it's
    within the specified tolerance of the circle's radius. Also computes the RMSE of
    the compatible points' distances from the ideal circle radius.

    Args:
        points: List of 2D points, each as [x, y] coordinates
        circle: Circle parameters as [center_x, center_y, radius]
        atol: Absolute tolerance for considering a point on the circle (default: 2*RADIUS_SCATTER)

    Returns:
        Tuple containing:
            - Number of points within radius tolerance (int)
            - RMSE of compatible points' distances (float). Returns NaN if no compatible points
    """
    # Convert points to a numpy array for efficient vectorized operations
    points = np.array(points)

    # Extract circle parameters (center and radius)
    c_x, c_y, radius = circle

    # Calculate the Euclidean distance from the circle's center to each point
    distances = np.linalg.norm(points - [c_x, c_y], axis=1)

    # Find points that are within the tolerance of the circle's radius
    compatible_points_mask = np.abs(distances - radius) <= atol
    compatible_points = points[compatible_points_mask]  # Extract compatible points
    num_compatible_points = len(compatible_points)  # Count compatible points

    # Calculate RMSE only for compatible points
    if num_compatible_points > 0:
        residuals = distances[compatible_points_mask] - radius  # Residuals (distance errors)
        rmse = np.sqrt(np.mean(residuals**2))  # Root Mean Square Error
    else:
        rmse = np.nan  # Return NaN if no compatible points are found

    return num_compatible_points, rmse

# ============================ VISUALIZATION ============================ #

def plot_quadrilateral_and_circle(points: List[np.ndarray], color: str = 'red') -> None:
    """
    Plots a quadrilateral formed by 4 points, its diagonals, and its circumcircle.

    This function visualizes:
    - The quadrilateral defined by the input points
    - Its two diagonals
    - The circumcircle passing through the four points (if possible)

    Args:
        points (List[np.ndarray]): A list of four 2D points (each of shape (2,)).
        color (str): Color used for the diagonals and the circumcircle.

    Returns:
        None
    """

    # Ensure we have exactly four points
    if len(points) != 4:
        print("Error: Requires exactly four points to define a quadrilateral.")
        return

    # Extract x and y coordinates
    point_x = [p[0] for p in points]
    point_y = [p[1] for p in points]

    # Plot the quadrilateral
    plt.plot(point_x + [point_x[0]], point_y + [point_y[0]], linestyle='-', markersize=8)

    # Plot the diagonals
    plt.plot([point_x[0], point_x[2]], [point_y[0], point_y[2]], linestyle='--', color=color)
    plt.plot([point_x[1], point_x[3]], [point_y[1], point_y[3]], linestyle='--', color=color)

    # Compute the circumcircle (center and radius)
    center, radius = circumcircle(points)

    # Plot the circumcircle as a dashed circle around the quadrilateral
    circle = plt.Circle(center, radius, color=color, fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    # Mark the center of the circle
    plt.plot(center[0], center[1], marker='o', markersize=10, color=color)

def process_points(points: np.ndarray,
                   sample_circles: np.ndarray,
                   k: int,
                   min_points: int = 0) -> None:
    """
    Attempt to detect circles within a set of 2D points by:
    - Randomly sampling quadruples of points
    - Verifying Ptolemy's theorem for potential cyclicity
    - Fitting a circle to the 4 points
    - Filtering out poor fits based on RMSE and compatibility
    - Matching detected circles with a list of known sample circles

    Args:
        points (np.ndarray): An array of shape (N, 2) containing 2D (x, y) coordinates.
        sample_circles (np.ndarray): Array of known circles [x, y, r] to match detected fits against
        k (int): Number of random samples (sets of 4 points) to draw and evaluate.
        min_points (int, optional): Minimum number of compatible points required to
            accept a detected circle. Defaults to 0.

    Returns:
        None. Prints diagnostic information and plots visualizations.
    """

    # Visualize the original points and known sample circles
    plot_points(points, color='blue', label="Original Points", hold=True)
    plot_circles(circles=sample_circles, title="Sample Circles", hold=False)

    #Then replot the original sample circles and sample points for next visualization
    plot_points(points, color='blue', label="Original Points", hold=True)
    plot_circles(sample_circles, title="Sample Circles", hold=True)

    ok_count = 0                    # Count of valid circles found
    found_circles = []             # Indices of sample circles matched to detected circles

    for i in range(k):
        # Randomly select 4 distinct points
        indices = np.random.choice(len(points), size=4, replace=False)
        point_a, point_b, point_c, point_d = points[indices]

        # Step 1: Check Ptolemy’s theorem (necessary for a cyclic quadrilateral)
        if not ptolemy_check(point_a, point_b, point_c, point_d):
            continue
        print(f"\nPtolemy's Theorem satisfied for points {indices}")

        # Step 2: Fit a circle to the 4 points
        result = fit_circle_to_four_points(point_a, point_b, point_c, point_d)
        if result is None:
            continue

        circle_tuple, rmse = result
        circle = np.array(circle_tuple)  # (x, y, r)
        print_circle(circle, rmse=rmse, label=f"Attempt {i+1:2d}")

        # Step 3: Filter by fit quality
        if not is_a_good_circle_fast(circle, rmse):
            print(f"RMSE: {rmse:.4f}; Circle discarded (not good)")
            continue

        # Step 4: Count how many other points are compatible with this circle
        compatible_points, compatible_rmse = count_points_on_circle(points, circle)
        if compatible_points < min_points:
            print(f"Compatible Points: {compatible_points:2d}, RMSE for CP: {compatible_rmse:.4f}")
            continue

        # Circle is valid — increment and visualize it
        ok_count += 1
        color = get_color(ok_count)

        plot_circle(circle,
                    linestyle='dashed',
                    color=color,
                    label=f"Circle {ok_count} ({i}/{compatible_points})",
                    hold=True)

        # Step 5: Find and visualize the closest known (sample) circle
        nearest_circle_tuple, index, distance = find_nearest_circle(sample_circles, circle)
        nearest_circle = np.array(nearest_circle_tuple)
        print(f"Nearest Circle {index}, Distance: {distance:.4f}")
        print_circle(nearest_circle, label=f"Nearest Circle {index}")

        found_circles.append(int(index))       # convert np.int64 to int
        print()  # For spacing

    # Summary
    print("\nIndices of detected circles:", found_circles)
    print(f"Unique circles found: {set(found_circles)}")

    # Show final plot
    plt.show()

# ============================ DATA GENERATION ============================ #

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
            - circles array has incorrect shape (not N×3)
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


# ======================== HERE IS THE MAIN ========================#

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(1)  # Set the random seed for reproducibility
    sample_circles_ptolemy = generate_circles(NR)  # Generate 2 sample circles
    sample_points = generate_rings(sample_circles_ptolemy,
                                   points_per_ring=PPR)  # Generate 10 points per ring

    # Print the original circles for reference
    print_circles(sample_circles_ptolemy, title='\nOriginal circles')

    # Process the points
    process_points(sample_points, sample_circles=sample_circles_ptolemy,
                   k=100 * NR, min_points=PPR/2)
