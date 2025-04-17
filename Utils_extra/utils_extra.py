# Copyright (C) 2023 a.fiorentino4@studenti.unipi.it
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

from typing import Tuple, Callable, Optional
import math
import multiprocessing as mp
from functools import partial

import numpy as np
import pycuda.driver as drv
from pycuda import curandom
from pycuda.compiler import SourceModule

# -------------------- Constants & Detector Parameters -------------------- #
N_H2O = 1.33           # Refractive index of water
BETA_NU = 1            # v_neutrino / c

# HyperKamiokande dimensions (in meters)
INNER_DIAMETER = 68    # inner diameter of the detector
INNER_HEIGHT = 71      # inner height of the detector
FIDUCIAL_DIAMETER = 64 # fiducial (active) diameter
FIDUCIAL_HEIGHT = 66   # fiducial (active) height

# Total PMTs on the detector
TOTAL_PMTS = 20000

RMAX_SCALE = 1  # Change this if you want to enlarge rmax because of non-orthogonal hits
ALPHA = 10      # Change if you consider the discrete min radius too small

def calculate_radii_in_kamiokande(n_h2o: float, beta_nu: float,
                   inner_diameter: float, inner_height: float,
                   fiducial_diameter: float, total_pmts: int,
                   rmax_scale: float, alpha: int,
                   rounding_mode: str = "round",
                   verbose: bool = True) -> Tuple[float, float, float]:
    """
    Computes the minimum and maximum detectable radii for the HyperKamiokande experiment.

    Args:
        n_h2o (float): Refractive index of water.
        beta_nu (float): Ratio of neutrino velocity to speed of light (v_neutrino / c).
        inner_diameter (float): Inner diameter of the detector in meters.
        inner_height (float): Inner height of the detector in meters.
        fiducial_diameter (float): Fiducial (active) diameter in meters.
        total_pmts (int): Total number of PMTs in the detector.
        rmax_scale (float): Scaling factor to increase max radius in case of
            non-perpendicular neutrino hits.
        alpha (int): Scaling factor for adjusting too small discrete min radius.
        rounding_mode (str): "round" (default), "ceil" (excess), or "floor" (defect)
            to control radius rounding.
        verbose (bool): If True, prints details of the computed values.

    Returns:
        tuple: (r_min, r_max, radius_scatter) - Normalized minimum and maximum radii,
            and the normalized radius scatter.
    """
    # -------------------- Cherenkov Angle Calculation -------------------- #
    cos_theta = 1 / (n_h2o * beta_nu)
    theta_radians = math.acos(cos_theta)
    theta_degrees = math.degrees(theta_radians)

    # -------------------- Continuous Radius Calculations -------------------- #
    min_radius = ((inner_diameter - fiducial_diameter) / 2) * math.sin(math.radians(theta_degrees))
    max_radius = (fiducial_diameter + 2) * math.sin(math.radians(theta_degrees))

    # -------------------- Discretization -------------------- #
    cap_area = math.pi * (inner_diameter / 2) ** 2  # Area of the top and bottom caps
    inner_surface_area = math.pi * inner_diameter * inner_height  # Lateral surface area
    total_surface_area = 2 * cap_area + inner_surface_area  # Total inner surface area

    # Compute the actual PMT spacing based on the total number of PMTs
    pmt_spacing_eff = total_surface_area / total_pmts  # Corrected PMT spacing

    # -------------------- Discrete Radii Calculation -------------------- #
    if rounding_mode == "ceil":
        discrete_min_radius = math.ceil(min_radius / pmt_spacing_eff) * pmt_spacing_eff
        discrete_max_radius = math.ceil((max_radius * rmax_scale)
                                         / pmt_spacing_eff) * pmt_spacing_eff
    elif rounding_mode == "floor":
        discrete_min_radius = math.floor(min_radius / pmt_spacing_eff) * pmt_spacing_eff
        discrete_max_radius = math.floor((max_radius * rmax_scale)
                                         / pmt_spacing_eff) * pmt_spacing_eff
    else:
        discrete_min_radius = round(min_radius / pmt_spacing_eff) * pmt_spacing_eff
        discrete_max_radius = round((max_radius * rmax_scale) / pmt_spacing_eff) * pmt_spacing_eff

    # Ensure min radius is at least 3x the PMT spacing, or alpha * spacing if still too small
    if discrete_min_radius < 3 * pmt_spacing_eff:
        discrete_min_radius = alpha * pmt_spacing_eff

    # -------------------- Normalization -------------------- #
    r_min = discrete_min_radius / inner_diameter
    r_max = discrete_max_radius / inner_diameter
    radius_scatter = pmt_spacing_eff / inner_diameter  # Normalized radius scatter (resolution)

    # -------------------- Verbose Output -------------------- #
    if verbose:
        print(f"Theta (degrees): {theta_degrees:.3f}")
        print(f"Effective PMT spacing: {pmt_spacing_eff:.3f} m")
        print(f"Discrete min_radius (adjusted): {discrete_min_radius:.3f} m")
        print(f"Discrete max_radius: {discrete_max_radius:.3f} m")
        print(f"Normalized r_min: {r_min:.3f}, r_max: {r_max:.3f}")
        print(f"Normalized radius scatter (resolution): {radius_scatter:.3f}")

    return r_min, r_max, radius_scatter

# ================================ Generate rings with Multiprocessing ====================

def generate_ring_points(circle: np.ndarray,
                         points_per_ring: int,
                         radius_scatter: float) -> np.ndarray:
    """
    Generates points for a single ring.

    Args:
        circle (np.ndarray): A numpy array representing the circle as [x, y, r].
        points_per_ring (int): Number of points to generate for the ring.
        radius_scatter (float): Allowed variation in the radius (R ± radius_scatter).

    Returns:
        np.ndarray: A numpy array of (x, y) coordinates representing
            the generated points for the ring.

    Raises:
        ValueError: If the input circle array is invalid or the radius scatter is negative.
    """
    # Input validation for circle
    if not isinstance(circle, np.ndarray) or circle.shape != (3,):
        raise ValueError("circle must be a 1D NumPy array of shape (3,): [x, y, r].")

    # Input validation for points_per_ring
    if not isinstance(points_per_ring, int) or points_per_ring <= 0:
        raise ValueError("points_per_ring must be a positive integer.")

    # Input validation for radius_scatter
    if not isinstance(radius_scatter, (int, float)) or radius_scatter < 0:
        raise ValueError("radius_scatter must be a non-negative number.")

    x_center, y_center, radius = circle

    # Generate random angles for the points
    angles = np.random.uniform(0, 2 * np.pi, points_per_ring)

    # Generate random radii within the allowed scatter
    radii = radius + np.random.uniform(-radius_scatter, radius_scatter, points_per_ring)

    # Convert polar coordinates (radii, angles) to Cartesian coordinates (x, y)
    x_points = x_center + radii * np.cos(angles)
    y_points = y_center + radii * np.sin(angles)

    # Combine x and y coordinates into a single array
    return np.column_stack((x_points, y_points))

def generate_rings(circles: np.ndarray,
                   points_per_ring: int = 500,
                   radius_scatter: float = 0.01,
                   filter_func: Optional[Callable] = None) -> np.ndarray:
    """
    Generates multiple rings and concatenates the points in a single
        numpy array using multiprocessing.

    Args:
        circles (np.ndarray): A numpy array of circles, where each circle is
            represented as [x, y, r].
        points_per_ring (int): Number of points to generate for each ring. Defaults to 1000.
        radius_scatter (float): Allowed variation in the radius (R ± radius_scatter).
            Defaults to 0.01.
        filter_func (callable, optional): A function to filter points.
            Should accept a point (x, y) and return True if the point is valid.

    Returns:
        np.ndarray: A numpy array of (x, y) coordinates representing
            all generated points for the rings.

    Raises:
        ValueError: If the input circles array is invalid or the
            points_per_ring or radius_scatter are invalid.
    """
    # Input validation for circles
    if not isinstance(circles, np.ndarray) or circles.ndim != 2 or circles.shape[1] != 3:
        raise ValueError("circles must be a 2D NumPy array of shape"
        "(N, 3), where each row is [x, y, r].")

    # Input validation for points_per_ring
    if not isinstance(points_per_ring, int) or points_per_ring <= 0:
        raise ValueError("points_per_ring must be a positive integer.")

    # Input validation for radius_scatter
    if not isinstance(radius_scatter, (int, float)) or radius_scatter < 0:
        raise ValueError("radius_scatter must be a non-negative number.")

    # Use multiprocessing to generate points for each ring in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        # Pass additional arguments to `generate_ring_points` using `functools.partial`
        func = partial(generate_ring_points, points_per_ring=points_per_ring,
                       radius_scatter=radius_scatter)
        all_points = pool.map(func, circles)

    # Concatenate all points into a single numpy array
    all_points = np.concatenate(all_points)

    # Filter points if a filter function is provided
    if filter_func:
        if not callable(filter_func):
            raise ValueError("filter_func must be a callable function.")
        return np.array(list(filter(filter_func, all_points)))
    return all_points

# ================================ Generate rings with Pycuda ====================

# PyCUDA Kernel
KERNEL_CODE = """
#include <math.h>

extern "C"
__global__ void generate_rings_kernel(float *circles, float *points,
                                      int points_per_ring, int num_circles,
                                      float radius_scatter, float *rand_angles, float *rand_radii) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = num_circles * points_per_ring;
    if (idx >= total_points) return;

    // Determine which circle this thread corresponds to
    int circle_idx = idx / points_per_ring;

    // Get circle parameters
    float cx = circles[3 * circle_idx + 0];
    float cy = circles[3 * circle_idx + 1];
    float r  = circles[3 * circle_idx + 2];

    // Fetch pre-generated random values
    float angle = rand_angles[idx] * 2.0f * 3.14159265f;
    float delta_r = (rand_radii[idx] * 2.0f - 1.0f) * radius_scatter;
    float final_r = r + delta_r;

    // Compute (x, y)
    points[2 * idx + 0] = final_r * cosf(angle) + cx;
    points[2 * idx + 1] = final_r * sinf(angle) + cy;
}
"""

# Compile CUDA module
mod = SourceModule(KERNEL_CODE)
generate_kernel = mod.get_function("generate_rings_kernel")

def generate_rings_gpu(circles: np.ndarray,
                       points_per_ring: int = 1000,
                       radius_scatter: float = 0.01) -> np.ndarray:
    """
    Generates points for multiple rings in parallel using PyCUDA.

    Args:
        circles (np.ndarray): A numpy array of circles,
            where each circle is represented as [x, y, r].
        points_per_ring (int): Number of points to generate for each ring. Defaults to 1000.
        radius_scatter (float): Allowed variation in the radius (R ± radius_scatter).
            Defaults to 0.05.

    Returns:
        np.ndarray: A numpy array of (x, y) coordinates representing all
            generated points for the rings.

    Raises:
        ValueError: If the input circles array is invalid or the
            points_per_ring or radius_scatter are invalid.
    """
    # Input validation for circles
    if not isinstance(circles, np.ndarray) or circles.ndim != 2 or circles.shape[1] != 3:
        raise ValueError("circles must be a 2D NumPy array of shape (N, 3),"
        "where each row is [x, y, r].")

    # Input validation for points_per_ring
    if not isinstance(points_per_ring, int) or points_per_ring <= 0:
        raise ValueError("points_per_ring must be a positive integer.")

    # Input validation for radius_scatter
    if not isinstance(radius_scatter, (int, float)) or radius_scatter < 0:
        raise ValueError("radius_scatter must be a non-negative number.")

    num_circles = circles.shape[0]
    total_points = num_circles * points_per_ring

    # Flatten circles array
    circles_flat = circles.astype(np.float32).flatten()

    # Allocate memory for output points
    points_out = np.empty(total_points * 2, dtype=np.float32)

    # Generate random values using PyCUDA (faster than custom RNG in kernel)
    rand_angles = curandom.rand((total_points,), dtype=np.float32).get()
    rand_radii = curandom.rand((total_points,), dtype=np.float32).get()

    # Set CUDA block and grid sizes
    block_size = 256
    grid_size = (total_points + block_size - 1) // block_size

    # Launch kernel
    generate_kernel(
        drv.In(circles_flat), drv.Out(points_out),
        np.int32(points_per_ring), np.int32(num_circles),
        np.float32(radius_scatter), drv.In(rand_angles), drv.In(rand_radii),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )

    # Reshape output
    return points_out.reshape((total_points, 2))

# ====================== Calculate rings metrics =================================

def calculate_ring_metrics(points: np.ndarray,
                           circle: np.ndarray,
                           verbose: bool = False) -> Tuple[float, float]:
    """
    Calculates the RMSE (Root Mean Square Error) and standard deviation of the distances
    from the center of the ring to the points.

    Args:
        points (np.ndarray): A 2D array of points, where each row is [x, y].
            Must have shape (N, 2), where N is the number of points.
        circle (np.ndarray): A 1D array representing the circle as [x_center, y_center, radius].
            Must have shape (3,).
        verbose (bool): If True, print additional information and plot the points.

    Returns:
        Tuple[float, float]: A tuple containing:
            - RMSE: The root mean square error of the distances from the points to the circle.
            - std_dev: The standard deviation of the distances.

    Raises:
        ValueError: If the input arrays have incorrect shapes or invalid values.
    """
    # Input validation for points
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be a 2D NumPy array of shape (N, 2).")

    # Input validation for circle
    if not isinstance(circle, np.ndarray) or circle.shape != (3,):
        raise ValueError("circle must be a 1D NumPy array of shape (3,):"
        "[x_center, y_center, radius].")

    # Extract the center and radius of the ring
    center_x, center_y, radius = circle

    # Validate radius
    if radius <= 0:
        raise ValueError("radius must be a positive number.")

    # Calculate distances from the center to each point
    distances = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)

    # Calculate RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.sum((distances - radius) ** 2))  # Corrected RMSE calculation

    # Calculate standard deviation of the distances
    std_dev = np.std(distances)

    if verbose:
        # Plot the points for visualization
        plot_points(points, title="Sample Data", label="Sample Data", hold=False)

        # Print RMSE and standard deviation
        print(f"RMSE: {rmse:.4f}, Standard Deviation: {std_dev:.4f}\n")

    return rmse, std_dev

def calculate_average_ring_metrics(num_rings: int, points_per_ring: int = 500,
                                   radius_scatter: float = 0.01,
                                   use_complete_generation: bool = False,
                                   verbose: bool = False
                                   ) -> Tuple[Tuple[float,float], Tuple[float, float]]:
    """
    Generates multiple rings, calculates the RMSE and standard deviation for each ring,
    and returns the average RMSE and average standard deviation along with their uncertainties.

    Args:
        num_rings (int): Number of rings to generate.
        points_per_ring (int): Number of points to generate for each ring. Defaults to 500.
        radius_scatter (float): Allowed variation in the radius (R ± radius_scatter).
            Defaults to 0.01.
        use_complete_generation (bool): If True, uses generate_rings_complete for point generation.
            Defaults to False.
        verbose (bool): If True, print additional information.

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float]]:
            - (Average RMSE, Standard Deviation of RMSE)
            - (Average Standard Deviation, Standard Deviation of Standard Deviation)

    Raises:
        ValueError: If the input parameters are invalid.
    """
    # Input validation for num_rings
    if not isinstance(num_rings, int) or num_rings <= 0:
        raise ValueError("num_rings must be a positive integer.")

    # Input validation for points_per_ring
    if not isinstance(points_per_ring, int) or points_per_ring <= 0:
        raise ValueError("points_per_ring must be a positive integer.")

    # Input validation for radius_scatter
    if not isinstance(radius_scatter, (int, float)) or radius_scatter < 0:
        raise ValueError("radius_scatter must be a non-negative number.")

    # Input validation for use_complete_generation
    if not isinstance(use_complete_generation, bool):
        raise ValueError("use_complete_generation must be a boolean value.")

    # Lists to store RMSE and standard deviation for each ring
    rmse_list = []
    std_dev_list = []

    for _ in range(num_rings):
        # Generate a single ring
        sample_circle = generate_circles(1)

        # Generate points for the ring using selected method
        if use_complete_generation:
            sample_points = generate_rings_complete(
                circles=sample_circle,
                points_per_ring=points_per_ring,
                radius_scatter=radius_scatter
            )
        else:
            sample_points = generate_rings(
                circles=sample_circle,
                points_per_ring=points_per_ring,
                radius_scatter=radius_scatter
            )

        # Calculate RMSE and standard deviation for the ring
        rmse, std_dev = calculate_ring_metrics(points=sample_points,
                                               circle=sample_circle[0], verbose=False)

        # Append the results to the lists
        rmse_list.append(rmse)
        std_dev_list.append(std_dev)

    # Calculate the average RMSE and average standard deviation
    avg_rmse = np.mean(rmse_list)
    avg_std_dev = np.mean(std_dev_list)

    # Calculate the standard deviation of RMSE and standard deviation values
    std_rmse = np.std(rmse_list)
    std_std_dev = np.std(std_dev_list)

    if verbose:
        print(f"Average RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
        print(f"Average Standard Deviation: {avg_std_dev:.4f} ± {std_std_dev:.4f}")

    return (avg_rmse, std_rmse), (avg_std_dev, std_std_dev)

def post_process_clusters(labels: np.ndarray,
                            points: np.ndarray,
                            min_samples: int) -> Tuple[np.ndarray, int]:
    """
    Post-processes the clusters to remove small clusters and reassign labels.

    Args:
        labels (np.ndarray): Cluster labels for each point (shape: (N,))
        points (np.ndarray): Input points array (shape: (N, 2))
        min_samples (int): Minimum points required for valid cluster (must be > 0)

    Returns:
        Tuple[np.ndarray, int]:
            - new_labels: Updated labels with contiguous IDs (noise as -1)
            - cluster_count: Number of valid clusters (excluding noise)

    Raises:
        ValueError: If inputs are invalid:
            - labels and points length mismatch
            - min_samples not positive
            - invalid array shapes
    """

    # Input validation
    if len(labels) != len(points):
        raise ValueError("Labels and points must have same length")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must be 2D array with shape (N, 2)")
    if labels.ndim != 1:
        raise ValueError("Labels must be 1D array")

    unique_labels = set(labels)  # Get all unique cluster labels

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points

        # Get the points in the current cluster
        cluster_mask = labels == label  # Boolean mask for points in the cluster
        cluster_points = points[cluster_mask]  # Extract points using the mask

        # If the cluster is too small, reassign its points to noise
        if len(cluster_points) < min_samples:
            labels[cluster_mask] = -1  # Reassign points to noise

    # Reassign labels to ensure contiguous cluster IDs
    valid_labels = []  # List to store valid cluster labels
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points

        # Count the number of points in the cluster
        cluster_size = np.sum(labels == label)
        if cluster_size >= min_samples:
            valid_labels.append(label)  # Add to valid labels if large enough

    # Create new labels with contiguous IDs
    new_labels = np.full_like(labels, -1)  # Initialize all labels as noise

    # Create a mapping from old labels to new contiguous labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}

    # Assign new labels to points in the valid clusters
    for old_label, new_label in label_mapping.items():
        new_labels[labels == old_label] = new_label

    # Update cluster count
    cluster_count = len(valid_labels)

    return new_labels, cluster_count


# ====================== Generate Rings ============================= #

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

def generate_rings_vectorized(circles: np.ndarray,
                             points_per_ring: int = 500,
                             radius_scatter: float = 0.01,
                             bounds: Tuple[float, float, float, float] = (0, 1, 0, 1),
                             verbose: bool = False) -> np.ndarray:
    """
    Vectorized generation of point clouds representing circular rings with controlled scatter.

    Creates a set of points for each input circle, with points randomly distributed
    around each ring's circumference with controlled radial variation, filtered to stay
    within specified bounds.

    Args:
        circles (np.ndarray): Array of shape (N,3) where each row contains:
            [x_center, y_center, radius] defining a circle
        points_per_ring (int): Number of points to generate per circle (default: 500)
        radius_scatter (float): Maximum radial variation from perfect circle (default: 0.01)
            - Points are generated with radii in [radius-scatter, radius+scatter]
            - Must be non-negative
        bounds (Tuple): (x_min, x_max, y_min, y_max) boundaries for point filtering
        verbose (bool): Whether to print generation statistics

    Returns:
        np.ndarray: A numpy array of (x, y) coordinates representing
            all generated points for the rings, filtered to stay within bounds.

    Raises:
        ValueError: If input validation fails:
            - circles array has incorrect shape (not N×3)
            - points_per_ring is negative
            - radius_scatter is negative
            - invalid bounds specification
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

    # Generate all angles and radii at once
    angles = np.random.uniform(0, 2*np.pi, (n_circles, points_per_ring))
    radii = circles[:, 2][:, np.newaxis] + np.random.uniform(
        -radius_scatter, radius_scatter, (n_circles, points_per_ring))

    # Convert to Cartesian coordinates
    x_coords = radii * np.cos(angles) + circles[:, 0][:, np.newaxis]
    y_coords = radii * np.sin(angles) + circles[:, 1][:, np.newaxis]

    # Combine and reshape
    all_points = np.column_stack((
        x_coords.ravel(),
        y_coords.ravel()
    ))

    # Vectorized boundary filtering
    in_bounds_mask = (
        (all_points[:, 0] >= x_min) &
        (all_points[:, 0] <= x_max) &
        (all_points[:, 1] >= y_min) &
        (all_points[:, 1] <= y_max))
    filtered_points = all_points[in_bounds_mask]


    if verbose:
        print(f"Generated {total_points:,} points total")
        print(f"Kept {len(filtered_points):,} points within bounds "
              f"({len(filtered_points)/total_points:.1%})\n")

    return filtered_points

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

def merge_clusters(cdict: Dict[int, Dict[str, Union[np.ndarray, bool, float]]],
                  keys: List[int],
                  verbose: bool = False) -> Optional[Dict[str, Union[np.ndarray, bool, float]]]:
    """
    Merges multiple clusters into a single cluster by fitting a new circle to their combined points.

    Args:
        cdict: Cluster dictionary where keys are cluster IDs and values contain:
               - 'points': np.ndarray of cluster points
               - 'circle': np.ndarray of fitted circle parameters [x, y, r]
               - 'errors': np.ndarray of fitting errors [err_x, err_y, err_r]
               - 'rmse': float of fitting RMSE
               - 'valid': bool indicating cluster validity
        keys: List of cluster IDs to merge
        verbose: Whether to print progress information (default: False)

    Returns:
        Dictionary with merged cluster information or None if merge failed.
        Contains same keys as input clusters with updated values.

    Raises:
        KeyError: If any cluster ID in keys is not found in cdict
        ValueError: If keys list is empty
    """

    # Input validation
    if not keys:
        raise ValueError("Keys list cannot be empty")
    for k in keys:
        if k not in cdict:
            raise KeyError(f"Cluster {k} not found in dictionary")

    if len(keys) == 1:
        if verbose:
            print(f"No need to merge cluster {keys}")
        return cdict[keys[0]].copy()
    else:
        # merge points
        points = np.vstack([cdict[k]['points'] for k in keys])

        # fit circle to points
        fit_result = fit_circle_to_points(points)

        # check result
        if fit_result is None:
            if verbose:
                print(f"Failed to merge clusters {keys}")
            return None
        else:
            if verbose:
                print(f"Merged clusters {keys} into a new cluster")

            # unpack fit results
            new_circle, new_errors, new_rmse = fit_result

            if verbose:
                print(f"Successfully merged clusters {keys}")
                print(f"New circle: center=({new_circle[0]:.3f}, {new_circle[1]:.3f}), radius={new_circle[2]:.3f}")

            return {
                'points': points,
                'circle': new_circle,
                'errors': new_errors,
                'rmse': new_rmse,
                'valid': True
            }

def compare_and_merge_clusters_2(cluster_dict: Dict[int, Dict],
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
            if not other_cluster['valid'] or other_key <= current_key or other_key in processed_keys:
                continue

            # Check compatibility
            are_compatible = compatible_clusters(merged_dict, current_key, other_key, sigma, verbose)

            if verbose:
                print(f"Clusters {current_key} and {other_key} are compatible? {are_compatible}")

            # Merge if compatible
            if are_compatible:
                # Merge current and other cluster
                merged_cluster = merge_clusters(merged_dict, [current_key, other_key], verbose)
                # Check if merge was successful
                if merged_cluster is None or merged_cluster['rmse'] >= current_cluster['rmse']:
                    # Skip other cluster if no fit or rmse does not improve
                    continue
                else:
                    # Accept merge
                    current_cluster.update(merged_cluster)

                    # append other_key to merged_keys
                    merged_keys.append(other_key)

        if len(merged_keys) == 1:
            current_cluster['merged_from'] = None
            # PAPO non serve c'è la copia globale all'inizio
            merged_dict[current_key] = current_cluster.copy()
        else:

            # get maximum rmse from ccompatible clusters
            max_rmse = max([merged_dict[k]['rmse'] for k in merged_keys])

            # merge clusters
            merged_cluster = merge_clusters(merged_dict, merged_keys, verbose)

            # check results
            if merged_cluster is None or merged_cluster['rmse'] >= max_rmse:
                # skip current cluster if no fit or rmse does not improve
                continue
            else:
                # accept merge
                current_cluster.update(merged_cluster)


            current_cluster['merged_from'] = merged_keys

            # Mark non-primary members as merged
            for member_key in merged_keys[1:]:
                merged_dict[member_key]['valid'] = False
                merged_dict[member_key]['merged_into'] = current_key

        # add merged_keys to processed set
        processed_keys.update(merged_keys)

    return merged_dict
# ============================ DATA STRUCTURES ============================ #
@dataclass
class RatiiData:
    """
    Container for storing statistical and diagnostic results of parameter evaluations.

    Attributes:
        mean_ratii_x (np.ndarray): Mean values of the x-coordinate ratios across trials.
        mean_ratii_y (np.ndarray): Mean values of the y-coordinate ratios across trials.
        mean_ratii_r (np.ndarray): Mean values of the radius ratios across trials.
        std_err_x (np.ndarray): Standard errors of the x-coordinate ratios.
        std_err_y (np.ndarray): Standard errors of the y-coordinate ratios.
        std_err_r (np.ndarray): Standard errors of the radius ratios.
        num_nan_inf (np.ndarray): Array of dictionaries recording NaN and Inf counts per trial.
        total_times (np.ndarray): Execution times for each evaluation.
        efficiencies (np.ndarray): Efficiency metrics corresponding to each parameter configuration.
    """
    mean_ratii_x: np.ndarray
    mean_ratii_y: np.ndarray
    mean_ratii_r: np.ndarray
    std_err_x: np.ndarray
    std_err_y: np.ndarray
    std_err_r: np.ndarray
    num_nan_inf: np.ndarray
    total_times: np.ndarray
    efficiencies: np.ndarray

# ============================ CORE FUNCTIONS ============================ #

def update_parameter_value(parameter_name: str,
                         param_value: Union[int, float]) -> None:
    """
    Update the value of a specific parameter and ensure
        dependencies between parameters are respected.

    Args:
        parameter_name: The name of the parameter to update
        param_value: The new value to assign to the parameter

    Notes:
        - Updates global variables for all parameters
        - Maintains constraints between MIN/MAX_CLUSTERS_PER_RING
        - Maintains minimum delta between R_MIN and R_MAX (default 0.1)
        - Converts integer parameters to int type
    """

    # Declare global variables to ensure they are updated correctly
    global S_SCALE, SIGMA_THRESHOLD, SIGMA_THRESHOLD_RM, MIN_SAMPLES
    global MIN_CLUSTERS_PER_RING, MAX_CLUSTERS_PER_RING, NUM_RINGS
    global POINTS_PER_RING, RADIUS_SCATTER, R_MIN, R_MAX

    # Round floating point parameters to avoid precision issues
    if parameter_name in ["S_SCALE", "SIGMA_THRESHOLD", "SIGMA_THRESHOLD_RM",
                         "RADIUS_SCATTER", "R_MIN", "R_MAX"]:
        param_value = round(param_value, 3)  # Round to 3 decimal places


    # Update the parameter value based on the parameter name
    if parameter_name == "S_SCALE":
        S_SCALE = param_value

    elif parameter_name == "SIGMA_THRESHOLD":
        SIGMA_THRESHOLD = param_value

    elif parameter_name == "SIGMA_THRESHOLD_RM":
        SIGMA_THRESHOLD_RM = param_value

    elif parameter_name == "MIN_SAMPLES":
        MIN_SAMPLES = int(param_value)  # Convert to integer

    elif parameter_name == "MIN_CLUSTERS_PER_RING":
        MIN_CLUSTERS_PER_RING = int(param_value)  # Convert to integer
        # Ensure MAX_CLUSTERS_PER_RING is at least equal to MIN_CLUSTERS_PER_RING
        MAX_CLUSTERS_PER_RING = max(MAX_CLUSTERS_PER_RING, MIN_CLUSTERS_PER_RING)

    elif parameter_name == "MAX_CLUSTERS_PER_RING":
        MAX_CLUSTERS_PER_RING = int(param_value)  # Convert to integer
        # Ensure MIN_CLUSTERS_PER_RING is not greater than MAX_CLUSTERS_PER_RING
        MIN_CLUSTERS_PER_RING = min(MIN_CLUSTERS_PER_RING, MAX_CLUSTERS_PER_RING)

    elif parameter_name == "NUM_RINGS":
        NUM_RINGS = int(param_value)  # Convert to integer

    elif parameter_name == "POINTS_PER_RING":
        POINTS_PER_RING = int(param_value)

    elif parameter_name == "RADIUS_SCATTER":
        RADIUS_SCATTER = param_value

    elif parameter_name == "R_MIN":
        R_MIN = param_value
        # Ensure R_MAX is at least R_MIN + 0.1 (minimum delta)
        R_MAX = max(R_MAX, R_MIN + 0.1)

    elif parameter_name == "R_MAX":
        R_MAX = param_value
        # Ensure R_MIN is at most R_MAX - 0.1 (minimum delta)
        R_MIN = min(R_MIN, R_MAX - 0.1)

def get_current_parameters() -> Dict[str, Union[int, float]]:
    """
    Collect and return the current configuration parameters used in the simulation or analysis.

    This function gathers all relevant global constants into a single dictionary for easy access,
    logging, or saving. It includes spatial boundaries, ring generation settings, DBSCAN clustering
    parameters, fitting thresholds, and more.

    Returns:
        Dict[str, Union[int, float]]: A dictionary mapping parameter names to their current values.
    """
    return {
        # Number of rings to generate
        "NUM_RINGS": NUM_RINGS,

        # X-axis bounds for ring centers
        "X_MIN": X_MIN,
        "X_MAX": X_MAX,

        # Y-axis bounds for ring centers
        "Y_MIN": Y_MIN,
        "Y_MAX": Y_MAX,

        # Minimum and maximum possible ring radii
        "R_MIN": R_MIN,
        "R_MAX": R_MAX,

        # Number of data points to generate per ring
        "POINTS_PER_RING": POINTS_PER_RING,

        # Standard deviation for radial noise (radius scatter)
        "RADIUS_SCATTER": RADIUS_SCATTER,

        # Range of DBSCAN epsilon values for adaptive clustering
        "MIN_DBSCAN_EPS": MIN_DBSCAN_EPS,
        "MAX_DBSCAN_EPS": MAX_DBSCAN_EPS,

        # Expected bounds on number of clusters per ring (for adaptive DBSCAN)
        "MIN_CLUSTERS_PER_RING": MIN_CLUSTERS_PER_RING,
        "MAX_CLUSTERS_PER_RING": MAX_CLUSTERS_PER_RING,

        # Minimum number of samples required for a DBSCAN core point
        "MIN_SAMPLES": MIN_SAMPLES,

        # Sigma threshold for residual-based outlier removal
        "SIGMA_THRESHOLD_RM": SIGMA_THRESHOLD_RM,

        # Sigma threshold for accepting a circle fit
        "SIGMA_THRESHOLD": SIGMA_THRESHOLD,

        # Scaling factor used during scoring or optimization
        "S_SCALE": S_SCALE,

        # Threshold for deciding whether two points form a viable fitting pair
        "FITTING_PAIR_TRESHOLD": FITTING_PAIR_TRESHOLD,

        # Number of parameter repetitions for statistical evaluation
        "N_FT": N_FT
    }

# ============================ ANALYSIS FUNCTIONS ============================ #

def run_fine_tuning(parameter_name: str,
                    parameter_values: List[Union[int, float]],
                    n_ft: int,
                    verbose: bool = False
                    ) -> Tuple[
                        List[float], List[float], List[float],  # mean_ratii_x, y, r
                        List[float], List[float], List[float],  # std_dev_x, y, r
                        List[float], List[float], List[float],  # std_err_x, y, r
                        List[Dict[str, Union[int, float]]],     # num_nan_inf
                        List[Dict[str, Union[int, float]]],     # all_results
                        List[float],                            # total_times
                        List[float]                             # efficiencies
                    ]:
    """
    Run the fine-tuning process for a given parameter.

    Args:
        parameter_name (str): The name of the parameter to fine-tune.
        parameter_values (List[Union[int, float]]): List of parameter values to test.
        n_ft (int): Number of fine-tuning runs per parameter value.
        verbose (bool): If True, print detailed progress messages.

    Returns:
        Tuple: Contains statistics, error metrics, and diagnostic data:
            - mean_ratii_x/y/r (List[float])
            - std_dev_x/y/r (List[float])
            - std_err_x/y/r (List[float])
            - num_nan_inf (List[Dict])
            - all_results (List[Dict])
            - total_times (List[float])
            - efficiencies (List[float])
    """

    # Initialize storage for results
    mean_ratii_x, mean_ratii_y, mean_ratii_r = [], [], []
    std_dev_x, std_dev_y, std_dev_r = [], [], []
    std_err_x, std_err_y, std_err_r = [], [], []
    num_nan_inf, all_results = [], []
    total_times = []
    efficiencies = []

    for param_value in parameter_values:
        # Clean up the display of floating-point parameter values
        if isinstance(param_value, float):
            formatted = f"{param_value:.3f}"
            display_value = (
                formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted
            )
        else:
            display_value = str(param_value)

        print(f"\n\n\n============================Testing {parameter_name} = {display_value}..."
              "==============================================================")
        update_parameter_value(parameter_name, param_value)

        # Print all parameter values before each run
        if verbose:
            print_all_parameters(PARAMETER_NAMES,
                               [S_SCALE, SIGMA_THRESHOLD, SIGMA_THRESHOLD_RM,
                                MIN_SAMPLES, MIN_CLUSTERS_PER_RING,
                                MAX_CLUSTERS_PER_RING, NUM_RINGS, POINTS_PER_RING,
                                RADIUS_SCATTER, R_MIN, R_MAX],
                               parameter_name, param_value)

        # Initialize storage for this parameter value
        all_ratii = []
        nan_inf_count = 0
        start_time = time.time()

        # Process each seed with progress bar
        for seed in tqdm(range(n_ft), total=n_ft, desc=f"Testing {parameter_name}={display_value}"):
            np.random.seed(seed + 1)

            # Get results from main procedure
            ratii = main_procedure_adaptive(verbose=False, seed=seed + 1)

            # Track invalid values
            if len(ratii) > 0:
                nan_inf_count += np.sum(~np.isfinite(ratii))
            all_ratii.append(ratii)

        elapsed_time = time.time() - start_time
        total_times.append(elapsed_time)

        # Process all ratii from all seeds
        if len(all_ratii) > 0:
            # Stack all ratii arrays (each is n x 3)
            stacked_ratii = np.vstack([r for r in all_ratii if len(r) > 0])

            if len(stacked_ratii) > 0:
                # Calculate statistics for each component
                means = np.nanmean(stacked_ratii, axis=0)
                std_devs = np.nanstd(stacked_ratii, axis=0)
                std_errs = std_devs / np.sqrt(len(stacked_ratii))

                # Calculate efficiency
                efficiency = (len(stacked_ratii) / (n_ft * NUM_RINGS)) * 100

                # Store results
                mean_ratii_x.append(means[0])
                mean_ratii_y.append(means[1])
                mean_ratii_r.append(means[2])
                std_dev_x.append(std_devs[0])
                std_dev_y.append(std_devs[1])
                std_dev_r.append(std_devs[2])
                std_err_x.append(std_errs[0])
                std_err_y.append(std_errs[1])
                std_err_r.append(std_errs[2])
                efficiencies.append(efficiency)
            else:
                # No valid ratii found
                means = [np.nan, np.nan, np.nan]
                std_devs = [np.nan, np.nan, np.nan]
                std_errs = [np.nan, np.nan, np.nan]
                efficiency = 0.0
        else:
            means = [np.nan, np.nan, np.nan]
            std_devs = [np.nan, np.nan, np.nan]
            std_errs = [np.nan, np.nan, np.nan]
            efficiency = 0.0

        # Store counts and results
        num_nan_inf.append({
            "parameter_value": param_value,
            "nan_inf_count_x": (
                np.sum(~np.isfinite(stacked_ratii[:, 0])) if len(stacked_ratii) > 0 else 0
            ),
            "nan_inf_count_y": (
                np.sum(~np.isfinite(stacked_ratii[:, 1])) if len(stacked_ratii) > 0 else 0
            ),
            "nan_inf_count_r": (
                np.sum(~np.isfinite(stacked_ratii[:, 2])) if len(stacked_ratii) > 0 else 0
            )
        })

        all_results.append({
            "parameter_value": param_value,
            "mean_ratii_x": means[0],
            "mean_ratii_y": means[1],
            "mean_ratii_r": means[2],
            "std_dev_x": std_devs[0],
            "std_dev_y": std_devs[1],
            "std_dev_r": std_devs[2],
            "std_err_x": std_errs[0],
            "std_err_y": std_errs[1],
            "std_err_r": std_errs[2],
            "nan_inf_count": nan_inf_count,
            "elapsed_time": elapsed_time,
            "efficiency": efficiency
        })

        if verbose:
            print("\nStatistics for Ratii:")
            print(f"Ratio X: Mean = {means[0]:.3f}, Std Dev = {std_devs[0]:.3f},"
                  f"SEM = {std_errs[0]:.3f}")
            print(f"Ratio Y: Mean = {means[1]:.3f}, Std Dev = {std_devs[1]:.3f},"
                  f"SEM = {std_errs[1]:.3f}")
            print(f"Ratio R: Mean = {means[2]:.3f}, Std Dev = {std_devs[2]:.3f},"
                  f"SEM = {std_errs[2]:.3f}")
            print(f"\nTotal Elapsed Time: {elapsed_time:.2f} seconds")
            print(f"Average Time per Seed: {elapsed_time / n_ft:.2f} seconds")
            print(f"Efficiency: {efficiency:.2f}%")

    return (mean_ratii_x, mean_ratii_y, mean_ratii_r,
            std_dev_x, std_dev_y, std_dev_r,
            std_err_x, std_err_y, std_err_r,
            num_nan_inf, all_results, total_times, efficiencies)

# ============================ VISUALIZATION ============================ #

def plot_mean_ratii_vs_parameter(parameter_values: np.ndarray,
                                ratii_data: RatiiData,
                                parameter_name: str,
                                save_plot: bool = False) -> None:
    """
    Plot and optionally save the mean ratii vs parameter values.

    Args:
        parameter_values: The values of the parameter being tested
        ratii_data: Contains mean ratii, errors, and other metrics
        parameter_name: Name of the parameter being fine-tuned
        save_plot: If True, saves the plot to file
    """
    fig, a_x = plt.subplots(figsize=(12, 8))

    # Convert inputs to NumPy arrays
    parameter_values = np.asarray(parameter_values)
    mean_ratii_x = np.asarray(ratii_data.mean_ratii_x)
    mean_ratii_y = np.asarray(ratii_data.mean_ratii_y)
    mean_ratii_r = np.asarray(ratii_data.mean_ratii_r)
    std_err_x = np.asarray(ratii_data.std_err_x)
    std_err_y = np.asarray(ratii_data.std_err_y)
    std_err_r = np.asarray(ratii_data.std_err_r)
    efficiencies = np.asarray(ratii_data.efficiencies)
    total_times = np.asarray(ratii_data.total_times)

    # Plotting with error bars
    a_x.errorbar(parameter_values, mean_ratii_x, yerr=std_err_x,
               fmt='o-', label='Mean Ratio X', color='blue', capsize=5)
    a_x.errorbar(parameter_values, mean_ratii_y, yerr=std_err_y,
               fmt='o-', label='Mean Ratio Y', color='green', capsize=5)
    a_x.errorbar(parameter_values, mean_ratii_r, yerr=std_err_r,
               fmt='o-', label='Mean Ratio R', color='red', capsize=5)

    # Uncertainty regions
    a_x.fill_between(parameter_values,
                   mean_ratii_x - std_err_x,
                   mean_ratii_x + std_err_x,
                   color='blue', alpha=0.2)
    a_x.fill_between(parameter_values,
                   mean_ratii_y - std_err_y,
                   mean_ratii_y + std_err_y,
                   color='green', alpha=0.2)
    a_x.fill_between(parameter_values,
                   mean_ratii_r - std_err_r,
                   mean_ratii_r + std_err_r,
                   color='red', alpha=0.2)

    # Point labels
    for i, (x_coord, y_x, y_y, y_r) in enumerate(zip(parameter_values,
                                             mean_ratii_x, mean_ratii_y, mean_ratii_r)):
        label = (f"Eff: {efficiencies[i]:.1f}%\n"
                f"NaN/Inf: {ratii_data.num_nan_inf[i]['nan_inf_count_x']}\n"
                f"Time: {total_times[i]:.1f}s")
        a_x.text(x_coord, y_x+0.05, label, fontsize=8, color='blue',
               ha='center', va='top', rotation=45)
        a_x.text(x_coord, y_y+0.03, label, fontsize=8, color='green',
               ha='center', va='center', rotation=45)
        a_x.text(x_coord, y_r+0.01, label, fontsize=8, color='red',
               ha='center', va='bottom', rotation=45)

    # Formatting
    a_x.set_xlabel(parameter_name)
    a_x.set_ylabel('Mean Ratii')
    a_x.set_title(f'Mean Ratii vs {parameter_name} with Error Bars')
    a_x.legend()
    a_x.grid(True)

    # Save before showing if requested
    if save_plot:
        plot_filename = os.path.join(DRIVE_RESULTS_PATH, f"plot_{parameter_name}.png")
        fig.savefig(plot_filename)
        print(f"\nPlot saved as {plot_filename}")

    plt.show()
    plt.close(fig)  # Ensure clean-up

def print_all_parameters(parameter_names: List[str],
                         parameter_values: List[Union[int, float]],
                         parameter_to_tune: str,
                         param_value: Union[int, float]) -> None:
    """
    Print the values of all parameters before each run, highlighting the parameter being fine-tuned.

    Args:
        parameter_names (List[str]): A list of names of all parameters.
        parameter_values (List[Union[int, float]]): A list of current values for all parameters.
        parameter_to_tune (str): The name of the parameter being fine-tuned.
        param_value (Union[int, float]): The current value of the parameter being fine-tuned.
    """
    # Print a header to indicate the start of the parameter values section
    print("\nCurrent Parameter Values:")

    # Loop through each parameter name and its corresponding value
    for nam, val in zip(parameter_names, parameter_values):
        # Format floating point numbers cleanly
        if isinstance(val, float):
            formatted = f"{param_value:.3f}"
            display_value = (
                formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted
            )
        else:
            display_value = str(val)

        # Check if the current parameter is the one being fine-tuned
        if nam == parameter_to_tune:
            # Print the parameter being fine-tuned with a special note
            print(f"{nam}: {display_value} (being analized)")
        else:
            # Print the parameter and its current value
            print(f"{nam}: {display_value}")

    print("\n")

def print_nan_inf_counts(num_nan_inf_: List[Dict[str, Union[float, int]]]) -> None:
    """
    Print the number of NaN and inf values encountered for each parameter value.

    Args:
        num_nan_inf_ (List[Dict[str, Union[float, int]]]): A list of dictionaries
            containing the counts of NaN and inf values
            for each parameter value. Each dictionary should have the keys:
            - "parameter_value": The value of the parameter being tested.
            - "nan_inf_count_x": The count of NaN/inf values for Ratio X.
            - "nan_inf_count_y": The count of NaN/inf values for Ratio Y.
            - "nan_inf_count_r": The count of NaN/inf values for Ratio R.
    """
    # Print a header to indicate the start of the NaN/inf counts section
    print("\nNumber of NaN and inf values encountered for each parameter value:")

    # Loop through each entry in the num_nan_inf_ list
    for count in num_nan_inf_:

        # Format parameter value for display
        param_value = count['parameter_value']
        if isinstance(param_value, float):
            formatted = f"{param_value:.3f}"
            display_value = (
                formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted
            )
        else:
            display_value = str(param_value)

        # Print the parameter value being tested
        print(f"Parameter Value: {display_value}")

        # Print the count of NaN/inf values for Ratio X
        print(f"NaN/Inf Count for Ratio X: {count['nan_inf_count_x']}")

        # Print the count of NaN/inf values for Ratio Y
        print(f"NaN/Inf Count for Ratio Y: {count['nan_inf_count_y']}")

        # Print the count of NaN/inf values for Ratio R
        print(f"NaN/Inf Count for Ratio R: {count['nan_inf_count_r']}")

        # Print an empty line for better readability between entries
        print()

# ============================ HERE IS THE MAIN============================ #

if __name__ == "__main__":

    # Extended parameter configuration
    PARAMETER_NAMES = [
        "S_SCALE", "SIGMA_THRESHOLD", "SIGMA_THRESHOLD_RM", "MIN_SAMPLES",
        "MIN_CLUSTERS_PER_RING", "MAX_CLUSTERS_PER_RING", "NUM_RINGS",
        "POINTS_PER_RING", "RADIUS_SCATTER", "R_MIN", "R_MAX"
    ]

    # Default ranges for each parameter
    PARAMETER_STARTS =[1.2, 1.0,  1.0,   5, 2, 3, 1, 100, 0.01,  0.1,  0.2]
    PARAMETER_ENDS =  [2.0, 10.0, 10.0, 15, 6, 8, 3, 500, 0.2,   0.7,  0.8]
    PARAMETER_STEPS = [0.1, 1.0,  1.0,   1, 1, 1, 1, 50,  0.01, 0.05, 0.05]

    PARAMETER_INDEX = 0  # Index of parameter to tune, a number from 0 to 10
    PARAMETER_TO_TUNE = PARAMETER_NAMES[PARAMETER_INDEX]
    PARAMETER_START = PARAMETER_STARTS[PARAMETER_INDEX]
    PARAMETER_END = PARAMETER_ENDS[PARAMETER_INDEX]
    PARAMETER_STEP = PARAMETER_STEPS[PARAMETER_INDEX]

    PARAMETER_VALUES = np.linspace(PARAMETER_START, PARAMETER_END,
                                   num=int((PARAMETER_END - PARAMETER_START) / PARAMETER_STEP) + 1)

    # Run fine-tuning and get results
    (
    mean_ratii_x_res, mean_ratii_y_res, mean_ratii_r_res, std_dev_x_res, std_dev_y_res,
    std_dev_r_res, std_err_x_res, std_err_y_res, std_err_r_res, num_nan_inf_res, all_results_res,
    total_times_res, efficiencies_res
    ) = run_fine_tuning(PARAMETER_TO_TUNE, PARAMETER_VALUES, N_FT, verbose=True)

    mean_ratii_x_a= np.array(mean_ratii_x_res)
    mean_ratii_y_a= np.array(mean_ratii_y_res)
    mean_ratii_r_a= np.array(mean_ratii_r_res)

    # Print the number of NaN and inf values for each parameter value
    #print_nan_inf_counts(num_nan_inf_res)

    ratii_da = RatiiData(
    mean_ratii_x=mean_ratii_x_a,
    mean_ratii_y=mean_ratii_y_a,
    mean_ratii_r=mean_ratii_r_a,
    std_err_x=np.array(std_err_x_res),
    std_err_y=np.array(std_err_y_res),
    std_err_r=np.array(std_err_r_res),
    num_nan_inf=np.array(num_nan_inf_res),
    total_times=np.array(total_times_res),
    efficiencies=np.array(efficiencies_res)
    )

    # Plot the results
    plot_mean_ratii_vs_parameter(PARAMETER_VALUES, ratii_da, PARAMETER_TO_TUNE,
                                save_plot=SAVE_RESULTS)

    # Save results summary if requested
    if SAVE_RESULTS:
        RESULTS_FILENAME = os.path.join(DRIVE_RESULTS_PATH, f"results_{PARAMETER_TO_TUNE}.txt")
        with open(RESULTS_FILENAME, 'w', encoding='utf-8') as f:
            # Write header with timestamp
            f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write all current parameter values
            f.write("=== CURRENT PARAMETER VALUES ===\n")
            params = get_current_parameters()
            for name, value in params.items():
                if isinstance(value, float):
                    f.write(f"{name}: {value:.3f}\n")
                else:
                    f.write(f"{name}: {value}\n")

            # Write tuning information
            f.write("\n=== TUNING INFORMATION ===\n")
            f.write(f"Parameter Tuned: {PARAMETER_TO_TUNE}\n")
            f.write(f"Parameter Values Tested: {PARAMETER_VALUES}\n")
            f.write(f"Number of Seeds per Value: {N_FT}\n\n")

            # Write summary statistics
            f.write("=== SUMMARY STATISTICS ===\n")
            f.write(f"Total Elapsed Time: {np.sum(total_times_res):.2f} seconds\n")
            f.write(f"Average Time per Seed: {np.mean(total_times_res)/N_FT:.2f} seconds\n")
            f.write(f"Average Efficiency: {np.mean(efficiencies_res):.2f}%\n\n")

            # Write detailed results
            f.write("=== DETAILED RESULTS ===\n")
            for idx, param_val in enumerate(PARAMETER_VALUES):
                f.write(f"\nValue: {param_val}\n")
                f.write(
                    f"Mean Ratii:\n X: {mean_ratii_x_res[idx]:.3f} ± {std_err_x_res[idx]:.3f}\n "
                    f"Y: {mean_ratii_y_res[idx]:.3f} ± {std_err_y_res[idx]:.3f} \n "
                    f"R: {mean_ratii_r_res[idx]:.3f} ± {std_err_r_res[idx]:.3f}\n"
                )
                f.write(f"Efficiency: {efficiencies_res[idx]:.2f}%\n")
                f.write(f"Time: {total_times_res[idx]:.2f}s\n")

        print(f"\nResults summary saved to {RESULTS_FILENAME}")

    # Call at the end of your simulation
    print("\n🎉 SIMULATION COMPLETE! 🎉")
# ============================ IMPORTS ============================ #

# Standard library imports
from typing import Tuple
import warnings

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================ EXPORT LIST ============================ #
__all__ = [
    # Data generation
    'generate_circles',
    'generate_rings',

    # Core functions for the CNN
    'points_to_image',
    'image_to_points',
    'create_dataset',
    'build_cnn',
    'train_cnn',
    'predict_rings',
    'test_cnn_efficiency'
]

# ============================ CONSTANTS ============================ #
DEFAULT_IMG_SIZE = (64, 64)          # Default image dimensions (height, width)
DEFAULT_POINTS_PER_RING = 500        # Default points per generated ring
DEFAULT_MAX_CIRCLES = 3              # Default maximum number of circles to classify
DEFAULT_BATCH_SIZE = 32              # Default training batch size
DEFAULT_EPOCHS = 20                  # Default training epochs

# ============================ DATA GENERATION ============================ #

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

# ============================ CORE FUNCTIONS FOR THE CNN============================ #

def points_to_image(points: np.ndarray,
                    img_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Converts an array of 2D points into a grayscale image with marked points.

    Transforms normalized coordinates [0,1]×[0,1] to discrete pixel locations in a
    grayscale image, with points represented as white pixels (255). Coordinates are
    automatically scaled to the target image dimensions.

    Args:
        points: Input array of shape (N, 2) containing normalized point coordinates.
               Each point should be in the range [0,1] for both x and y dimensions.
        img_size: Target image dimensions as (width, height) tuple. Defaults to (64, 64).

    Returns:
        Grayscale image of specified size with dtype uint8, where points are marked
        as white pixels (255) on a black background (0).
    """

    # Initialize blank image
    img = np.zeros(img_size, dtype=np.uint8)
    width, height = img_size

    # Convert normalized coordinates to pixel indices
    scaled_x = (np.round(points[:, 0] * (width - 1))).astype(np.uint8)
    scaled_y = (np.round(points[:, 1] * (height - 1))).astype(np.uint8)

    # Mark points in image (using Cartesian coordinates - origin at bottom-left)
    for x_coord, y_coord in zip(scaled_x, scaled_y):
        img[y_coord, x_coord] = 255

    return img

def image_to_points(image: np.ndarray,
                    n_points: int = 500) -> np.ndarray:

    """
    Converts a grayscale image to normalized point coordinates by intensity-weighted sampling.

    Samples pixel locations with probability proportional to their intensity values,
    then normalizes the coordinates to the [0,1]×[0,1] range. Brighter pixels have
    higher probability of being selected.

    Args:
        image: Input grayscale image array of shape (H, W) with values in [0, 255]
        n_points: Number of points to sample (default: 100)

    Returns:
        Array of shape (n_points, 2) containing normalized coordinates in [0,1] range
    """
    # Normalize image and prepare coordinate grids
    normalized_image = image / 255.0
    height, width = image.shape

    # Create coordinate grids and flatten
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    intensities = normalized_image.flatten()

     # Sample points with intensity-weighted probability
    sampled_indices = np.random.choice(len(intensities),
                                       size=n_points, p=intensities / np.sum(intensities))

    # Extract coordinates
    sampled_x = x_coords[sampled_indices]
    sampled_y = y_coords[sampled_indices]

    # Normalize the sampled coordinates to [0,1] range
    points = np.column_stack((sampled_x / (width - 1), sampled_y / (height - 1)))

    return points

def create_dataset(num_samples_per_count: int,
                   max_circles: int,
                   points_per_ring: int = 500,
                   x_min: float = 0.2,
                   x_max: float = 0.8,
                   y_min: float = 0.2,
                   y_max: float = 0.8,
                   r_min: float = 0.15,
                   r_max: float = 0.8,
                   img_size: Tuple[int, int] = (64, 64)
                   ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Generates a labeled dataset of synthetic circle images for classification tasks.

    Creates images containing 1 to max_circles randomly generated circles, with each
    image labeled by its circle count. Circles are generated with random centers and
    radii within specified bounds, then converted to grayscale images.

    Args:
        num_samples_per_count: Number of samples to generate per circle count
        max_circles: Maximum number of circles per image (inclusive)
        points_per_ring: Points to generate per circle (default: 300)
        x_min: Minimum x-coordinate for circle centers (default: 0.2)
        x_max: Maximum x-coordinate for circle centers (default: 0.8)
        y_min: Minimum y-coordinate for circle centers (default: 0.2)
        y_max: Maximum y-coordinate for circle centers (default: 0.8)
        r_min: Minimum circle radius (default: 0.05)
        r_max: Maximum circle radius (default: 0.8)
        img_size: Output image dimensions (default: (64, 64))

    Returns:
        A tuple containing:
            - images: Float32 array of shape (N, H, W, 1) with normalized pixel values [0,1]
            - labels: UInt8 array of shape (N,) with zero-based circle counts
    """
    images = []  # List to store generated images
    labels = []  # List to store corresponding labels

    # Generate samples for each circle count (1 to max_circles)
    for circle_count in range(1, max_circles+1):
        # Generate 'num_samples_per_count' samples for the current circle count
        for _ in range(num_samples_per_count):
            # Generate random circles with the current circle count
            circles = generate_circles(circle_count, x_min, x_max, y_min, y_max, r_min, r_max)

            # Generate points on the rings of the circles
            sample_points = generate_rings(circles, points_per_ring, radius_scatter=0.01)

            # Convert the points to a grayscale image
            img = points_to_image(sample_points, img_size=img_size)

            # Append the image and its label to the lists
            images.append(img)
            labels.append(circle_count-1)#Labels are zero-based(0 for 1 circle,1 for 2 circles,etc.)

    # Convert to numpy arrays with appropriate types
    images = np.array(images, dtype=np.float32) / 255.0  # Normalize to [0,1]
    labels = np.array(labels, dtype=np.uint8)

    # Add a channel dimension for compatibility with CNNs (batch, height, width, channels)
    images = np.expand_dims(images, axis=-1)  # Shape: (N, 64, 64, 1)

    return images, labels


def build_cnn(input_shape: tuple = (64, 64, 1),
              num_classes: int = 3
              ) -> tf.keras.Sequential:
    """
    Build and compile a simple Convolutional Neural Network (CNN) for image classification.

    The architecture includes two convolutional layers, each followed by max pooling,
    a dense hidden layer, and an output layer with softmax activation.

    Args:
        input_shape (tuple): Shape of input images, in (height, width, channels).
            Default is (64, 64, 1).
        num_classes (int): Number of output classes. Default is 3.

    Returns:
        tf.keras.Sequential: A compiled Keras CNN model.
    """
    # Initialize a Sequential model
    model = models.Sequential()

    # First Convolutional Layer
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    # - 16 filters, each of size 3x3
    # - ReLU activation function to introduce non-linearity
    # - Input shape is (64, 64, 1) for grayscale images

    # First MaxPooling Layer
    model.add(layers.MaxPooling2D((2, 2)))
    # - Reduces the spatial dimensions (height and width) by taking
    #   the maximum value in each 2x2 window
    # - Helps reduce computational complexity and control overfitting

    # Second Convolutional Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # - 32 filters, each of size 3x3
    # - ReLU activation function

    # Second MaxPooling Layer
    model.add(layers.MaxPooling2D((2, 2)))
    # - Further reduces spatial dimensions

    # Flatten Layer
    model.add(layers.Flatten())
    # - Converts the 2D feature maps into a 1D vector
    # - Prepares the data for the fully connected (Dense) layers

    # Fully Connected (Dense) Layer
    model.add(layers.Dense(64, activation='relu'))
    # - 64 neurons with ReLU activation
    # - Learns high-level features from the flattened data

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    # - 'num_classes' neurons (one for each class)
    # - Softmax activation function to output probabilities for each class

    # Compile the Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # - Optimizer: Adam (adaptive learning rate optimization algorithm)
    # - Loss: Sparse Categorical Crossentropy (for integer labels)
    # - Metrics: Accuracy (to monitor during training)

    return model

def train_cnn(num_samples_per_count: int,
              max_circles: int,
              points_per_ring: int,
              img_size: Tuple[int, int],
              epochs: int = 20,
              batch_size: int = 32
              ) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:

    """
    Generate a synthetic dataset, build a CNN model, train it, and visualize training history.

    This function generates ring-based images for classification (based on number of circles),
    builds a simple CNN classifier, trains it on the dataset, and plots both training accuracy
    and loss over epochs.

    Args:
        num_samples_per_count (int): Number of images to generate for each circle count
            (1 to max_circles).
        max_circles (int): Maximum number of circles per image (determines the number of classes).
        points_per_ring (int): Number of points sampled per ring.
        img_size (tuple): Shape of output image as (height, width).
        epochs (int): Number of training epochs. Default is 20.
        batch_size (int): Size of training batches. Default is 32.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]:
            The trained CNN model and its training history.
    """

    # 1) Generate dataset
    images, labels = create_dataset(num_samples_per_count, max_circles, points_per_ring,
                                    x_min=0.2, x_max=0.8, y_min=0.2, y_max=0.8,
                                    r_min=0.05, r_max=0.15, img_size=img_size)
    print(f"Dataset shape: {images.shape}, Labels shape: {labels.shape}")

    # Shuffle dataset
    perm = np.random.permutation(len(images))
    images, labels = images[perm], labels[perm]

    # 2) Build CNN
    model = build_cnn(input_shape=(img_size[0], img_size[1], 1), num_classes=max_circles)
    model.summary()

    # 3) Train CNN with callbacks for early stopping and reduced learning rate
    history = model.fit(
        images, labels,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=7, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=7, verbose=1)
        ]
    )


    # 4) Plot training history (Accuracy and Loss)
    _, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title("Model Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()

    # Loss plot
    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].set_title("Model Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return model, history

def predict_rings(model: tf.keras.Model,
                  img: np.ndarray,
                  verbose: bool = False) -> int:

    """
    Predicts the number of circles in an input image using a trained CNN model.

    The model is expected to output a probability distribution over classes where
    each class corresponds to the number of rings (starting from 1). The function
    returns the predicted number of rings by taking the argmax of the output and
    adding 1 (to adjust for 1-based labeling).

    Args:
        model: Trained Keras model for circle count prediction
        img: Input image array of shape (1, height, width, 1) or (height, width, 1)
        verbose: Whether to print prediction details. Default: False

    Returns:
        Predicted number of circles (1-based count)
    """

    # Generate the prediction probabilities for the input image.
    pred = model.predict(img)

    # Determine the predicted class and adjust index (e.g. 0 -> 1, 1 -> 2, ...)
    label = np.argmax(pred[0]) + 1  # Convert to 1-based index

    if verbose:
        print(f"Predicted {label} circle{'s' if label != 1 else ''} in the image.")
        print(f"Class probabilities: {dict(enumerate(pred[0].round(3), start=1))}")

    return label


def test_cnn_efficiency(model: tf.keras.Model,
                       num_trials: int = 200,
                       max_circles: int = 3,
                       img_size: Tuple[int, int] = (64, 64),
                       verbose: bool = False) -> float:
    """
    Tests CNN's accuracy in counting circles through randomized trials.

    Generates fresh images with random circle counts (1 to max_circles) using:
    - generate_circles() for circle generation
    - generate_rings_complete() for point generation
    - points_to_image() for image conversion

    Args:
        model: Trained CNN model for circle counting
        num_trials: Number of test images to generate (default: 200)
        max_circles: Maximum number of circles to generate (default: 3)
        img_size: Image dimensions (default: (64, 64))
        verbose: Whether to print detailed progress (default: False)

    Returns:
        Accuracy score between 0-1 representing correct prediction ratio

    """

    # Circle generation parameters
    gen_params = {
        'x_min': 0.2,
        'x_max': 0.8,
        'y_min': 0.2,
        'y_max': 0.8,
        'r_min': 0.15,  # Using your default from generate_circles()
        'r_max': 0.8    # Using your default from generate_circles()
    }

    # Ring generation parameters
    ring_params = {
        'points_per_ring': 500,     # Your default value
        'radius_scatter': 0.01      # Your default value
    }

    correct_predictions = 0
    confusion_matrix = np.zeros((max_circles, max_circles), dtype=int)

    for trial in range(num_trials):
        # Randomize circle count for each trial (1 to max_circles)
        true_count = np.random.randint(1, max_circles + 1)

        # Generate circles using your function
        circles = generate_circles(num_circles=true_count, **gen_params)

        # Generate points on rings using your function
        points = generate_rings(circles, **ring_params)

        # Create and preprocess image
        img = points_to_image(points, img_size)
        img_processed = img.astype(np.float32) / 255.0
        img_processed = np.expand_dims(img_processed, axis=[0, -1])  # Add batch + channel dims

        # Get prediction (convert from zero-based to one-based count)
        predicted_count = predict_rings(model, img_processed)

        # Update statistics
        if predicted_count == true_count:
            correct_predictions += 1
        confusion_matrix[true_count-1][predicted_count-1] += 1

        if verbose and (trial+1) % 50 == 0:
            print(f"Completed {trial+1}/{num_trials} trials...")

    # Calculate final accuracy
    accuracy = correct_predictions / num_trials

    # Print summary
    print("\n=== CNN Evaluation Summary ===")
    print(f"Total trials: {num_trials}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}\n")

    # Print confusion matrix
    print("Confusion Matrix (rows=true, cols=predicted):")
    header = "   " + " ".join(f"{i+1:3}" for i in range(max_circles))
    print(header)
    print("-" * len(header))
    for i in range(max_circles):
        print(f"{i+1:2} " + " ".join(f"{confusion_matrix[i][j]:3}"
                                   for j in range(max_circles)))

    return accuracy

def ptolemy_check(A: Tuple[float, float],
                  B: Tuple[float, float],
                  C: Tuple[float, float],
                  D: Tuple[float, float],
                  rtol: float = 1e-3) -> bool:

    """
    Check if four points in a plane satisfy Ptolemy's inequality as an equality.

    Ptolemy's theorem states that for four points A, B, C, D in a plane:
    AC * BD ≤ AB * CD + BC * DA, with equality if and only if the points are concyclic
    (lie on the same circle in order A-B-C-D or any cyclic permutation).

    Args:
        A, B, C, D: 1D numpy arrays representing 2D points (x,y coordinates)
        rtol: Relative tolerance for floating point comparison

    Returns:
        bool: True if the points satisfy AC * BD ≈ AB * CD + BC * DA within tolerance
    """
    # Calculate all pairwise distances between points
    AC = np.linalg.norm(A - C)
    BD = np.linalg.norm(B - D)
    AB = np.linalg.norm(A - B)
    BC = np.linalg.norm(B - C)
    CD = np.linalg.norm(C - D)
    DA = np.linalg.norm(D - A)

    # Check if the product of diagonals equals the sum of products of opposite sides
    return np.isclose(AC * BD, AB * CD + BC * DA, rtol=rtol)

def fit_circle_to_four_points(A: Tuple[float, float],
                              B: Tuple[float, float],
                              C: Tuple[float, float],
                              D: Tuple[float, float],
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
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D

    # Calculate the determinant of the matrix (to check for collinearity)
    determinant = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)

    # If the determinant is near zero, the points are nearly collinear
    if abs(determinant) < tolerance:
        return None  # Return None to indicate invalid input (collinear points)

    # Calculate the center of the circle using coordinate geometry formulas
    center_x = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / (2 * determinant)
    center_y = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / (2 * determinant)

    # Calculate the radius for each point
    radius1 = np.sqrt((center_x - x1)**2 + (center_y - y1)**2)
    radius2 = np.sqrt((center_x - x2)**2 + (center_y - y2)**2)
    radius3 = np.sqrt((center_x - x3)**2 + (center_y - y3)**2)
    radius4 = np.sqrt((center_x - x4)**2 + (center_y - y4)**2)

    # Calculate the average radius
    radius = np.mean([radius1, radius2, radius3, radius4])

    # Calculate the Root Mean Square Error (RMSE) of the fit
    rmse = np.sqrt(np.mean([(radius - radius1)**2, (radius - radius2)**2, (radius - radius3)**2, (radius - radius4)**2]))

    # Return the circle parameters and RMSE
    return [center_x, center_y, radius], rmse

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
    cx, cy, r = circle

    # Calculate the Euclidean distance from the circle's center to each point
    distances = np.linalg.norm(points - [cx, cy], axis=1)

    # Find points that are within the tolerance of the circle's radius
    compatible_points_mask = np.abs(distances - r) <= atol
    compatible_points = points[compatible_points_mask]  # Extract compatible points
    num_compatible_points = len(compatible_points)  # Count compatible points

    # Calculate RMSE only for compatible points
    if num_compatible_points > 0:
        residuals = distances[compatible_points_mask] - r  # Residuals (distance errors)
        rmse = np.sqrt(np.mean(residuals**2))  # Root Mean Square Error
    else:
        rmse = np.nan  # Return NaN if no compatible points are found

    return num_compatible_points, rmse

def process_points(points: np.ndarray, 
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
        k (int): Number of random samples (sets of 4 points) to draw and evaluate.
        min_points (int, optional): Minimum number of compatible points required to accept a detected circle. Defaults to 0.

    Returns:
        None. Prints diagnostic information and plots visualizations.
    """

    # Visualize the original points and known sample circles
    plot_points(points, color='blue', label="Original Points", hold=True)
    plot_circles(circles=sample_circles, title="Sample Circles", hold=False)
    plot_points(points, color='blue', label="Original Points", hold=True)
    plot_circles(sample_circles, title="Sample Circles", hold=True)

    ok_count = 0                    # Count of valid circles found
    found_circles = []             # Indices of sample circles matched to detected circles

    for i in range(k):
        # Randomly select 4 distinct points
        indices = np.random.choice(len(points), size=4, replace=False)
        A, B, C, D = points[indices]

        # Step 1: Check Ptolemy’s theorem (necessary for a cyclic quadrilateral)
        if not ptolemy_check(A, B, C, D):
            continue
        print(f"\nPtolemy's Theorem satisfied for points {indices}")

        # Step 2: Fit a circle to the 4 points
        result = fit_circle_to_four_points(A, B, C, D)
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

        found_circles.append(index)
        print()  # For spacing

    # Summary
    print("\nIndices of detected circles:", found_circles)
    print(f"Unique circles found: {set(found_circles)}")

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
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    # Plot the quadrilateral
    plt.plot(x + [x[0]], y + [y[0]], linestyle='-', markersize=8)

    # Plot the diagonals
    plt.plot([x[0], x[2]], [y[0], y[2]], linestyle='--', color=color)
    plt.plot([x[1], x[3]], [y[1], y[3]], linestyle='--', color=color)

    # Compute the circumcircle (center and radius)
    center, radius = circumcircle(points)

    # Plot the circumcircle as a dashed circle around the quadrilateral
    circle = plt.Circle(center, radius, color=color, fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    # Mark the center of the circle
    plt.plot(center[0], center[1], marker='o', markersize=10, color=color)


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
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]  # Not used in calculation

    # Compute determinant to check collinearity and prepare for the formula
    determinant = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
    if abs(determinant) < 1e-6:
        # If determinant is very small, points are almost collinear
        print("Warning: Points are nearly collinear. Circumcircle might not be accurate.")
        determinant = 1e-6

    # Apply the coordinate geometry formula to find the circumcenter
    center_x = ((x1**2 + y1**2) * (y2 - y3) +
                (x2**2 + y2**2) * (y3 - y1) +
                (x3**2 + y3**2) * (y1 - y2)) / (2 * determinant)

    center_y = ((x1**2 + y1**2) * (x3 - x2) +
                (x2**2 + y2**2) * (x1 - x3) +
                (x3**2 + y3**2) * (x2 - x1)) / (2 * determinant)

    # Compute radius using distance from center to one of the points
    radius = np.sqrt((center_x - x1)**2 + (center_y - y1)**2)

    return (center_x, center_y), radius
