# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
# For license terms see LICENSE file.
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software under GPLv2+
# See https://www.gnu.org/licenses/gpl-2.0.html

"""
Parameter Optimization Toolkit for Ring Detection

This module provides functionality for systematically evaluating and optimizing
parameters in the ring detection pipeline. Key features include:

- Parameter sweep analysis
- Statistical evaluation of parameter effects
- Visualization of parameter-performance relationships
- Automated parameter tuning
"""

# ============================ IMPORTS ============================ #
# Standard library imports
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Dict, Tuple, List

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from google.colab import drive

# Local imports
from complete_ring_detection import main_procedure_adaptive

# ============================ GOOGLE DRIVE SETUP ============================ #

# Mount Google Drive to access files from your Drive in Colab
#drive.mount('/content/drive')

# Define the path where fine-tuning results will be saved inside your Google Drive
DRIVE_RESULTS_PATH = '/content/drive/MyDrive/Ring_Detection/Fine_Tuning_results'

# Create the directory if it doesn't already exist (avoid error if it already exists)
os.makedirs(DRIVE_RESULTS_PATH, exist_ok=True)

# ============================ CONSTANTS ============================ #
DEBUG = False           # Global debug flag for additional output

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

# -------------------- OPTIMIZATION PARAMETERS -------------------- #
N_PR = 200                  # Number of Seeds in normal procedure
N_FT = 500                  # Number of parameter repetitions
SAVE_RESULTS = True         # Flag to control result saving

# Parameter names for display and reference
PARAMETER_NAMES = [
    "S_SCALE", "SIGMA_THRESHOLD", "SIGMA_THRESHOLD_RM",
    "MIN_SAMPLES", "MIN_CLUSTERS_PER_RING", "MAX_CLUSTERS_PER_RING",
    "NUM_RINGS", "POINTS_PER_RING", "RADIUS_SCATTER", "R_MIN", "R_MAX"
]

# ============================ EXPORT LIST ============================ #
__all__ = [
    'RatiiData',
    'update_parameter_value',
    'get_current_parameters',
    'run_fine_tuning',
    'plot_mean_ratii_vs_parameter',
    'print_nan_inf_counts',
    'print_all_parameters',
]

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
