# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
#
# For license terms see LICENSE file.
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Unit tests for printing and clustering utilities

"""
Test Suite for Ring Detection Utilities

Contains:
- Output formatting validation tests
- Clustering algorithm verification
- Error handling test cases
"""

# ============================ IMPORTS ============================ #
# Standard library imports
import io
import sys
import unittest
from typing import Optional, Tuple
from unittest.mock import patch

# Third-party imports
import numpy as np


from RingDetectionToolkit.RingDetectionToolkit.ringdetection import (
    print_circle,
    print_circles,
    adaptive_clustering,
    is_a_good_circle,
    filter_labels,
    analyze_ratii_efficiency,
    compatible_clusters,
    compare_and_merge_clusters,
    find_fitting_pairs
)

# ========================= TEST PRINT CIRCLE FUNCTIONS ========================= #

class TestPrintCircle(unittest.TestCase):
    """Unit tests for the print_circle() and print_circles() functions.
    """

    @staticmethod
    def capture_print_output(circle: np.ndarray,
                            errors: Optional[np.ndarray] = None,
                            title: Optional[str] = None,
                            label: Optional[str] = None,
                            rmse: Optional[float] = None) -> str:
        """
        Helper function to capture the printed output of print_circle.

        Args:
            circle (np.ndarray): A NumPy array representing the circle as [x, y, r].
            errors (np.ndarray, optional): A NumPy array representing the errors
                as [err_x, err_y, err_r].
            title (str, optional): A title to print before the circle details.
            label (str, optional): A label to prepend to the circle details.
            rmse (float, optional): The Root Mean Square Error (RMSE).

        Returns:
            str: The captured output as a string.
        """
        # Redirect stdout to a StringIO object
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Call the function
        print_circle(circle, errors, title, label, rmse)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Return the captured output
        return captured_output.getvalue()

    @staticmethod
    def capture_print_circles_output(circles: np.ndarray,
                                     errors: Optional[np.ndarray] = None,
                                     title: Optional[str] = None,
                                     label: Optional[str] = None,
                                     enum: Optional[np.ndarray] = None,
                                     rmse: Optional[np.ndarray] = None) -> str:
        """
        Helper function to capture the printed output of print_circles.

        Args:
            circles (np.ndarray): A NumPy array of circles, each row is [x, y, r].
            errors (np.ndarray, optional): A NumPy array of errors, each row is
                [err_x, err_y, err_r].
            title (str, optional): A title to print before the circle details.
            label (str, optional): A label to prepend to each circle's details.
            enum (np.ndarray, optional): A NumPy array of custom enumeration labels for each circle.
            rmse (np.ndarray, optional): A NumPy array of RMSE values for each circle.

        Returns:
            str: The captured output as a string.
        """
        # Redirect stdout to a StringIO object
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Call the function
        print_circles(circles, errors, title, label, enum, rmse)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Return the captured output
        return captured_output.getvalue()

    def test_print_circle_basic(self):
        """Test print_circle with minimal arguments.
        """
        circle = np.array([1.0, 2.0, 3.0])
        expected_output = "Center = (1.0000, 2.0000), Radius = 3.0000\n"
        output = self.capture_print_output(circle)
        self.assertEqual(output, expected_output)

    def test_print_circle_with_errors(self):
        """Test print_circle with errors.
        """
        circle = np.array([1.0, 2.0, 3.0])
        errors = np.array([0.1, 0.2, 0.3])
        expected_output = "Center = (1.0000 ± 0.1000, 2.0000 ± 0.2000), Radius = 3.0000 ± 0.3000\n"
        output = self.capture_print_output(circle, errors=errors)
        self.assertEqual(output, expected_output)

    def test_print_circle_with_rmse(self):
        """Test print_circle with RMSE.
        """
        circle = np.array([1.0, 2.0, 3.0])
        rmse = 0.05
        expected_output = "Center = (1.0000, 2.0000), Radius = 3.0000, RMSE: 0.0500\n"
        output = self.capture_print_output(circle, rmse=rmse)
        self.assertEqual(output, expected_output)

    def test_print_circle_with_all_arguments(self):
        """Test print_circle with all optional arguments.
        """
        circle = np.array([1.0, 2.0, 3.0])
        errors = np.array([0.1, 0.2, 0.3])
        title = "Test Circle"
        label = "Cluster 1"
        rmse = 0.05
        expected_output = (
            "Test Circle\n"
            "Cluster 1: Center = (1.0000 ± 0.1000, 2.0000 ± 0.2000), Radius = 3.0000 ± 0.3000, "
            "RMSE: 0.0500\n"
        )
        output = self.capture_print_output(circle, errors=errors,
                                           title=title, label=label, rmse=rmse)
        self.assertEqual(output, expected_output)

    def test_print_circles_basic(self):
        """Test print_circles with minimal arguments.
        """
        circles = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_output = (
            "0: Center = (1.0000, 2.0000), Radius = 3.0000\n"
            "1: Center = (4.0000, 5.0000), Radius = 6.0000\n"
        )
        output = self.capture_print_circles_output(circles)
        self.assertEqual(output, expected_output)

    def test_print_circles_with_enumeration(self):
        """Test print_circles with custom enumeration.
        """
        circles = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        enum = np.array([4, 7])  # Updated to use NumPy array
        label = "Cluster"
        expected_output = (
            "Cluster 4: Center = (1.0000, 2.0000), Radius = 3.0000\n"
            "Cluster 7: Center = (4.0000, 5.0000), Radius = 6.0000\n"
        )
        output = self.capture_print_circles_output(circles, label=label, enum=enum)
        self.assertEqual(output, expected_output)

    def test_print_circles_with_all_arguments(self):
        """Test print_circles with all optional arguments.
        """
        circles = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        errors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        title = "Test Circles"
        label = "Cluster"
        enum = np.array([4, 7])  # Updated to use NumPy array
        rmse = np.array([0.05, 0.06])
        expected_output = (
            "Test Circles\n"
            "Cluster 4: Center = (1.0000 ± 0.1000, 2.0000 ± 0.2000), Radius = 3.0000 ± 0.3000, "
            "RMSE: 0.0500\n"
            "Cluster 7: Center = (4.0000 ± 0.4000, 5.0000 ± 0.5000), Radius = 6.0000 ± 0.6000, "
            "RMSE: 0.0600\n"
        )
        output = self.capture_print_circles_output(circles, errors=errors, title=title, label=label,
                                                   enum=enum, rmse=rmse)
        self.assertEqual(output, expected_output)

# ============================= TEST ADAPTIVE CLUSTERING ========================= #

class TestAdaptiveClustering(unittest.TestCase):
    """Unit tests for the adaptive_clustering function."""

    def setUp(self):
        """Set up test data."""
        # Generate sample points for testing
        self.sample_points = np.array([
            [0.1, 0.1], [0.11, 0.11], [0.12, 0.12],  # Cluster 1
            [0.5, 0.5], [0.51, 0.51], [0.52, 0.52],  # Cluster 2
            [0.9, 0.9], [0.91, 0.91], [0.92, 0.92]   # Cluster 3
        ])

    def test_adaptive_clustering_valid_input(self):
        """Test adaptive_clustering with valid input parameters."""
        labels, cluster_count = adaptive_clustering(
            self.sample_points, min_clusters=1, max_clusters=3, verbose=False
        )

        # Check that the number of clusters is within the desired range
        self.assertGreaterEqual(cluster_count, 1)
        self.assertLessEqual(cluster_count, 3)
        # Check that labels are assigned correctly
        self.assertEqual(len(labels), len(self.sample_points))

    def test_adaptive_clustering_min_clusters_not_reached(self):
        """Test adaptive_clustering when min_clusters cannot be reached."""
        # Points are too spread out to form clusters with the given parameters
        sparse_points = np.array([
            [0.1, 0.1], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]
        ])
        _, cluster_count = adaptive_clustering(
            sparse_points, min_clusters=2, max_clusters=3, verbose=False
        )
        # Expect only one cluster (or noise)
        self.assertLessEqual(cluster_count, 1)

    def test_adaptive_clustering_max_clusters_exceeded(self):
        """Test adaptive_clustering when max_clusters is exceeded."""
        # Points are densely packed, leading to many small clusters
        dense_points = np.array([
            [0.1, 0.1], [0.11, 0.11], [0.12, 0.12],  # Cluster 1
            [0.2, 0.2], [0.21, 0.21], [0.22, 0.22],  # Cluster 2
            [0.3, 0.3], [0.31, 0.31], [0.32, 0.32]   # Cluster 3
        ])
        _, cluster_count = adaptive_clustering(
            dense_points, min_clusters=1, max_clusters=2, verbose=False
        )
        # Expect the number of clusters to be within the desired range
        self.assertGreaterEqual(cluster_count, 1)
        self.assertLessEqual(cluster_count, 2)

    def test_adaptive_clustering_filter_small_clusters(self):
        """Test adaptive_clustering for filtering small clusters.
        Previously, post_process=True was used to remove small clusters.
        With the update, filtering is always applied, and small clusters are removed.
        """
        # Create a dataset with a small cluster
        points = np.array([
            [0.1, 0.1], [0.11, 0.11], [0.12, 0.12], [0.13, 0.13],  # Cluster 1 (4 points)
            [0.3, 0.3], [0.31, 0.31], [0.32, 0.32], [0.33, 0.33],  # Cluster 2 (4 points)
            [0.9, 0.9], [0.91, 0.91], [0.92, 0.92], [0.93, 0.93],  # Cluster 3 (4 points)
            [0.7, 0.7], [0.71, 0.71]                               # Small cluster (2 points)
        ])
        # Remove post_process parameter (the filtering is always enabled now)
        labels, cluster_count = adaptive_clustering(
            points, min_clusters=3, max_clusters=4, min_samples=4, verbose=False
        )

        # Check that the small cluster was removed (filtered to noise)
        # Expect 3 valid clusters and that the 2 points of the small cluster were set to -1
        self.assertEqual(cluster_count, 3)
        self.assertEqual(np.sum(labels == -1), 2)

    def test_adaptive_clustering_invalid_points(self):
        """Test adaptive_clustering with invalid points input."""
        invalid_points = np.array([0.1, 0.1])  # Not a 2D array
        with self.assertRaises(ValueError):
            adaptive_clustering(invalid_points)

    def test_adaptive_clustering_invalid_min_clusters(self):
        """Test adaptive_clustering with invalid min_clusters."""
        with self.assertRaises(ValueError):
            adaptive_clustering(self.sample_points, min_clusters=0)

    def test_adaptive_clustering_invalid_max_clusters(self):
        """Test adaptive_clustering with invalid max_clusters."""
        with self.assertRaises(ValueError):
            adaptive_clustering(self.sample_points, max_clusters=0)

    def test_adaptive_clustering_max_clusters_less_than_min_clusters(self):
        """Test adaptive_clustering when max_clusters < min_clusters."""
        with self.assertRaises(ValueError):
            adaptive_clustering(self.sample_points, min_clusters=3, max_clusters=2)

    def test_adaptive_clustering_invalid_min_samples(self):
        """Test adaptive_clustering with invalid min_samples."""
        with self.assertRaises(ValueError):
            adaptive_clustering(self.sample_points, min_samples=2)

    def test_adaptive_clustering_invalid_max_iter(self):
        """Test adaptive_clustering with invalid max_iter."""
        with self.assertRaises(ValueError):
            adaptive_clustering(self.sample_points, max_iter=0)

    def test_adaptive_clustering_verbose_output(self):
        """Test adaptive_clustering with verbose output."""
        # Redirect stdout to capture verbose output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Call the function with verbose=True
        adaptive_clustering(self.sample_points, verbose=True)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Check that verbose output was printed
        output = captured_output.getvalue()
        self.assertIn("Target clusters", output)
        self.assertIn("Final eps", output)

    def test_filter_labels(self):
        """Test filter_labels with a typical case."""
        labels = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, -1, -1]
        min_points = 3

        # Capture verbose output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Call the function
        filtered_labels, num_valid_labels = filter_labels(
            labels=labels,
            min_points=min_points,
            verbose=True
        )

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Check outputs
        expected_labels = np.array([-1, -1, -1, 3, 3, 3, 4, 4, 4, 4, -1, -1])
        expected_valid_labels = 2

        # Verify outputs
        np.testing.assert_array_equal(filtered_labels, expected_labels)
        self.assertEqual(num_valid_labels, expected_valid_labels)

        # Verify verbose output
        verbose_output = captured_output.getvalue()
        self.assertIn("Initial clusters: 4", verbose_output)
        self.assertIn("Final clusters:   2", verbose_output)

    def test_filter_labels_negative_min_points(self):
        """Test with negative min_points."""
        with self.assertRaises(ValueError):
            filter_labels(labels=[1, 2, 3], min_points=-1)

    def test_filter_labels_all_noise(self):
        """Test case where all labels are noise (-1)."""
        labels = [-1, -1, -1]
        filtered_labels, num_valid_labels = filter_labels(labels, min_points=1)
        np.testing.assert_array_equal(filtered_labels, np.array([-1, -1, -1]))
        self.assertEqual(num_valid_labels, 0)

    def test_filter_labels_numpy_input(self):
        """Test with numpy array input."""
        labels = np.array([1, 2, 2, 3, 3, 3])
        filtered_labels, _ = filter_labels(labels, min_points=2)
        expected = np.array([-1, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(filtered_labels, expected)

# ============================= TEST IS A GOOD CIRCLE ========================= #

class TestIsAGoodCircle(unittest.TestCase):
    """Unit tests for the is_a_good_circle() function."""

    def test_basic_check_numpy_arrays(self):
        """Test basic check with NumPy arrays."""
        result = is_a_good_circle(
            circle=np.array([0.5, 0.5, 0.3]),
            errors=np.array([0.01, 0.02, 0.01])
        )
        self.assertTrue(result)

    def test_with_rmse_check(self):
        """Test with RMSE and avg_rmse check."""
        result = is_a_good_circle(
            circle=np.array([0.5, 0.5, 0.3]),
            errors=np.array([0.01, 0.02, 0.01]),
            rmse=np.array(0.05),
            avg_rmse=0.1
        )
        self.assertTrue(result)

    def test_custom_radius_thresholds(self):
        """Test with custom radius thresholds."""
        result = is_a_good_circle(
            circle=np.array([0.5, 0.5, 0.25]),
            errors=np.array([0.01, 0.02, 0.01]),
            r_max=0.5,
            r_min=0.1
        )
        self.assertTrue(result)

    def test_failing_case_radius_too_large(self):
        """Test failing case (radius too large)."""
        result = is_a_good_circle(
            circle=np.array([0.5, 0.5, 1.5]),
            errors=np.array([0.01, 0.02, 0.01])
        )
        self.assertFalse(result)

    def test_custom_alpha_value(self):
        """Test with custom alpha value."""
        result = is_a_good_circle(
            circle=np.array([0.5, 0.5, 0.3]),
            errors=np.array([0.01, 0.02, 0.01]),
            rmse=np.array(0.05),
            avg_rmse=0.1,
            alpha=2
        )
        self.assertTrue(result)

    def test_invalid_input(self):
        """Test with invalid input (None values)."""
        result = is_a_good_circle(
            circle=(None, None, None),
            errors=(0.01, 0.01, 0.02)
        )
        self.assertFalse(result)

    def test_radius_too_small(self):
        """Test case where radius is too small."""
        result = is_a_good_circle(
            circle=np.array([0.5, 0.5, 0.1]),
            errors=np.array([0.01, 0.02, 0.01]),
            r_min=0.2
        )
        self.assertFalse(result)

    def test_high_relative_error(self):
        """Test case with high relative error."""
        result = is_a_good_circle(
            circle=np.array([0.5, 0.5, 0.3]),
            errors=np.array([0.01, 0.02, 0.2])  # err_r/radius = 0.2/0.3 > 0.5
        )
        self.assertFalse(result)

    def test_rmse_too_high(self):
        """Test case where RMSE is too high."""
        result = is_a_good_circle(
            circle=np.array([0.5, 0.5, 0.3]),
            errors=np.array([0.01, 0.02, 0.01]),
            rmse=0.5,
            avg_rmse=0.1,
            alpha=3  # 0.5 > 3*0.1
        )
        self.assertFalse(result)

    def test_missing_avg_rmse_with_rmse(self):
        """Test that assertion is raised when rmse is provided without avg_rmse."""
        with self.assertRaises(AssertionError):
            is_a_good_circle(
                circle=np.array([0.5, 0.5, 0.3]),
                errors=np.array([0.01, 0.02, 0.01]),
                rmse=0.1
            )

    def test_tuple_input(self):
        """Test that function works with tuple input."""
        result = is_a_good_circle(
            circle=(0.5, 0.5, 0.3),
            errors=(0.01, 0.02, 0.01)
        )
        self.assertTrue(result)

    def test_non_finite_values(self):
        """Test with non-finite values in circle parameters."""
        result = is_a_good_circle(
            circle=np.array([0.5, 0.5, np.nan]),
            errors=np.array([0.01, 0.02, 0.01])
        )
        self.assertFalse(result)

    def test_invalid_alpha(self):
        """Test that assertion is raised when alpha is non-positive."""
        with self.assertRaises(AssertionError):
            is_a_good_circle(
                circle=np.array([0.5, 0.5, 0.3]),
                errors=np.array([0.01, 0.02, 0.01]),
                alpha=0
            )

# ============================= TEST ANALIZE RATII EFFICIENCY ========================= #

class TestAnalyzeRatiiEfficiency(unittest.TestCase):
    """Unit tests for the analyze_ratii_efficiency function."""

    def capture_print_output(self, *args, **kwargs) -> Tuple[float, str]:
        """Helper to capture printed output."""
        captured_output = io.StringIO()
        sys.stdout = captured_output
        result = analyze_ratii_efficiency(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return result, captured_output.getvalue()

    def test_normal_case(self):
        """Test with typical good input data."""
        num_seeds = 5
        num_rings = 3
        combined_ratii = np.random.rand(10, 3)  # 10 good rings found

        # Call function and capture output
        total_eff, output = self.capture_print_output(
            combined_ratii, num_rings, num_seeds
        )

        # Verify calculations
        expected_eff = (10 / 15) * 100  # 10 found out of 15 expected
        self.assertAlmostEqual(total_eff, expected_eff)

        # Verify printed output
        self.assertIn("Total expected rings: 15", output)
        self.assertIn("Good rings found: 10", output)
        self.assertIn(f"Total Efficiency: {expected_eff:.2f}%", output)

    def test_invalid_inputs(self):
        """Test input validation."""
        good_ratii = np.random.rand(5, 3)

        with self.subTest("Invalid num_rings"):
            with self.assertRaises(ValueError):
                analyze_ratii_efficiency(good_ratii, num_rings=0, num_seeds=5)

        with self.subTest("Invalid num_seeds"):
            with self.assertRaises(ValueError):
                analyze_ratii_efficiency(good_ratii, num_rings=3, num_seeds=-1)

# ============================= TEST COMPARE AND MERGE CLUSTERS ========================= #

class TestCompareAndMergeClusters(unittest.TestCase):
    """Tests for compare_and_merge_clusters and its helper functions."""

    def setUp(self):
        """Create test cluster dictionaries."""
        self.base_cluster_dict = {
            1: {
                'circle': np.array([0.5, 0.5, 0.2]),
                'errors': np.array([0.01, 0.01, 0.005]),
                'points': np.random.rand(10, 2),
                'rmse': 0.02,
                'valid': True
            },
            2: {
                'circle': np.array([0.52, 0.51, 0.19]),
                'errors': np.array([0.02, 0.015, 0.006]),
                'points': np.random.rand(10, 2),
                'rmse': 0.025,
                'valid': True
            },
            3: {
                'circle': np.array([0.7, 0.7, 0.3]),
                'errors': np.array([0.01, 0.01, 0.01]),
                'points': np.random.rand(10, 2),
                'rmse': 0.03,
                'valid': True
            },
            4: {
                'circle': np.array([0.9, 0.9, 0.4]),
                'errors': np.array([0.02, 0.02, 0.02]),
                'points': np.random.rand(10, 2),
                'rmse': 0.04,
                'valid': False  # Invalid cluster for testing
            }
        }

    # ================== compatible_clusters tests ==================

    def test_compatible_clusters_basic_compatibility(self):
        """Test when clusters should be compatible."""
        result = compatible_clusters(self.base_cluster_dict, 1, 2, sigma=3.0)
        self.assertTrue(result)

    def test_compatible_clusters_basic_incompatibility(self):
        """Test when clusters should not be compatible."""
        result = compatible_clusters(self.base_cluster_dict, 1, 3, sigma=3.0)
        self.assertFalse(result)

    def test_compatible_clusters_invalid_cluster_id(self):
        """Test with invalid cluster IDs."""
        with self.assertRaises(KeyError):
            compatible_clusters(self.base_cluster_dict, 1, 99)

    def test_compatible_clusters_negative_sigma(self):
        """Test with invalid sigma value."""
        with self.assertRaises(ValueError):
            compatible_clusters(self.base_cluster_dict, 1, 2, sigma=-1.0)

    # ================== compare_and_merge_clusters tests ==================

    def test_compare_and_merge_no_merges(self):
        """Test when no clusters should be merged."""
        # Make clusters incompatible by increasing distance
        test_dict = self.base_cluster_dict.copy()
        test_dict[1]['circle'] = np.array([0.5, 0.5, 0.2])
        test_dict[1]['errors'] = np.array([0.001, 0.001, 0.001])  # Small errors

        test_dict[2]['circle'] = np.array([0.7, 0.7, 0.19])  # Far from cluster 1
        test_dict[2]['errors'] = np.array([0.001, 0.001, 0.001])  # Small errors

        result = compare_and_merge_clusters(test_dict, sigma=1.0)  # Strict threshold

        # Verify no merges occurred
        self.assertEqual(len(result), len(test_dict))
        self.assertTrue(result[1]['valid'])
        self.assertTrue(result[2]['valid'])
        self.assertTrue(result[3]['valid'])
        self.assertIsNone(result[1]['merged_from'])
        self.assertIsNone(result[2]['merged_from'])
        self.assertIsNone(result[3]['merged_from'])

    def test_compare_and_merge_successful_merge(self):
        """Test successful cluster merging."""
        # Patch fit_circle_to_points to return a new circle with RMSE lower than max(0.02, 0.025)
        # so that clusters 1 and 2 merge.
        with patch('RingDetectionToolkit.ringdetection.fit_circle_to_points',
                return_value=(np.array([0.51, 0.505, 0.195]),
                                np.array([0.015, 0.012, 0.0055]),
                                0.02)):
            result = compare_and_merge_clusters(self.base_cluster_dict, sigma=3.0)

        # Verify clusters 1 and 2 were merged (keeping cluster 1 as primary)
        self.assertFalse(result[2]['valid'])
        self.assertEqual(result[2]['merged_into'], 1)
        self.assertEqual(result[1]['merged_from'], [1, 2])

        # Verify cluster 3 remains unchanged
        self.assertTrue(result[3]['valid'])
        self.assertIsNone(result[3]['merged_from'])

    def test_compare_and_merge_invalid_sigma(self):
        """Test with invalid sigma parameter."""
        with self.assertRaises(ValueError):
            compare_and_merge_clusters(self.base_cluster_dict, sigma=-1.0)

    def test_compare_and_merge_rmse_condition(self):
        """Test merge rejection when RMSE would increase."""
        test_dict = self.base_cluster_dict.copy()

        # Set high RMSE for cluster 2 to prevent merge.
        test_dict[2]['rmse'] = 0.5

        # Patch fit_circle_to_points to simulate a fit that yields a RMSE
        # higher than max(0.02, 0.5) i.e. 0.6, so that the merge is rejected.
        with patch('RingDetectionToolkit.ringdetection.fit_circle_to_points',
                return_value=(np.array([0.51, 0.505, 0.195]),
                                np.array([0.015, 0.012, 0.0055]),
                                0.6)):
            result = compare_and_merge_clusters(test_dict, sigma=3.0)

        # Verify no merge occurred even if the clusters are spatially compatible.
        # Both cluster 1 and cluster 2 should remain valid, and no merge info is recorded.
        self.assertTrue(result[1]['valid'])
        self.assertTrue(result[2]['valid'])
        self.assertIsNone(result[1].get('merged_from'))

    def test_compare_and_merge_skip_invalid_clusters(self):
        """Test that invalid clusters are skipped."""
        result = compare_and_merge_clusters(self.base_cluster_dict)

        # Verify invalid cluster 4 was not processed
        self.assertFalse(result[4]['valid'])
        self.assertNotIn('merged_from', result[4])

    def test_compare_and_merge_multiple_merges(self):
        """Test scenario with multiple merges."""
        test_dict = self.base_cluster_dict.copy()

        # Modify clusters 1, 2, and 3 so that they are close and compatible.
        # Cluster 1 (reference)
        test_dict[1]['circle'] = np.array([0.5, 0.5, 0.2])
        test_dict[1]['errors'] = np.array([0.05, 0.05, 0.02])
        # Cluster 2 - slightly offset but within error bounds
        test_dict[2]['circle'] = np.array([0.52, 0.51, 0.19])
        test_dict[2]['errors'] = np.array([0.05, 0.05, 0.02])
        # Cluster 3 - slightly offset but within error bounds
        test_dict[3]['circle'] = np.array([0.51, 0.52, 0.21])
        test_dict[3]['errors'] = np.array([0.05, 0.05, 0.02])

        # Ensure RMSE values allow merging.
        test_dict[1]['rmse'] = 0.02
        test_dict[2]['rmse'] = 0.02
        test_dict[3]['rmse'] = 0.02

        # Patch fit_circle_to_points to always return a new circle with low RMSE (0.02)
        # so that merging is accepted for every compatible pair.
        with patch('RingDetectionToolkit.ringdetection.fit_circle_to_points',
                return_value=(np.array([0.51, 0.51, 0.2]),
                                np.array([0.04, 0.04, 0.015]),
                                0.02)):
            result = compare_and_merge_clusters(test_dict, sigma=3.0)

        # Verify that all clusters merged into cluster 1.
        self.assertTrue(result[1]['valid'])
        self.assertFalse(result[2]['valid'])
        self.assertFalse(result[3]['valid'])
        self.assertEqual(result[1]['merged_from'], [1, 2, 3])
        self.assertEqual(result[2]['merged_into'], 1)
        self.assertEqual(result[3]['merged_into'], 1)

        # Verify the merged cluster has updated properties.
        self.assertIsNotNone(result[1]['circle'])
        self.assertIsNotNone(result[1]['errors'])
        self.assertIsNotNone(result[1]['rmse'])

# ============================= TEST FIND FITTING PAIRS ========================= #

class TestFindFittingPairs(unittest.TestCase):
    """Unit tests for the find_fitting_pairs function and its helper calculate_ratii."""

    def setUp(self):
        # Create sample original circles (each row: [x, y, r])
        self.original_circles = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ])

        # Create sample fitted circles and corresponding errors
        self.fitted_circles = np.array([
            [1.1, 1.1, 1.1],
            [10.0, 10.0, 10.0]
        ])
        self.fitted_errors = np.array([
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2]
        ])
        # Tuple as expected by find_fitting_pairs (placeholder set to None)
        self.new_best_fit = (self.fitted_circles, self.fitted_errors, None)
        self.threshold = 3.0

    def test_find_fitting_pairs_basic(self):
        """Test basic matching between original and fitted circles."""
        fitted_pairs, good_ratii, messages = find_fitting_pairs(
            self.original_circles, self.new_best_fit, threshold=self.threshold, verbose=False
        )

        # Calculate expected distance and total error for the first fitted circle
        expected_distance = np.linalg.norm(self.original_circles[0] - self.fitted_circles[0])
        expected_tot_error = np.linalg.norm(self.fitted_errors[0])

        # For original circle 0, the first fitted circle is within the threshold.
        # For original circle 1, no fitted circle should match (distance exceeds threshold*error)
        self.assertAlmostEqual(fitted_pairs[0][1], expected_distance, places=4)
        self.assertEqual(fitted_pairs[0][2], 0)
        self.assertAlmostEqual(fitted_pairs[0][3], expected_tot_error, places=4)

        # Original circle 1 must have no match: distance is infinity and fitted_idx is None
        self.assertEqual(fitted_pairs[1][1], np.inf)
        self.assertIsNone(fitted_pairs[1][2])
        self.assertIsNone(fitted_pairs[1][3])

        # Check good_ratii: only one pair is accepted so it should have one row with 3 columns
        self.assertEqual(good_ratii.shape, (1, 3))
        # Check that messages contain the expected header (first message)
        self.assertTrue(len(messages) > 0)
        self.assertTrue(messages[0].startswith("=" * 25))

    def test_find_fitting_pairs_invalid_threshold(self):
        """Test that a non-positive threshold raises a ValueError."""
        with self.assertRaises(ValueError):
            find_fitting_pairs(
                self.original_circles, self.new_best_fit, threshold=0, verbose=False
            )

    def test_find_fitting_pairs_invalid_fitted_arrays(self):
        """Test that non-numpy or mismatched fitted arrays raise errors."""
        # Pass non-numpy object instead of fitted_circles (should raise TypeError)
        new_best_fit_invalid = (list(self.fitted_circles), self.fitted_errors, None)
        with self.assertRaises(TypeError):
            find_fitting_pairs(
                self.original_circles, new_best_fit_invalid, threshold=self.threshold, verbose=False
            )

        # Pass fitted_errors with mismatched shape compared to fitted_circles
        # (should raise ValueError)
        new_best_fit_invalid2 = (self.fitted_circles, self.fitted_errors[:1], None)
        with self.assertRaises(ValueError):
            find_fitting_pairs(
                self.original_circles, new_best_fit_invalid2,
                threshold=self.threshold, verbose=False
            )
