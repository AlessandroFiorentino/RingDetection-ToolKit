

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
        )
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

    def test_adaptive_clustering_post_process_true(self):
        """Test adaptive_clustering with post-processing enabled."""
        # Create a dataset with a small cluster
        points = np.array([
            [0.1, 0.1], [0.11, 0.11], [0.12, 0.12], [0.13, 0.13],  # Cluster 1 (4 points)
            [0.3, 0.3], [0.31, 0.31], [0.32, 0.32], [0.33, 0.33],  # Cluster 2 (4 points)
            [0.9, 0.9], [0.91, 0.91], [0.92, 0.92], [0.93, 0.93],  # Cluster 3 (4 points)
            [0.8, 0.8], [0.81, 0.81]                               # Small cluster (2 points)
        ])
        labels, cluster_count = adaptive_clustering(
            points, min_clusters=2, max_clusters=4, min_samples=4, verbose=False, post_process=True
        )

        # Check that the small cluster was removed (reassigned to noise)
        self.assertEqual(cluster_count, 2)  # Only 2 valid clusters (small cluster removed)
        self.assertEqual(np.sum(labels == -1), 2)  # 2 points reassigned to noise

    def test_adaptive_clustering_post_process_false(self):
        """Test adaptive_clustering with post-processing disabled."""
        # Create a dataset with a small cluster
        points = np.array([
            [0.1, 0.1], [0.11, 0.11], [0.12, 0.12], [0.13, 0.13],  # Cluster 1 (4 points)
            [0.3, 0.3], [0.31, 0.31], [0.32, 0.32], [0.33, 0.33],  # Cluster 2 (4 points)
            [0.9, 0.9], [0.91, 0.91], [0.92, 0.92], [0.93, 0.93],  # Cluster 3 (4 points)
            [0.8, 0.8], [0.81, 0.81]                               # Small cluster (2 points)
        ])
        labels, cluster_count = adaptive_clustering(
            points, min_clusters=2, max_clusters=4, min_samples=4, verbose=False, post_process=False
        )

        # Check that the small cluster was not removed
        self.assertEqual(cluster_count, 2)  # Only 2 valid clusters (small cluster retained)
        self.assertEqual(np.sum(labels == -1), 0)  # No points reassigned to noise



    def test_adaptive_clustering_post_process_true(self):
        """Test adaptive_clustering with post-processing enabled."""
        # Create a dataset with a small cluster
        points = np.array([
            [0.1, 0.1], [0.11, 0.11], [0.12, 0.12], [0.13, 0.13],
            [0.3, 0.3], [0.31, 0.31], [0.32, 0.32], [0.33, 0.33],
            [0.9, 0.9], [0.91, 0.91], [0.92, 0.92], [0.93, 0.93],
            [0.8, 0.8], [0.81, 0.81]
        ])
        labels, cluster_count = adaptive_clustering(
            points, min_clusters=3, max_clusters=3, min_samples=4, verbose=False, post_process=True
        )

        # Check that the small cluster was removed (reassigned to noise)
        self.assertEqual(cluster_count, 3)  # Only 3 valid clusters
        self.assertEqual(np.sum(labels == -1), 1)  # 2 points reassigned to noise

    def test_adaptive_clustering_post_process_true(self):
        """Test adaptive_clustering with post-processing enabled."""
        # Create a dataset with a small cluster
        points = np.array([
            [0.1, 0.1], [0.11, 0.11], [0.12, 0.12], [0.13, 0.13],  # Cluster 1 (4 points)
            [0.3, 0.3], [0.31, 0.31], [0.32, 0.32], [0.33, 0.33],  # Cluster 2 (4 points)
            [0.9, 0.9], [0.91, 0.91], [0.92, 0.92], [0.93, 0.93],  # Cluster 3 (4 points)
            [0.7, 0.7], [0.71, 0.71]                               # Small cluster (2 points)
        ])
        labels, cluster_count = adaptive_clustering(
            points, min_clusters=3, max_clusters=4, min_samples=4, verbose=False, post_process=True
        )

        # Check that the small cluster was removed (reassigned to noise)
        self.assertEqual(cluster_count, 3)  # Only 3 valid clusters
        self.assertEqual(np.sum(labels == -1), 2)  # 2 points reassigned to noise
