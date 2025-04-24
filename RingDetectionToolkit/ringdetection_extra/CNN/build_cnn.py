# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
# For license terms see LICENSE file.
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software under GPLv2+
# See https://www.gnu.org/licenses/gpl-2.0.html

"""
Ring Detection and Classification Toolkit

This module provides comprehensive functionality for:
- Synthetic ring generation and visualization
- Image/point cloud conversion utilities
- CNN-based ring counting models
- Model training and evaluation pipelines

Key Features:
1. Data Generation:
   - generate_circles(): Creates random circle configurations
   - generate_rings(): Converts circles to point clouds
   - create_dataset(): Generates labeled training data

2. Image Processing:
   - points_to_image(): Converts point clouds to images
   - image_to_points(): Converts images back to point clouds

3. CNN Pipeline:
   - build_cnn(): Creates classification model architecture
   - train_cnn(): Full training workflow
   - predict_rings(): Makes predictions on new images
   - test_cnn_efficiency(): Evaluates model performance
"""

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

# ======================== HERE IS THE MAIN ========================#

if __name__ == "__main__":
    # ======================= CONFIG ======================= #
    NUM_SAMPLES_PER_COUNT = 300
    MAX_CIRCLES = 3
    POINTS_PER_RING = 500
    IMG_SIZE = (64, 64)
    EPOCHS = 15
    BATCH_SIZE = 32

    # ======================= TRAIN ======================= #
    model, _ = train_cnn(
        num_samples_per_count=NUM_SAMPLES_PER_COUNT,
        max_circles=MAX_CIRCLES,
        points_per_ring=POINTS_PER_RING,
        img_size=IMG_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # ======================= SINGLE TEST ======================= #
    test_circles = generate_circles(np.random.randint(1, MAX_CIRCLES + 1))
    test_points = generate_rings(test_circles)
    test_img = points_to_image(test_points, img_size=IMG_SIZE).astype(np.float32) / 255.0
    test_img = np.expand_dims(test_img, axis=(0, -1))

    predicted = predict_rings(model, test_img, verbose=True)

    plt.imshow(test_img[0, :, :, 0], cmap='gray')
    plt.title(f"Predicted Circles: {predicted}")
    plt.axis("off")
    plt.show()

    # ======================= EFFICIENCY ======================= #
    ACCURACY = test_cnn_efficiency(
        model=model,
        num_trials=200,
        max_circles=MAX_CIRCLES,
        img_size=IMG_SIZE,
        verbose=True
    )

    print(f"\n Final CNN accuracy over 200 trials: {ACCURACY:.2%}")
