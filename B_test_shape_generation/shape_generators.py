"""
shape_generators.py

This module provides functions for generating 2D and 3D geometric shapes
using Latin Hypercube Sampling (LHS), along with boundary-aware labeling noise.

Available shapes:
    - 2D: Rectangle, Radial Segment, Barbell, Sine Wave, Star
    - 3D: Saddle, Radial Segment, Barbell

Each shape generator returns:
    - df_x: pd.DataFrame of sample coordinates
    - y: binary label array (1 = inside, 0 = outside)
    - samples: raw sample points (np.ndarray)

Features:
    - Latin Hypercube Sampling (LHS)
    - Inverse rotation for evaluation
    - Label border noise control via distance-based label flipping
"""

# Standard library
from math import radians, pi

# Third-party libraries
import numpy as np
import pandas as pd
from pyDOE2 import lhs
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point, Polygon

# Local configuration
from src.config.settings import DEFAULT_VARIABLE_SEEDS


# --------------------------------------------------------
# 2D Sampling, Rotation, and Label Noise
# --------------------------------------------------------


def sample_points_2d(num_samples, rng=None):
    """
    Generate 2D sample points using Latin Hypercube Sampling (LHS).

    Args:
        num_samples (int): Number of samples to generate.
        rng (np.random.Generator, optional): Random generator for reproducibility.

    Returns:
        np.ndarray: Array of shape (num_samples, 2)
    """
    if rng is not None:
        np.random.seed(rng.integers(0, 1e6))
    return lhs(n=2, samples=num_samples)


def apply_rotation_2d(point, angle_rad):
    """
    Rotate a 2D point counterclockwise by a given angle.

    Args:
        point (np.ndarray): 2D coordinate to rotate.
        angle_rad (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotated 2D coordinate.
    """
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return rotation_matrix @ point


def apply_label_noise_2d(X, y, shape_polygon, label_noise=0.0, rng=None):
    """
    Apply label noise by flipping labels from 0 → 1 (outside → inside) near the polygon boundary.

    The closer a point labeled as 'outside' is to the boundary, the higher the probability it gets flipped to 'inside'.
    Points already labeled as 'inside' (label 1) are never flipped.

    Args:
        X (np.ndarray): Sample points of shape (N, 2).
        y (np.ndarray): Binary labels (0 = outside, 1 = inside), shape (N,).
        shape_polygon (shapely.geometry.Polygon): The geometric shape used for computing boundary distance.
        label_noise (float): Controls how fast the flip probability decays with distance to boundary.
                           Bigger values = more flips near the boundary.
        rng (np.random.Generator, optional): Random number generator. If None, a new generator is used.

    Returns:
        np.ndarray: Array of same shape as `y`, with flipped labels (0 → 1) based on label_noise value and proximity to
        the boundary.
    """
    if rng is None:
        rng = np.random.default_rng()

    y_noisy = y.copy()
    for i, point in enumerate(X):
        if y[i] == 0:
            d = shape_polygon.boundary.distance(Point(point))
            flip_prob = np.exp(-d / label_noise) if label_noise > 0 else 0.0
            if rng.random() < flip_prob:
                y_noisy[i] = 1

    return y_noisy


# --------------------------------------------------------
# Helper Functions - Sampling, Rotating and Noise 3D
# --------------------------------------------------------

def sample_points_3d(num_samples, rng=None):
    """
    Generate 3D sample points using Latin Hypercube Sampling (LHS).

    Parameters:
        num_samples (int): Number of sample points to generate.
        rng (np.random.Generator, optional): Random number generator for reproducibility.
                                             If None, a new default generator is used.

    Returns:
        np.ndarray: Array of shape (num_samples, 3) with 3D sample coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()
    return lhs(n=3, samples=num_samples)


def apply_rotation_3d(points, rotation_angles):
    """
    Apply a 3D rotation to a set of points using Euler angles (in degrees).

    Parameters:
        points (np.ndarray): Input array of shape (N, 3), representing 3D points.
        rotation_angles (tuple or list of float): Euler angles (x, y, z) in degrees.

    Returns:
        np.ndarray: Rotated 3D points with the same shape as input.
    """
    rotation = R.from_euler('xyz', rotation_angles, degrees=True)
    return rotation.apply(points)


def apply_label_noise_3d(X, y, distance_fn, label_noise=0.0, rng=None):
    """
    Apply label noise by flipping labels from 0 → 1 (outside → inside) near the polygon boundary.

    The closer a point labeled as 'outside' is to the boundary, the higher the probability it gets flipped to 'inside'.
    Points already labeled as 'inside' (label 1) are never flipped.

    Args:
        X (np.ndarray): Sample points of shape (N, 3).
        y (np.ndarray): Binary labels (0 = outside, 1 = inside), shape (N,).
        distance_fn (Callable): A function that returns signed distance to boundary (negative for the inside).
        label_noise (float): Controls the exponential decay of flip probability based on distance to boundary.
        rng (np.random.Generator, optional): Random number generator. If None, a new generator is created.

    Returns:
        np.ndarray: Modified label array (shape N,) where only 0s may be flipped to 1s based on label_noise value.
    """
    if rng is None:
        rng = np.random.default_rng()

    y_noisy = y.copy()
    for i, point in enumerate(X):
        if y[i] == 0:
            d = abs(distance_fn(point))
            flip_prob = np.exp(-d / label_noise) if label_noise > 0 else 0.0
            if rng.random() < flip_prob:
                y_noisy[i] = 1

    return y_noisy


# --------------------------------------------------------
# 2D Shape Generation Functions (Using Inverse Rotation for Boundary Evaluation)
# --------------------------------------------------------

def generate_2d_rectangle(num_samples=5000, center=(0.5, 0.5), ribs=(0.5, 0.5), rotation=45, samples=None,
                          label_noise=0.0, random_state=DEFAULT_VARIABLE_SEEDS[0]):
    """
    Generate 2D sample points for a rotatable rectangle with label noise.
    """
    rng = np.random.default_rng(random_state)
    samples = sample_points_2d(num_samples, rng)
    df_x = pd.DataFrame(samples, columns=['x1', 'x2'])

    center = np.array(center)
    rotation_rad = radians(rotation)
    half_width, half_height = ribs[0] / 2.0, ribs[1] / 2.0
    y = np.zeros(len(samples), dtype=int)
    rotated_samples = []

    for i, sample in enumerate(samples):
        p_transformed = apply_rotation_2d(sample - center, -rotation_rad)
        rotated_samples.append(p_transformed)
        if abs(p_transformed[0]) <= half_width and abs(p_transformed[1]) <= half_height:
            y[i] = 1

    rectangle = Polygon([
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height)
    ])

    y = apply_label_noise_2d(np.array(rotated_samples), y, rectangle, label_noise=label_noise, rng=rng)

    return df_x, y, samples


def generate_2d_radial_segment(num_samples=5000, center=(0.5, 0.5), outer_radius=0.4, inner_radius=0.2,
                               arc_span_degrees=300, rotation=90, samples=None, label_noise=0.0,
                               random_state=DEFAULT_VARIABLE_SEEDS[0]):
    """
    Generate 2D sample points for a radial segment (partial annulus) with label noise.
    """
    rng = np.random.default_rng(random_state)
    samples = sample_points_2d(num_samples, rng)
    df_x = pd.DataFrame(samples, columns=['x1', 'x2'])

    center = np.array(center)
    arc_span_rad = radians(arc_span_degrees)
    rotation_rad = radians(rotation)
    y = np.zeros(len(samples), dtype=int)

    def is_within_arc(angle, lower, upper):
        if lower < upper:
            return lower <= angle <= upper
        else:  # Arc crosses 0
            return angle >= lower or angle <= upper

    # Define arc bounds based on rotation
    mid_angle = rotation_rad % (2 * pi)
    lower_bound = (mid_angle - arc_span_rad / 2) % (2 * pi)
    upper_bound = (mid_angle + arc_span_rad / 2) % (2 * pi)

    for i, sample in enumerate(samples):
        rel = sample - center
        distance = np.linalg.norm(rel)
        angle = np.arctan2(rel[1], rel[0]) % (2 * pi)
        if inner_radius <= distance <= outer_radius and is_within_arc(angle, lower_bound, upper_bound):
            y[i] = 1

    # Build polygon to approximate the radial segment (arc wedge)
    if lower_bound < upper_bound:
        arc_outer = np.linspace(lower_bound, upper_bound, 100)
    else:
        arc_outer = np.linspace(lower_bound, upper_bound + 2 * pi, 100) % (2 * pi)

    arc_points = [center + outer_radius * np.array([np.cos(t), np.sin(t)]) for t in arc_outer]
    arc_inner = arc_outer[::-1]
    arc_points += [center + inner_radius * np.array([np.cos(t), np.sin(t)]) for t in arc_inner]
    arc_polygon = Polygon(arc_points)

    # Apply label noise (only outside→inside flips)
    y = apply_label_noise_2d(samples, y, arc_polygon, label_noise=label_noise, rng=rng)

    return df_x, y, samples


def generate_2d_barbell(num_samples=5000, center=(0.5, 0.5), barbell_length=0.6, sphere_radius=0.2,
                        connector_thickness=0.04, rotation=50, samples=None, label_noise=0.0,
                        random_state=DEFAULT_VARIABLE_SEEDS[0]):
    """
    Generate 2D sample points for a barbell shape.

    The barbell consists of two circles and a rectangular connector.
    The boundary evaluation applies an inverse rotation to the sample point.
    """
    rng = np.random.default_rng(random_state)
    samples = sample_points_2d(num_samples, rng)

    df_x = pd.DataFrame(samples, columns=['x1', 'x2'])
    center = np.array(center)
    rotation_rad = radians(rotation)

    # Define geometry in the unrotated frame.
    circle_A = np.array([-barbell_length / 2, 0])
    circle_B = np.array([barbell_length / 2, 0])
    rectangle = Polygon([
        (-barbell_length / 2, -connector_thickness),
        (barbell_length / 2, -connector_thickness),
        (barbell_length / 2, connector_thickness),
        (-barbell_length / 2, connector_thickness)
    ])

    y = np.zeros(len(samples), dtype=int)
    transformed_points = []
    for i, (x_val, y_val) in enumerate(samples):
        p_transformed = apply_rotation_2d(np.array([x_val, y_val]) - center, -rotation_rad)
        transformed_points.append(p_transformed)
        in_circle_A = np.linalg.norm(p_transformed - circle_A) <= sphere_radius
        in_circle_B = np.linalg.norm(p_transformed - circle_B) <= sphere_radius
        in_rectangle = rectangle.contains(Point(p_transformed))
        if in_circle_A or in_circle_B or in_rectangle:
            y[i] = 1

    # Combine all regions into one polygon approximation
    barbell_union = rectangle.buffer(0)
    barbell_union = barbell_union.union(Point(circle_A).buffer(sphere_radius))
    barbell_union = barbell_union.union(Point(circle_B).buffer(sphere_radius))

    y = apply_label_noise_2d(np.array(transformed_points), y, shape_polygon=barbell_union,
                             label_noise=label_noise, rng=rng)

    return df_x, y, samples


def generate_2d_sine_wave(num_samples=5000, x_range=(0.1, 0.9), vertical_offset=0.5, amplitude=0.2, frequency=0.5,
                          thickness=0.10, rotation=0, samples=None, label_noise=0.0,
                          random_state=DEFAULT_VARIABLE_SEEDS[0]):
    """
    Generate 2D sample points for a sine wave.

    The sine function's boundary is transformed using an inverse rotation for evaluation.
    """
    rng = np.random.default_rng(random_state)
    samples = sample_points_2d(num_samples, rng)

    df_x = pd.DataFrame(samples, columns=['x1', 'x2'])
    x_min, x_max = x_range
    rotation_rad = radians(rotation)
    center = [(x_max + x_min) / 2, vertical_offset]  # Center for the sine function
    y = np.zeros(len(samples), dtype=int)

    transformed_points = []
    shape_coords = []

    for i, (x_val, y_val) in enumerate(samples):
        p_transformed = apply_rotation_2d(np.array([x_val, y_val]) - center, -rotation_rad) + center
        transformed_points.append(p_transformed)
        x_rot, y_rot = p_transformed
        if x_min <= x_rot <= x_max:
            f_val = vertical_offset + amplitude * np.sin(2 * np.pi * frequency * (x_rot - x_min) / (x_max - x_min))
            if abs(y_rot - f_val) < thickness:
                y[i] = 1

    # Build polygon approximation of a sine-band
    x_vals = np.linspace(x_min, x_max, 300)
    y_top = vertical_offset + amplitude * np.sin(2 * np.pi * frequency * (x_vals - x_min) / (x_max - x_min)) + thickness
    y_bot = vertical_offset + amplitude * np.sin(2 * np.pi * frequency * (x_vals - x_min) / (x_max - x_min)) - thickness
    upper = np.stack([x_vals, y_top], axis=1)
    lower = np.stack([x_vals[::-1], y_bot[::-1]], axis=1)
    band = np.concatenate([upper, lower], axis=0)
    shape_polygon = Polygon(band)

    y = apply_label_noise_2d(np.array(transformed_points), y, shape_polygon=shape_polygon,
                             label_noise=label_noise, rng=rng)

    return df_x, y, samples


def generate_2d_star(num_samples=5000, center=(0.5, 0.5), num_points=5, star_size=1.0,
                     outer_radius=0.4, inner_radius=0.2, rotation=0, samples=None, label_noise=0.0,
                     random_state=DEFAULT_VARIABLE_SEEDS[0]):
    """
    Generate 2D sample points for a star shape.

    The star is defined by vertices calculated with alternating radii.
    An inverse rotation is applied to the vertex coordinates before classification.
    """
    rng = np.random.default_rng(random_state)
    samples = sample_points_2d(num_samples, rng)

    df_x = pd.DataFrame(samples, columns=['x1', 'x2'])

    # Generate star polygon points
    angles = np.linspace(0, 2 * pi, num_points * 2, endpoint=False)
    radii = np.array([outer_radius if i % 2 == 0 else inner_radius for i in range(len(angles))])
    star_x1 = radii * np.cos(angles) * star_size
    star_x2 = radii * np.sin(angles) * star_size
    star_x1 = np.append(star_x1, star_x1[0])
    star_x2 = np.append(star_x2, star_x2[0])

    rotation_rad = radians(rotation)
    rotated_coords = [apply_rotation_2d(np.array([x, y]), -rotation_rad) for x, y in zip(star_x1, star_x2)]
    rotated_star_x1, rotated_star_x2 = zip(*rotated_coords)
    final_star_x1 = np.array(rotated_star_x1) + center[0]
    final_star_x2 = np.array(rotated_star_x2) + center[1]

    # Create the shape polygon
    star_polygon = Polygon(zip(final_star_x1, final_star_x2))

    # Classify points
    y = np.array([1 if star_polygon.contains(Point(x, y)) else 0 for x, y in samples], dtype=int)

    # Apply label_noise
    y = apply_label_noise_2d(samples, y, shape_polygon=star_polygon, label_noise=label_noise, rng=rng)

    return df_x, y, samples


# --------------------------------------------------------
# 3D Shape Generation Functions (Using Inverse Rotation for Boundary Evaluation)
# --------------------------------------------------------

def generate_3d_radial_segment(num_samples=10000, center=(0.5, 0.5, 0.5), outer_radius=0.4, inner_radius=0.2,
                               arc_span_degrees=300, rotation_x1=35, rotation_x2=0, rotation_x3=60, samples=None,
                               label_noise=0.0, random_state=DEFAULT_VARIABLE_SEEDS[0]):
    rng = np.random.default_rng(random_state)
    samples = sample_points_3d(num_samples, rng)

    df_x = pd.DataFrame(samples, columns=['x1', 'x2', 'x3'])
    center = np.array(center)
    arc_span_rad = radians(arc_span_degrees)
    R_val = outer_radius
    r_val = outer_radius - inner_radius
    inv_rot_angles = (-rotation_x1, -rotation_x2, -rotation_x3)

    def distance_fn(point):
        rel_point = point - center
        p_transformed = apply_rotation_3d(np.array([rel_point]), inv_rot_angles)[0]
        distance_xy = np.sqrt(p_transformed[0]**2 + p_transformed[1]**2)
        angle = np.arctan2(p_transformed[1], p_transformed[0])
        angle = angle + 2 * pi if angle < 0 else angle
        d_torus = np.sqrt((distance_xy - R_val) ** 2 + p_transformed[2] ** 2) - r_val
        if arc_span_degrees not in (0, 360):
            lower_bound = pi - arc_span_rad / 2
            upper_bound = pi + arc_span_rad / 2
            in_arc = lower_bound <= angle <= upper_bound
            if not in_arc:
                return float('inf')  # Considered outside
        return d_torus

    y = np.array([1 if distance_fn(p) <= 0 else 0 for p in samples], dtype=int)

    y = apply_label_noise_3d(samples, y, distance_fn=distance_fn, label_noise=label_noise, rng=rng)

    return df_x, y, samples


def generate_3d_barbell(num_samples=10000, center=(0.5, 0.5, 0.5), barbell_length=0.8, sphere_radius=0.25,
                        connector_thickness=0.1, rotation_angle_x1=50, rotation_angle_x2=50, rotation_angle_x3=0,
                        samples=None, label_noise=0.0, random_state=DEFAULT_VARIABLE_SEEDS[0]):
    rng = np.random.default_rng(random_state)
    samples = sample_points_3d(num_samples, rng)

    df_x = pd.DataFrame(samples, columns=['x1', 'x2', 'x3'])
    center = np.array(center)
    sphere_center_A = np.array([-barbell_length / 2, 0, 0])
    sphere_center_B = np.array([barbell_length / 2, 0, 0])
    inv_rot_angles = (-rotation_angle_x1, -rotation_angle_x2, -rotation_angle_x3)

    def distance_fn(point):
        rel_point = point - center
        p_transformed = apply_rotation_3d(np.array([rel_point]), inv_rot_angles)[0]
        d_sphere_A = np.linalg.norm(p_transformed - sphere_center_A) - sphere_radius
        d_sphere_B = np.linalg.norm(p_transformed - sphere_center_B) - sphere_radius
        in_cylinder_x = np.clip(p_transformed[0], -barbell_length/2, barbell_length/2)
        radial_d = np.sqrt(p_transformed[1]**2 + p_transformed[2]**2)
        d_cylinder = max(
            float(abs(p_transformed[0]) - barbell_length / 2),
            float(radial_d - connector_thickness)
        )
        return min(d_sphere_A, d_sphere_B, d_cylinder)

    y = np.array([1 if distance_fn(p) <= 0 else 0 for p in samples], dtype=int)

    y = apply_label_noise_3d(samples, y, distance_fn=distance_fn, label_noise=label_noise, rng=rng)

    return df_x, y, samples


def generate_3d_saddle(num_samples=10000, center=(0.5, 0.5, 0.5), saddle_height=0.5,
                       curve_sharpness_x1=1.0, curve_sharpness_x2=1.0, surface_thickness=0.2,
                       rotate_x1_deg=0, rotate_x2_deg=0, rotate_x3_deg=0, samples=None,
                       label_noise=0.0, random_state=DEFAULT_VARIABLE_SEEDS[0]):
    """
    Generate 3D sample points for a saddle shape with label noise.
    The saddle surface is defined as a quadratic surface and rotated.
    """
    rng = np.random.default_rng(random_state)
    samples = sample_points_3d(num_samples, rng)

    df_x = pd.DataFrame(samples, columns=['x1', 'x2', 'x3'])
    center = np.array(center)
    rotation = R.from_euler('xyz', [-rotate_x1_deg, -rotate_x2_deg, -rotate_x3_deg], degrees=True)

    def distance_fn(point):
        rel = point - center
        inv_rot = rotation.apply([rel])[0] + center
        x, y, z = inv_rot
        saddle_z = (curve_sharpness_x1 * (x - center[0]) ** 2 -
                    curve_sharpness_x2 * (y - center[1]) ** 2)
        saddle_z = ((saddle_z - saddle_z_minmax[0]) / (saddle_z_minmax[1] - saddle_z_minmax[0]) *
                    saddle_range + saddle_min)
        return abs(z - saddle_z) - surface_thickness

    # Precompute normalized saddle Z-range
    rel_points = samples - center
    inv_rotated_points = rotation.apply(rel_points) + center
    raw_saddle_z = (curve_sharpness_x1 * ((inv_rotated_points[:, 0] - center[0]) ** 2) -
                    curve_sharpness_x2 * ((inv_rotated_points[:, 1] - center[1]) ** 2))
    saddle_min = center[2] - saddle_height / 2
    saddle_max = center[2] + saddle_height / 2
    saddle_z_minmax = (np.min(raw_saddle_z), np.max(raw_saddle_z))
    saddle_range = saddle_max - saddle_min

    y = np.array([1 if distance_fn(p) <= 0 else 0 for p in samples], dtype=int)

    y = apply_label_noise_3d(samples, y, distance_fn, label_noise=label_noise, rng=rng)

    return df_x, y, samples
