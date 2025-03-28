"""
shape_generators.py

This module generates sample points for various 2D and 3D shapes using Latin Hypercube Sampling (LHS).
It provides functions for:
  - 2D Radial Segment
  - 2D Barbell
  - 2D Sine Wave
  - 2D Star
  - 3D Saddle
  - 3D Radial Segment
  - 3D Barbell

Each function returns:
  - df_x: A pandas DataFrame with sample coordinates.
  - y: A binary classification array (1 = inside the shape, 0 = outside).
  - samples: The raw sample points as a NumPy array.

All functions use:
  - LHS for 2D and 3D sampling.
  - Rotation functions for consistent transformations.

"""

import numpy as np
import pandas as pd
from pyDOE2 import lhs
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point, Polygon
from math import radians, pi


# --------------------------------------------------------
# Helper Functions - Sampling and Rotating 2D
# --------------------------------------------------------


def sample_points_2d(num_samples):
    """Generate 2D sample points using Latin Hypercube Sampling (LHS)."""
    return lhs(n=2, samples=num_samples)


def apply_rotation_2d(point, angle_rad):
    """Apply a 2D rotation to a point using a given angle (in radians)."""
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return rot_matrix @ point


# --------------------------------------------------------
# Helper Functions - Sampling and Rotating 3D
# --------------------------------------------------------

def sample_points_3d(num_samples):
    """Generate 3D sample points using Latin Hypercube Sampling (LHS)."""
    return lhs(n=3, samples=num_samples)


def apply_rotation_3d(points, rotation_angles):
    """Apply 3D rotation to a set of points using Euler angles (in degrees)."""
    rotation = R.from_euler('xyz', rotation_angles, degrees=True)
    return rotation.apply(points)


# --------------------------------------------------------
# 2D Shape Generation Functions (Using Inverse Rotation for Boundary Evaluation)
# --------------------------------------------------------


def generate_2d_rectangle(num_samples=1000, center=(0.5, 0.5), ribs=(0.4, 0.2), rotation=0, samples=None):
    """
    Generate 2D sample points for a rotatable rectangle.

    Parameters:
        num_samples (int): Number of sample points.
        center (tuple): The (x, y) coordinates of the rectangle's center.
        ribs (tuple): A tuple (width, height) representing the rectangle dimensions.
        rotation (float): Rotation angle in degrees.
        samples (np.array, optional): Precomputed sample points.

    Returns:
        df_x (pd.DataFrame): DataFrame with sample points.
        y (np.array): Binary array (1 if point is inside the rectangle, 0 otherwise).
        samples (np.array): The raw sample points.
    """
    if samples is None:
        samples = sample_points_2d(num_samples)
    df_x = pd.DataFrame(samples, columns=['x1', 'x2'])
    center = np.array(center)
    rotation_rad = radians(rotation)
    half_width = ribs[0] / 2.0
    half_height = ribs[1] / 2.0
    y = np.zeros(len(samples), dtype=int)

    for i, sample in enumerate(samples):
        # Subtract center and apply inverse rotation
        p_transformed = apply_rotation_2d(sample - center, -rotation_rad)
        # Check if the transformed point lies within the axis-aligned rectangle
        if abs(p_transformed[0]) <= half_width and abs(p_transformed[1]) <= half_height:
            y[i] = 1

    return df_x, y, samples


def generate_2d_radial_segment(num_samples=1000, center=(0.5, 0.5), outer_radius=0.4, inner_radius=0.2,
                               arc_span_degrees=180, rotation=0, samples=None):
    """
    Generate 2D sample points for a radial segment (partial annulus).

    The sample points remain fixed in the unit square.
    For each point, an inverse rotation is applied for boundary evaluation.
    """
    if samples is None:
        samples = sample_points_2d(num_samples)
    df_x = pd.DataFrame(samples, columns=['x1', 'x2'])
    center = np.array(center)
    arc_span_rad = radians(arc_span_degrees)
    rotation_rad = radians(rotation)
    y = np.zeros(len(samples), dtype=int)

    for i, sample in enumerate(samples):
        # Transform sample for evaluation: subtract centre and apply inverse rotation.
        p_transformed = apply_rotation_2d(sample - center, -rotation_rad)
        distance = np.linalg.norm(p_transformed)
        angle = np.arctan2(p_transformed[1], p_transformed[0])
        if angle < 0:
            angle += 2 * pi
        lower_bound, upper_bound = -arc_span_rad / 2, arc_span_rad / 2
        in_arc = (lower_bound <= angle <= upper_bound) or ((lower_bound + 2 * pi) <= angle <= (upper_bound + 2 * pi))
        if inner_radius <= distance <= outer_radius and in_arc:
            y[i] = 1

    return df_x, y, samples


def generate_2d_barbell(num_samples=1000, center=(0.5, 0.5), barbell_length=0.3, sphere_radius=0.15,
                        connector_thickness=0.07, rotation=0, samples=None):
    """
    Generate 2D sample points for a barbell shape.

    The barbell consists of two circles and a rectangular connector.
    The boundary evaluation applies an inverse rotation to the sample point.
    """
    if samples is None:
        samples = sample_points_2d(num_samples)
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
    for i, (x_val, y_val) in enumerate(samples):
        p_transformed = apply_rotation_2d(np.array([x_val, y_val]) - center, -rotation_rad)
        in_circle_A = np.linalg.norm(p_transformed - circle_A) <= sphere_radius
        in_circle_B = np.linalg.norm(p_transformed - circle_B) <= sphere_radius
        in_rectangle = rectangle.contains(Point(p_transformed))
        if in_circle_A or in_circle_B or in_rectangle:
            y[i] = 1

    return df_x, y, samples


def generate_2d_sine_wave(num_samples=1000, x_range=(0.2, 0.8), vertical_offset=0.5, amplitude=0.1, frequency=1,
                          thickness=0.05, rotation=0, samples=None):
    """
    Generate 2D sample points for a sine wave.

    The sine function's boundary is transformed using an inverse rotation for evaluation.
    """
    if samples is None:
        samples = sample_points_2d(num_samples)
    df_x = pd.DataFrame(samples, columns=['x1', 'x2'])
    x_min, x_max = x_range
    rotation_rad = radians(rotation)
    center = [(x_max + x_min) / 2, vertical_offset]  # Center for the sine function
    y = np.zeros(len(samples), dtype=int)

    for i, (x_val, y_val) in enumerate(samples):
        p_transformed = apply_rotation_2d(np.array([x_val, y_val]) - center, -rotation_rad) + center
        x_rot, y_rot = p_transformed
        if x_min <= x_rot <= x_max:
            f_val = vertical_offset + amplitude * np.sin(2 * np.pi * frequency * (x_rot - x_min) / (x_max - x_min))
            if abs(y_rot - f_val) < thickness:
                y[i] = 1
    return df_x, y, samples


def generate_2d_star(num_samples=2000, center=(0.5, 0.5),
                     num_points=5, star_size=0.8,
                     outer_radius=0.4, inner_radius=0.2,
                     rotation=0, samples=None):
    """
    Generate 2D sample points for a star shape.

    The star is defined by vertices calculated with alternating radii.
    An inverse rotation is applied to the vertex coordinates before classification.
    """
    if samples is None:
        samples = sample_points_2d(num_samples)
    df_x = pd.DataFrame(samples, columns=['x1', 'x2'])

    angles = np.linspace(0, 2 * pi, num_points * 2, endpoint=False)
    radii = np.array([outer_radius if i % 2 == 0 else inner_radius for i in range(len(angles))])
    star_x = radii * np.cos(angles) * star_size
    star_y = radii * np.sin(angles) * star_size
    star_x = np.append(star_x, star_x[0])
    star_y = np.append(star_y, star_y[0])

    rotation_rad = radians(rotation)
    rotated_coords = [apply_rotation_2d(np.array([x, y]), -rotation_rad) for x, y in zip(star_x, star_y)]
    rotated_star_x, rotated_star_y = zip(*rotated_coords)
    final_star_x = np.array(rotated_star_x) + center[0]
    final_star_y = np.array(rotated_star_y) + center[1]

    star_polygon = Polygon(zip(final_star_x, final_star_y))
    y = np.array([1 if star_polygon.contains(Point(x, y)) else 0 for x, y in samples], dtype=int)
    return df_x, y, samples


# --------------------------------------------------------
# 3D Shape Generation Functions (Using Inverse Rotation for Boundary Evaluation)
# --------------------------------------------------------

def generate_3d_radial_segment(num_samples=2000, center=(0.5, 0.5, 0.5),
                               outer_radius=0.5, inner_radius=0.2,
                               arc_span_degrees=180,
                               rotation_x=0, rotation_y=0, rotation_z=0, samples=None):
    """
    Generate 3D sample points for a radial segment (partial torus).

    Instead of rotating the sample points, an inverse rotation (via Euler angles) is applied for boundary evaluation.
    """
    if samples is None:
        samples = sample_points_3d(num_samples)
    df_x = pd.DataFrame(samples, columns=['x1', 'x2', 'x3'])
    center = np.array(center)
    arc_span_rad = radians(arc_span_degrees)
    R_val = outer_radius
    r_val = outer_radius - inner_radius
    y = np.zeros(len(samples), dtype=int)
    inv_rot_angles = (-rotation_x, -rotation_y, -rotation_z)

    for i, sample in enumerate(samples):
        rel_point = sample - center
        p_transformed = apply_rotation_3d(np.array([rel_point]), inv_rot_angles)[0]
        distance_xy = np.sqrt(p_transformed[0] ** 2 + p_transformed[1] ** 2)
        inside_torus = (distance_xy - R_val) ** 2 + p_transformed[2] ** 2 <= r_val ** 2
        angle = np.arctan2(p_transformed[1], p_transformed[0])
        if angle < 0:
            angle += 2 * pi
        if arc_span_degrees == 360:
            in_arc = True
        elif arc_span_degrees == 0:
            in_arc = False
        else:
            lower_bound = pi - arc_span_rad / 2
            upper_bound = pi + arc_span_rad / 2
            in_arc = lower_bound <= angle <= upper_bound
        if inside_torus and in_arc:
            y[i] = 1

    return df_x, y, samples


def generate_3d_barbell(num_samples=2000, center=(0.5, 0.5, 0.5),
                        barbell_length=0.3, sphere_radius=0.15,
                        connector_thickness=0.07,
                        rotation_angle_x=0, rotation_angle_y=0, rotation_angle_z=0, samples=None):
    """
    Generate 3D sample points for a barbell shape.

    The 3D barbell is composed of two spheres connected by a cylinder.
    An inverse rotation (using Euler angles) is applied for boundary evaluation.
    """
    if samples is None:
        samples = sample_points_3d(num_samples)
    df_x = pd.DataFrame(samples, columns=['x1', 'x2', 'x3'])
    center = np.array(center)
    sphere_center_A = np.array([-barbell_length / 2, 0, 0])
    sphere_center_B = np.array([barbell_length / 2, 0, 0])
    inv_rot_angles = (-rotation_angle_x, -rotation_angle_y, -rotation_angle_z)

    def is_inside_barbell(point):
        rel_point = point - center
        p_transformed = apply_rotation_3d(np.array([rel_point]), inv_rot_angles)[0]
        in_sphere_A = np.linalg.norm(p_transformed - sphere_center_A) <= sphere_radius
        in_sphere_B = np.linalg.norm(p_transformed - sphere_center_B) <= sphere_radius
        in_cylinder = (-barbell_length / 2 <= p_transformed[0] <= barbell_length / 2) and \
                      (np.sqrt(p_transformed[1] ** 2 + p_transformed[2] ** 2) <= connector_thickness)
        return in_sphere_A or in_sphere_B or in_cylinder

    y = np.array([1 if is_inside_barbell(point) else 0 for point in samples], dtype=int)
    return df_x, y, samples


def generate_3d_saddle(num_samples=2000, center=(0.5, 0.5, 0.5),
                       saddle_height=0.5, curve_sharpness_x=1.0, curve_sharpness_y=1.0,
                       surface_thickness=0.05, rotate_x_deg=0, rotate_y_deg=0, rotate_z_deg=0, samples=None):
    """
    Generate 3D sample points for a saddle shape.

    The sample points remain fixed in the unit cube, and the saddle's boundary is evaluated by applying
    an inverse rotation (via Euler angles) to each point.
    The saddle surface is defined by:
      z_saddle = c_x (x - x_c)^2 - c_y (y - y_c)^2,
    and normalized to span a total height.
    """
    if samples is None:
        samples = sample_points_3d(num_samples)
    rel_points = samples - np.array(center)
    rotation = R.from_euler('xyz', [-rotate_x_deg, -rotate_y_deg, -rotate_z_deg], degrees=True)
    inv_rotated_points = rotation.apply(rel_points) + np.array(center)
    saddle_z = (curve_sharpness_x * ((inv_rotated_points[:, 0] - center[0]) ** 2) -
                curve_sharpness_y * ((inv_rotated_points[:, 1] - center[1]) ** 2))
    min_z = center[2] - saddle_height / 2
    max_z = center[2] + saddle_height / 2
    saddle_z = (saddle_z - np.min(saddle_z)) / (np.max(saddle_z) - np.min(saddle_z)) * (max_z - min_z) + min_z

    y = np.array([1 if abs(samples[i, 2] - saddle_z[i]) < surface_thickness else 0 for i in range(len(samples))],
                 dtype=int)
    df_x = pd.DataFrame(samples, columns=['x1', 'x2', 'x3'])
    return df_x, y, samples
