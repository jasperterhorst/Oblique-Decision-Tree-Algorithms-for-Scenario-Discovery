"""
utils.py

This module provides utility functions for plotting and saving shapes, and saving generated _data.
"""

import os
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
from src.config.colors import (
    PRIMARY_LIGHT, PRIMARY_DARK,
    AXIS_LINE_COLOR, GRID_COLOR
)


# --------------------------------------------------------
# Helper Function: Automatic Text Wrapping for Notes
# --------------------------------------------------------

# def format_note_text(note, dimension="2D"):
#     max_length = 65 if dimension == "2D" else 60
#     words = note.split()
#     lines = []
#     current_line = ""
#     for word in words:
#         if len(current_line) + len(word) + 1 <= max_length:
#             current_line += " " + word if current_line else word
#         else:
#             lines.append(current_line)
#             current_line = word
#     if current_line:
#         lines.append(current_line)
#     return "\n".join(lines)


def format_note_text(note, dimension="2D"):
    max_length = 65 if dimension == "2D" else 60
    words = note.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_length:
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
        if len(lines) == 3:
            break  # Stop at 3 lines
    if current_line and len(lines) < 3:
        lines.append(current_line)

    # Pad with empty lines if less than 3
    while len(lines) < 3:
        lines.append(" ")

    return "\n".join(lines)


# --------------------------------------------------------
# 2D Plotting Function
# --------------------------------------------------------
def plot_2d_shape(samples, y, title="2D Shape", save_path=None, note=""):
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(samples[y == 0, 0], samples[y == 0, 1], c=PRIMARY_LIGHT,
               label='Not of Interest', alpha=0.2)
    ax.scatter(samples[y == 1, 0], samples[y == 1, 1], c=PRIMARY_DARK,
               label='Of Interest', alpha=0.6)

    ax.set_title(title, fontsize=24, pad=10)
    ax.set_xlabel('X1', fontsize=20)
    ax.set_ylabel('X2', fontsize=20)
    ax.legend(loc='upper right', fontsize=15)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(AXIS_LINE_COLOR)
    ax.spines['bottom'].set_color(AXIS_LINE_COLOR)
    ax.tick_params(axis='both', colors=AXIS_LINE_COLOR)

    if note:
        formatted_note = format_note_text(note, dimension="2D")
        fig.text(0.5, -0.01, formatted_note, va='top', ha='center',
                 fontsize=14, color=AXIS_LINE_COLOR)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    display(fig)
    plt.close(fig)


# --------------------------------------------------------
# 3D Plotting Function
# --------------------------------------------------------

def plot_3d_shape(samples, y, title="3D Shape", save_path=None, note=""):
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(samples[y == 0, 0], samples[y == 0, 1], samples[y == 0, 2],
               c=PRIMARY_LIGHT, label='Not of Interest', alpha=0.1, s=10)
    ax.scatter(samples[y == 1, 0], samples[y == 1, 1], samples[y == 1, 2],
               c=PRIMARY_DARK, label='Of Interest', alpha=1.0, s=10)

    ax.set_title(title, fontsize=22, pad=10)
    ax.set_xlabel('X1', fontsize=18, labelpad=8)
    ax.set_ylabel('X2', fontsize=18, labelpad=8)
    ax.set_zlabel('X3', fontsize=18, labelpad=8)
    ax.legend(loc='upper right', fontsize=14)

    ax.xaxis.set_tick_params(labelsize=14, colors=AXIS_LINE_COLOR)
    ax.yaxis.set_tick_params(labelsize=14, colors=AXIS_LINE_COLOR)
    ax.zaxis.set_tick_params(labelsize=14, colors=AXIS_LINE_COLOR)

    ax.xaxis.line.set_color(AXIS_LINE_COLOR)
    ax.yaxis.line.set_color(AXIS_LINE_COLOR)
    ax.zaxis.line.set_color(AXIS_LINE_COLOR)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    ax.set_box_aspect([1, 1, 1], zoom=0.89)

    ax.grid(True, linestyle='--', linewidth=0.5, color=GRID_COLOR)

    if note:
        formatted_note = format_note_text(note, dimension="3D")
        fig.text(0.5, 0.08, formatted_note, va='top', ha='center',
                 fontsize=14, color=AXIS_LINE_COLOR)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    display(fig)
    plt.close(fig)


# --------------------------------------------------------
# Data Saving Function
# --------------------------------------------------------

def save_data(df_x, y, file_prefix, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Save the individual _data files for x and y
    df_x.to_csv(os.path.join(save_dir, f"{file_prefix}_x.csv"), index=False)
    pd.DataFrame({"y": y}).to_csv(os.path.join(save_dir, f"{file_prefix}_y.csv"), index=False)

    # Create and save a combined DataFrame with both x and y
    df_combined = df_x.copy()
    df_combined["y"] = y
    df_combined.to_csv(os.path.join(save_dir, f"{file_prefix}_xy_combined.csv"), index=False)

    print(f"Data saved to: {save_dir}")
