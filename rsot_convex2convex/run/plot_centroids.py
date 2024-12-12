#!/usr/bin/env python3
"""
Visualize cell centroids from Power Diagram computation.

This script reads centroids from a space-separated file and creates a 3D scatter plot
using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional

def load_centroids(filename: str) -> np.ndarray:
    """
    Load centroids from a space-separated file.

    Args:
        filename (str): Path to the file containing centroids

    Returns:
        np.ndarray: Array of shape (n_points, 3) containing the centroids

    Raises:
        ValueError: If the data doesn't have exactly 3 columns
        FileNotFoundError: If the specified file doesn't exist
    """
    try:
        data = np.loadtxt(filename)
        if data.shape[1] != 3:
            raise ValueError(f"Expected 3 coordinates per point, got {data.shape[1]}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find centroid file: {filename}")

def plot_centroids(centroids: np.ndarray,
                  fig_size: Tuple[int, int] = (10, 10),
                  marker_size: int = 20,
                  save_path: Optional[str] = None) -> None:
    """
    Create a 3D scatter plot of the centroids.

    Args:
        centroids (np.ndarray): Array of shape (n_points, 3) containing the centroids
        fig_size (Tuple[int, int]): Figure size in inches
        marker_size (int): Size of the scatter plot markers
        save_path (Optional[str]): If provided, save the plot to this path
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot
    scatter = ax.scatter(centroids[:, 0],
                        centroids[:, 1],
                        centroids[:, 2],
                        c=centroids[:, 2],  # Color by z-coordinate
                        cmap='viridis',
                        s=marker_size)

    # Add colorbar
    plt.colorbar(scatter)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Power Diagram Cell Centroids')

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def main():
    """Main function to load and plot centroids."""
    # Configuration
    outputdir = "output/power_diagram/"
    input_file = outputdir + "centroids.txt"
    output_file = "centroids_plot.png"

    try:
        # Load the centroids
        centroids = load_centroids(input_file)

        # Create the plot
        plot_centroids(centroids,
                      fig_size=(12, 12),
                      marker_size=30,
                      save_path=output_file)

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
