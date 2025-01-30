"""
Visualize and compare cell centroids from Power Diagram computations.

This script reads centroids from both Deal.II and Geogram implementations and creates
comparative 3D scatter plots using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List
import os
import glob

def list_epsilon_folders(base_dir: str = "output") -> List[str]:
    """
    List all epsilon and exact_sot folders in the base directory.

    Args:
        base_dir (str): Base directory to search in

    Returns:
        List[str]: List of folder names
    """
    folders = []

    # Find all epsilon folders and exact_sot
    for entry in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, entry)):
            if entry.startswith("epsilon_") or entry == "exact_sot":
                folders.append(entry)

    if not folders:
        raise FileNotFoundError(f"No epsilon or exact_sot folders found in {base_dir}")

    # Sort folders for consistent ordering
    folders.sort()
    return folders

def select_folder(folders: List[str], allow_all: bool = True) -> List[str]:
    """
    Let user select a folder from the list.

    Args:
        folders (List[str]): List of available folders
        allow_all (bool): Whether to allow selecting all folders

    Returns:
        List[str]: Selected folder(s)
    """
    print("\nAvailable folders:")
    for i, folder in enumerate(folders, 1):
        print(f"[{i}] {folder}")
    if allow_all:
        print(f"[{len(folders) + 1}] ALL FOLDERS")

    while True:
        try:
            selection = int(input(f"\nSelect a folder (1-{len(folders) + 1 if allow_all else len(folders)}): "))
            if 1 <= selection <= (len(folders) + 1 if allow_all else len(folders)):
                break
        except ValueError:
            pass
        print("Invalid selection. Please try again.")

    if allow_all and selection == len(folders) + 1:
        return folders
    return [folders[selection - 1]]

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

def plot_centroids_comparison(dealii_centroids: np.ndarray,
                            geogram_centroids: np.ndarray,
                            title: str,
                            fig_size: Tuple[int, int] = (10, 10),
                            marker_size: int = 20,
                            save_path: Optional[str] = None) -> None:
    """
    Create three 3D scatter plots: Deal.II, Geogram, and comparison.

    Args:
        dealii_centroids (np.ndarray): Array of Deal.II centroids
        geogram_centroids (np.ndarray): Array of Geogram centroids
        title (str): Base title for the plots
        fig_size (Tuple[int, int]): Figure size in inches
        marker_size (int): Size of the scatter plot markers
        save_path (Optional[str]): If provided, save the plots to this path
    """
    # Create figure with three subplots
    fig = plt.figure(figsize=(fig_size[0] * 3, fig_size[1]))

    # Deal.II plot
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(dealii_centroids[:, 0],
                          dealii_centroids[:, 1],
                          dealii_centroids[:, 2],
                          c='blue',
                          label='Deal.II',
                          s=marker_size)
    ax1.set_title(f'{title}\nDeal.II Centroids')
    ax1.legend()
    ax1.set_box_aspect([1, 1, 1])

    # Geogram plot
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(geogram_centroids[:, 0],
                          geogram_centroids[:, 1],
                          geogram_centroids[:, 2],
                          c='red',
                          label='Geogram',
                          s=marker_size)
    ax2.set_title(f'{title}\nGeogram Centroids')
    ax2.legend()
    ax2.set_box_aspect([1, 1, 1])

    # Comparison plot
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3_1 = ax3.scatter(dealii_centroids[:, 0],
                            dealii_centroids[:, 1],
                            dealii_centroids[:, 2],
                            c='blue',
                            label='Deal.II',
                            s=marker_size,
                            alpha=0.6)
    scatter3_2 = ax3.scatter(geogram_centroids[:, 0],
                            geogram_centroids[:, 1],
                            geogram_centroids[:, 2],
                            c='red',
                            label='Geogram',
                            s=marker_size,
                            alpha=0.6)
    ax3.set_title(f'{title}\nComparison')
    ax3.legend()
    ax3.set_box_aspect([1, 1, 1])

    # Set labels for all plots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def main():
    """Main function to load and plot centroids."""
    try:
        # Get available folders
        folders = list_epsilon_folders()
        selected_folders = select_folder(folders)

        for folder in selected_folders:
            # Construct paths
            dealii_file = f"output/{folder}/power_diagram_dealii/centroids.txt"
            geogram_file = f"output/{folder}/power_diagram_geogram/centroids.txt"

            # Load the centroids
            dealii_centroids = load_centroids(dealii_file)
            geogram_centroids = load_centroids(geogram_file)

            # Create the plots
            output_file = f"centroids_comparison_{folder}.png"
            plot_centroids_comparison(
                dealii_centroids,
                geogram_centroids,
                title=folder,
                fig_size=(6, 6),
                marker_size=30,
                save_path=output_file
            )

            print(f"Plots saved to: {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
