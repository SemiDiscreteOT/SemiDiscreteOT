import numpy as np
import os
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class PointCloudHierarchyManager:
    def __init__(self, min_points=200, max_points=5000):
        self.min_points = min_points
        self.max_points = max_points
        self.num_levels = 0
        self.level_point_counts = []
        self.parent_indices = []
        self.child_indices = []

    def set_max_points(self, max_points):
        self.max_points = max_points

    def set_min_points(self, min_points):
        self.min_points = min_points

    def get_num_levels(self):
        return self.num_levels

    def get_point_count(self, level):
        if level < 0 or level >= len(self.level_point_counts):
            raise IndexError("Level index out of range")
        return self.level_point_counts[level]

    def get_parent_indices(self, level):
        if level <= 0 or level >= self.num_levels:
            raise IndexError("Level index out of range for parent indices")
        return self.parent_indices[level-1]

    def get_child_indices(self, level):
        if level < 0 or level >= self.num_levels-1:
            raise IndexError("Level index out of range for child indices")
        return self.child_indices[level]

    def ensure_directory_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_points_for_level(self, base_points, level):
        # Level 0 always uses original point cloud size
        if level == 0:
            return base_points
        # Level 1 starts with max_points (if smaller than base_points)
        if level == 1:
            return min(base_points, self.max_points)
        # For subsequent levels, use a reduction factor of 4 based on level 1's size
        level1_points = min(base_points, self.max_points)
        points = int(level1_points / (4.0 ** (level - 1)))
        return max(points, self.min_points)

    def kmeans_clustering(self, points, density, k):
        n_points = len(points)
        print(f"Clustering {n_points} points into {k} clusters")

        # If fewer points than clusters, return original points
        if n_points <= k:
            assignments = list(range(n_points))
            return points, density, assignments

        # Start timing
        start_time = time.time()

        # Run exact K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        assignments = kmeans.fit_predict(points)
        centers = kmeans.cluster_centers_

        # Stop timing
        end_time = time.time()
        duration = end_time - start_time

        print(f"  K-means clustering took {duration:.2f} seconds")

        # Compute density for each cluster
        cluster_density = np.zeros(k)
        for i in range(len(points)):
            cluster = assignments[i]
            point_density = 1.0 if len(density) == 0 else density[i]
            cluster_density[cluster] += point_density

        return centers, cluster_density, assignments

    def generate_hierarchy(self, input_points, input_density, output_dir):
        # Validate inputs
        if len(input_points) == 0:
            raise ValueError("Input point cloud is empty")

        # Start timing the entire hierarchy generation
        start_total = time.time()

        # Create uniform density if none provided
        density = []
        if len(input_density) == 0:
            density = np.ones(len(input_points)) / len(input_points)
        else:
            if len(input_density) != len(input_points):
                raise ValueError("Density vector size doesn't match point cloud size")
            density = input_density

        # Create output directories
        self.ensure_directory_exists(output_dir)

        # Calculate number of levels
        total_points = len(input_points)
        level1_points = min(total_points, self.max_points)

        self.num_levels = min(5, int(math.log(level1_points / self.min_points) / math.log(4.0)) + 2)
        self.num_levels = max(1, self.num_levels)  # At least one level

        print(f"Initial point cloud has {total_points} points")
        print(f"Level 1 will have maximum of {self.max_points} points")
        print(f"Creating {self.num_levels} levels of point clouds")

        # Reset level point counts and parent-child relationships
        self.level_point_counts = []
        self.parent_indices = []
        self.child_indices = []

        # Resize parent and child indices containers
        self.parent_indices = [[] for _ in range(self.num_levels - 1)]
        self.child_indices = [[] for _ in range(self.num_levels - 1)]

        # Store all point clouds for each level
        level_points_vec = [[] for _ in range(self.num_levels)]
        level_density_vec = [[] for _ in range(self.num_levels)]

        # Level 0 is always the original point cloud
        level_points_vec[0] = input_points
        level_density_vec[0] = density
        self.level_point_counts.append(len(input_points))

        print(f"Level 0: using original point cloud with {len(level_points_vec[0])} points")

        # Generate coarser levels
        for level in range(1, self.num_levels):
            points_for_level = self.get_points_for_level(total_points, level)
            print(f"Level {level}: targeting {points_for_level} points")

            # Use k-means clustering to create coarser point cloud with parent-child tracking
            coarse_points, coarse_density, assignments = self.kmeans_clustering(
                level_points_vec[level-1], level_density_vec[level-1], points_for_level)

            level_points_vec[level] = coarse_points
            level_density_vec[level] = coarse_density
            self.level_point_counts.append(len(coarse_points))

            print(f"  Generated {len(coarse_points)} points after clustering")

            # Build parent-child relationships
            n_fine_points = len(level_points_vec[level-1])
            n_coarse_points = len(coarse_points)

            # Initialize parent indices for previous (finer) level
            self.parent_indices[level-1] = [[] for _ in range(n_fine_points)]

            # Initialize child indices for this (coarser) level
            self.child_indices[level-1] = [[] for _ in range(n_coarse_points)]

            # Populate parent-child relationships
            for i in range(n_fine_points):
                coarse_point_idx = assignments[i]
                if 0 <= coarse_point_idx < n_coarse_points:
                    # Add the coarse point as parent of this fine point
                    self.parent_indices[level-1][i].append(coarse_point_idx)

                    # Add this fine point as child of the coarse point
                    self.child_indices[level-1][coarse_point_idx].append(i)

        # Save the point clouds and parent-child relationships for each level
        for level in range(self.num_levels):
            points_file = os.path.join(output_dir, f"level_{level}_points.txt")
            density_file = os.path.join(output_dir, f"level_{level}_density.txt")

            with open(points_file, 'w') as points_out, open(density_file, 'w') as density_out:
                # Write points and density
                for i in range(len(level_points_vec[level])):
                    # Write points
                    point_str = ' '.join(map(str, level_points_vec[level][i]))
                    points_out.write(f"{point_str}\n")

                    # Write density
                    density_out.write(f"{level_density_vec[level][i]}\n")

            # Save parent-child relationships for non-boundary levels
            # Parents: For each point at level L, save its parent at level L+1
            if level < self.num_levels - 1:
                parents_file = os.path.join(output_dir, f"level_{level}_parents.txt")
                with open(parents_file, 'w') as parents_out:
                    # Write parent indices for points at current level
                    for i in range(len(self.parent_indices[level])):
                        parents_line = f"{len(self.parent_indices[level][i])}"
                        for parent_idx in self.parent_indices[level][i]:
                            parents_line += f" {parent_idx}"
                        parents_out.write(f"{parents_line}\n")

            # Children: For each point at level L, save its children at level L-1
            if level > 0:
                children_file = os.path.join(output_dir, f"level_{level}_children.txt")
                with open(children_file, 'w') as children_out:
                    # Write child indices for points at current level
                    for i in range(len(self.child_indices[level-1])):
                        children_line = f"{len(self.child_indices[level-1][i])}"
                        for child_idx in self.child_indices[level-1][i]:
                            children_line += f" {child_idx}"
                        children_out.write(f"{children_line}\n")

            print(f"Saved level {level} point cloud and relationships to {output_dir}")

        # Stop timing the entire hierarchy generation
        end_total = time.time()
        duration_total = end_total - start_total

        print(f"Total hierarchy generation took {duration_total:.2f} seconds")

        return self.num_levels


def read_parameters_from_prm(prm_file="parameters.prm"):
    """Read parameters from parameters.prm file."""
    min_points = 100  # Default value
    max_points = 2000  # Default value
    output_dir = "output/data_multilevel/target_multilevel"  # Default value from ParameterManager.h
    
    try:
        with open(prm_file, 'r') as f:
            lines = f.readlines()
            
        in_multilevel_section = False
        for line in lines:
            line = line.strip()
            
            # Check if we're in the multilevel_parameters section
            if "subsection multilevel_parameters" in line:
                in_multilevel_section = True
                continue
            elif in_multilevel_section and "end" in line:
                in_multilevel_section = False
                continue
            
            # Parse parameters within the multilevel_parameters section
            if in_multilevel_section:
                if "set target_min_points =" in line:
                    # Extract the value and remove any comments
                    value_part = line.split("=")[1].strip()
                    if "#" in value_part:
                        value_part = value_part.split("#")[0].strip()
                    min_points = int(value_part)
                elif "set target_max_points =" in line:
                    # Extract the value and remove any comments
                    value_part = line.split("=")[1].strip()
                    if "#" in value_part:
                        value_part = value_part.split("#")[0].strip()
                    max_points = int(value_part)
                elif "set target_hierarchy_dir =" in line:
                    # Extract the value and remove any comments
                    value_part = line.split("=")[1].strip()
                    if "#" in value_part:
                        value_part = value_part.split("#")[0].strip()
                    # Remove quotes if present
                    output_dir = value_part.strip('"').strip("'")
        
        print(f"Read from parameters.prm:")
        print(f"  min_points = {min_points}")
        print(f"  max_points = {max_points}")
        print(f"  target_hierarchy_dir = {output_dir}")
    except Exception as e:
        print(f"Error reading parameters.prm: {e}")
        print(f"Using default values:")
        print(f"  min_points = {min_points}")
        print(f"  max_points = {max_points}")
        print(f"  target_hierarchy_dir = {output_dir}")
    
    return min_points, max_points, output_dir


def main():
    # Load the target points
    points_file = "output/data_points/target_points.txt"
    density_file = "output/data_density/target_density.txt"
    
    try:
        points = np.loadtxt(points_file)
        density = np.loadtxt(density_file)
        print(f"Loaded {len(points)} target points")
        print(f"Loaded target density values")
    except Exception as e:
        print(f"Error loading points or density: {e}")
        return
    
    # Read parameters from parameters.prm
    min_points, max_points, output_dir = read_parameters_from_prm()
    
    # Create hierarchy manager
    hierarchy_manager = PointCloudHierarchyManager(min_points, max_points)
    
    # Generate hierarchy
    try:
        num_levels = hierarchy_manager.generate_hierarchy(points, density, output_dir)
        print(f"Successfully generated {num_levels} levels of point cloud hierarchy")
    except Exception as e:
        print(f"Error generating hierarchy: {e}")
        return


if __name__ == "__main__":
    main()
