#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include "SemiDiscreteOT/utils/ColorDefinitions.h"

/**
 * @namespace Utils
 * @brief Collection of utility functions for file I/O, mesh handling and data management
 */
namespace Utils {

/**
 * @brief Convert a double to a string in scientific notation
 * @param value The double value to convert
 * @param precision Number of significant digits (default: 6)
 * @return String representation in scientific notation
 */
inline std::string to_scientific_string(double value, int precision = 6) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(precision) << value;
    return oss.str();
}

/**
 * @brief Write a vector container to a file in binary or text format
 * @tparam VectorContainer Type of vector container
 * @param points Vector container to write
 * @param filepath Path to output file (without extension)
 * @param fileMode Output format ("txt" or "bin")
 */
template<typename VectorContainer>
void write_vector(const VectorContainer& points, 
                 const std::string& filepath, 
                 const std::string& fileMode = "txt") {
    std::ofstream file;
    
    // Create directories if they don't exist
    size_t pos = filepath.find_last_of("/\\");
    if (pos != std::string::npos) {
        std::string dir = filepath.substr(0, pos);
        #ifdef _WIN32
            system(("mkdir " + dir + " 2>nul").c_str());
        #else
            system(("mkdir -p " + dir).c_str());
        #endif
    }

    if (fileMode == "bin") {
        std::cout << "Writing: " << filepath << ".bin" << std::endl;
        file.open(filepath + ".bin", std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file for writing in binary mode." << std::endl;
            return;
        }
        for (const auto& point : points) {
            file.write(reinterpret_cast<const char*>(&point), sizeof(point));
        }
    } else if (fileMode == "txt") {
        std::cout << "Writing: " << filepath << ".txt" << std::endl;
        file.open(filepath + ".txt");
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file for writing in text mode." << std::endl;
            return;
        }
        file << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (const auto& point : points) {
            file << point << std::endl;
        }
    } else {
        std::cerr << "Error: Invalid file mode specified." << std::endl;
    }
    file.close();
}

/**
 * @brief Read a vector container from a file in binary or text format
 * @tparam VectorContainer Type of vector container
 * @param points Vector container to store read data
 * @param filepath Path to input file (with or without extension)
 * @param fileMode Input format ("txt" or "bin"), if empty will be inferred from filepath
 * @return true if read successful, false otherwise
 */
template<typename VectorContainer>
bool read_vector(VectorContainer& points, 
                const std::string& filepath, 
                const std::string& fileMode = "") {
    std::string actualFilepath = filepath;
    std::string actualFileMode = fileMode;
    
    // If fileMode is not provided, infer it from the filepath extension
    if (actualFileMode.empty()) {
        size_t dotPos = filepath.find_last_of('.');
        if (dotPos != std::string::npos) {
            std::string extension = filepath.substr(dotPos + 1);
            if (extension == "txt" || extension == "bin") {
                actualFileMode = extension;
                // Use the filepath as is since it already has the extension
                actualFilepath = filepath;
            } else {
                // Unknown extension, default to txt
                actualFileMode = "txt";
                actualFilepath = filepath + ".txt";
            }
        } else {
            // No extension found, default to txt
            actualFileMode = "txt";
            actualFilepath = filepath + ".txt";
        }
    } else {
        // File mode is specified, append extension if not already present
        if (filepath.find('.' + actualFileMode) != filepath.length() - actualFileMode.length() - 1) {
            actualFilepath = filepath + '.' + actualFileMode;
        }
    }
    
    std::ifstream file;
    
    if (actualFileMode == "bin") {
        file.open(actualFilepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file for reading: " << actualFilepath << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        using value_type = typename VectorContainer::value_type;
        points.resize(size / sizeof(value_type));
        file.read(reinterpret_cast<char*>(points.data()), size);
        
    } else if (actualFileMode == "txt") {
        file.open(actualFilepath);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file for reading: " << actualFilepath << std::endl;
            return false;
        }
        
        points.clear();
        typename VectorContainer::value_type value;
        while (file >> value) {
            points.push_back(value);
        }
        
    } else {
        std::cerr << "Error: Invalid file mode specified: " << actualFileMode << std::endl;
        return false;
    }
    
    file.close();
    return true;
}

/**
 * @brief List available epsilon folders and allow user selection
 * @param base_dir Base directory containing epsilon folders
 * @param allow_all Whether to allow selecting all folders
 * @return Vector of selected folder names
 */
inline std::vector<std::string> select_folder(const std::string& base_dir = "output", bool allow_all = true) {
    std::vector<std::string> epsilon_folders;
    
    // List all epsilon folders and exact_sot
    for (const auto& entry : std::filesystem::directory_iterator(base_dir)) {
        if (entry.is_directory()) {
            std::string folder_name = entry.path().filename().string();
            if (folder_name.find("epsilon_") == 0 || folder_name == "exact_sot") {
                epsilon_folders.push_back(folder_name);
            }
        }
    }
    
    if (epsilon_folders.empty()) {
        throw std::runtime_error("No epsilon or exact_sot folders found in " + base_dir);
    }
    
    // Sort folders to ensure consistent ordering
    std::sort(epsilon_folders.begin(), epsilon_folders.end());
    
    // Print available options
    std::cout << "\nAvailable folders:\n";
    for (size_t i = 0; i < epsilon_folders.size(); ++i) {
        std::cout << "[" << i + 1 << "] " << epsilon_folders[i] << "\n";
    }
    if (allow_all) {
        std::cout << "[" << epsilon_folders.size() + 1 << "] ALL FOLDERS\n";
    }
    
    // Get user selection
    size_t selection;
    while (true) {
        std::cout << "\nSelect a folder (1-" << (allow_all ? epsilon_folders.size() + 1 : epsilon_folders.size()) << "): ";
        if (std::cin >> selection && selection >= 1 && 
            selection <= (allow_all ? epsilon_folders.size() + 1 : epsilon_folders.size())) {
            break;
        }
        std::cout << "Invalid selection. Please try again.\n";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    
    // Return all folders if "ALL" was selected, otherwise return a vector with just the selected folder
    if (allow_all && selection == epsilon_folders.size() + 1) {
        return epsilon_folders;
    }
    return {epsilon_folders[selection - 1]};
}

/**
 * @brief Get target point cloud hierarchy files
 * @param dir Directory containing hierarchy files
 * @return Vector of pairs (points_file, density_file)
 */
inline std::vector<std::pair<std::string, std::string>> get_target_hierarchy_files(const std::string& dir)
{
    std::vector<std::pair<std::string, std::string>> files;
    
    // Check if directory exists
    if (!std::filesystem::exists(dir)) {
        throw std::runtime_error("Target point cloud hierarchy directory does not exist: " + dir);
    }
    
    // Collect all level point files
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().filename().string().find("level_") == 0 && 
            entry.path().filename().string().find("_points.txt") != std::string::npos) {
            std::string points_file = entry.path().string();
            points_file = points_file.substr(0, points_file.length() - 4);
            
            std::string level_num = points_file.substr(
                points_file.find("level_") + 6, 
                points_file.find("_points") - points_file.find("level_") - 6
            );
            std::string density_file = dir + "/level_" + level_num + "_density";
            
            if (std::filesystem::exists(density_file + ".txt")) {
                files.push_back({points_file, density_file});
            }
        }
    }

    // Sort in reverse order (coarsest to finest)
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        int level_a = std::stoi(a.first.substr(a.first.find("level_") + 6, 1));
        int level_b = std::stoi(b.first.substr(b.first.find("level_") + 6, 1));
        return level_a > level_b;
    });

    
    return files;
}

/**
 * @brief Load hierarchy data from files
 * @param hierarchy_dir Directory containing hierarchy files
 * @param child_indices Vector to store parent-child relationships
 * @param specific_level Level to load (-1 for all levels)
 * @param mpi_communicator MPI communicator
 * @param pcout Parallel console output
 * @return true if load successful, false otherwise
 */
template <int dim, int spacedim>
bool load_hierarchy_data(const std::string& hierarchy_dir,
                        std::vector<std::vector<std::vector<size_t>>>& child_indices,
                        int specific_level,
                        const MPI_Comm& mpi_communicator,
                        dealii::ConditionalOStream& pcout)
{
    // Only rank 0 checks directory and counts levels
    int num_levels = 0;
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        while (true) {
            std::string points_file = hierarchy_dir + "/level_" + std::to_string(num_levels) + "_points.txt";
            if (!std::filesystem::exists(points_file)) {
                break;
            }
            num_levels++;
        }
    }
    
    num_levels = dealii::Utilities::MPI::broadcast(mpi_communicator, num_levels, 0);
    
    if (num_levels == 0) {
        pcout << "No hierarchy data found in " << hierarchy_dir << std::endl;
        return false;
    }

    // If loading all levels, just resize and return
    if (specific_level == -1) {
        child_indices.resize(num_levels - 1);
        return true;
    }

    if (specific_level >= num_levels - 1) {
        return true; // No parent-child relationships for the last level
    }

    if (static_cast<int>(child_indices.size()) <= specific_level) {
        child_indices.resize(num_levels - 1);
    }

    // Process 0 loads data
    std::vector<std::vector<size_t>> level_data;
    bool level_load_success = true;
    
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        std::string children_file = hierarchy_dir + "/level_" + std::to_string(specific_level+1) + "_children.txt";
        std::ifstream children_in(children_file);
        
        if (!children_in) {
            level_load_success = false;
        } else {
            std::string line;
            
            while (std::getline(children_in, line)) {
                std::istringstream iss(line);
                int num_children;
                iss >> num_children;
                
                std::vector<size_t> children(num_children);
                
                size_t child_idx;
                int idx = 0;

                while (idx < num_children && iss >> child_idx) {
                    children[idx] = child_idx;
                    ++idx;
                }
                
                level_data.push_back(children);
            }
        }
    }

    // Broadcast success status
    level_load_success = dealii::Utilities::MPI::broadcast(mpi_communicator, level_load_success, 0);
    
    if (!level_load_success) {
        pcout << "Error loading hierarchy data at level " << specific_level << std::endl;
        return false;
    }

    // First broadcast the size of the level data
    unsigned int n_parents = 0;
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        n_parents = level_data.size();
    }
    n_parents = dealii::Utilities::MPI::broadcast(mpi_communicator, n_parents, 0);
    
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) != 0) {
        level_data.resize(n_parents);
    }
    
    for (unsigned int i = 0; i < n_parents; ++i) {
        unsigned int n_children = 0;
        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
            n_children = level_data[i].size();
        }
        n_children = dealii::Utilities::MPI::broadcast(mpi_communicator, n_children, 0);
        
        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) != 0) {
            level_data[i].resize(n_children);
        }

        // mpi broadcast level data[i]
        MPI_Bcast(&level_data[i][0], n_children, MPI_UNSIGNED_LONG, 0, mpi_communicator);
        
    }
    
    child_indices[specific_level] = level_data;
    return true;
}

/**
 * @brief Write mesh to file in specified formats with optional cell data
 * @tparam dim Dimension of the mesh
 * @tparam spacedim Ambient space dimension
 * @param mesh Triangulation to write
 * @param filepath Base path for output files (without extension)
 * @param formats Vector of output formats ("vtk", "msh", "vtu")
 * @param cell_data Optional vector of cell data to include
 * @param data_name Name for the cell data field
 * @return true if write successful, false otherwise
 */
template<int dim, int spacedim = dim>
bool write_mesh(const dealii::Triangulation<dim, spacedim>& mesh,
               const std::string& filepath,
               const std::vector<std::string>& formats,
               const std::vector<double>* cell_data = nullptr,
               const std::string& data_name = "cell_data")
{
    try {
        std::filesystem::path path(filepath);
        std::filesystem::create_directories(path.parent_path());

        dealii::GridOut grid_out;
        
        for (const auto& format : formats) {
            if (format == "vtk") {
                std::ofstream out_vtk(filepath + ".vtk");
                if (!out_vtk.is_open()) {
                    std::cerr << "Error: Unable to open file for writing: " 
                              << filepath + ".vtk" << std::endl;
                    return false;
                }
                grid_out.write_vtk(mesh, out_vtk);
                std::cout << "Mesh saved to: " << filepath + ".vtk" << std::endl;
            }
            else if (format == "msh") {
                // First write in default format
                std::ofstream out_msh(filepath + ".msh");
                if (!out_msh.is_open()) {
                    std::cerr << "Error: Unable to open file for writing: " 
                              << filepath + ".msh" << std::endl;
                    return false;
                }
                grid_out.write_msh(mesh, out_msh);
                out_msh.close();
                
                // Convert to MSH2 format using gmsh
                std::string cmd = "gmsh " + filepath + ".msh -format msh2 -save_all -3 -o " + 
                                filepath + "_msh2.msh && mv " + filepath + "_msh2.msh " + 
                                filepath + ".msh";
                int ret = system(cmd.c_str());
                if (ret != 0) {
                    std::cerr << "Error: Failed to convert mesh to MSH2 format" << std::endl;
                    return false;
                }
                std::cout << "Mesh saved and converted to MSH2 format: " << filepath + ".msh" << std::endl;
            }
            else if (format == "vtu") {
                dealii::DataOut<dim, spacedim> data_out;
                data_out.attach_triangulation(mesh);
                
                // Add cell data if provided
                if (cell_data != nullptr) {
                    Assert(cell_data->size() == mesh.n_active_cells(),
                           dealii::ExcDimensionMismatch(cell_data->size(), 
                                                      mesh.n_active_cells()));
                    
                    dealii::Vector<double> cell_data_vector(cell_data->begin(), cell_data->end());
                    data_out.add_data_vector(cell_data_vector, data_name);
                }
                
                data_out.build_patches();
                
                std::ofstream out_vtu(filepath + ".vtu");
                if (!out_vtu.is_open()) {
                    std::cerr << "Error: Unable to open file for writing: " 
                              << filepath + ".vtu" << std::endl;
                    return false;
                }
                data_out.write_vtu(out_vtu);
                std::cout << "Mesh saved to: " << filepath + ".vtu" << std::endl;
            }
            else {
                std::cerr << "Warning: Unsupported format '" << format 
                          << "' requested" << std::endl;
            }
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error writing mesh: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Interpolate a field from a source mesh to a target mesh using nearest neighbor approach
 * @tparam dim Dimension of the mesh
 * @tparam spacedim Spatial dimension
 * @param source_dh Source DoFHandler
 * @param source_field Source field values
 * @param target_dh Target DoFHandler
 * @param target_field Target field values (output)
 */
template <int dim, int spacedim>
void interpolate_non_conforming_nearest(
    const dealii::DoFHandler<dim, spacedim> &source_dh,
    const dealii::Vector<double>            &source_field,
    const dealii::DoFHandler<dim, spacedim> &target_dh,
    dealii::LinearAlgebra::distributed::Vector<double> &target_field)
{
  using namespace dealii;
  
  Assert(source_field.size() == source_dh.n_dofs(),
         ExcDimensionMismatch(source_field.size(), source_dh.n_dofs()));
  Assert(target_field.size() == target_dh.n_dofs(),
         ExcDimensionMismatch(target_field.size(), target_dh.n_dofs()));

  const auto &source_fe = source_dh.get_fe();
  const auto &target_fe = target_dh.get_fe();

  std::unique_ptr<Mapping<dim, spacedim>> source_mapping;
  if (source_fe.reference_cell() == ReferenceCells::get_hypercube<dim>())
    source_mapping = std::make_unique<MappingQ1<dim, spacedim>>();
  else
    source_mapping = std::make_unique<MappingFE<dim, spacedim>>(FE_SimplexP<dim, spacedim>(1));

  std::unique_ptr<Mapping<dim, spacedim>> target_mapping;
  if (target_fe.reference_cell() == ReferenceCells::get_hypercube<dim>())
    target_mapping = std::make_unique<MappingQ1<dim, spacedim>>();
  else
    target_mapping = std::make_unique<MappingFE<dim, spacedim>>(FE_SimplexP<dim, spacedim>(1));

  const Quadrature<dim> target_quadrature(target_fe.get_generalized_support_points());
  FEValues<dim, spacedim> target_fe_values(*target_mapping,
                                target_fe,
                                target_quadrature,
                                update_quadrature_points);

  std::vector<types::global_dof_index> source_dof_indices(source_fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> target_dof_indices(target_fe.n_dofs_per_cell());
  Vector<double> source_cell_values(source_fe.n_dofs_per_cell());

  Vector<double> target_values(target_field.size());
  Vector<double> weights(target_field.size());

  std::vector<std::pair<Point<spacedim>, typename DoFHandler<dim, spacedim>::active_cell_iterator>> cell_centers;

  for (const auto &cell : source_dh.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;
    cell_centers.emplace_back(cell->center(), cell);
  }

  std::vector<std::pair<Point<spacedim>, unsigned int>> indexed_points;
  indexed_points.reserve(cell_centers.size());
  for (unsigned int i = 0; i < cell_centers.size(); ++i)
    indexed_points.emplace_back(cell_centers[i].first, i);

  using RTreeParams = boost::geometry::index::rstar<8>;
  boost::geometry::index::rtree<std::pair<Point<spacedim>, unsigned int>, RTreeParams>
    rtree(indexed_points.begin(), indexed_points.end());

  for (const auto &target_cell : target_dh.active_cell_iterators())
  {
    if (!target_cell->is_locally_owned())
      continue;

    target_fe_values.reinit(target_cell);
    const std::vector<Point<spacedim>> &target_points = target_fe_values.get_quadrature_points();
    target_cell->get_dof_indices(target_dof_indices);

    for (unsigned int q = 0; q < target_points.size(); ++q)
    {
      const Point<spacedim> &target_point = target_points[q];

      std::vector<std::pair<Point<spacedim>, unsigned int>> nearest;
      rtree.query(boost::geometry::index::nearest(target_point, 1), std::back_inserter(nearest));

      if (nearest.empty()) 
        continue;

      const unsigned int nearest_index = nearest.front().second;
      auto chosen_source_cell = cell_centers[nearest_index].second;

      Point<dim> p_unit;
      try
      {
        p_unit = source_mapping->transform_real_to_unit_cell(chosen_source_cell, target_point);
      }
      catch (...)
      {
        continue;
      }

      chosen_source_cell->get_dof_indices(source_dof_indices);
      for (unsigned int i = 0; i < source_dof_indices.size(); ++i)
        source_cell_values[i] = source_field[source_dof_indices[i]];

      double source_value = 0.0;
      for (unsigned int i = 0; i < source_fe.n_dofs_per_cell(); ++i)
        source_value += source_cell_values[i] * source_fe.shape_value(i, p_unit);

      const unsigned int target_dof = target_dof_indices[q];
      target_values[target_dof] += source_value;
      weights[target_dof] += 1.0;
    }
  }

  for (types::global_dof_index i = 0; i < target_field.size(); ++i)
    if (weights[i] > 0)
      target_field[i] = target_values[i] / weights[i];

  target_field.compress(VectorOperation::insert);
}

/**
 * @brief Interpolate a field from a source mesh to a target mesh
 * @tparam dim Dimension of the mesh
 * @tparam spacedim Spatial dimension
 * @param source_dh Source DoFHandler
 * @param source_field Source field values
 * @param target_dh Target DoFHandler
 * @param target_field Target field values (output)
 */
template <int dim, int spacedim>
void interpolate_non_conforming(const dealii::DoFHandler<dim, spacedim> &source_dh,
                              const dealii::Vector<double>              &source_field,
                              const dealii::DoFHandler<dim, spacedim>   &target_dh,
                              dealii::LinearAlgebra::distributed::Vector<double> &target_field)
{
  using namespace dealii;
  
  Assert(source_field.size() == source_dh.n_dofs(),
         ExcDimensionMismatch(source_field.size(), source_dh.n_dofs()));
  Assert(target_field.size() == target_dh.n_dofs(),
         ExcDimensionMismatch(target_field.size(), target_dh.n_dofs()));

  const auto &source_fe = source_dh.get_fe();
  const auto &target_fe = target_dh.get_fe();

  std::unique_ptr<Mapping<dim, spacedim>> source_mapping;
  if (source_dh.get_fe().reference_cell() == ReferenceCells::get_hypercube<dim>())
    source_mapping = std::make_unique<MappingQ1<dim, spacedim>>();
  else
    source_mapping = std::make_unique<MappingFE<dim, spacedim>>(FE_SimplexP<dim, spacedim>(1));

  std::unique_ptr<Mapping<dim, spacedim>> target_mapping;
  if (target_dh.get_fe().reference_cell() == ReferenceCells::get_hypercube<dim>())
    target_mapping = std::make_unique<MappingQ1<dim, spacedim>>();
  else
    target_mapping = std::make_unique<MappingFE<dim, spacedim>>(FE_SimplexP<dim, spacedim>(1));

  std::vector<std::pair<Point<spacedim>, typename DoFHandler<dim, spacedim>::active_cell_iterator>>
    cell_centers;

  for (const auto &cell : source_dh.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;
    cell_centers.emplace_back(cell->center(), cell);
  }

  std::vector<std::pair<Point<spacedim>, unsigned int>> indexed_points;
  indexed_points.reserve(cell_centers.size());
  for (unsigned int i = 0; i < cell_centers.size(); ++i)
    indexed_points.emplace_back(cell_centers[i].first, i);

  using RTreeParams = boost::geometry::index::rstar<8>;
  boost::geometry::index::rtree<std::pair<Point<spacedim>, unsigned int>, RTreeParams>
    rtree(indexed_points.begin(), indexed_points.end());

  const Quadrature<dim> quadrature(target_fe.get_unit_support_points());

  FEValues<dim> source_fe_values(*source_mapping,
                                source_fe,
                                quadrature,
                                update_values | update_quadrature_points);

  FEValues<dim> target_fe_values(*target_mapping,
                                target_fe,
                                quadrature,
                                update_quadrature_points);

  std::vector<types::global_dof_index> source_dof_indices(source_fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> target_dof_indices(target_fe.n_dofs_per_cell());
  Vector<double> source_cell_values(source_fe.n_dofs_per_cell());

  Vector<double> target_values(target_dh.n_dofs());
  Vector<double> weights(target_dh.n_dofs());

  for (const auto &target_cell : target_dh.active_cell_iterators())
  {
    if (!target_cell->is_locally_owned())
      continue;

    target_fe_values.reinit(target_cell);
    const std::vector<Point<spacedim>> &target_points = target_fe_values.get_quadrature_points();
    target_cell->get_dof_indices(target_dof_indices);

    for (unsigned int q = 0; q < target_points.size(); ++q)
    {
      const Point<spacedim> &target_point = target_points[q];

      std::vector<std::pair<Point<spacedim>, unsigned int>> nearest;
      rtree.query(boost::geometry::index::nearest(target_point, 1),
                   std::back_inserter(nearest));

      Assert(!nearest.empty(), ExcInternalError("RTree nearest neighbor search failed"));

      const auto &source_cell = cell_centers[nearest[0].second].second;

      source_fe_values.reinit(source_cell);
      source_cell->get_dof_indices(source_dof_indices);

      for (unsigned int i = 0; i < source_dof_indices.size(); ++i)
        source_cell_values[i] = source_field[source_dof_indices[i]];

      double source_value = 0;
      for (unsigned int i = 0; i < source_fe.n_dofs_per_cell(); ++i)
        source_value += source_cell_values[i] * source_fe_values.shape_value(i, q);

      const unsigned int target_dof = target_dof_indices[q];
      target_values[target_dof] += source_value;
      weights[target_dof] += 1.0;
    }
  }

  for (types::global_dof_index i = 0; i < target_dh.n_dofs(); ++i)
    if (weights[i] > 0)
      target_field[i] = target_values[i] / weights[i];

  target_field.compress(VectorOperation::insert);
}

/**
 * @brief Detect cell types in a triangulation and create appropriate finite element
 * @tparam dim Dimension of the mesh
 * @param triangulation Input triangulation to analyze
 * @param degree Polynomial degree for the finite element (defaults to 1)
 * @return Unique pointer to appropriate finite element
 * @throw std::runtime_error if cell type cannot be determined
 */
template <int dim, int spacedim = dim>
std::unique_ptr<dealii::FiniteElement<dim, spacedim>> create_fe_for_mesh(
    const dealii::Triangulation<dim, spacedim>& triangulation,
    const unsigned int degree = 1)
{
    bool has_quads_or_hexes = false;
    bool has_triangles_or_tets = false;
    
    for (const auto& cell : triangulation.active_cell_iterators()) {
        if (cell->reference_cell() == dealii::ReferenceCells::get_hypercube<dim>())
            has_quads_or_hexes = true;
        if (cell->reference_cell() == dealii::ReferenceCells::get_simplex<dim>())
            has_triangles_or_tets = true;
    }

    if (has_triangles_or_tets && !has_quads_or_hexes) {
        return std::make_unique<dealii::FE_SimplexP<dim, spacedim>>(degree);
    } else if (has_quads_or_hexes && !has_triangles_or_tets) {
        return std::make_unique<dealii::FE_Q<dim, spacedim>>(degree);
    } else {
        throw std::runtime_error("Mixed cell types or no cells found in triangulation");
    }
}

/**
 * @brief Create appropriate finite element and mapping for a triangulation
 * @tparam dim Dimension of the mesh
 * @param triangulation Input triangulation to analyze
 * @param fe_degree Polynomial degree for the finite element (defaults to 1)
 * @param mapping_degree Polynomial degree for the mapping (defaults to 1)
 * @return Pair of unique pointers to appropriate finite element and mapping
 * @throw std::runtime_error if cell type cannot be determined
 */
template<int dim, int spacedim>
std::pair<std::unique_ptr<dealii::FiniteElement<dim, spacedim>>, 
          std::unique_ptr<dealii::Mapping<dim, spacedim>>> 
create_fe_and_mapping_for_mesh(
    const dealii::Triangulation<dim, spacedim>& triangulation,
    const unsigned int fe_degree = 1,
    const unsigned int mapping_degree = 1)
{
    bool has_quads_or_hexes = false;
    bool has_triangles_or_tets = false;
    
    for (const auto& cell : triangulation.active_cell_iterators()) {
        if (cell->reference_cell() == dealii::ReferenceCells::get_hypercube<dim>())
            has_quads_or_hexes = true;
        if (cell->reference_cell() == dealii::ReferenceCells::get_simplex<dim>())
            has_triangles_or_tets = true;
    }

    std::unique_ptr<dealii::FiniteElement<dim, spacedim>> fe;
    std::unique_ptr<dealii::Mapping<dim, spacedim>> mapping;

    if (has_triangles_or_tets) {
        fe = std::make_unique<dealii::FE_SimplexP<dim, spacedim>>(fe_degree);
        mapping = std::make_unique<dealii::MappingFE<dim, spacedim>>(dealii::FE_SimplexP<dim, spacedim>(mapping_degree));
    } else if (has_quads_or_hexes) {
        fe = std::make_unique<dealii::FE_Q<dim, spacedim>>(fe_degree);
        mapping = std::make_unique<dealii::MappingQ1<dim, spacedim>>();
    } else {
        throw std::runtime_error("Could not determine mesh cell type in triangulation");
    }

    return {std::move(fe), std::move(mapping)};
}

/**
 * @brief Create appropriate quadrature for a triangulation based on cell types
 * @tparam dim Dimension of the mesh
 * @param triangulation Input triangulation to analyze
 * @param order Quadrature order (defaults to 2)
 * @return Unique pointer to appropriate quadrature
 * @throw std::runtime_error if cell type cannot be determined
 */
template <int dim, int spacedim = dim>
std::unique_ptr<dealii::Quadrature<dim>> create_quadrature_for_mesh(
    const dealii::Triangulation<dim, spacedim>& triangulation,
    const unsigned int order = 2)
{
    bool has_quads_or_hexes = false;
    bool has_triangles_or_tets = false;
    
    for (const auto& cell : triangulation.active_cell_iterators()) {
        if (cell->reference_cell() == dealii::ReferenceCells::get_hypercube<dim>())
            has_quads_or_hexes = true;
        if (cell->reference_cell() == dealii::ReferenceCells::get_simplex<dim>())
            has_triangles_or_tets = true;
    }

    if (has_triangles_or_tets) {
        return std::make_unique<dealii::QGaussSimplex<dim>>(order);
    } else if (has_quads_or_hexes) {
        return std::make_unique<dealii::QGauss<dim>>(order);
    } else {
        throw std::runtime_error("Could not determine mesh cell type for quadrature creation");
    }
}

/**
 * @brief Write points with associated density to VTK file
 * @tparam spacedim Spatial dimension
 * @param points Vector of points to write
 * @param density Vector of density values associated with points
 * @param filename Output VTK filename
 * @param description Description for the VTK file header
 * @param density_name Name for the density scalar field
 * @return true if write successful, false otherwise
 */
template <int spacedim>
bool write_points_with_density_vtk(
    const std::vector<dealii::Point<spacedim>>& points,
    const std::vector<double>& density,
    const std::string& filename,
    const std::string& description = "Points with density values",
    const std::string& density_name = "density")
{
    if (points.size() != density.size()) {
        std::cerr << "Error: Points and density vectors must have the same size" << std::endl;
        return false;
    }
    
    if (points.empty()) {
        std::cerr << "Error: Cannot write empty point set" << std::endl;
        return false;
    }

    // Create directories if they don't exist
    std::filesystem::path path(filename);
    std::filesystem::create_directories(path.parent_path());

    std::ofstream vtk_file(filename);
    if (!vtk_file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        return false;
    }

    // Write VTK header
    vtk_file << "# vtk DataFile Version 3.0\n"
             << description << "\n"
             << "ASCII\n"
             << "DATASET UNSTRUCTURED_GRID\n"
             << "POINTS " << points.size() << " double\n";

    // Write point coordinates
    for (const auto& p : points) {
        vtk_file << p[0] << " " << p[1];
        if (spacedim == 3) {
            vtk_file << " " << p[2];
        } else if (spacedim == 2) {
            vtk_file << " 0.0";  // Add z=0 for 2D
        }
        vtk_file << "\n";
    }

    // Write cells (point cells)
    const unsigned int n_points = points.size();
    vtk_file << "CELLS " << n_points << " " << 2 * n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << "1 " << i << "\n";
    }

    // Write cell types
    vtk_file << "CELL_TYPES " << n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << "1\n";  // VTK_VERTEX
    }

    // Write density as scalar data
    vtk_file << "POINT_DATA " << n_points << "\n"
             << "SCALARS " << density_name << " double 1\n"
             << "LOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < n_points; ++i) {
        vtk_file << density[i] << "\n";
    }

    vtk_file.close();
    
    std::cout << "Points with density saved to: " << filename << std::endl;
    return true;
}

/**
 * @brief Write points with displacement vectors and density to VTK file
 * @tparam spacedim Spatial dimension
 * @param source_points Vector of source points
 * @param mapped_points Vector of mapped points (targets)
 * @param density Vector of density values associated with points
 * @param filename Output VTK filename
 * @param description Description for the VTK file header
 * @param density_name Name for the density scalar field
 * @return true if write successful, false otherwise
 */
template <int spacedim>
bool write_points_with_displacement_vtk(
    const std::vector<dealii::Point<spacedim>>& source_points,
    const std::vector<dealii::Point<spacedim>>& mapped_points,
    const std::vector<double>& density,
    const std::string& filename,
    const std::string& description = "Points with displacement vectors",
    const std::string& density_name = "density")
{
    if (source_points.size() != mapped_points.size() || 
        source_points.size() != density.size()) {
        std::cerr << "Error: Source points, mapped points, and density vectors must have the same size" << std::endl;
        return false;
    }
    
    if (source_points.empty()) {
        std::cerr << "Error: Cannot write empty point set" << std::endl;
        return false;
    }

    // Create directories if they don't exist
    std::filesystem::path path(filename);
    std::filesystem::create_directories(path.parent_path());

    std::ofstream vtk_file(filename);
    if (!vtk_file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        return false;
    }

    // Write VTK header
    vtk_file << "# vtk DataFile Version 3.0\n"
             << description << "\n"
             << "ASCII\n"
             << "DATASET UNSTRUCTURED_GRID\n"
             << "POINTS " << source_points.size() << " double\n";

    // Write source point coordinates
    for (const auto& p : source_points) {
        vtk_file << p[0] << " " << p[1];
        if (spacedim == 3) {
            vtk_file << " " << p[2];
        } else if (spacedim == 2) {
            vtk_file << " 0.0";  // Add z=0 for 2D
        }
        vtk_file << "\n";
    }

    // Write cells (point cells)
    const unsigned int n_points = source_points.size();
    vtk_file << "CELLS " << n_points << " " << 2 * n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << "1 " << i << "\n";
    }

    // Write cell types
    vtk_file << "CELL_TYPES " << n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << "1\n";  // VTK_VERTEX
    }

    // Write point data
    vtk_file << "POINT_DATA " << n_points << "\n";
    
    // Write displacement vectors
    vtk_file << "VECTORS displacement double\n";
    for (std::size_t i = 0; i < n_points; ++i) {
        dealii::Tensor<1, spacedim> displacement = mapped_points[i] - source_points[i];
        vtk_file << displacement[0] << " " << displacement[1];
        if (spacedim == 3) {
            vtk_file << " " << displacement[2];
        } else if (spacedim == 2) {
            vtk_file << " 0.0";  // Add z=0 for 2D
        }
        vtk_file << "\n";
    }

    // Write density as scalar data
    vtk_file << "SCALARS " << density_name << " double 1\n"
             << "LOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < n_points; ++i) {
        vtk_file << density[i] << "\n";
    }

    vtk_file.close();
    
    std::cout << "Points with displacement and density saved to: " << filename << std::endl;
    return true;
}

/**
 * @brief Read a scalar field from a VTK file
 * 
 * @param filename Path to the VTK file
 * @param vtk_dof_handler DoF handler for the VTK mesh
 * @param vtk_field Vector for the VTK field data
 * @param vtk_tria Triangulation for the VTK mesh
 * @param mpi_communicator MPI communicator
 * @param pcout Parallel output stream
 * @param broadcast_field Whether to broadcast the field to all processes (true for source, false for target)
 * @param field_name Name of the field to read from the VTK file
 * @return bool Success status
 */
template <int dim, int spacedim = dim>
bool read_vtk_field(
    const std::string& filename,
    dealii::DoFHandler<dim, spacedim>& vtk_dof_handler,
    dealii::Vector<double>& vtk_field,
    dealii::Triangulation<dim, spacedim>& vtk_tria,
    const MPI_Comm& mpi_communicator,
    dealii::ConditionalOStream& pcout,
    bool broadcast_field = false,
    const std::string& field_name = "normalized_density")
{
    bool success = false;
    // Clear any existing triangulation
    vtk_tria.clear();

    // Read mesh from VTK file
    dealii::GridIn<dim, spacedim> grid_in;
    grid_in.attach_triangulation(vtk_tria);

    std::ifstream vtk_file(filename);
    if (!vtk_file) {
        pcout << Color::red << Color::bold
              << "Error: Could not open VTK file: " << filename
              << Color::reset << std::endl;
        return false;
    }

    // Read VTK file
    grid_in.read_vtk(vtk_file);
    pcout << "VTK mesh: " << vtk_tria.n_active_cells() << " cells" << std::endl;

    // Create and store FE (must outlive DoFHandler usage)
    auto fe = create_fe_for_mesh(vtk_tria);
    
    // Clear and reinitialize DoFHandler with triangulation
    vtk_dof_handler.clear();
    vtk_dof_handler.reinit(vtk_tria);
    vtk_dof_handler.distribute_dofs(*fe);

    // Initialize field vector
    vtk_field.reinit(vtk_dof_handler.n_dofs());

    // Only rank 0 reads the file
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        try {
            // Read scalar field data
            std::ifstream vtk_reader(filename);
            std::string line;
            bool found_point_data = false;
            bool found_scalars = false;
            std::string found_field_name;

            while (std::getline(vtk_reader, line)) {
                if (line.find("POINT_DATA") != std::string::npos) {
                    found_point_data = true;
                }

                if (found_point_data && line.find("SCALARS") != std::string::npos) {
                    std::istringstream iss(line);
                    std::string dummy;
                    iss >> dummy >> found_field_name;
                    
                    // Check if this is the field we're looking for
                    if (found_field_name == field_name) {
                        found_scalars = true;
                        pcout << "Found scalar field: " << found_field_name << std::endl;

                        // Skip LOOKUP_TABLE line
                        std::getline(vtk_reader, line);

                        // Read scalar values
                        for (unsigned int i = 0; i < vtk_dof_handler.n_dofs(); ++i) {
                            if (!(vtk_reader >> vtk_field[i])) {
                                pcout << Color::red << Color::bold
                                      << "Error reading scalar field data from VTK"
                                      << Color::reset << std::endl;
                                found_scalars = false;
                                break;
                            }
                        }
                        break;
                    }
                }
            }

            success = found_scalars;
            if (!success) {
                pcout << Color::red << Color::bold
                      << "Error: Could not find scalar field '" << field_name << "' in VTK file"
                      << Color::reset << std::endl;
            }
        } catch (const std::exception& e) {
            pcout << Color::red << Color::bold
                  << "Exception during VTK file reading: " << e.what()
                  << Color::reset << std::endl;
            success = false;
        }
    }

    if (broadcast_field) {
        success = dealii::Utilities::MPI::broadcast(mpi_communicator, success, 0);
        if (success) {
            // For source case: Broadcast field data to all processes
            vtk_field = dealii::Utilities::MPI::broadcast(mpi_communicator, vtk_field, 0);
        }
    }

    return success;
}

} 
#endif 