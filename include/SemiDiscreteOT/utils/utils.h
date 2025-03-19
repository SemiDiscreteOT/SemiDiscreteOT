#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

/**
 * @namespace Utils
 * @brief Collection of utility functions for file I/O, mesh handling and data management
 */
namespace Utils {

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
 * @param filepath Path to input file (without extension)
 * @param fileMode Input format ("txt" or "bin")
 * @return true if read successful, false otherwise
 */
template<typename VectorContainer>
bool read_vector(VectorContainer& points, 
                const std::string& filepath, 
                const std::string& fileMode = "txt") {
    std::ifstream file;
    
    if (fileMode == "bin") {
        file.open(filepath + ".bin", std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file for reading: " << filepath << ".bin" << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        using value_type = typename VectorContainer::value_type;
        points.resize(size / sizeof(value_type));
        file.read(reinterpret_cast<char*>(points.data()), size);
        
    } else if (fileMode == "txt") {
        file.open(filepath + ".txt");
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file for reading: " << filepath << ".txt" << std::endl;
            return false;
        }
        
        points.clear();
        typename VectorContainer::value_type value;
        while (file >> value) {
            points.push_back(value);
        }
        
    } else {
        std::cerr << "Error: Invalid file mode specified." << std::endl;
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
 * @return Vector of pairs (points_file, weights_file)
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
            std::string weights_file = dir + "/level_" + level_num + "_weights";
            
            if (std::filesystem::exists(weights_file + ".txt")) {
                files.push_back({points_file, weights_file});
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
 * @tparam dim Dimension of the mesh
 * @param hierarchy_dir Directory containing hierarchy files
 * @param child_indices Vector to store parent-child relationships
 * @param specific_level Level to load (-1 for all levels)
 * @param mpi_communicator MPI communicator
 * @param pcout Parallel console output
 * @return true if load successful, false otherwise
 */
template <int dim>
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
 * @param mesh Triangulation to write
 * @param filepath Base path for output files (without extension)
 * @param formats Vector of output formats ("vtk", "msh", "vtu")
 * @param cell_data Optional vector of cell data to include
 * @param data_name Name for the cell data field
 * @return true if write successful, false otherwise
 */
template<int dim>
bool write_mesh(const dealii::Triangulation<dim>& mesh,
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
                dealii::DataOut<dim> data_out;
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

} 
#endif 