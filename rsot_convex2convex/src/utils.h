#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>

namespace Utils {

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