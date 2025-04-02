#ifndef MESH_MANAGER_TEMPLATES_H
#define MESH_MANAGER_TEMPLATES_H

template <int dim, int spacedim>
MeshManager<dim, spacedim>::MeshManager(const MPI_Comm& comm)
    : mpi_communicator(comm)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
    , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
    , pcout(std::cout, this_mpi_process == 0)
{}

template <int dim, int spacedim>
template <typename TriangulationType>
void MeshManager<dim, spacedim>::generate_mesh(TriangulationType& tria,
                                   const std::string& grid_generator_function,
                                   const std::string& grid_generator_arguments,
                                   const unsigned int n_refinements,
                                   const bool use_tetrahedral_mesh)
{
    if constexpr (std::is_same_v<TriangulationType, parallel::fullydistributed::Triangulation<dim, spacedim>>) {
        // For fullydistributed triangulation, first create a serial triangulation
        Triangulation<dim, spacedim> serial_tria;
        GridGenerator::generate_from_name_and_arguments(
            serial_tria,
            grid_generator_function,
            grid_generator_arguments);

        if (use_tetrahedral_mesh && dim == 3 && spacedim == 3) {
            GridGenerator::convert_hypercube_to_simplex_mesh(serial_tria, serial_tria);
        }

        serial_tria.refine_global(n_refinements);

        // Set up the partitioner to use z-order curve
        tria.set_partitioner([](Triangulation<dim, spacedim>& tria_to_partition, const unsigned int n_partitions) {
            GridTools::partition_triangulation_zorder(n_partitions, tria_to_partition);
        }, TriangulationDescription::Settings::construct_multigrid_hierarchy);

        // Create the construction data
        auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(
            serial_tria, mpi_communicator,
            TriangulationDescription::Settings::construct_multigrid_hierarchy);

        // Actually create the distributed triangulation
        tria.create_triangulation(construction_data);
    } else {
        // For regular triangulation
        GridGenerator::generate_from_name_and_arguments(
            tria,
            grid_generator_function,
            grid_generator_arguments);

        if (use_tetrahedral_mesh && dim == 3 && spacedim == 3) {
            GridGenerator::convert_hypercube_to_simplex_mesh(tria, tria);
        }

        tria.refine_global(n_refinements);
    }
}

template <int dim, int spacedim>
void MeshManager<dim, spacedim>::load_source_mesh(parallel::fullydistributed::Triangulation<dim, spacedim>& source_mesh)
{
    // First load source mesh into a serial triangulation
    Triangulation<dim, spacedim> serial_source;
    GridIn<dim, spacedim> grid_in_source;
    grid_in_source.attach_triangulation(serial_source);
    bool source_loaded = false;

    // First try VTK
    std::ifstream in_vtk_source(mesh_directory + "/source.vtk");
    if (in_vtk_source.good()) {
        try {
            grid_in_source.read_vtk(in_vtk_source);
            source_loaded = true;
            pcout << "Source mesh loaded from VTK format" << std::endl;
        } catch (const std::exception& e) {
            pcout << "Failed to load source mesh from VTK format: " << e.what() << std::endl;
        }
    }

    // If VTK failed, try MSH
    if (!source_loaded) {
        std::ifstream in_msh_source(mesh_directory + "/source.msh");
        if (in_msh_source.good()) {
            try {
                grid_in_source.read_msh(in_msh_source);
                source_loaded = true;
                pcout << "Source mesh loaded from MSH format" << std::endl;
            } catch (const std::exception& e) {
                pcout << "Failed to load source mesh from MSH format: " << e.what() << std::endl;
            }
        }
    }

    if (!source_loaded) {
        throw std::runtime_error("Failed to load source mesh from either VTK or MSH format");
    }

    // Partition the serial source mesh using z-order curve
    GridTools::partition_triangulation_zorder(n_mpi_processes, serial_source);

    // Convert serial source mesh to fullydistributed without multigrid hierarchy
    auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(
        serial_source, mpi_communicator,
        TriangulationDescription::Settings::default_setting);
    source_mesh.create_triangulation(construction_data);
}

template <int dim, int spacedim>
void MeshManager<dim, spacedim>::load_target_mesh(Triangulation<dim, spacedim>& target_mesh)
{
    // Only rank 0 loads the target mesh
    if (this_mpi_process != 0) {
        return;
    }

    // Load target mesh (stays serial)
    GridIn<dim, spacedim> grid_in_target;
    grid_in_target.attach_triangulation(target_mesh);
    bool target_loaded = false;

    // Try VTK for target
    std::ifstream in_vtk_target(mesh_directory + "/target.vtk");
    if (in_vtk_target.good()) {
        try {
            grid_in_target.read_vtk(in_vtk_target);
            target_loaded = true;
            pcout << "Target mesh loaded from VTK format" << std::endl;
        } catch (const std::exception& e) {
            pcout << "Failed to load target mesh from VTK format: " << e.what() << std::endl;
        }
    }

    // If VTK failed, try MSH for target
    if (!target_loaded) {
        std::ifstream in_msh_target(mesh_directory + "/target.msh");
        if (in_msh_target.good()) {
            try {
                grid_in_target.read_msh(in_msh_target);
                target_loaded = true;
                pcout << "Target mesh loaded from MSH format" << std::endl;
            } catch (const std::exception& e) {
                pcout << "Failed to load target mesh from MSH format: " << e.what() << std::endl;
            }
        }
    }

    if (!target_loaded) {
        throw std::runtime_error("Failed to load target mesh from either VTK or MSH format");
    }
}

template <int dim, int spacedim>
void MeshManager<dim, spacedim>::load_mesh_at_level(parallel::fullydistributed::Triangulation<dim, spacedim>& source_mesh,
                                        DoFHandler<dim, spacedim>& dof_handler_source,
                                        const std::string& mesh_file)
{
    pcout << "Attempting to load mesh from: " << mesh_file << std::endl;
    
    // Check if file exists
    if (!std::filesystem::exists(mesh_file)) {
        throw std::runtime_error("Mesh file does not exist: " + mesh_file);
    }

    // Check if file is readable and non-empty
    std::ifstream input(mesh_file);
    if (!input.good()) {
        throw std::runtime_error("Cannot open mesh file: " + mesh_file);
    }
    
    input.seekg(0, std::ios::end);
    if (input.tellg() == 0) {
        throw std::runtime_error("Mesh file is empty: " + mesh_file);
    }
    input.seekg(0, std::ios::beg);

    try {
        // First load into a serial triangulation
        Triangulation<dim, spacedim> serial_source;
        GridIn<dim, spacedim> grid_in;
        grid_in.attach_triangulation(serial_source);
        
        grid_in.read_msh(input);
        
        // Verify the mesh was loaded properly
        if (serial_source.n_active_cells() == 0) {
            throw std::runtime_error("Loaded mesh contains no cells");
        }
        
        pcout << "Successfully loaded serial mesh with "
              << serial_source.n_active_cells() << " cells and "
              << serial_source.n_vertices() << " vertices" << std::endl;
        
        // Partition the serial mesh using z-order curve
        GridTools::partition_triangulation_zorder(n_mpi_processes, serial_source);
        
        // Convert to fullydistributed triangulation
        auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(
            serial_source, mpi_communicator,
            TriangulationDescription::Settings::default_setting);
        
        // Clear old DoFHandler first
        dof_handler_source.clear();
        // Then clear and recreate triangulation
        source_mesh.clear();
        source_mesh.create_triangulation(construction_data);
        
        // Verify the distributed mesh
        const unsigned int n_global_active_cells = 
            Utilities::MPI::sum(source_mesh.n_locally_owned_active_cells(), mpi_communicator);
            
        if (n_global_active_cells == 0) {
            throw std::runtime_error("Distributed mesh contains no cells");
        }
        
        pcout << "Successfully created distributed mesh with "
              << n_global_active_cells << " total cells across "
              << n_mpi_processes << " processes" << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load mesh from " + mesh_file + 
                               "\nError: " + e.what());
    }
}

template <int dim, int spacedim>
void MeshManager<dim, spacedim>::save_meshes(const parallel::fullydistributed::Triangulation<dim, spacedim>& source_mesh,
                                 const Triangulation<dim, spacedim>& target_mesh)
{
    write_mesh(source_mesh,
              mesh_directory + "/source",
              std::vector<std::string>{"vtk", "msh"});

    write_mesh(target_mesh,
              mesh_directory + "/target",
              std::vector<std::string>{"vtk", "msh"});

    pcout << "Meshes saved in VTK and MSH formats" << std::endl;
}

template <int dim, int spacedim>
template <typename TriangulationType>
bool MeshManager<dim, spacedim>::write_mesh(const TriangulationType& mesh,
                                const std::string& filepath,
                                const std::vector<std::string>& formats,
                                const std::vector<double>* cell_data,
                                const std::string& data_name)
{
    try {
        std::filesystem::path path(filepath);
        std::filesystem::create_directories(path.parent_path());

        GridOut grid_out;
        
        for (const auto& format : formats) {
            if (format == "vtk") {
                std::ofstream out_vtk(filepath + ".vtk");
                if (!out_vtk.is_open()) {
                    pcout << "Error: Unable to open file for writing: " 
                          << filepath + ".vtk" << std::endl;
                    return false;
                }
                grid_out.write_vtk(mesh, out_vtk);
                pcout << "Mesh saved to: " << filepath + ".vtk" << std::endl;
            }
            else if (format == "msh") {
                // First write in default format
                std::ofstream out_msh(filepath + ".msh");
                if (!out_msh.is_open()) {
                    pcout << "Error: Unable to open file for writing: " 
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
                    pcout << "Error: Failed to convert mesh to MSH2 format" << std::endl;
                    return false;
                }
                pcout << "Mesh saved and converted to MSH2 format: " << filepath + ".msh" << std::endl;
            }
            else if (format == "vtu") {
                DataOut<dim, spacedim> data_out;
                data_out.attach_triangulation(mesh);
                
                // Add cell data if provided
                if (cell_data != nullptr) {
                    Assert(cell_data->size() == mesh.n_active_cells(),
                           ExcDimensionMismatch(cell_data->size(), 
                                              mesh.n_active_cells()));
                    
                    Vector<double> cell_data_vector(cell_data->begin(), cell_data->end());
                    data_out.add_data_vector(cell_data_vector, data_name);
                }
                
                data_out.build_patches();
                
                std::ofstream out_vtu(filepath + ".vtu");
                if (!out_vtu.is_open()) {
                    pcout << "Error: Unable to open file for writing: " 
                          << filepath + ".vtu" << std::endl;
                    return false;
                }
                data_out.write_vtu(out_vtu);
                pcout << "Mesh saved to: " << filepath + ".vtu" << std::endl;
            }
            else {
                pcout << "Warning: Unsupported format '" << format 
                      << "' requested" << std::endl;
            }
        }
        
        return true;
    }
    catch (const std::exception& e) {
        pcout << "Error writing mesh: " << e.what() << std::endl;
        return false;
    }
}

template <int dim, int spacedim>
std::vector<std::string> MeshManager<dim, spacedim>::get_mesh_hierarchy_files(const std::string& dir)
{
    std::vector<std::string> mesh_files;
    
    // List all .msh files in the hierarchy directory
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".msh") {
            mesh_files.push_back(entry.path().string());
        }
    }
    
    // Sort in reverse order (coarsest to finest)
    std::sort(mesh_files.begin(), mesh_files.end(), std::greater<std::string>());
    return mesh_files;
}

#endif // MESH_MANAGER_TEMPLATES_H 