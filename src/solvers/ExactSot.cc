#include "SemiDiscreteOT/solvers/ExactSot.h"
#include "SemiDiscreteOT/utils/utils.h"

ExactSot::ExactSot() 
    : source_mesh(std::make_unique<GEO::Mesh>()),
      max_iterations_(1000),
      epsilon_(0.01)
{
    // Initialize Geogram
    GEO::initialize();
    using namespace GEO;

    CmdLine::import_arg_group("standard");
    CmdLine::import_arg_group("algo");
    CmdLine::import_arg_group("opt");
    CmdLine::declare_arg("nb_iter", 1000, "number of iterations for OTM");
 
    CmdLine::declare_arg_group(
        "RVD", "RVD output options", CmdLine::ARG_ADVANCED);
    CmdLine::declare_arg("RVD", false, "save restricted Voronoi diagram");
    CmdLine::declare_arg(
        "RVD_iter", false, "save restricted Voronoi diagram at each iteration");
    CmdLine::declare_arg(
        "RVD:borders_only", false, "save only border of RVD");
    CmdLine::declare_arg(
        "RVD:integration_simplices", true, "export RVD as integration simplices");

    CmdLine::declare_arg("multilevel", true, "use multilevel algorithm");
    CmdLine::declare_arg("BRIO", true,
                         "use BRIO reordering to compute the levels");
    CmdLine::declare_arg("ratio", 0.125, "ratio between levels");
    CmdLine::declare_arg("epsilon", 0.01, "relative measure error in a cell");
    CmdLine::declare_arg(
        "lock", true, "Lock lower levels when sampling shape");
    CmdLine::declare_arg(
        "fitting_degree", 2, "degree for interpolating weights");
    CmdLine::declare_arg(
        "project", true, "project sampling on border");
    CmdLine::declare_arg(
        "feature_sensitive", true, "attempt to recover hard edges");
    CmdLine::declare_arg(
        "singular", false, "compute and save singular surface");
    CmdLine::set_arg("algo:delaunay", "BPOW");
    CmdLine::declare_arg(
        "recenter", true, "recenter target onto source mesh");
    CmdLine::declare_arg(
        "rescale", true, "rescale target to match source volume");
}

ExactSot::~ExactSot() = default;

bool ExactSot::load_volume_mesh(const std::string& filename, GEO::Mesh& mesh) {
    GEO::MeshIOFlags flags;
    flags.set_element(GEO::MESH_CELLS);
    flags.set_attribute(GEO::MESH_CELL_REGION);

    std::cout << "Loading mesh from " << filename << std::endl;
    if (!GEO::mesh_load(filename, mesh, flags)) {
        return false;
    }
    
    if (!mesh.cells.are_simplices()) {
        std::cerr << "File " << filename 
                 << " should only have tetrahedra" << std::endl;
        return false;
    }
    
    if (mesh.cells.nb() == 0) {
        std::cout << "File " << filename 
                 << " does not contain a volume" << std::endl;
        std::cout << "Trying to tetrahedralize..." << std::endl;
        if (!GEO::mesh_tetrahedralize(mesh, true, false)) {
            return false;
        }
    }
    return true;
}

bool ExactSot::set_source_mesh(const std::string& filename) {
    return load_volume_mesh(filename, *source_mesh);
}

bool ExactSot::set_target_points(const std::string& filename, const std::string& io_coding) {
    return Utils::read_vector(target_points, filename, io_coding);
}

void ExactSot::set_parameters(unsigned int max_iterations,
                            double epsilon) {
    max_iterations_ = max_iterations;
    epsilon_ = epsilon;
}

bool ExactSot::run() {
    try {
        if (target_points.empty()) {
            std::cerr << "No target points loaded. Please load target points before running." << std::endl;
            return false;
        }

        std::cout << "Setting up optimal transport computation..." << std::endl;
        
        // Set density on source mesh (uniform density for now)
        GEO::Attribute<double> density(source_mesh->vertices.attributes(), "density");
        for(GEO::index_t v=0; v < source_mesh->vertices.nb(); ++v) {
            density[v] = 1.0;
        }
        
        // Everything happens in dimension 4 (power diagram is seen
        // as Voronoi diagram in dimension 4)
        source_mesh->vertices.set_dimension(4);
        std::cout << "Source mesh vertices dimension: " << source_mesh->vertices.dimension() << std::endl;
        
        // Create and setup OTM
        GEO::OptimalTransportMap3d OTM(source_mesh.get());
        std::cout << "Optimal transport map created" << std::endl;
        
        // Convert target points to Geogram format
        std::vector<double> target_points_data;
        target_points_data.reserve(target_points.size() * 3);
        for (const auto& point : target_points) {
            target_points_data.push_back(point[0]);
            target_points_data.push_back(point[1]);
            target_points_data.push_back(point[2]);
        }
        
        OTM.set_points(
            target_points.size(),
            target_points_data.data()
        );
        
        OTM.set_epsilon(epsilon_);
        
        std::cout << "Running optimization with " << max_iterations_ 
                 << " iterations and epsilon = " << epsilon_ << std::endl;
        
        // Run optimization
        OTM.optimize(max_iterations_);
        
        // Store results
        potential.resize(OTM.nb_points());
        for (GEO::index_t i = 0; i < OTM.nb_points(); ++i) {
            potential[i] = OTM.weight(i);
        }
        
        std::cout << "Optimization completed successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during computation: " << e.what() << std::endl;
        return false;
    }
}

std::vector<double> ExactSot::get_potential() const {
    return std::vector<double>(potential.begin(), potential.end());
}

std::vector<dealii::Point<3>> ExactSot::get_target_points() const {
    return target_points;
}

bool ExactSot::save_results(const std::string& potential_file,
                          const std::string& points_file,
                          const std::string& io_coding) const {
    try {
        std::cout << "Saving potential to " << potential_file << std::endl;
        // Save potential using Utils
        std::vector<double> potential_vec(potential.begin(), potential.end());
        Utils::write_vector(potential_vec, potential_file, io_coding);
        
        std::cout << "Saving points to " << points_file << std::endl;
        // Save points directly using Utils
        Utils::write_vector(target_points, points_file, io_coding);
 
        std::cout << "Results saved successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << std::endl;
        return false;
    }
}
