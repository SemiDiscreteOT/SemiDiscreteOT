#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/lac/vector.h>

#include <SemiDiscreteOT/core/SemiDiscreteOT.h>

#include <map>

using namespace dealii;

int main(int argc, char *argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0);

    pcout << "========================================================" << std::endl;
    pcout << "      SemiDiscreteOT: Tutorial 0      " << std::endl;
    pcout << "========================================================" << std::endl;

    const int dim = 3;
    const int spacedim = 3;

    pcout << "\n--- User: Preparing source data ---" << std::endl;

    Triangulation<dim,spacedim> tria;
    GridIn<dim,spacedim> grid_in;
    grid_in.attach_triangulation(tria);

    try {
        std::ifstream input_file("source.msh");
        if (!input_file.is_open()) {
            pcout << "Warning: Could not open mesh file. Using a simple generated mesh instead." << std::endl;
            GridGenerator::hyper_cube(tria, 0, 1);
            tria.refine_global(2);
        } else {
            grid_in.read_msh(input_file);
            pcout << "Mesh loaded from file: source.msh" << std::endl;
        }
    } catch (const std::exception& e) {
        pcout << "Warning: Error loading mesh file: " << e.what() << std::endl;
        pcout << "Using a simple generated mesh instead." << std::endl;
        GridGenerator::hyper_cube(tria, 0, 1);
        tria.refine_global(2);
    }

    pcout << "Triangulation created with " << tria.n_active_cells() << " cells" << std::endl;

    FE_SimplexP<dim,spacedim> fe(1);
    DoFHandler<dim,spacedim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    Vector<double> density_vector(dof_handler.n_dofs());

    MappingFE<dim,spacedim> mapping(FE_SimplexP<dim,spacedim>(1));

    std::map<types::global_dof_index, Point<spacedim>> support_points;
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    for (const auto& [dof_index, point] : support_points)
    {
        const Point<spacedim> &p = point;
        density_vector = 1.0;
    }

    pcout << "Density vector computed with " << dof_handler.n_dofs() << " DoFs" << std::endl;
    std::vector<Point<spacedim>> target_points;
    if (!Utils::read_vector(target_points, "target_points.txt"))
    {
        pcout << "Warning: Could not read target points from file. Using default points instead." << std::endl;
        target_points = {
            Point<spacedim>(0.1, 0.1, 0.8),
            Point<spacedim>(0.8, 0.1, -0.1),
            Point<spacedim>(0.1, 0.8, -0.8)
        };
    }

    Vector<double> target_weights(target_points.size());
    target_weights = 1.0;

    pcout << "\n--- Setting up SemiDiscreteOT problem ---" << std::endl;

    SemiDiscreteOT<dim,spacedim> ot_problem(mpi_communicator);

    ot_problem.configure([&](SotParameterManager &params) {
        params.multilevel_params.source_enabled = true;
        params.multilevel_params.target_enabled = false;
        params.multilevel_params.source_min_vertices = 100;
        params.multilevel_params.source_max_vertices = 10000;
        params.solver_params.epsilon = 0.01;
        params.solver_params.verbose_output = false;
        params.solver_params.tau = 1e-12;
        params.solver_params.distance_threshold_type = "pointwise";
        params.solver_params.use_log_sum_exp_trick = false;
        params.solver_params.tolerance = 1e-2;
    });

    ot_problem.setup_source_measure(tria, dof_handler, density_vector);
    ot_problem.setup_target_measure(target_points, target_weights);
    ot_problem.prepare_multilevel_hierarchies();
    Vector<double> potentials = ot_problem.solve();

    pcout << "\n--- Tutorial completed successfully ---" << std::endl;
    pcout << "Computed " << potentials.size() << " optimal transport potentials" << std::endl;

  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << "Exception on processing: " << exc.what() << std::endl;
    return 1;
  }
  catch(...)
  {
      std::cerr << std::endl << "Unknown exception!" << std::endl;
      return 1;
  }
  return 0;
}
