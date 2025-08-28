#include <SemiDiscreteOT/core/SemiDiscreteOT.h>
#include <SemiDiscreteOT/utils/utils.h>
#include <SemiDiscreteOT/core/WassersteinBarycenters.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <iomanip>

using namespace dealii;

class SphericalGaussian : public Function<3>
{
public:
  SphericalGaussian(const Point<3> &center,
                    const double sigma)
    : Function<3>(), center(center), sigma(sigma)
  {
    // Ensure center is normalized
    Assert(std::abs(center.norm() - 1.0) < 1e-10,
          ExcMessage("Center must be on the unit sphere."));
  }

  virtual double value(const Point<3> &p,
                      const unsigned int = 0) const override
  {
    const double norm_p = p.norm();
    if (norm_p == 0.0)
      return 0.0;

    // Project to the sphere if needed
    Point<3> x = p / norm_p;

    // Compute geodesic distance via arccos(dot product)
    const double dot = center * x;
    const double theta = std::acos(std::min(1.0, std::max(-1.0, dot)));

    return std::exp(- (theta * theta) / (2.0 * sigma * sigma));
  }

  Point<3> center;
  double sigma;
};

void load_source_meshes_and_set_target(
  WassersteinBarycenters<2, 3, UpdateMode::TargetMeasureOnly> &wb)
{
  Triangulation<2, 3> source_tria;
  {
    GridIn<2, 3> grid_in;
    grid_in.attach_triangulation(source_tria);
    std::ifstream input_file(wb.file_params.source_filenames[0]);
    if (!input_file.is_open()) {
      wb.pcout << "Error: Could not open source file " << 0 << ": " << wb.file_params.source_filenames[0] << std::endl;
    }
    grid_in.read_msh(input_file);

    if (wb.barycenter_params.volume_scaling) {
      const double volume = GridTools::volume(source_tria);
      if (volume > 1e-12) {
        GridTools::scale(1.0 / std::cbrt(volume), source_tria);
        wb.pcout << "Rescaled source " << 1 << " to unit volume." << std::endl;
      }
    }
  }

  auto [source_fe, map] = Utils::create_fe_and_mapping_for_mesh<2, 3>(source_tria);

  DoFHandler<2, 3> source_dof_handler(source_tria);
  source_dof_handler.distribute_dofs(*source_fe);
  wb.sot_problems[0]->setup_source_mesh(source_tria, "source_" + std::to_string(0 + 1));
  
  // setup barycenters support points (fixed for this example)
  std::map<types::global_dof_index, Point<3>> support_points; //locally_relevant support_points
  DoFTools::map_dofs_to_support_points(
    *map, wb.sot_problems[0]->dof_handler_source, support_points);

  std::vector<Point<3>> barycenter_points_local;
  barycenter_points_local.resize(wb.sot_problems[0]->source_fine_loc_owned_dofs.n_elements());
  for(unsigned int i=0; i < wb.sot_problems[0]->source_fine_loc_owned_dofs.n_elements(); ++i)
    barycenter_points_local[i] = support_points[wb.sot_problems[0]->source_fine_loc_owned_dofs.nth_index_in_set(i)];

  // Gather all support points
  auto all_barycenter_points = Utilities::MPI::all_gather(
    wb.mpi_comm, barycenter_points_local);
  wb.barycenter_points.clear();
  // locally_owned dofs have contiguous indices for increasing ranks order
  for (const auto &local_points : all_barycenter_points)
    wb.barycenter_points.insert(
      wb.barycenter_points.end(), local_points.begin(), local_points.end());

  // source_1
  Vector<double> source_density(
    wb.sot_problems[0]->dof_handler_source.n_dofs());
  Point<3> center_on_sphere(1.0, 0.0, 0.0);
  double sigma = 0.3;
  SphericalGaussian gaussian(center_on_sphere, sigma);
  for (const auto &[key, value] : support_points)
    source_density[key] = gaussian.value(value);

  wb.sot_problems[0]->setup_source_measure(source_density);
  
  for (unsigned int i = 1; i < wb.sot_problems.size(); ++i)
  {
    Triangulation<2, 3> source_tria;
    {
      GridIn<2, 3> grid_in;
      grid_in.attach_triangulation(source_tria);
      std::ifstream input_file(wb.file_params.source_filenames[i]);
      if (!input_file.is_open()) {
        wb.pcout << "Error: Could not open source file " << i << ": " << wb.file_params.source_filenames[i] << std::endl;
        continue;
      }
      grid_in.read_msh(input_file);

      if (wb.barycenter_params.volume_scaling) {
        const double volume = GridTools::volume(source_tria);
        if (volume > 1e-12) {
          GridTools::scale(1.0 / std::cbrt(volume), source_tria);
          wb.pcout << "Rescaled source " << i + 1 << " to unit volume." << std::endl;
        }
      }
    }

    auto [source_fe, map] = Utils::create_fe_and_mapping_for_mesh<2, 3>(source_tria);

    source_dof_handler.reinit(source_tria);
    source_dof_handler.distribute_dofs(*source_fe);
    wb.sot_problems[i]->setup_source_mesh(source_tria, "source_" + std::to_string(i + 1));
    
    center_on_sphere = Point<3>(-1.0, 0.0, 0.0);
    SphericalGaussian gaussian_(center_on_sphere, sigma);
    
    std::map<types::global_dof_index, Point<3>> support_points_;
    DoFTools::map_dofs_to_support_points(*map, wb.sot_problems[i]->dof_handler_source, support_points_);

    source_density.reinit(wb.sot_problems[i]->dof_handler_source.n_dofs());
    for (const auto &[key, value] : support_points)
      source_density[key] = gaussian_.value(value);
    
    wb.sot_problems[i]->setup_source_measure(source_density);

    if (Utilities::MPI::this_mpi_process(wb.mpi_comm) == 0)
    {
      wb.save_vtk_output(wb.barycenter_points, source_density, "source_tutorial_5.vtk");
    }
  }
}

int main(int argc, char *argv[])
{
  try
  {    
    deallog.depth_console(0);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    const unsigned int this_mpi_process = Utilities::MPI::this_mpi_process(comm);

    // Create conditional output stream
    ConditionalOStream pcout(std::cout, this_mpi_process == 0);
    pcout << "=== Wasserstein Barycenter Tutorial 5 === " << std::endl;
    
    if (Utilities::MPI::this_mpi_process(comm) == 0 && !std::ifstream("sphere_tria.msh").good() && !std::ifstream("sphere_quad.msh").good())
    {
      Triangulation<2, 3> quad_tria;
      GridGenerator::hyper_sphere(quad_tria);
      quad_tria.refine_global(4);
      
      std::ofstream output_file_quad("sphere_quad.msh");
      GridOut grid_out_quad;
      grid_out_quad.write_msh(quad_tria, output_file_quad);
      std::cout << "Saved tria sphere_quad.msh" << std::endl;

      Triangulation<2, 3> tri_mesh;
      GridGenerator::convert_hypercube_to_simplex_mesh(quad_tria, tri_mesh);

      std::ofstream output_file_tria("sphere_tria.msh");
      GridOut grid_out_tria;
      grid_out_tria.write_msh(tri_mesh, output_file_tria);
      std::cout << "Saved tria sphere_tria.msh" << std::endl;

      std::cout << "Mesh have been saved, re-run\n";
      return 0;
    }

    WassersteinBarycenters<2, 3, UpdateMode::TargetMeasureOnly> wb(comm);
    std::string prm_file = (argc > 1) ? argv[1] : "parameters.prm";
    pcout << "Using parameter file: " << prm_file << std::endl;
    ParameterAcceptor::initialize(prm_file);
    
    wb.configure();
    AssertThrow(wb.n_measures == 2, ExcMessage("Expected wb.n_measures to be 2."));

    load_source_meshes_and_set_target(wb);

    // Init target with uniform distribution
    wb.barycenter_weights.reinit(wb.barycenter_points.size());
    Point<3> center_on_sphere(1.0/std::sqrt(2), 1.0/std::sqrt(2), 0.0);
    double sigma = 0.3;
    SphericalGaussian gaussian(center_on_sphere, sigma);
    double sum = 0.0;
    for (unsigned int i = 0; i < wb.barycenter_weights.size(); ++i)
    {
      wb.barycenter_weights[i] = gaussian.value(wb.barycenter_points[i]);
      sum += wb.barycenter_weights[i];
    }
    wb.barycenter_weights = 1.0 / sum; //wb.barycenter_points.size();
    pcout << "Initialized barycenter with " << wb.barycenter_points.size() << " points." << std::endl;

    if (Utilities::MPI::this_mpi_process(wb.mpi_comm) == 0)
    {
      wb.save_vtk_output(wb.barycenter_points, wb.barycenter_weights, "barycenter_init.vtk");
    }

    wb.run_wasserstein_barycenters();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
