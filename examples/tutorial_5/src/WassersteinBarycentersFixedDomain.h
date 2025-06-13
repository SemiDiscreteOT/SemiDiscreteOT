#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <SemiDiscreteOT/core/WassersteinBarycenters.h>
#include <SemiDiscreteOT/solvers/Distance.h>

namespace fs = std::filesystem;

namespace LA
{
#if defined(DEAL_II_WITH_PETSC)
  using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
#else
#error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

namespace Applications
{
  using namespace dealii;

  class WassersteinBarycentersFixedDomain : public ParameterAcceptor, public WassersteinBarycenters<2, 3>
  {
    using Iterator = typename DoFHandler<2, 3>::active_cell_iterator;
    using VectorType = LA::MPI::Vector;
    using MatrixType = LA::MPI::SparseMatrix;

  public:
    WassersteinBarycentersFixedDomain(
      const unsigned int n_measures,
      const std::vector<double> weights,
      const MPI_Comm &comm
    );
    void run();

  private:
    void setup_system();
    void output_results() const;
    
    MPI_Comm mpi_communicator;
    ConditionalOStream pcout;

    const std::string vtk_folder = "vtk";
    const std::string output_dir = "output/density_field/";

    parallel::distributed::Triangulation<2, 3> tria;

    DoFHandler<2, 3> dof_handler;
    FE_Q<2, 3> fe;
    MappingQ<2, 3> mapping;

    AffineConstraints<double> constraints;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    VectorType source_1;
    VectorType source_2;

    unsigned int n_refinements;
  };

}