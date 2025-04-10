#pragma once

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

#include <filesystem>
#include <fstream>
#include <iostream>

#include "PressureDensityField.h"

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

  template <int dim>
  class Mixed : ParameterAcceptor
  {
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    using VectorType = LA::MPI::BlockVector;
    using MatrixType = LA::MPI::BlockSparseMatrix;

  public:
    Mixed();
    void run();

  private:
    void setup_system(
        std::string &name,
        parallel::distributed::Triangulation<dim> &tria,
        DoFHandler<dim> &dof_handler,
        IndexSet &locally_owned_dofs,
        IndexSet &locally_relevant_dofs,
        std::vector<IndexSet> &block_owned_dofs,
        std::vector<IndexSet> &block_relevant_dofs,
        AffineConstraints<double> &constraints,
        std::string &grid_generator_function,
        std::string &grid_generator_arguments,
        unsigned int n_refinements,
        VectorType &solution,
        VectorType &locally_relevant_solution,
        VectorType &rhs,
        MatrixType &mat,
        LA::MPI::BlockSparseMatrix &preconditioner_matrix);
    
    void assemble_system(
        DoFHandler<dim> &dof_handler,
        AffineConstraints<double> &constraints,
        MatrixType &mat,
        MatrixType &pmat,
        VectorType &rhs,
        FunctionParser<dim> &forcing_term);
    
    void solve_system(
        MatrixType &mat,
        VectorType &rhs,
        VectorType &solution,
        VectorType &locally_relevant_solution,
        LA::MPI::BlockSparseMatrix &preconditioner_matrix,
        AffineConstraints<double> &constraints);
    
    void output_results(
        std::string &field_name,
        parallel::distributed::Triangulation<dim> &triangulation,
        DoFHandler<dim> &dof_handler,
        VectorType &locally_relevant_solution) const;
    
    void run_source();
    void run_target();

    MPI_Comm mpi_communicator;
    ConditionalOStream pcout;

    const std::string vtk_folder = "vtk";

    parallel::distributed::Triangulation<dim> triangulation_1;
    parallel::distributed::Triangulation<dim> triangulation_2;

    DoFHandler<dim> dof_handler_1;
    DoFHandler<dim> dof_handler_2;
    FESystem<dim> fe;

    MappingQ<dim> mapping;

    AffineConstraints<double> constraints_1;
    AffineConstraints<double> constraints_2;

    IndexSet locally_owned_dofs_1;
    IndexSet locally_relevant_dofs_1;
    std::vector<IndexSet> block_owned_dofs_1;
    std::vector<IndexSet> block_relevant_dofs_1;

    IndexSet locally_owned_dofs_2;
    IndexSet locally_relevant_dofs_2;
    std::vector<IndexSet> block_owned_dofs_2;
    std::vector<IndexSet> block_relevant_dofs_2;

    VectorType solution_1;
    VectorType locally_relevant_solution_1;
    VectorType rhs_1;
    MatrixType mat_1;
    MatrixType preconditioner_matrix_1;

    VectorType solution_2;
    VectorType locally_relevant_solution_2;
    VectorType rhs_2;
    MatrixType mat_2;
    MatrixType preconditioner_matrix_2;

    // parameters
    unsigned int n_refinements_1 = 0;
    std::string grid_generator_arguments_1 = "2: 0.5: 8";
    std::string grid_generator_function_1 = "torus";
    std::string forcing_term_expression_1 = "1";
    std::map<std::string, double> function_constants_1;
    FunctionParser<dim> forcing_term_1;

    unsigned int n_refinements_2 = 0;
    std::string grid_generator_arguments_2 = "2: 0.5: 8";
    std::string grid_generator_function_2 = "torus";
    std::string forcing_term_expression_2 = "1";
    std::map<std::string, double> function_constants_2;
    FunctionParser<dim> forcing_term_2;

    unsigned int max_iterations_outer = 1000;
    unsigned int max_iterations_inner = 1000;
    double tolerance_outer = 1e-8;
    double tolerance_inner = 1e-10;

    bool use_tet;

    std::unique_ptr<PressureDensityField<dim>> source_density_field;
    std::unique_ptr<PressureDensityField<dim>> target_density_field;

    void setup_density_fields();
    void compute_optimal_transport();
  };

  template <int dim>
  struct ScratchData
  {
    ScratchData(
        const Mapping<dim> &mapping,
        const FiniteElement<dim> &fe,
        const unsigned int quadrature_degree,
        const UpdateFlags update_flags = update_values |
                                         update_gradients |
                                         update_quadrature_points |
                                         update_JxW_values,
        const UpdateFlags interface_update_flags =
            update_values |
            update_gradients | update_quadrature_points |
            update_JxW_values |
            update_normal_vectors)
        : fe_values(
              mapping, fe, QGauss<dim>(quadrature_degree), update_flags),
          fe_interface_values(
              mapping, fe, QGauss<dim - 1>(quadrature_degree), interface_update_flags) {}

    ScratchData(const ScratchData<dim> &scratch_data)
        : fe_values(scratch_data.fe_values.get_mapping(),
                    scratch_data.fe_values.get_fe(),
                    scratch_data.fe_values.get_quadrature(),
                    scratch_data.fe_values.get_update_flags()),
          fe_interface_values(
              scratch_data.fe_values.get_mapping(),
              scratch_data.fe_values.get_fe(),
              scratch_data.fe_interface_values.get_quadrature(),
              scratch_data.fe_interface_values.get_update_flags())
    {
    }

    FEValues<dim> fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };

  struct CopyDataFace
  {
    FullMatrix<double> cell_matrix, cell_pmatrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  };

  struct CopyData
  {
    FullMatrix<double> cell_matrix, cell_pmatrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace> face_data;

    template <class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_pmatrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };
}