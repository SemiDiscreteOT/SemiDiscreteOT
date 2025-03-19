#pragma once

#include <deal.II/base/smartpointer.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>

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
  class PressureDensityField
  {
  public:
    PressureDensityField(const DoFHandler<dim>& dof_handler,
                         const LA::MPI::BlockVector& solution,
                         const MPI_Comm& mpi_communicator);

    void extract_pressure();
    void normalize();
    void output_density(const std::string& filename) const;

    const Vector<double>& get_density() const { return normalized_density; }
    const Vector<double>& get_raw_pressure() const { return pressure_field; }

  private:
    SmartPointer<const DoFHandler<dim>> dof_handler;
    SmartPointer<const LA::MPI::BlockVector> solution;
    const MPI_Comm& mpi_communicator;

    Vector<double> pressure_field;
    Vector<double> normalized_density;

    void compute_normalization_factor();
    void ensure_positivity();
  };

} // namespace Applications 