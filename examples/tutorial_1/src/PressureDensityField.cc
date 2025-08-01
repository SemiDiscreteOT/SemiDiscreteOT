#include "PressureDensityField.h"

namespace Applications
{
  template <int dim>
  PressureDensityField<dim>::PressureDensityField(
      const DoFHandler<dim>& dof_handler_in,
      const LA::MPI::BlockVector& solution_in,
      const MPI_Comm& mpi_communicator_in)
      : dof_handler(&dof_handler_in)
      , solution(&solution_in)
      , mpi_communicator(mpi_communicator_in)
  {
  }

    template <int dim>
  void PressureDensityField<dim>::extract_pressure()
  {
    // We only need n_p since we're only extracting the pressure component
    const unsigned int n_p = solution->block(1).size();
    pressure_field.reinit(n_p);
    for (unsigned int i = 0; i < n_p; ++i)
      pressure_field[i] = (*solution).block(1)[i];
  }


  template <int dim>
  void PressureDensityField<dim>::normalize()
  {
    compute_normalization_factor();
    ensure_positivity();
  }

  template <int dim>
  void PressureDensityField<dim>::compute_normalization_factor()
  {
    const FESystem<dim>& fe = dof_handler->get_fe();
    
    // Create FEValues for integration - only for the pressure component
    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, 
                           quadrature_formula,
                           update_values | update_JxW_values | 
                           update_quadrature_points);

    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<double> pressure_values(n_q_points);
    
    // Extract pressure component (component dim for mixed system)
    const FEValuesExtractors::Scalar pressure(dim);
    
    double local_integral = 0.0;
    double global_integral = 0.0;

    // Compute the integral of the pressure field
    for (const auto& cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        // Use the full solution vector and extract pressure component
        fe_values[pressure].get_function_values(*solution, pressure_values);
        
        for (unsigned int q = 0; q < n_q_points; ++q)
          local_integral += pressure_values[q] * fe_values.JxW(q);
      }

    // Sum up the contributions from all processors
    global_integral = Utilities::MPI::sum(local_integral, mpi_communicator);

    // Normalize the density field
    normalized_density = pressure_field;
    if (std::abs(global_integral) > 1e-15)
      normalized_density *= 1.0 / global_integral;
  }

  template <int dim>
  void PressureDensityField<dim>::ensure_positivity()
  {
    // Find the minimum value
    double local_min = std::numeric_limits<double>::max();
    for (const auto& val : normalized_density)
      local_min = std::min(local_min, val);
    
    const double global_min = Utilities::MPI::min(local_min, mpi_communicator);

    // Shift values to ensure positivity if necessary
    if (global_min < 0)
    {
      const double shift = -global_min + 1e-10;  // Small buffer to ensure strictly positive
      for (auto& val : normalized_density)
        val += shift;
      
      // Renormalize after shifting
      compute_normalization_factor();
    }
  }

  template <int dim>
  void PressureDensityField<dim>::output_density(const std::string& filename) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(*dof_handler);
    
    // Add the normalized density to the output
    data_out.add_data_vector(normalized_density, "normalized_density");
    
    data_out.build_patches();
    
    // Write to VTK file
    std::ofstream output(filename);
    data_out.write_vtk(output);
  }

  // Explicit instantiation
  template class PressureDensityField<2>;
  template class PressureDensityField<3>;

} // namespace Applications 