#include "SoftmaxRefinement.h"
#include <deal.II/base/quadrature_lib.h>

template <int dim>
SoftmaxRefinement<dim>::SoftmaxRefinement(
    MPI_Comm mpi_comm,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const FiniteElement<dim>& fe,
    const LinearAlgebra::distributed::Vector<double>& source_density,
    unsigned int quadrature_order,
    double distance_threshold)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout, this_mpi_process == 0)
    , dof_handler(dof_handler)
    , mapping(mapping)
    , fe(fe)
    , source_density(source_density)
    , quadrature_order(quadrature_order)
    , current_distance_threshold(distance_threshold)
{}

template <int dim>
void SoftmaxRefinement<dim>::setup_rtrees()
{
    namespace bgi = boost::geometry::index;
    
    // Setup R-tree for coarse points
    std::vector<IndexedPoint> indexed_coarse_points;
    indexed_coarse_points.reserve(current_target_points_coarse->size());
    pcout << "Setting up R-tree for coarse points" << std::endl;
    pcout << "Number of coarse points: " << current_target_points_coarse->size() << std::endl;
    
    for (std::size_t i = 0; i < current_target_points_coarse->size(); ++i) {
        indexed_coarse_points.emplace_back((*current_target_points_coarse)[i], i);
    }
    
    coarse_points_rtree = RTree(indexed_coarse_points.begin(), indexed_coarse_points.end());
    
    // Setup R-tree for fine points
    std::vector<IndexedPoint> indexed_fine_points;
    indexed_fine_points.reserve(current_target_points_fine->size());
    pcout << "Setting up R-tree for fine points" << std::endl;
    pcout << "Number of fine points: " << current_target_points_fine->size() << std::endl;
    
    for (std::size_t i = 0; i < current_target_points_fine->size(); ++i) {
        indexed_fine_points.emplace_back((*current_target_points_fine)[i], i);
    }
    
    fine_points_rtree = RTree(indexed_fine_points.begin(), indexed_fine_points.end());
    // print also the first value of density for each density please
    pcout << "First value of density for coarse points: " << (*current_target_density_coarse)[0] << std::endl;
    pcout << "First value of density for fine points: " << (*current_target_density_fine)[0] << std::endl;
    // print also the first coordinates of each point please
    pcout << "First coordinates of coarse points: " << (*current_target_points_coarse)[0] << std::endl;
    pcout << "First coordinates of fine points: " << (*current_target_points_fine)[0] << std::endl;
}

template <int dim>
std::vector<std::size_t> SoftmaxRefinement<dim>::find_nearest_coarse_points(
    const Point<dim>& query_point) const
{
    namespace bgi = boost::geometry::index;
    std::vector<std::size_t> indices;

    for (const auto& indexed_point : coarse_points_rtree |
         bgi::adaptors::queried(bgi::satisfies([&](const IndexedPoint& p) {
             return (p.first - query_point).norm() <= current_distance_threshold;
         })))
    {
        indices.push_back(indexed_point.second);
    }

    return indices;
}

template <int dim>
std::vector<std::size_t> SoftmaxRefinement<dim>::find_nearest_fine_points(
    const Point<dim>& query_point) const
{
    namespace bgi = boost::geometry::index;
    std::vector<std::size_t> indices;

    for (const auto& indexed_point : fine_points_rtree |
         bgi::adaptors::queried(bgi::satisfies([&](const IndexedPoint& p) {
             return (p.first - query_point).norm() <= current_distance_threshold;
         })))
    {
        indices.push_back(indexed_point.second);
    }

    return indices;
}

template <int dim>
void SoftmaxRefinement<dim>::local_assemble(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &scratch_data,
    CopyData &copy_data)
{
    if (!cell->is_locally_owned())
        return;

    scratch_data.fe_values.reinit(cell);
    const std::vector<Point<dim>> &q_points = scratch_data.fe_values.get_quadrature_points();
    scratch_data.fe_values.get_function_values(source_density, scratch_data.density_values);

    copy_data.weight_values = 0;

    const unsigned int n_q_points = q_points.size();
    const double lambda_inv = 1.0 / current_lambda;
    const double threshold_sq = current_distance_threshold * current_distance_threshold;

    // Get relevant coarse target points for this cell
    std::vector<std::size_t> cell_target_indices_coarse = find_nearest_coarse_points(cell->center());
    
    if (cell_target_indices_coarse.empty()) return;

    const unsigned int n_target_points_coarse = cell_target_indices_coarse.size();
    std::vector<Point<dim>> target_positions_coarse(n_target_points_coarse);
    std::vector<double> target_densities_coarse(n_target_points_coarse);
    std::vector<double> weight_values_coarse(n_target_points_coarse);
    
    // Load coarse target point data
    for (size_t i = 0; i < n_target_points_coarse; ++i) {
        const size_t idx = cell_target_indices_coarse[i];
        target_positions_coarse[i] = (*current_target_points_coarse)[idx];
        target_densities_coarse[i] = (*current_target_density_coarse)[idx];
        weight_values_coarse[i] = (*current_weights_coarse)[idx];
    }

    // Get fine points near this cell
    std::vector<std::size_t> cell_target_indices_fine = find_nearest_fine_points(cell->center());
    
    if (cell_target_indices_fine.empty()) return;
    
    const unsigned int n_target_points_fine = cell_target_indices_fine.size();
    std::vector<Point<dim>> target_positions_fine(n_target_points_fine);
    
    // Load fine target point positions
    for (size_t i = 0; i < n_target_points_fine; ++i) {
        const size_t idx = cell_target_indices_fine[i];
        target_positions_fine[i] = (*current_target_points_fine)[idx];
    }

    // For each quadrature point
    for (unsigned int q = 0; q < n_q_points; ++q) {
        const Point<dim> &x = q_points[q];
        const double density_value = scratch_data.density_values[q];
        const double JxW = scratch_data.fe_values.JxW(q);
        
        // First compute normalization using coarse points
        double total_sum_exp = 0.0;
        std::vector<double> exp_terms_coarse(n_target_points_coarse);

        #pragma omp simd reduction(+:total_sum_exp)
        for (size_t i = 0; i < n_target_points_coarse; ++i) {
            const double local_dist2 = (x - target_positions_coarse[i]).norm_square();
            if (local_dist2 <= threshold_sq) {
                exp_terms_coarse[i] = target_densities_coarse[i] * 
                    std::exp((weight_values_coarse[i] - 0.5 * local_dist2) * lambda_inv);
                total_sum_exp += exp_terms_coarse[i];
            }
        }

        if (total_sum_exp <= 0.0) continue;

        // Now update weights for fine points using the coarse points' exp term for normalization
        const double scale = density_value * JxW / total_sum_exp;
        
        #pragma omp simd
        for (size_t i = 0; i < n_target_points_fine; ++i) {
            const double local_dist2_fine = (x - target_positions_fine[i]).norm_square();
            if (local_dist2_fine <= threshold_sq) {
                const double exp_term_fine = std::exp((- 0.5 * local_dist2_fine) * lambda_inv);
                copy_data.weight_values[cell_target_indices_fine[i]] += scale * exp_term_fine;
            }
        }
    }
}

template <int dim>
Vector<double> SoftmaxRefinement<dim>::compute_refinement(
    const std::vector<Point<dim>>& target_points_fine,
    const Vector<double>& target_density_fine,
    const std::vector<Point<dim>>& target_points_coarse,
    const Vector<double>& target_density_coarse,
    const Vector<double>& weights_coarse,
    double regularization_param)
{
    // Store computation parameters
    current_target_points_fine = &target_points_fine;
    current_target_density_fine = &target_density_fine;
    current_target_points_coarse = &target_points_coarse;
    current_target_density_coarse = &target_density_coarse;
    current_weights_coarse = &weights_coarse;
    current_lambda = regularization_param;

    // Initialize RTrees for spatial queries
    setup_rtrees();

    // Initialize output weights
    Vector<double> weights_fine(target_points_fine.size());
    Vector<double> local_process_weights(target_points_fine.size());

    // Create appropriate quadrature
    std::unique_ptr<Quadrature<dim>> quadrature;
    const bool use_simplex = (dynamic_cast<const FE_SimplexP<dim>*>(&fe) != nullptr);
    if (use_simplex) {
        quadrature = std::make_unique<QGaussSimplex<dim>>(quadrature_order);
    } else {
        quadrature = std::make_unique<QGauss<dim>>(quadrature_order);
    }

    // Create scratch and copy data objects
    ScratchData scratch_data(fe, mapping, *quadrature);
    CopyData copy_data(target_points_fine.size());

    // Create filtered iterator for locally owned cells
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
        begin_filtered(IteratorFilters::LocallyOwnedCell(),
                      dof_handler.begin_active()),
        end_filtered(IteratorFilters::LocallyOwnedCell(),
                    dof_handler.end());

    // Parallel assembly
    WorkStream::run(
        begin_filtered,
        end_filtered,
        [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
               ScratchData &scratch_data,
               CopyData &copy_data) {
            this->local_assemble(cell, scratch_data, copy_data);
        },
        [&local_process_weights](const CopyData &copy_data) {
            local_process_weights += copy_data.weight_values;
        },
        scratch_data,
        copy_data);

    // Sum up contributions across all MPI processes
    weights_fine = 0;
    Utilities::MPI::sum(local_process_weights, mpi_communicator, weights_fine);

    // Apply epsilon scaling to weights
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        for (unsigned int i = 0; i < target_points_fine.size(); ++i) {
            if (weights_fine[i] > 0.0) {
                weights_fine[i] = -regularization_param * std::log(weights_fine[i]);
            }
        }
    }

    // Broadcast final weights to all processes
    weights_fine = Utilities::MPI::broadcast(mpi_communicator, weights_fine, 0);

    return weights_fine;
}

// Explicit instantiation
template class SoftmaxRefinement<2>;
template class SoftmaxRefinement<3>; 