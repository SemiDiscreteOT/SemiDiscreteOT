#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/manifold_lib.h>
#include <iostream>
#include <iomanip>

using namespace dealii;

int main() {
    try {
        const int dim = 3;
        Triangulation<dim> triangulation;
        GridGenerator::hyper_ball(triangulation, Point<dim>(), 1.0);
        triangulation.set_all_manifold_ids_on_boundary(0);
        triangulation.set_manifold(0, SphericalManifold<dim>());
        triangulation.refine_global(3);  

        // Vector to store cell markers
        std::vector<bool> marked_cells;
        marked_cells.reserve(triangulation.n_active_cells());
        
        // Mark cells where x > 1
        for (const auto& cell : triangulation.active_cell_iterators()) {
            Point<dim> center = cell->center();
            marked_cells.push_back(center[0] > 0.0);
        }

        // Method 1: Sum of cell measures
        double volume1 = 0.0;
        {
            unsigned int cell_index = 0;
            for (const auto& cell : triangulation.active_cell_iterators()) {
                if (marked_cells[cell_index++]) {
                    volume1 += cell->measure();
                }
            }
        }

        // Method 2: Vector-based approach with DG0
        FE_DGQ<dim> fe_dg(0);
        DoFHandler<dim> dof_handler_dg(triangulation);
        dof_handler_dg.distribute_dofs(fe_dg);
        
        Vector<double> cell_measures(triangulation.n_active_cells());
        Vector<double> indicator(triangulation.n_active_cells());
        {
            unsigned int cell_index = 0;
            for (const auto& cell : triangulation.active_cell_iterators()) {
                cell_measures[cell_index] = cell->measure();
                indicator[cell_index] = marked_cells[cell_index] ? 1.0 : 0.0;
                ++cell_index;
            }
        }
        double volume2 = cell_measures * indicator;

        // Method 3: FE Integration with Q1 elements
        FE_Q<dim> fe_q(1);
        DoFHandler<dim> dof_handler_q(triangulation);
        dof_handler_q.distribute_dofs(fe_q);
        
        QGauss<dim> quadrature_formula(2);
        FEValues<dim> fe_values(fe_q,
                             quadrature_formula,
                             update_JxW_values | update_quadrature_points);
        
        const unsigned int n_q_points = quadrature_formula.size();
        double volume3 = 0;
        
        unsigned int cell_index = 0;
        for (const auto& cell : dof_handler_q.active_cell_iterators()) {
            if (marked_cells[cell_index++]) {
                fe_values.reinit(cell);
                for (unsigned int q = 0; q < n_q_points; ++q) {
                    volume3 += 1.0 * fe_values.JxW(q);
                }
            }
        }

        // Output results with high precision
        std::cout << std::setprecision(16) << std::scientific;
        std::cout << "Volume of marked cells (x > 1) using different methods:" << std::endl;
        std::cout << "Method 1 (sum of measures):    " << volume1 << std::endl;
        std::cout << "Method 2 (vector approach):    " << volume2 << std::endl;
        std::cout << "Method 3 (FE integration):     " << volume3 << std::endl;
        
        // Compare differences between methods
        std::cout << "\nDifferences between methods:" << std::endl;
        std::cout << "Methods 1-2: " << std::abs(volume1 - volume2) << std::endl;
        std::cout << "Methods 1-3: " << std::abs(volume1 - volume3) << std::endl;
        std::cout << "Methods 2-3: " << std::abs(volume2 - volume3) << std::endl;

        // Also output how many cells were marked
        unsigned int n_marked = std::count(marked_cells.begin(), marked_cells.end(), true);
        std::cout << "\nNumber of marked cells: " << n_marked 
                  << " out of " << triangulation.n_active_cells() << std::endl;
        
    } catch (std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
    
    return 0;
}