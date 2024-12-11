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

        // Method 1: GridTools::volume()
        double volume1 = GridTools::volume(triangulation);

        // Method 2: Sum of cell measures
        double volume2 = 0.0;
        for (const auto& cell : triangulation.active_cell_iterators()) {
            volume2 += cell->measure();
        }

        // Method 3: Vector-based approach with DG0
        FE_DGQ<dim> fe_dg(0);
        DoFHandler<dim> dof_handler_dg(triangulation);
        dof_handler_dg.distribute_dofs(fe_dg);
        
        Vector<double> cell_measures(triangulation.n_active_cells());
        Vector<double> ones(triangulation.n_active_cells());
        {
            unsigned int cell_index = 0;
            for (const auto& cell : triangulation.active_cell_iterators()) {
                cell_measures[cell_index] = cell->measure();
                ones[cell_index] = 1.0;
                ++cell_index;
            }
        }
        double volume3 = cell_measures * ones;

        // Method 4: FE Integration with Q1 elements
        FE_Q<dim> fe_q(1);
        DoFHandler<dim> dof_handler_q(triangulation);
        dof_handler_q.distribute_dofs(fe_q);
        
        QGauss<dim> quadrature_formula(2);
        FEValues<dim> fe_values(fe_q,
                             quadrature_formula,
                             update_JxW_values);
        
        const unsigned int n_q_points = quadrature_formula.size();
        double volume4 = 0;
        
        for (const auto& cell : dof_handler_q.active_cell_iterators()) {
            fe_values.reinit(cell);
            for (unsigned int q = 0; q < n_q_points; ++q) {
                volume4 += 1.0 * fe_values.JxW(q);
            }
        }

        // Output results with high precision
        std::cout << std::setprecision(16) << std::scientific;
        std::cout << "Volume computation using different methods:" << std::endl;
        std::cout << "Method 1 (GridTools::volume):  " << volume1 << std::endl;
        std::cout << "Method 2 (sum of measures):    " << volume2 << std::endl;
        std::cout << "Method 3 (vector approach):    " << volume3 << std::endl;
        std::cout << "Method 4 (FE integration):     " << volume4 << std::endl;
        std::cout << "Exact value:                   " << (4.0/3.0 * M_PI) << std::endl;
        
        // Compare differences between methods
        std::cout << "\nDifferences between methods:" << std::endl;
        std::cout << "Methods 1-2: " << std::abs(volume1 - volume2) << std::endl;
        std::cout << "Methods 1-3: " << std::abs(volume1 - volume3) << std::endl;
        std::cout << "Methods 1-4: " << std::abs(volume1 - volume4) << std::endl;
        
    } catch (std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
    
    return 0;
}