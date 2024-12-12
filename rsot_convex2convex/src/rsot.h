#ifndef RSOT_H
#define RSOT_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/optimization/solver_bfgs.h>

#include <filesystem>

using namespace dealii;

template <int dim>
class Convex2Convex : public ParameterAcceptor {
public:
    Convex2Convex();
    void run();

private:
    void mesh_generation();
    void print_parameters();
    void load_meshes();
    void run_sot();
    void compute_power_diagram();

    std::string selected_task;
    std::string io_coding = "txt"; 


    struct MeshParameters {
        unsigned int n_refinements = 0 ;
        std::string grid_generator_function;
        std::string grid_generator_arguments;
    };

    MeshParameters source_params;
    MeshParameters target_params;

    Triangulation<dim> source_mesh;
    Triangulation<dim> target_mesh;
    DoFHandler<dim> dof_handler_source;
    DoFHandler<dim> dof_handler_target;
    FESystem<dim> fe_system;
    Vector<double> source_density;
    std::vector<Point<dim>> target_points;
    std::vector<Point<dim>> source_points;
    Vector<double> target_weights;

    struct SolverParameters {
        unsigned int max_iterations = 1000;
        double tolerance = 1e-5;
        double regularization_param = 0.1;
        bool verbose_output = true;
        std::string solver_type = "BFGS";
        unsigned int quadrature_order = 3;
    } solver_params;

    void generate_mesh(Triangulation<dim> &tria,
                      const std::string &grid_generator_function,
                      const std::string &grid_generator_arguments,
                      const unsigned int n_refinements);
    void save_meshes();
    void setup_finite_elements();
    double evaluate_sot_functional(const Vector<double>& weights, Vector<double>& gradient);
    void save_results(const Vector<double>& weights, const std::string& filename);
};

#endif
