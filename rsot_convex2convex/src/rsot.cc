#include "rsot.h"
#include "PowerDiagram.h"
#include <deal.II/base/timer.h>

template <int dim>
Convex2Convex<dim>::Convex2Convex()
    : ParameterAcceptor("Convex2Convex"), source_mesh(), target_mesh(), dof_handler_source(source_mesh), dof_handler_target(target_mesh), fe_system(FE_Q<dim>(1), 1)
{
    add_parameter("selected_task", selected_task);

    enter_subsection("mesh_generation");
    {
        enter_subsection("source");
        {
            add_parameter("number of refinements", source_params.n_refinements);
            add_parameter("grid generator function", source_params.grid_generator_function);
            add_parameter("grid generator arguments", source_params.grid_generator_arguments);
        }
        leave_subsection();

        enter_subsection("target");
        {
            add_parameter("number of refinements", target_params.n_refinements);
            add_parameter("grid generator function", target_params.grid_generator_function);
            add_parameter("grid generator arguments", target_params.grid_generator_arguments);
        }
        leave_subsection();
    }
    leave_subsection();

    enter_subsection("rsot_solver");
    {
        add_parameter("max_iterations", 
                     solver_params.max_iterations,
                     "Maximum number of iterations for the optimization solver");
        
        add_parameter("tolerance", 
                     solver_params.tolerance,
                     "Convergence tolerance for the optimization solver");
        
        add_parameter("regularization_parameter", 
                     solver_params.regularization_param,
                     "Entropy regularization parameter (lambda)");
        
        add_parameter("verbose_output", 
                     solver_params.verbose_output,
                     "Enable detailed solver output");
        
        add_parameter("solver_type",
                     solver_params.solver_type,
                     "Type of optimization solver (BFGS)");
        
        add_parameter("quadrature_order",
                     solver_params.quadrature_order,
                     "Order of quadrature formula for numerical integration");
    }
    leave_subsection();
}

template <int dim>
void Convex2Convex<dim>::print_parameters()
{
    std::cout << "Selected Task: " << selected_task << std::endl;

    std::cout << "Source Mesh Parameters:" << std::endl;
    std::cout << "  Grid Generator Function: " << source_params.grid_generator_function << std::endl;
    std::cout << "  Grid Generator Arguments: " << source_params.grid_generator_arguments << std::endl;
    std::cout << "  Number of Refinements: " << source_params.n_refinements << std::endl;

    std::cout << "Target Mesh Parameters:" << std::endl;
    std::cout << "  Grid Generator Function: " << target_params.grid_generator_function << std::endl;
    std::cout << "  Grid Generator Arguments: " << target_params.grid_generator_arguments << std::endl;
    std::cout << "  Number of Refinements: " << target_params.n_refinements << std::endl;

    std::cout << "RSOT Solver Parameters:" << std::endl;
    std::cout << "  Max Iterations: " << solver_params.max_iterations << std::endl;
    std::cout << "  Tolerance: " << solver_params.tolerance << std::endl;
    std::cout << "  Regularization Parameter (λ): " << solver_params.regularization_param << std::endl;
    std::cout << "  Verbose Output: " << (solver_params.verbose_output ? "Yes" : "No") << std::endl;
    std::cout << "  Solver Type: " << solver_params.solver_type << std::endl;
    std::cout << "  Quadrature Order: " << solver_params.quadrature_order << std::endl;
}

template <int dim>
void Convex2Convex<dim>::generate_mesh(Triangulation<dim> &tria,
                                       const std::string &grid_generator_function,
                                       const std::string &grid_generator_arguments,
                                       const unsigned int n_refinements)
{
    GridGenerator::generate_from_name_and_arguments(
        tria,
        grid_generator_function,
        grid_generator_arguments);
    tria.refine_global(n_refinements);
}

template <int dim>
void Convex2Convex<dim>::save_meshes()
{
    const std::string directory = "output/data_mesh";
    std::filesystem::create_directories(directory);

    GridOut grid_out;

    std::ofstream out_vtk_source(directory + "/source.vtk");
    std::ofstream out_msh_source(directory + "/source.msh");
    grid_out.write_vtk(source_mesh, out_vtk_source);
    grid_out.write_msh(source_mesh, out_msh_source);

    std::ofstream out_vtk_target(directory + "/target.vtk");
    std::ofstream out_msh_target(directory + "/target.msh");
    grid_out.write_vtk(target_mesh, out_vtk_target);
    grid_out.write_msh(target_mesh, out_msh_target);

    std::cout << "Meshes saved in VTK and MSH formats" << std::endl;
}

template <int dim>
void Convex2Convex<dim>::mesh_generation()
{
    generate_mesh(source_mesh,
                  source_params.grid_generator_function,
                  source_params.grid_generator_arguments,
                  source_params.n_refinements);

    generate_mesh(target_mesh,
                  target_params.grid_generator_function,
                  target_params.grid_generator_arguments,
                  target_params.n_refinements);

    save_meshes();
}

template <int dim>
void Convex2Convex<dim>::load_meshes()
{
    const std::string directory = "output/data_mesh";

    std::ifstream in_vtk_source(directory + "/source.vtk");
    std::ifstream in_msh_source(directory + "/source.msh");
    GridIn<dim> grid_in_source;
    grid_in_source.attach_triangulation(source_mesh);
    grid_in_source.read_vtk(in_vtk_source);

    std::ifstream in_vtk_target(directory + "/target.vtk");
    std::ifstream in_msh_target(directory + "/target.msh");
    GridIn<dim> grid_in_target;
    grid_in_target.attach_triangulation(target_mesh);
    grid_in_target.read_vtk(in_vtk_target);

    std::cout << "Meshes loaded from VTK and MSH formats" << std::endl;
    std::cout << "Source mesh: " << source_mesh.n_active_cells() << " cells, " << source_mesh.n_vertices() << " vertices" << std::endl;
    std::cout << "Target mesh: " << target_mesh.n_active_cells() << " cells, " << target_mesh.n_vertices() << " vertices" << std::endl;
}

// TODO: controllare da qua in poi
template <int dim>
void Convex2Convex<dim>::setup_finite_elements()
{
    dof_handler_target.distribute_dofs(fe_system);
    dof_handler_source.distribute_dofs(fe_system);
    
    source_density.reinit(dof_handler_source.n_dofs());
    source_density = 1.0;
    
    // Compute actual L1 norm using quadrature
    QGauss<dim> quadrature(solver_params.quadrature_order);
    FEValues<dim> fe_values(fe_system, quadrature,
                           update_values | update_JxW_values);
    
    std::vector<double> density_values(quadrature.size());
    double l1_norm = 0.0;
    
    for (const auto &cell : dof_handler_source.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(source_density, density_values);
        
        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            l1_norm += std::abs(density_values[q]) * fe_values.JxW(q);
        }
    }
    
    std::cout << "Source density L1 norm: " << l1_norm << std::endl;
    source_density /= l1_norm; // Normalize to mass 1

    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler_target, support_points);
    target_points.clear();
    target_weights.reinit(support_points.size());
    target_weights = 1.0 / support_points.size();

    for (const auto &point_pair : support_points)
    {
        target_points.push_back(point_pair.second);
    }
}


template <int dim>
double Convex2Convex<dim>::evaluate_sot_functional(const Vector<double> &weights, Vector<double> &gradient)
{
    Timer timer;
    timer.start();
    
    const double lambda = solver_params.regularization_param;
    double functional = 0.0;
    gradient = 0;

    QGauss<dim> quadrature(dim);
    FEValues<dim> fe_values(fe_system, quadrature,
                            update_values | update_quadrature_points | update_JxW_values);

    std::vector<double> density_values(quadrature.size());
    for (const auto &cell : dof_handler_source.active_cell_iterators())
    {
        fe_values.reinit(cell);
        const std::vector<Point<dim>> &q_points = fe_values.get_quadrature_points();
        fe_values.get_function_values(source_density, density_values);


        for (unsigned int q = 0; q < q_points.size(); ++q)
        {
            const Point<dim> &x = q_points[q];
            double sum_exp = 0.0;
            std::vector<double> exp_terms(target_points.size());

            // Compute exp terms
            for (unsigned int i = 0; i < target_points.size(); ++i)
            {
                const double dist2 = (x - target_points[i]).norm_square();
                exp_terms[i] = target_weights[i] * std::exp((weights[i] - 0.5 * dist2) / lambda);
                sum_exp += exp_terms[i];
            }

            functional += density_values[q] * lambda * std::log(sum_exp) * fe_values.JxW(q);

            for (unsigned int i = 0; i < target_points.size(); ++i)
            {
                gradient[i] += density_values[q] * (exp_terms[i] / sum_exp) * fe_values.JxW(q);

            }
        }
    }

    // Add linear term
    for (unsigned int i = 0; i < target_points.size(); ++i)
    {
        functional -= weights[i] * target_weights[i];
        gradient[i] -= target_weights[i];
    }

    timer.stop();
    if (solver_params.verbose_output)
        std::cout << "Evaluation time: " << timer.wall_time() << " seconds" << std::endl;

    return functional;
}

template <int dim>
void Convex2Convex<dim>::run_sot()
{
    setup_finite_elements();

    std::cout << "Starting SOT optimization with " << target_points.size()
              << " target points..." << std::endl;

    Vector<double> weights(target_points.size());
    Vector<double> gradient(target_points.size());

    // Define VerboseSolverControl class
    class VerboseSolverControl : public SolverControl
    {
    public:
        VerboseSolverControl(unsigned int n, double tol)
            : SolverControl(n, tol) {}

        virtual State check(unsigned int step, double value) override
        {
            std::cout << "Iteration " << step
                      << " - Function value: " << value
                      << " - Relative residual: " << value / initial_value() << std::endl;
            return SolverControl::check(step, value);
        }
    };

    VerboseSolverControl solver_control(solver_params.max_iterations, 
                                      solver_params.tolerance);
    
    if (!solver_params.verbose_output)
    {
        solver_control.log_history(false);
        solver_control.log_result(false);
    }

    SolverBFGS<Vector<double>> solver(solver_control);

    try
    {
        std::cout << "Using regularization parameter λ = " 
                  << solver_params.regularization_param << std::endl;
        solver.solve(
            [&](const Vector<double> &w, Vector<double> &grad) {
                return evaluate_sot_functional(w, grad);
            },
            weights
        );
        
        std::cout << "\nOptimization completed successfully!" << std::endl;
        std::cout << "Final number of iterations: " << solver_control.last_step() << std::endl;
        std::cout << "Final function value: " << solver_control.last_value() << std::endl;

        save_results(weights, "sot_results.txt");
        std::cout << "Results saved to sot_results.txt" << std::endl;
    }
    catch (std::exception &exc)
    {
        std::cerr << "Error in SOT computation: " << exc.what() << std::endl;
    }
}

template <int dim>
void Convex2Convex<dim>::save_results(const Vector<double> &weights, const std::string &filename)
{
    std::ofstream file(filename);
    for (unsigned int i = 0; i < weights.size(); ++i)
    {
        file << weights[i] << std::endl;
    }
}

template <int dim>
void Convex2Convex<dim>::compute_power_diagram()
{
    load_meshes();
    setup_finite_elements();

    // Read weights from SOT results
    std::ifstream weights_file("sot_results.txt");
    Assert(weights_file.is_open(), 
           ExcMessage("Could not open sot_results.txt for reading weights"));

    Vector<double> weights(target_points.size());
    for (unsigned int i = 0; i < target_points.size(); ++i)
    {
        if (!(weights_file >> weights[i]))
        {
            AssertThrow(false, 
                ExcMessage("Error reading weights from sot_results.txt"));
        }
    }
    weights_file.close();

    // Create and configure power diagram
    PowerDiagramSpace::PowerDiagram<dim> power_diagram(source_mesh);
    power_diagram.set_generators(target_points, weights);
    
    // Compute power diagram and its properties
    power_diagram.compute_power_diagram();
    power_diagram.compute_cell_centroids();
    
    // Save results
    const std::string output_dir = "output/power_diagram";
    std::filesystem::create_directories(output_dir);
    
    power_diagram.save_centroids_to_file(output_dir + "/centroids.txt");
    power_diagram.output_vtu(output_dir + "/power_diagram.vtu");
    
    std::cout << "Power diagram computation completed." << std::endl;
    std::cout << "Results saved in " << output_dir << std::endl;
}

template <int dim>
void Convex2Convex<dim>::run()
{
    print_parameters();

    if (selected_task == "mesh_generation")
    {
        mesh_generation();
    }
    else if (selected_task == "load_meshes")
    {
        load_meshes();
    }
    else if (selected_task == "sot")
    {
        load_meshes();
        run_sot();
    }
    else if (selected_task == "power_diagram")
    {
        compute_power_diagram();
    }
    else
    {
        std::cout << "No valid task selected" << std::endl;
    }
}

template class Convex2Convex<2>;
template class Convex2Convex<3>;