#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/numerics/data_out.h>
#include <vector>
#include <string>

namespace PowerDiagramSpace {

using namespace dealii;

template <int dim>
class PowerDiagram
{
public:
    PowerDiagram(const Triangulation<dim> &source_mesh);
    
    void set_generators(const std::vector<Point<dim>> &points,
                       const std::vector<double> &weights);
                       
    void compute_power_diagram();
    
    void output_vtu(const std::string& filename) const;
    
    unsigned int get_cell_assignment(const unsigned int cell_index) const;
    
    const std::vector<unsigned int>& get_cell_assignments() const;
    
    void compute_cell_centroids();
    void save_centroids_to_file(const std::string& filename) const;
    const std::vector<Point<dim>>& get_cell_centroids() const;

private:
    const Triangulation<dim>* source_triangulation;
    std::vector<Point<dim>> generator_points;
    std::vector<double> generator_weights;
    std::vector<unsigned int> cell_assignments;
    std::vector<Point<dim>> cell_centroids;
    double power_distance(const Point<dim> &point,
                         const unsigned int generator_idx) const;
};

}