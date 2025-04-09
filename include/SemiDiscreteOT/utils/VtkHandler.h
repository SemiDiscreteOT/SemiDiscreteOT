#ifndef VTK_HANDLER_HPP
#define VTK_HANDLER_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkPointLocator.h>
#include <vtkCellLocator.h>
#include <vtkDoubleArray.h>

#include <string>
#include <memory>

using namespace dealii;

template <int dim, int spacedim = dim>
class VTKHandler : public Function<spacedim>
{
public:
    // Enum for data location (points or cells)
    enum class DataLocation {
        PointData,
        CellData
    };

    // Constructor
    VTKHandler(const std::string& filename,
               const bool is_binary = false,
               const double scaling_factor = 1.0);

    // Read VTK file and setup data structures
    void read_file();

    // Setup field data for interpolation
    void setup_field(const std::string& field_name,
                    const DataLocation data_location = DataLocation::PointData,
                    const unsigned int component = 0);

    // Get field value at a point (implements Function interface)
    virtual double value(const Point<spacedim>& p,
                        const unsigned int component = 0) const override;

    // Get the underlying VTK grid
    vtkSmartPointer<vtkUnstructuredGrid> get_grid() const { return vtk_grid; }

    // Get field data
    vtkSmartPointer<vtkDataArray> get_field_data() const { return field_data; }

    // Get number of components in the field
    int get_num_components() const { return field_data ? field_data->GetNumberOfComponents() : 0; }

private:
    std::string filename;
    bool is_binary;
    double scaling_factor;
    
    vtkSmartPointer<vtkUnstructuredGrid> vtk_grid;
    vtkSmartPointer<vtkDataArray> field_data;
    vtkSmartPointer<vtkPointLocator> point_locator;
    vtkSmartPointer<vtkCellLocator> cell_locator;
    
    DataLocation data_location;
    std::string field_name;
    unsigned int selected_component;
};

#endif // VTK_HANDLER_HPP 