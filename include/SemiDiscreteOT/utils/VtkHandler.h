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

/**
 * @brief A handler class for reading and interpolating data from VTK files.
 *
 * This class inherits from `dealii::Function` and provides functionality to
 * read VTK unstructured grids, extract scalar or vector fields from point or
 * cell data, and interpolate these fields at arbitrary points in space.
 *
 * @tparam dim The dimension of the mesh.
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int dim, int spacedim = dim>
class VTKHandler : public Function<spacedim>
{
public:
    /**
     * @brief Enum to specify whether the data is associated with points or cells.
     */
    enum class DataLocation {
        PointData, ///< Data is associated with the vertices of the grid.
        CellData   ///< Data is associated with the cells of the grid.
    };

    /**
     * @brief Constructor for the VTKHandler.
     *
     * @param filename The path to the VTK file.
     * @param is_binary Whether the VTK file is in binary format.
     * @param scaling_factor A factor to scale the data by upon reading.
     */
    VTKHandler(const std::string& filename,
               const bool is_binary = false,
               const double scaling_factor = 1.0);

    /**
     * @brief Reads the VTK file and initializes the internal data structures.
     * @throw std::runtime_error if the file cannot be opened or is invalid.
     */
    void read_file();

    /**
     * @brief Sets up the field to be used for interpolation.
     *
     * @param field_name The name of the data array in the VTK file.
     * @param data_location Whether the data is point or cell data.
     * @param component The component of the data array to use (for vector fields).
     */
    void setup_field(const std::string& field_name,
                    const DataLocation data_location = DataLocation::PointData,
                    const unsigned int component = 0);

    /**
     * @brief Interpolates the value of the selected field at a given point.
     *
     * This method overrides the `value` method of the `dealii::Function` base class.
     *
     * @param p The point at which to interpolate the value.
     * @param component The component of the function to evaluate (unused, as the component is pre-selected in setup_field).
     * @return The interpolated value of the field at point p.
     */
    virtual double value(const Point<spacedim>& p,
                        const unsigned int component = 0) const override;

    /**
     * @brief Returns the underlying VTK unstructured grid.
     * @return A `vtkSmartPointer` to the `vtkUnstructuredGrid`.
     */
    vtkSmartPointer<vtkUnstructuredGrid> get_grid() const { return vtk_grid; }

    /**
     * @brief Returns the data array for the selected field.
     * @return A `vtkSmartPointer` to the `vtkDataArray`.
     */
    vtkSmartPointer<vtkDataArray> get_field_data() const { return field_data; }

    /**
     * @brief Returns the number of components in the selected field data.
     * @return The number of components.
     */
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