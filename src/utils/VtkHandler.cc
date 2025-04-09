#include "SemiDiscreteOT/utils/VtkHandler.h"

#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkGenericCell.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>

#include <stdexcept>
#include <filesystem>
#include <string>
#include <algorithm>

template <int dim, int spacedim>
VTKHandler<dim, spacedim>::VTKHandler(const std::string& filename,
                       const bool is_binary,
                       const double scaling_factor)
    : Function<spacedim>(1)  // scalar function
    , filename(filename)
    , is_binary(is_binary)
    , scaling_factor(scaling_factor)
    , selected_component(0)
{
    // Check if file exists
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("VTK file not found: " + filename);
    }
    
    read_file();
}

template <int dim, int spacedim>
void VTKHandler<dim, spacedim>::read_file()
{
    // Get file extension
    std::string extension = std::filesystem::path(filename).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == ".vtu") {
        // For VTU files (XML format)
        auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        reader->SetFileName(filename.c_str());
        reader->Update();
        vtk_grid = reader->GetOutput();
        
        if (!vtk_grid || vtk_grid->GetNumberOfPoints() == 0) {
            throw std::runtime_error("Failed to read VTU file or file is empty");
        }
    } else if (extension == ".vtk") {
        // For legacy VTK files
        auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
        reader->SetFileName(filename.c_str());
        reader->Update();
        vtk_grid = reader->GetOutput();
        
        if (!vtk_grid || vtk_grid->GetNumberOfPoints() == 0) {
            throw std::runtime_error("Failed to read VTK file or file is empty");
        }
    } else {
        throw std::runtime_error("Unsupported file extension: " + extension + 
                               ". Supported extensions are .vtk and .vtu");
    }

    // Apply scaling if needed
    if (scaling_factor != 1.0) {
        auto transform = vtkSmartPointer<vtkTransform>::New();
        transform->Scale(scaling_factor, scaling_factor, scaling_factor);

        auto transform_filter = vtkSmartPointer<vtkTransformFilter>::New();
        transform_filter->SetInputData(vtk_grid);
        transform_filter->SetTransform(transform);
        transform_filter->Update();

        vtk_grid = transform_filter->GetUnstructuredGridOutput();
    }

    // Print some information about the loaded grid
    std::cout << "Loaded VTK grid with:" << std::endl
              << "  Points: " << vtk_grid->GetNumberOfPoints() << std::endl
              << "  Cells: " << vtk_grid->GetNumberOfCells() << std::endl;
              
    // Print available point data arrays
    vtkPointData* point_data = vtk_grid->GetPointData();
    std::cout << "Available point data arrays:" << std::endl;
    for (int i = 0; i < point_data->GetNumberOfArrays(); ++i) {
        vtkDataArray* array = point_data->GetArray(i);
        std::cout << "  - " << array->GetName() 
                 << " (components: " << array->GetNumberOfComponents() << ")" 
                 << std::endl;
    }
}

template <int dim, int spacedim>
void VTKHandler<dim, spacedim>::setup_field(const std::string& field_name_,
                            const DataLocation data_location_,
                            const unsigned int component)
{
    field_name = field_name_;
    data_location = data_location_;
    selected_component = component;

    // Get the field data based on location
    if (data_location == DataLocation::PointData) {
        field_data = vtk_grid->GetPointData()->GetArray(field_name.c_str());
        if (!field_data) {
            throw std::runtime_error("Point data array '" + field_name + "' not found");
        }
        
        // Setup point locator
        point_locator = vtkSmartPointer<vtkPointLocator>::New();
        point_locator->SetDataSet(vtk_grid);
        point_locator->BuildLocator();
    } else {
        field_data = vtk_grid->GetCellData()->GetArray(field_name.c_str());
        if (!field_data) {
            throw std::runtime_error("Cell data array '" + field_name + "' not found");
        }
        
        // Setup cell locator
        cell_locator = vtkSmartPointer<vtkCellLocator>::New();
        cell_locator->SetDataSet(vtk_grid);
        cell_locator->BuildLocator();
    }

    // Check if the component is valid
    if (selected_component >= static_cast<unsigned int>(field_data->GetNumberOfComponents())) {
        throw std::runtime_error("Invalid component index " + std::to_string(selected_component) + 
                               " for field '" + field_name + "' with " + 
                               std::to_string(field_data->GetNumberOfComponents()) + " components");
    }
}

template <int dim, int spacedim>
double VTKHandler<dim, spacedim>::value(const Point<spacedim>& p, const unsigned int /*component*/) const
{
    double closest_point[3];
    vtkIdType closest_id;
    double closest_dist2;
    double value = 0.0;

    if (data_location == DataLocation::PointData) {
        // Convert dealii::Point to double array for VTK
        double point[3] = {p[0], p[1], p[2]};
        closest_id = point_locator->FindClosestPoint(point);
        
        // Get the selected component from the field data
        double tuple[3];  // Assuming max 3 components
        field_data->GetTuple(closest_id, tuple);
        value = tuple[selected_component];
    } else {
        // Find closest cell
        auto cell = vtkSmartPointer<vtkGenericCell>::New();
        double x[3] = {p[0], p[1], p[2]};
        int subId;
        
        cell_locator->FindClosestPoint(x, closest_point, cell, closest_id,
                                     subId, closest_dist2);
        
        // Get the selected component from the field data
        double tuple[3];  // Assuming max 3 components
        field_data->GetTuple(closest_id, tuple);
        value = tuple[selected_component];
    }

    return value;
}

// Explicit template instantiations
template class VTKHandler<2, 2>;
template class VTKHandler<2, 3>;
template class VTKHandler<3, 3>; 