#include "SemiDiscreteOT/core/OptimalTransportPlan.h"

namespace fs = std::filesystem;

namespace OptimalTransportPlanSpace {

template <int spacedim>
OptimalTransportPlan<spacedim>::OptimalTransportPlan(const std::string& strategy_name)
    : ParameterAcceptor("OptimalTransportPlan"),
    distance_function(euclidean_distance<spacedim>)
{
    add_parameter("n_samples", params.n_samples);
    add_parameter("n_neighbors", params.n_neighbors);
    add_parameter("kernel_width", params.kernel_width);
    add_parameter("interpolation_type", params.interpolation_type);

    strategy = create_strategy(strategy_name);
}

template <int spacedim>
void OptimalTransportPlan<spacedim>::set_source_measure(
    const std::vector<Point<spacedim>>& points,
    const std::vector<double>& density)
{
    AssertDimension(points.size(), density.size());
    source_points = points;
    source_density = density;
}

template <int spacedim>
void OptimalTransportPlan<spacedim>::set_target_measure(
    const std::vector<Point<spacedim>>& points,
    const std::vector<double>& density)
{
    AssertDimension(points.size(), density.size());
    target_points = points;
    target_density = density;
}

template <int spacedim>
void OptimalTransportPlan<spacedim>::set_potential(
    const Vector<double>& potential,
    const double regularization_param)
{
    AssertDimension(potential.size(), target_points.size());
    transport_potential = potential;
    regularization_parameter = regularization_param;
}

template <int spacedim>
void OptimalTransportPlan<spacedim>::compute_map()
{
    Assert(strategy, ExcMessage("No strategy selected"));
    Assert(!source_points.empty(), ExcMessage("Source measure not set"));
    Assert(!target_points.empty(), ExcMessage("Target measure not set"));
    Assert(transport_potential.size() > 0, ExcMessage("Transport potential not set"));

    strategy->compute_map(distance_function, source_points, source_density,
                         target_points, target_density,
                         transport_potential, regularization_parameter);
}

template <int spacedim>
void OptimalTransportPlan<spacedim>::save_map(const std::string& output_dir) const
{
    Assert(strategy, ExcMessage("No strategy selected"));
    fs::create_directories(output_dir);
    strategy->save_results(output_dir);
}

template <int spacedim>
void OptimalTransportPlan<spacedim>::set_strategy(const std::string& strategy_name)
{
    strategy = create_strategy(strategy_name);
}

template <int spacedim>
std::vector<std::string> OptimalTransportPlan<spacedim>::get_available_strategies()
{
    return {"nearest_neighbor", "barycentric", "kernel"};
}

template <int spacedim>
std::unique_ptr<MapApproximationStrategy<spacedim>>
OptimalTransportPlan<spacedim>::create_strategy(const std::string& name)
{
    if (name == "nearest_neighbor")
        return std::make_unique<NearestNeighborStrategy<spacedim>>();
    else if (name == "barycentric")
        return std::make_unique<BarycentricStrategy<spacedim>>();
    else if (name == "kernel")
        return std::make_unique<KernelStrategy<spacedim>>();
    else
        throw std::runtime_error("Unknown strategy: " + name);
}

// Implementation of NearestNeighborStrategy
template <int spacedim>
void NearestNeighborStrategy<spacedim>::compute_map(
    const std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function,
    const std::vector<Point<spacedim>>& source_points,
    const std::vector<double>& source_density,
    const std::vector<Point<spacedim>>& target_points,
    const std::vector<double>& target_density,
    const Vector<double>& potential,
    const double regularization_param)
{
    using IndexedPoint = std::pair<Point<spacedim>, std::size_t>;
    using RTreeParams = boost::geometry::index::rstar<8>;
    using RTree = boost::geometry::index::rtree<IndexedPoint, RTreeParams>;

    // Build RTree for target points
    std::vector<IndexedPoint> indexed_points;
    indexed_points.reserve(target_points.size());
    for (std::size_t i = 0; i < target_points.size(); ++i) {
        indexed_points.emplace_back(target_points[i], i);
    }
    RTree target_rtree(indexed_points.begin(), indexed_points.end());

    // For each source point, find the nearest target point considering the potential
    this->mapped_points.resize(source_points.size());
    this->transport_density.resize(source_points.size());

    for (std::size_t i = 0; i < source_points.size(); ++i) {
        const Point<spacedim>& x = source_points[i];
        double min_cost = std::numeric_limits<double>::infinity();
        std::size_t best_idx = 0;

        // Search for k nearest neighbors
        // TODO: pass custom `distance_function`
        const std::size_t k = 10;  // Number of neighbors to consider
        std::vector<IndexedPoint> neighbors;
        target_rtree.query(boost::geometry::index::nearest(x, k), std::back_inserter(neighbors));

        // Find the one minimizing the cost
        for (const auto& neighbor : neighbors) {
            const Point<spacedim>& y = neighbor.first;
            const std::size_t j = neighbor.second;
            
            double squared_dist = std::pow(distance_function(x, y), 2);

            double cost = 0.5 * squared_dist - potential[j];
            if (cost < min_cost) {
                min_cost = cost;
                best_idx = j;
            }
        }

        this->mapped_points[i] = target_points[best_idx];
        this->transport_density[i] = source_density[i];
    }
}

template <int spacedim>
void NearestNeighborStrategy<spacedim>::save_results(const std::string& output_dir) const
{
    // Save mapped points
    Utils::write_vector(this->mapped_points, output_dir + "/mapped_points", "txt");
    Utils::write_vector(this->transport_density, output_dir + "/transport_density", "txt");

    // Save as VTK for visualization
    std::ofstream out(output_dir + "/transport_map.vtk");
    out << "# vtk DataFile Version 3.0\n"
        << "Transport map\n"
        << "ASCII\n"
        << "DATASET UNSTRUCTURED_GRID\n"
        << "POINTS " << this->mapped_points.size() << " double\n";

    // Write points
    for (const auto& p : this->mapped_points)
        out << p << "\n";

    out.close();
}

// Implementation of BarycentricStrategy
template <int spacedim>
void BarycentricStrategy<spacedim>::compute_map(
    const std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function,
    const std::vector<Point<spacedim>>& source_points,
    const std::vector<double>& source_density,
    const std::vector<Point<spacedim>>& target_points,
    const std::vector<double>& target_density,
    const Vector<double>& potential,
    const double regularization_param)
{
    using IndexedPoint = std::pair<Point<spacedim>, std::size_t>;
    using RTreeParams = boost::geometry::index::rstar<8>; // TODO: set as hyperparameter ?
    using RTree = boost::geometry::index::rtree<IndexedPoint, RTreeParams>;

    // Build RTree for target points
    std::vector<IndexedPoint> indexed_points;
    indexed_points.reserve(target_points.size());
    for (std::size_t i = 0; i < target_points.size(); ++i) {
        indexed_points.emplace_back(target_points[i], i);
    }
    RTree target_rtree(indexed_points.begin(), indexed_points.end());

    this->mapped_points.resize(source_points.size());
    this->transport_density.resize(source_points.size());

    // For each source point, compute barycentric interpolation
    for (std::size_t i = 0; i < source_points.size(); ++i) {
        const Point<spacedim>& x = source_points[i];
        
        // Find k nearest neighbors
        // TODO: pass custom `distance_function`
        const std::size_t k = 4;  // Number of neighbors for barycentric coordinates
        std::vector<IndexedPoint> neighbors;
        target_rtree.query(boost::geometry::index::nearest(x, k), std::back_inserter(neighbors));

        // Compute weights based on cost
        std::vector<double> weights(k);
        double total_weight = 0.0;

        for (std::size_t j = 0; j < k; ++j) {
            const Point<spacedim>& y = neighbors[j].first;
            const std::size_t idx = neighbors[j].second;
            
            double squared_dist = std::pow(distance_function(x, y), 2);

            // Use Gaussian kernel for weights
            // weights[j] = std::exp(-(squared_dist + potential[idx]) / regularization_param);
            weights[j] = std::exp((potential[idx] - 0.5 * squared_dist) / regularization_param);
            total_weight += weights[j];
        }

        // Normalize weights and compute interpolated point
        Point<spacedim> interpolated_point;
        for (std::size_t j = 0; j < k; ++j) {
            weights[j] /= total_weight;
            interpolated_point += weights[j] * neighbors[j].first;
        }

        this->mapped_points[i] = interpolated_point;
        this->transport_density[i] = source_density[i];
    }
}

template <int spacedim>
void BarycentricStrategy<spacedim>::save_results(const std::string& output_dir) const
{
    // Save mapped points and density
    Utils::write_vector(this->mapped_points, output_dir + "/mapped_points", "txt");
    Utils::write_vector(this->transport_density, output_dir + "/transport_density", "txt");

    // Save as VTK with interpolation weights
    std::ofstream out(output_dir + "/transport_map.vtk");
    out << "# vtk DataFile Version 3.0\n"
        << "Transport map with barycentric interpolation\n"
        << "ASCII\n"
        << "DATASET UNSTRUCTURED_GRID\n"
        << "POINTS " << this->mapped_points.size() << " double\n";

    // Write points
    for (const auto& p : this->mapped_points)
        out << p << "\n";

    out.close();
}

// Implementation of KernelStrategy
template <int spacedim>
void KernelStrategy<spacedim>::compute_map(
    const std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function,
    const std::vector<Point<spacedim>>& source_points,
    const std::vector<double>& source_density,
    const std::vector<Point<spacedim>>& target_points,
    const std::vector<double>& target_density,
    const Vector<double>& potential,
    const double regularization_param)
{
    this->mapped_points.resize(source_points.size());
    this->transport_density.resize(source_points.size());

    // For each source point, compute kernel-based approximation
    #pragma omp parallel for
    for (std::size_t i = 0; i < source_points.size(); ++i) {
        const Point<spacedim>& x = source_points[i];
        Point<spacedim> weighted_sum;
        double total_weight = 0.0;

        // Use all target points with kernel weighting
        for (std::size_t j = 0; j < target_points.size(); ++j) {
            const Point<spacedim>& y = target_points[j];
            
            double squared_dist = std::pow(distance_function(x, y), 2);

            // Compute kernel weight
            double weight = target_density[j] * 
                std::exp(-(squared_dist - potential[j]) / (2.0 * regularization_param));
            
            weighted_sum += weight * y;
            total_weight += weight;
        }

        // Normalize
        if (total_weight > 0) {
            this->mapped_points[i] = weighted_sum / total_weight;
        } else {
            // If all weights are zero, map to nearest target point
            double min_dist = std::numeric_limits<double>::infinity();
            std::size_t nearest_idx = 0;
            
            for (std::size_t j = 0; j < target_points.size(); ++j) {
                double dist = std::pow(distance_function(x, target_points[j]), 2);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_idx = j;
                }
            }
            
            this->mapped_points[i] = target_points[nearest_idx];
        }

        this->transport_density[i] = source_density[i];
    }
}

template <int spacedim>
void KernelStrategy<spacedim>::save_results(const std::string& output_dir) const
{
    // Save mapped points and density
    Utils::write_vector(this->mapped_points, output_dir + "/mapped_points", "txt");
    Utils::write_vector(this->transport_density, output_dir + "/transport_density", "txt");

    // Save as VTK with kernel information
    std::ofstream out(output_dir + "/transport_map.vtk");
    out << "# vtk DataFile Version 3.0\n"
        << "Transport map with kernel approximation\n"
        << "ASCII\n"
        << "DATASET UNSTRUCTURED_GRID\n"
        << "POINTS " << this->mapped_points.size() << " double\n";

    // Write points
    for (const auto& p : this->mapped_points)
        out << p << "\n";

    out.close();
}

// Explicit instantiation
template class OptimalTransportPlan<2>;
template class OptimalTransportPlan<3>;
template class NearestNeighborStrategy<2>;
template class NearestNeighborStrategy<3>;
template class BarycentricStrategy<2>;
template class BarycentricStrategy<3>;
template class KernelStrategy<2>;
template class KernelStrategy<3>;

} // namespace OptimalTransportPlanSpace 