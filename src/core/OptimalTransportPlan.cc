#include "SemiDiscreteOT/core/OptimalTransportPlan.h"
#include "SemiDiscreteOT/utils/utils.h"

namespace fs = std::filesystem;

namespace OptimalTransportPlanSpace {

template <int spacedim>
OptimalTransportPlan<spacedim>::OptimalTransportPlan(const std::string& strategy_name)
    : ParameterAcceptor("OptimalTransportPlan"),
    distance_function(euclidean_distance<spacedim>)
{
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
    epsilon = regularization_param;
}

template <int spacedim>
void OptimalTransportPlan<spacedim>::set_truncation_radius(double radius)
{
    truncation_radius = radius;
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
                         transport_potential, epsilon,
                         truncation_radius);
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
    return {"modal", "barycentric"};
}

template <int spacedim>
std::unique_ptr<MapApproximationStrategy<spacedim>>
OptimalTransportPlan<spacedim>::create_strategy(const std::string& name)
{
    if (name == "modal")
        return std::make_unique<ModalStrategy<spacedim>>();
    else if (name == "barycentric")
        return std::make_unique<BarycentricStrategy<spacedim>>();
    else
        throw std::runtime_error("Unknown strategy: " + name);
}

// Implementation of ModalStrategy
template <int spacedim>
void ModalStrategy<spacedim>::compute_map(
    const std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function,
    const std::vector<Point<spacedim>>& source_points,
    const std::vector<double>& source_density,
    const std::vector<Point<spacedim>>& target_points,
    const std::vector<double>& target_density,
    const Vector<double>& potential,
    const double regularization_param,
    const double truncation_radius)
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

    // For each source point, find the target point that maximizes the score
    this->mapped_points.resize(source_points.size());
    this->transport_density.resize(source_points.size());

    // Store source points for displacement computation
    this->source_points = source_points;

    // Determine if we use truncation or consider all points
    const bool use_truncation = (truncation_radius > 0.0);

    for (std::size_t i = 0; i < source_points.size(); ++i) {
        const Point<spacedim>& x = source_points[i];
        double max_score = -std::numeric_limits<double>::infinity();
        std::size_t best_idx = 0;

        // Set of points to consider (all or truncated)
        std::vector<IndexedPoint> candidates;
        
        if (use_truncation) {
            // Use truncation radius to limit points to consider
            target_rtree.query(
                boost::geometry::index::satisfies([&x, &distance_function, truncation_radius](const IndexedPoint& p) {
                    return distance_function(x, p.first) < truncation_radius;
                }),
                std::back_inserter(candidates)
            );
            
            // If no points within truncation radius, fall back to nearest neighbor
            if (candidates.empty()) {
                target_rtree.query(boost::geometry::index::nearest(x, 1), std::back_inserter(candidates));
            }
        } else {
            // Consider all target points
            candidates.reserve(target_points.size());
            for (std::size_t j = 0; j < target_points.size(); ++j) {
                candidates.push_back(indexed_points[j]);
            }
        }

        // Compute scores and find maximum
        for (const auto& candidate : candidates) {
            const Point<spacedim>& y = candidate.first;
            const std::size_t j = candidate.second;
            
            // Compute squared distance
            double squared_dist = std::pow(distance_function(x, y), 2);

            // Compute score: potential - c(x,y) + regularization_param * log(target_density)
            double log_term = 0.0;
            if (target_density[j] > 0) {
                log_term = regularization_param * std::log(target_density[j]);
            } else {
                // If target density is zero or negative, use negative infinity for log term
                log_term = -std::numeric_limits<double>::infinity();
            }

            double score = potential[j] - 0.5 * squared_dist + log_term;

            if (score > max_score) {
                max_score = score;
                best_idx = j;
            }
        }

        this->mapped_points[i] = target_points[best_idx];
        this->transport_density[i] = source_density[i];
    }
}

template <int spacedim>
void ModalStrategy<spacedim>::save_results(const std::string& output_dir) const
{
    // Save text files for backward compatibility
    Utils::write_vector(this->mapped_points, output_dir + "/mapped_points", "txt");
    Utils::write_vector(this->transport_density, output_dir + "/transport_density", "txt");

    // 1. Source triangulation with displacement field using utility function
    Utils::write_points_with_displacement_vtk<spacedim>(
        this->source_points, 
        this->mapped_points, 
        this->transport_density,
        output_dir + "/source_triangulation_with_displacement.vtk",
        "Source triangulation with displacement vectors",
        "source_density"
    );

    // 2. Mapped points with density scalars using utility function
    Utils::write_points_with_density_vtk<spacedim>(
        this->mapped_points, 
        this->transport_density,
        output_dir + "/mapped_points_with_density.vtk",
        "Mapped points with density values",
        "transport_density"
    );
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
    const double regularization_param,
    const double truncation_radius)
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

    // Store source points for displacement computation
    this->source_points = source_points;

    // Determine if we use truncation or consider all points
    const bool use_truncation = (truncation_radius > 0.0);

    // For each source point, compute barycentric interpolation
    for (std::size_t i = 0; i < source_points.size(); ++i) {
        const Point<spacedim>& x = source_points[i];
        Point<spacedim> weighted_sum;
        
        // Determine which target points to consider
        std::vector<IndexedPoint> candidates;
        
        if (use_truncation) {
            // Use truncation radius to limit points to consider
            target_rtree.query(
                boost::geometry::index::satisfies([&x, &distance_function, truncation_radius](const IndexedPoint& p) {
                    return distance_function(x, p.first) < truncation_radius;
                }),
                std::back_inserter(candidates)
            );
            
            // If no points within truncation radius, fall back to nearest neighbor
            if (candidates.empty()) {
                target_rtree.query(boost::geometry::index::nearest(x, 1), std::back_inserter(candidates));
                const Point<spacedim>& nearest = candidates[0].first;
                this->mapped_points[i] = nearest;
                this->transport_density[i] = source_density[i];
                continue;
            }
        } else {
            // Consider all target points
            candidates.reserve(target_points.size());
            for (std::size_t j = 0; j < target_points.size(); ++j) {
                candidates.push_back(indexed_points[j]);
            }
        }

        // Log-sum-exp trick for numerical stability
        // First compute all exponent terms and find maximum
        std::vector<double> exponent_terms;
        exponent_terms.reserve(candidates.size());
        double max_exponent = -std::numeric_limits<double>::infinity();
        
        for (const auto& candidate : candidates) {
            const Point<spacedim>& y = candidate.first;
            const std::size_t j = candidate.second;
            
            double squared_dist = std::pow(distance_function(x, y), 2);
            double exponent = (potential[j] - 0.5 * squared_dist) / regularization_param;
            
            exponent_terms.push_back(exponent);
            max_exponent = std::max(max_exponent, exponent);
        }
        
        // Compute weighted sum using the log-sum-exp trick
        double sum_exp = 0.0;
        for (std::size_t k = 0; k < candidates.size(); ++k) {
            const auto& candidate = candidates[k];
            const Point<spacedim>& y = candidate.first;
            const std::size_t j = candidate.second;
            
            // Subtract max_exponent for numerical stability
            double stable_exp = std::exp(exponent_terms[k] - max_exponent);
            double weight = target_density[j] * stable_exp;
            
            weighted_sum += weight * y;
            sum_exp += weight;
        }

        // Normalize the weighted sum
        if (sum_exp > 0) {
            this->mapped_points[i] = weighted_sum / sum_exp;
        } else {
            // If all weights are zero, map to nearest target point
            std::vector<IndexedPoint> nearest;
            target_rtree.query(boost::geometry::index::nearest(x, 1), std::back_inserter(nearest));
            this->mapped_points[i] = nearest[0].first;
        }
        
        this->transport_density[i] = source_density[i];
    }
}

template <int spacedim>
void BarycentricStrategy<spacedim>::save_results(const std::string& output_dir) const
{
    // Save text files for backward compatibility
    Utils::write_vector(this->mapped_points, output_dir + "/mapped_points", "txt");
    Utils::write_vector(this->transport_density, output_dir + "/transport_density", "txt");

    // 1. Source triangulation with displacement field using utility function
    Utils::write_points_with_displacement_vtk<spacedim>(
        this->source_points, 
        this->mapped_points, 
        this->transport_density,
        output_dir + "/source_triangulation_with_displacement.vtk",
        "Source triangulation with displacement vectors (barycentric)",
        "source_density"
    );

    // 2. Mapped points with density scalars using utility function
    Utils::write_points_with_density_vtk<spacedim>(
        this->mapped_points, 
        this->transport_density,
        output_dir + "/mapped_points_with_density.vtk",
        "Mapped points with density values (barycentric)",
        "transport_density"
    );
}

// Explicit instantiation
template class OptimalTransportPlan<2>;
template class OptimalTransportPlan<3>;
template class ModalStrategy<2>;
template class ModalStrategy<3>;
template class BarycentricStrategy<2>;
template class BarycentricStrategy<3>;

} // namespace OptimalTransportPlanSpace 