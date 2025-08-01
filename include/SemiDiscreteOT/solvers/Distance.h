#ifndef DISTANCE_H
#define DISTANCE_H

#include <deal.II/base/point.h>

/**
 * @file
 * @brief A collection of distance functions, their gradients, and exponential maps.
 */

#include <deal.II/base/point.h>

using namespace dealii;

// TODO implement factory

/**
 * @brief Computes the Euclidean distance between two points.
 * @param a The first point.
 * @param b The second point.
 * @return The Euclidean distance between a and b.
 */
template <int spacedim>
double euclidean_distance(
    const Point<spacedim> a, const Point<spacedim> b) {
    return (a-b).norm();
}

/**
 * @brief Computes the spherical distance between two points.
 * @param a The first point.
 * @param b The second point.
 * @return The spherical distance between a and b.
 */
template <int spacedim>
double spherical_distance(
    const Point<spacedim> a, const Point<spacedim> b) {
    double dot_product = 0.0;
    for (unsigned int i = 0; i < spacedim; ++i) {
        dot_product += a[i] * b[i];
    }
    
    // Normalize by the product of magnitudes
    double norm_a = a.norm();
    double norm_b = b.norm();
    
    // Ensure dot product is between -1 and 1 to avoid numerical issues
    double cosine = std::min(1.0, std::max(-1.0, dot_product / (norm_a * norm_b)));
    
    return std::acos(cosine);
}

// distance gradients: grad of d^2

/**
 * @brief Computes the gradient of the squared Euclidean distance.
 * @param a The first point.
 * @param b The second point.
 * @return The gradient of the squared Euclidean distance.
 */
template <int spacedim>
Vector<double> euclidean_distance_gradient(
    const Point<spacedim> a, const Point<spacedim> b) {
    Vector<double> gradient(spacedim);
    for (unsigned int i = 0; i < spacedim; ++i) {
        gradient[i] = (a[i] - b[i]);
    }
    return gradient;
}

// it is actually the log map and grad of d^2, mind the order, x is the evaluation point
/**
 * @brief Computes the gradient of the squared spherical distance.
 * @param a The first point.
 * @param b The second point.
 * @return The gradient of the squared spherical distance.
 */
template <int spacedim>
Vector<double> spherical_distance_gradient(
    const Point<spacedim> a, const Point<spacedim> b) {
    Vector<double> gradient(spacedim);
    double dist = spherical_distance(a, b);

    double dot_product = 0.0;
    for (unsigned int i = 0; i < spacedim; ++i) {
        dot_product += a[i] * b[i];
    }

    for (unsigned int i = 0; i < spacedim; ++i) {
        gradient[i] = -2 * (dist/std::sin(dist)) * (b[i]-dot_product*a[i]);
    }
    return gradient;
}

// exponential maps
/**
 * @brief Computes the exponential map for the Euclidean distance.
 * @param a The point.
 * @param v The vector.
 * @return The exponential map.
 */
template <int spacedim>
Point<spacedim> euclidean_distance_exp_map(
    const Point<spacedim> a, const Vector<double> v) {
    Point<spacedim> b;
    for (unsigned int i = 0; i < spacedim; ++i) {
        b[i] = a[i] + v[i];
    }
    return b;
}

// v must be different form 0, a perpendicular to v
/**
 * @brief Computes the exponential map for the spherical distance.
 * @param a The point.
 * @param v The vector.
 * @return The exponential map.
 */
template <int spacedim>
Point<spacedim> spherical_distance_exp_map(
    const Point<spacedim> a, const Vector<double> v) {
    Point<spacedim> b;
    for (unsigned int i = 0; i < spacedim; ++i) {
        b[i] = std::cos(v.l2_norm())*a[i] + std::sin(v.l2_norm()) * (v[i]/v.l2_norm());
    }
    return b;
}

#endif // DISTANCE_H
