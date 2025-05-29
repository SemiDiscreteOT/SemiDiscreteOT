#ifndef DISTANCE_H
#define DISTANCE_H

#include <deal.II/base/point.h>

using namespace dealii;

template <int spacedim>
double euclidean_distance(
    const Point<spacedim> a, const Point<spacedim> b) {
    return (a-b).norm();
}

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

// distance gradients
template <int spacedim>
Vector<double> euclidean_distance_gradient(
    const Point<spacedim> a, const Point<spacedim> b) {
    Vector<double> gradient(spacedim);
    for (unsigned int i = 0; i < spacedim; ++i) {
        gradient[i] = (a[i] - b[i]) / (a-b).norm();
    }
    return gradient;
}

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
template <int spacedim>
Point<spacedim> euclidean_distance_exp_map(
    const Point<spacedim> a, const Vector<double> v) {
    Point<spacedim> b;
    for (unsigned int i = 0; i < spacedim; ++i) {
        b[i] = a[i] + v[i];
    }
    return b;
}

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
