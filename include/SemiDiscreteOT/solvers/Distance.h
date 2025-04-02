#ifndef DISTANCE_H
#define DISTANCE_H

#include <deal.II/base/point.h>

using namespace dealii;

template <int spacedim>
double euclidean_distance(
    const Point<spacedim> a, const Point<spacedim> b) {
    return (a-b).norm();
}

#endif // DISTANCE_H
