#ifndef SAMPLING_H
#define SAMPLING_H

#include "BSPOT.h"
#include <random>

namespace BSPOT {

template<int D>
inline Points<D> sampleUnitBall(int N,int dim = D) {
    static std::mt19937 gen;
    static std::normal_distribution<double> gaussian_dist;
    static std::uniform_real_distribution<double> uniform_dist;

    Points<D> X(dim, N);
    for (auto i : range(N)){
        Vector<D> point(dim);
        for (int j = 0; j < dim; ++j)
            point[j] = gaussian_dist(gen);

        // Normalisation pour obtenir un point sur la sphère
        point.normalize();

        // Distance aléatoire à l'intérieur de la boule avec distribution uniforme
        double radius = std::pow(uniform_dist(gen), 1.0 / dim);

        X.col(i) = point * radius;
    }
    return X;
}

template<int D>
inline Vector<D> sampleUnitGaussian(int dim = D) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::normal_distribution<double> gaussian_dist(0,1);
    Vector<D> point(dim);
    for (int j = 0; j < dim; ++j)
        point[j] = gaussian_dist(gen);
    return point;
}

inline int randint(int a,int b) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(a,b);
    return dist(gen);
}


}

#endif // SAMPLING_H
