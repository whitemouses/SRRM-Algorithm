#pragma once


#include <random>
#include <thread>
#include "eigen3/Eigen/Dense"

using Vec = Eigen::VectorXd;
using Vecs = std::vector<Vec>;

// Fonction pour générer un point uniforme dans la boule unité en dimension d
Vec sample_point_in_unit_ball(int d) {
    static std::mt19937 gen;
    static std::normal_distribution<double> gaussian_dist;
    static std::uniform_real_distribution<double> uniform_dist;
    // Génère un point gaussien aléatoire
    Vec point(d);
    for (int i = 0; i < d; ++i) {
        point[i] = gaussian_dist(gen);
    }

    // Normalisation pour obtenir un point sur la sphère
    point.normalize();

    // Distance aléatoire à l'intérieur de la boule avec distribution uniforme
    double radius = std::pow(uniform_dist(gen), 1.0 / d);

    return point * radius;
}

// Fonction principale pour échantillonner N points dans la boule unité de dimension d
inline Vecs sample_unit_ball(int N, int d,double r = 1) {
    Vecs samples(N);

    for (int i = 0; i < N; ++i)
        samples[i] = sample_point_in_unit_ball(d)*r;

    return samples;
}
