#ifndef IMAGE_AS_MEASURE_H
#define IMAGE_AS_MEASURE_H

#include "../src/BSPOT.h"
#include "../src/coupling.h"
#include "load_image.h"
#include <random>

namespace BSPOT {

template<int dim>
struct Histogram {
    Points<dim> positions;
//    Atoms weights;
    Vec mass;
    Eigen::Vector<int,dim> dimensions;

    Points<dim> drawFrom(int n) {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::vector<scalar> W(mass.size());
        for (auto i : range(mass.size()))
            W[i] = mass(i);
        std::discrete_distribution<> d(W.begin(),W.end());
        Points<dim> X(dim,n);
        for (auto i : range(n)) {
            int j = d(gen);
            X.col(i) = positions.col(j);
        }
        return X;
    }

    Atoms getAtoms() const {
        return FromMass(mass);
    }

    // remove all atoms with mass below threshold
    void filterThreshold(scalar threshold) {
        ints keep;
        for (auto i : range(mass.size())) {
            if (mass(i) > threshold)
                keep.push_back(i);
        }
        Points<dim> new_pos = Points<dim>::Zero(dim,keep.size());
        Vec new_mass = Vec::Zero(keep.size());;
        for (auto i : range(keep.size())) {
            new_pos.col(i) = positions.col(keep[i]);
            new_mass(i) = mass(keep[i]);
        }
        positions = new_pos;
        mass = new_mass;
    }
};

Histogram<2> LoadImageAsMeasure(const std::string& filename,int max_w = -1,int max_h = -1,scalar threshold = 0);

Histogram<2> UniformGrid(int n);



};

#endif // IMAGE_AS_MEASURE_H
