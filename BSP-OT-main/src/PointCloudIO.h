#ifndef POINTCLOUDIO_H
#define POINTCLOUDIO_H

#include "BSPOT.h"
#include <fstream>
#include <iostream>

namespace BSPOT {

template<int D>
inline Points<D> ReadPointCloud(std::filesystem::path path) {
    std::ifstream infile(path);

    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << path.filename() << std::endl;
        throw std::runtime_error("File not found");
    }

    std::vector<double> data; // Store all values in a single contiguous array
    int rows = 0;
    std::string line;
    int dim = D;

    // First pass: Read the file and store numbers in a vector
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double num;
        int current_cols = 0;

        while (iss >> num) {
            data.push_back(num);
            ++current_cols;
        }


        if (dim == -1)
            dim = current_cols;
        if (current_cols != dim) {
            throw std::runtime_error("Inconsistent dimensions in point cloud file or static dim != point cloud dim");
        }
        ++rows;
    }

    // Second pass: Copy the data into an Eigen matrix
    // where each col is a point
    Points<D> pointCloud(dim, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < dim; ++j) {
            pointCloud(j, i) = data[i * dim + j];
        }
    }
    return pointCloud;
}

template<int D>
void WritePointCloud(std::filesystem::path path,const Points<D>& pts) {
    // each row is a point
    std::ofstream outfile(path);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << path.filename() << std::endl;
        return;
    }

    for (int i = 0; i < pts.cols(); ++i) {
        for (int j = 0; j < pts.rows(); ++j) {
            outfile << pts(j, i);
            if (j < pts.rows() - 1) {
                outfile << " ";
            }
        }
        outfile << "\n";
    }
}

}

#endif // POINTCLOUDIO_H
