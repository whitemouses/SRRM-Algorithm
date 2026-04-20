#ifndef POINTCLOUDIO_H
#define POINTCLOUDIO_H

#include "eigen3/Eigen/Dense"
#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>

namespace PointCloudIO
{

using vecs = std::vector<Eigen::Vector3d>;
using Vecs = std::vector<Eigen::VectorXd>;
using Mat = Eigen::MatrixXd;

inline void write_point_cloud(std::string filename,const Vecs& X)
{
    std::ofstream file(filename);
    for (const auto& x : X)
        file << x.transpose() << std::endl;
    file.close();
}

vecs read_point_cloud(std::string filename);

inline Vecs ReadPointCloud(std::filesystem::path path) {
    std::ifstream infile(path);

    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << path.filename() << std::endl;
        return {};
    }

    Vecs pointCloud;
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> point;
        double value;

        // Read each dimension of the point
        while (iss >> value) {
            point.push_back(value);
        }

        if (!point.empty()) {
            // Convert to Eigen::VectorXd
            Eigen::VectorXd eigenPoint = Eigen::VectorXd::Map(point.data(), point.size());
            pointCloud.push_back(eigenPoint);
        }
    }
    return pointCloud;
}

inline Mat ReadPointCloudMat(std::filesystem::path path) {
    std::ifstream infile(path);

    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << path.filename() << std::endl;
        return {};
    }

    std::vector<double> data; // Store all values in a single contiguous array
    int rows = 0, cols = -1;
    std::string line;

    // First pass: Read the file and store numbers in a vector
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double num;
        int current_cols = 0;

        while (iss >> num) {
            data.push_back(num);
            ++current_cols;
        }

        if (cols == -1) {
            cols = current_cols; // Set number of columns from first row
        } else if (current_cols != cols) {
            std::cerr << "Error: Inconsistent number of columns in file.\n";
            return {};
        }
        ++rows;
    }

    // Second pass: Copy the data into an Eigen matrix
    // where each col is a point
    Mat pointCloud(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            pointCloud(j, i) = data[i * cols + j];
        }
    }
    return pointCloud;
}


}

#endif // POINTCLOUDIO_H
