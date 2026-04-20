#ifndef SPARSE_MATRIX_IO_H
#define SPARSE_MATRIX_IO_H

#include "types.h"
#include <iostream>
#include <fstream>

// Define a Row-Major Sparse Matrix
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RowMajorSparseMatrix;

// Function to export a Row-Major sparse matrix to a file
void exportSparseMatrix(const RowMajorSparseMatrix &mat, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    file << mat.rows() << " " << mat.cols() << " " << mat.nonZeros() << "\n";

    for (int i = 0; i < mat.outerSize(); ++i) {
        for (typename RowMajorSparseMatrix::InnerIterator it(mat, i); it; ++it) {
            file << it.row() << " " << it.col() << " " << it.value() << "\n";
        }
    }
    file.close();
}

// Function to import a Row-Major sparse matrix from a file
RowMajorSparseMatrix importSparseMatrix(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return RowMajorSparseMatrix(0, 0);
    }

    int rows, cols, nonZeros;
    file >> rows >> cols >> nonZeros;

    RowMajorSparseMatrix mat(rows, cols);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nonZeros);

    int row, col;
    double value;
    while (file >> row >> col >> value) {
        triplets.emplace_back(row, col, value);
    }

    mat.setFromTriplets(triplets.begin(), triplets.end());
    file.close();
    return mat;
}

#endif // SPARSE_MATRIX_IO_H
