#ifndef BSPOT_H
#define BSPOT_H

#include "../common/types.h"
#include "spdlog/spdlog.h"

namespace BSPOT {

template<int dim>
using Points = Eigen::Matrix<scalar,dim,-1,Eigen::ColMajor>;
template<int dim>
using Vector = Eigen::Vector<scalar,dim>;

using cost_function = std::function<scalar(size_t,size_t)>;
template<class Point>
using geometric_cost = std::function<scalar(const Point&,const Point&)>;

template<int dim>
using CovType = Eigen::Matrix<scalar,dim,dim>;

template<int dim>
struct Moments {
    Vector<dim> mean;
    CovType<dim> Cov;
};


template<int D>
Vector<D> Mean(const Points<D>& X) {
    return X.rowwise().mean();
}

template<int D>
CovType<D> Covariance(const Points<D>& X) {
    Vector<D> mean = X.rowwise().mean();
    Points<D> centered = X.colwise() - mean;
    CovType<D> rslt = centered * centered.adjoint() / double(X.cols());
    return rslt;
}

template<int D>
CovType<D> Covariance(const Points<D>& X,const Points<D>& Y) {
    Vector<D> meanX = Mean(X);
    Points<D> centeredX = X.colwise() - meanX;
    Vector<D> meanY = Mean(Y);
    Points<D> centeredY = Y.colwise() - meanY;
    CovType<D> rslt = centeredX * centeredY.adjoint() / double(X.cols());
    return rslt;
}


template<int dim>
CovType<dim> sqrt(const CovType<dim> &A) {
    Eigen::SelfAdjointEigenSolver<CovType<dim>> root(A);
    return root.operatorSqrt();
}

template<int dim>
CovType<dim> W2GaussianTransportMap(const CovType<dim>& A,const CovType<dim>& B){
    Eigen::SelfAdjointEigenSolver<CovType<dim>> sasA(A);
    CovType<dim> root_A = sasA.operatorSqrt();
    CovType<dim> inv_root_A = sasA.operatorInverseSqrt();
    CovType<dim> C = root_A * B * root_A;
    C = sqrt(C);
    C = inv_root_A*C*inv_root_A;
    return C;
}


}

#endif // BSPOT_H
