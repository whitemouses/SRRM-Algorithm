#ifndef CLOUDUTILS_H
#define CLOUDUTILS_H

#include "BSPOT.h"
#include <random>

namespace BSPOT {

inline void NormalizeDyn(Points<-1> &X, scalar dilat = 1)
{
    Vector<-1> min = X.rowwise().minCoeff();
    Vector<-1> max = X.rowwise().maxCoeff();
    Vector<-1> scale = max - min;
    double f = dilat/scale.maxCoeff();
    Vector<-1> c = (min+max)*0.5;
    X.colwise() -= c;
    X *= f;
}


template<int dim>
void Normalize(Points<dim> &X, Vector<dim> offset = Vector<dim>::Zero(dim), scalar dilat = 1)
{
    if (dim == -1) {
        offset = Vector<dim>::Zero(X.rows());
    }
    Vector<dim> min = X.rowwise().minCoeff();
    Vector<dim> max = X.rowwise().maxCoeff();
    Vector<dim> scale = max - min;
    double f = dilat/scale.maxCoeff();
    Vector<dim> c = (min+max)*0.5;
    X.colwise() -= c;
    X *= f;
    X.colwise() += offset;
}


template<int dim>
Points<dim> concat(const Points<dim>& X,const Points<dim>& Y) {
    Points<dim> rslt(X.rows(),X.cols() + Y.cols());
    rslt << X,Y;
    return rslt;
}

template<int dim>
Points<dim> pad(const Points<dim>& X,int target) {
    int n = X.cols();
    Points<dim> rslt(dim,target);
    for (auto i : range(target))
        rslt.col(i) = X.col(rand()%n);
    return rslt;
}


template<int dim>
Points<dim> trunc(const Points<dim>& X,int target) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 g(rd());
    ints I = rangeVec(X.cols());
    ::std::shuffle(I.begin(),I.end(),g);
    Points<dim> rslt(X.rows(),target);
    for (auto i : range(target))
        rslt.col(i) = X.col(I[i]);
    return rslt;
}

template<int dim>
inline Points<dim> ForceToSize(const Points<dim>& X,int target) {
    if (X.size() == target)
        return X;
    if (X.size() < target)
        return pad(X,target);
    return trunc(X,target);
}

}


#endif // CLOUDUTILS_H
