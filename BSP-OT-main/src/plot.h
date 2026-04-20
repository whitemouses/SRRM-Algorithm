#ifndef PLOT_H
#define PLOT_H

#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"
#include "BSPOT.h"
#include "coupling.h"

namespace BSPOT {

template<int D>
inline polyscope::CurveNetwork* plotMatching(std::string name,const Points<D>& A,const Points<D>& B,const ints& T) {
    if (T.empty()){
        spdlog::error("plot empty plan");
        return nullptr;
    }
    std::vector<Vector<D>> combined;
    for (auto i : range(A.cols()))
        combined.push_back(A.col(i));
    for (auto i : range(B.cols()))
        combined.push_back(B.col(i));

    std::vector<std::array<int, 2>> edges;
    int N = A.cols();
    for (auto i : range(N)){
        std::array<int, 2> E = {i,T[i]+N};
        edges.push_back(E);
    }
    auto dim = A.rows();
    if (dim == 2)
        return polyscope::registerCurveNetwork2D("plan " + name,combined,edges)->setRadius(0.001);
    else if (dim == 3)
        return polyscope::registerCurveNetwork("plan " + name,combined,edges)->setRadius(0.001);
    else {
        spdlog::error("can't plot in dim {}",dim);
        return nullptr;
    }
}


template<int dim>
inline void plotPartialMatching(std::string name,const Points<dim>& A,const Points<dim>& B,const ints& T) {
    if (T.empty()){
        spdlog::error("plot empty plan");
        return;
    }
    std::vector<Vector<dim>> combined;
    for (auto i : range(A.cols()))
        combined.push_back(A.col(i));
    for (auto i : range(A.cols()))
        combined.push_back(B.col(T[i]));

    std::vector<std::array<int, 2>> edges;
    int N = A.cols();
    for (auto i : range(N)){
        std::array<int, 2> E = {i,i+N};
        edges.push_back(E);
    }
    if (dim == 2)
        polyscope::registerCurveNetwork2D("plan " + name,combined,edges)->setRadius(0.001);
    else if (dim == 3)
        polyscope::registerCurveNetwork("plan " + name,combined,edges)->setRadius(0.001);
    else {
        spdlog::error("can't plot in dim {}",dim);
    }
}


template<int dim>
inline polyscope::CurveNetwork* plotCoupling(std::string name,const Points<dim>& A,const Points<dim>& B,const Coupling& pi) {
    if (pi.size() == 0){
        spdlog::error("plot empty coupling");
        return nullptr;
    }
    std::vector<Vector<dim>> combined;
    for (auto i : range(A.cols()))
        combined.push_back(A.col(i));
    for (auto i : range(B.cols()))
        combined.push_back(B.col(i));

    std::vector<std::array<int, 2>> edges;
    scalars mass;
    int N = A.cols();
    for (int k = 0; k < pi.outerSize(); ++k){
        for (Coupling::InnerIterator it(pi, k); it; ++it){
            std::array<int, 2> E = {(int)it.row(),int(it.col()+N)};
            mass.push_back(it.value());
            edges.push_back(E);
        }
    }
    polyscope::CurveNetwork* pc;
    if (dim == 2){
        pc = polyscope::registerCurveNetwork2D("plan " + name,combined,edges);
    }
    else if (dim == 3)
        pc = polyscope::registerCurveNetwork("plan " + name,combined,edges)->setRadius(0.001);
    else {
        spdlog::error("can't plot in dim {}",dim);
        return nullptr;
    }
    pc->setRadius(0.001);
    pc->addEdgeScalarQuantity("mass",mass);
    return pc;
}


#include <spdlog/spdlog.h>

template<int dim>
inline polyscope::PointCloud* display(std::string label,const Points<dim>& X) {
    if (X.rows() == 3)
        return polyscope::registerPointCloud(label,X.transpose());
    else if (X.rows() == 2) {
        polyscope::view::style = polyscope::NavigateStyle::Planar;
        polyscope::view::projectionMode = polyscope::ProjectionMode::Orthographic;
        return polyscope::registerPointCloud2D(label,X.transpose());
    }
    else {
        spdlog::error("can't display points of dim {}",dim);
    }
    return nullptr;
}

template<int dim>
inline polyscope::PointCloud* display(std::string label,const Points<dim>& X,const Vec& mass) {
    if (dim == 3){
        auto pc = polyscope::registerPointCloud(label,X.transpose());
        pc->addScalarQuantity("mass",mass);
        pc->setPointRadiusQuantity("mass");
        return pc;
    }
    else if (dim == 2) {
        polyscope::view::style = polyscope::NavigateStyle::Planar;
        auto pc = polyscope::registerPointCloud2D(label,X.transpose());
        pc->addScalarQuantity("mass",mass);
        pc->setPointRadiusQuantity("mass");
        return pc;
    }
    else {
        spdlog::error("can't display points of dim {}",dim);
    }
    return nullptr;
}

}

#endif // PLOT_H
