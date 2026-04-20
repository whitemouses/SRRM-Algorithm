#ifndef MESHSAMPLING_H
#define MESHSAMPLING_H

#include "Mesh.h"
#include <geometrycentral/surface/surface_point.h>
#include <polyscope/point_cloud.h>
#include "../src/BSPOT.h"

namespace BSPOT {

struct PointOnMesh {
    vec pos;
    size_t face_id;
    vec barycentric_coord;
};

//using PointsOnMesh = std::vector<PointOnMesh>;
using SurfacePoint = geometrycentral::surface::SurfacePoint;
using SurfacePoints = std::vector<SurfacePoint>;

SurfacePoints sampleMesh(const Mesh &M, int sampleNum, const scalars &face_weights);

inline scalar drawUnit() {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    static thread_local std::uniform_real_distribution<double> dist(0.,1);
    return dist(gen);
}

inline vec drawBary() {
    scalar alpha, beta;
    alpha = (1 - 0) * drawUnit() + 0;
    beta = (1 - 0) *  drawUnit() + 0;

    scalar a, b, c;
    a = 1 - std::sqrt(beta);
    b = (std::sqrt(beta)) * (1 - alpha);
    c = std::sqrt(beta) * alpha;
    return vec(a,b,c);
}

inline Points<3> toPositions(const Mesh &M, const SurfacePoints &X) {
    Points<3> P(3,X.size());
    for (auto i : range(X.size()))
        P.col(i) = toVec(X[i].interpolate(M.geometry->vertexPositions));
    return P;
}

inline vec drawWeightedBary(const vec& P) {
    scalar s = P.maxCoeff();
    //Rejection sampling
    while (true) {
        auto b = drawBary();
        if (drawUnit() < b.dot(P)/s)
            return b;
    }
    return drawBary();
}

inline Vector3 toVec3(const vec& x) {
    return Vector3(x(0),x(1),x(2));
}

SurfacePoints sampleMeshVertexDensity(const Mesh& M,const Vec& V,int n);



polyscope::PointCloud* display(std::string label,const Mesh& M,const SurfacePoints& X);
}

#endif // MESHSAMPLING_H
