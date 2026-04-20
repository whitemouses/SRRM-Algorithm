#ifndef SPECTRALEMBEDDING_H
#define SPECTRALEMBEDDING_H

#include "Mesh.h"

#include "MeshSampling.h"
#include "../src/BSPOT.h"

#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseCholesky.h>

namespace BSPOT {


//using EigenSolver = Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEigsMode::Cholesky>;
std::pair<Mat,Vec> computeEigenModesSPD(const smat& A,const smat& M,int nb);

inline std::pair<Mat,Vec> filterFirst(std::pair<Mat,Vec> rslt) {
    int n = rslt.first.cols();
    int dim = rslt.first.rows();
    rslt.first = rslt.first.block(0,1,dim,n-1);
    rslt.second = rslt.second.block(1,0,n-1,1);
    return rslt;
}

template<class T>
using VertexData = geometrycentral::surface::VertexData<T>;

inline VertexData<scalar> toVertexData(const Mesh& mesh,const Vec& X) {
    return VertexData<scalar>(*mesh.topology,X);
}

template<int embedding_dim>
class GlobalPointSignature {
public:
    using SpectralPts = Points<embedding_dim>;
    GlobalPointSignature(const Mesh& _mesh,int embed_dim = embedding_dim) : mesh(_mesh),dim(embed_dim) {
        computeEigenModes();
    }

    SpectralPts computeEmbedding(const SurfacePoints& X) const
    {
        SpectralPts EX = SpectralPts::Zero(dim,X.size());
        for (auto i : range(dim)) {
            auto VGPS = toVertexData(mesh,eigenvectors.col(i));
            for (auto j : range(X.size())) {
                auto v = X[j].interpolate(VGPS);
                EX(i,j) = v;
            }
        }
        return EX;
    }

    scalar Varadhan(const Vec& GPS1,const Vec& GPS2) const;

    scalar L2(const Vec& GPS1,const Vec& GPS2) const;

    //const Diag& getMetric() const {return Metric;}

protected:
    Mat eigenvectors;
    Vec vals;
    Vec scale;
    smat Mass;
    //Diag Metric;

    scalar tau;

    const Mesh& mesh;
    int dim;


    void computeEigenModes()
    {
        auto V = mesh.topology->nVertices();
        mesh.geometry->requireCotanLaplacian();
        //mesh.geometry->requireVertexGalerkinMassMatrix();
        mesh.geometry->requireVertexLumpedMassMatrix();
        smat L = mesh.geometry->cotanLaplacian.cast<scalar>() + Identity(V)*1e-8;
        //smat M = mesh.geometry->vertexGalerkinMassMatrix;
        Mass = mesh.geometry->vertexLumpedMassMatrix.cast<scalar>();
        auto modes = computeEigenModesSPD(L,Mass,dim); //check if lambda 0 is necessary
        if (false) {
            dim--;
            eigenvectors = modes.first.rightCols(dim);
            vals.resize(dim);
            for (auto i : range(dim))
                vals(i) = modes.second(i+1);
        }
        else {
            eigenvectors = modes.first;
            vals = modes.second;
        }
        scale.resize(dim);

        mesh.geometry->requireMeshLengthScale();
        scalar h = mesh.geometry->meshLengthScale;
        tau = h;
        spdlog::info("tau {}",h*h);

        eigenvectors.col(0) = Vec::Zero(dim);

        for (auto i : range(1,dim)){
            eigenvectors.col(i) *= std::sqrt(vals[i])/std::sqrt(eigenvectors.col(i).dot(Mass*eigenvectors.col(i)));
            scale[i] = vals(i);//std::exp(-tau/vals[i]);
        }
    }

};


}




#endif // SPECTRALEMBEDDING_H
