#ifndef SPECTRAWRAPPER_H
#define SPECTRAWRAPPER_H
#include <iostream>

#include "eigen3/Eigen/Dense"

#include "types.h"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/MatOp/SparseGenRealShiftSolve.h>
#include <Spectra/MatOp/SparseCholesky.h>

using SpectraSPDSolver = Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEigsMode::Cholesky>;

inline SpectraSPDSolver computeEigenModesSPD(const smat& A,const smat& M,int nb) {
    using namespace Spectra;

    Spectra::SparseSymMatProd<double> Aop(M);
    Spectra::SparseCholesky<double> Bop(A);

    int nev = std::min(10*nb,(int)A.cols());

    std::cout << "computing eigenvectors;" << std::endl;

    SymGEigsSolver<Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, GEigsMode::Cholesky> eigs(Aop,Bop,nb,nev);

    eigs.init();
    eigs.compute(SortRule::LargestAlge);

    if (eigs.info() != CompInfo::Successful){
        exit(1);
    }
    return eigs;
}

inline Vecs computeEigenVectorsSPD(const smat& A,const smat& M,int nb) {
    auto&& eigs = computeEigenModesSPD(A,M,nb);
    auto E = eigs.eigenvectors(nb);
    Vecs rslt(nb);
    for (int i = 0;i<nb;i++)
        rslt[i] = E.col(i);

    return rslt;
}

inline Vecs computeEigenVectorsSPD(const smat& A,int nb)  {
    smat I(A.cols(),A.rows());I.setIdentity();
    return computeEigenVectorsSPD(A,I,nb);
}

inline Vecs computeEigenVectors(const smat& A,int nb) {
    using namespace Spectra;

//    Spectra::SparseGenMatProd<double> Aop(A);

    int nev = std::min(10*nb,(int)A.cols());

    std::cout << "computing eigenvectors;" << std::endl;
    using op = SparseGenMatProd<scalar>;
    //using op = SparseGenRealShiftSolve<scalar>;
    op Aop(A);
//    Spectra::GenEigsRealShiftSolver<op> eigs(Aop,nb,nev,1 + 1e-6);
    Spectra::GenEigsSolver<Spectra::SparseGenMatProd<double>> eigs(Aop,nb,nev);

    eigs.init();
    //eigs.compute(SortRule::LargestReal);
    eigs.compute(SortRule::LargestReal);

    if (eigs.info() != CompInfo::Successful){
        exit(1);
    }

    auto E = eigs.eigenvectors(nb);
    Vecs rslt(nb);
    for (int i = 0;i<nb;i++){
        const auto& tmp = E.col(i);
        rslt[i] = Vec::Ones(A.cols());
        for (int j = 0;j<A.cols();j++)
            rslt[i](j) = tmp(j).real();
    }

    return rslt;
}



#endif // SPECTRAWRAPPER_H
