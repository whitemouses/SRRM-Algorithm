#include "SpectralEmbedding.h"
#include <spdlog/spdlog.h>

using namespace BSPOT;


std::pair<Mat, Vec> BSPOT::computeEigenModesSPD(const smat &A, const smat &M, int nb) {
    using namespace Spectra;

    Spectra::SparseSymMatProd<scalar> Aop(M);
    Spectra::SparseCholesky<scalar> Bop(A);

    int nev = std::min(2*nb,(int)A.cols());

    //    std::cout << "computing eigenvectors;" << std::endl;

    SymGEigsSolver<Spectra::SparseSymMatProd<scalar>, Spectra::SparseCholesky<scalar>, GEigsMode::Cholesky> eigs(Aop,Bop,nb,nev);

    eigs.init();
    eigs.compute(SortRule::LargestAlge);

    if (eigs.info() != CompInfo::Successful){
        exit(1);
    }
    Mat rslt = eigs.eigenvectors(nb);

    return {rslt,eigs.eigenvalues()};
}


