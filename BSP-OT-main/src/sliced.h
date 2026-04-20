#ifndef SLICED_H
#define SLICED_H

#include "BSPOT.h"

namespace BSPOT {



template<int static_dim>
ints SlicedAssign(const Points<static_dim>& A,const Points<static_dim>& B) {
    int N = A.cols();
    int dim = A.rows();
    std::vector<std::pair<scalar,int>> dot_mu(N),dot_nu(N);
    Vector<static_dim> d = sampleUnitGaussian<static_dim>(dim);
    for (auto j : range(N)) {
        dot_mu[j] = {d.dot(A.col(j)),j};
        dot_nu[j] = {d.dot(B.col(j)),j};
    }
    std::sort(dot_mu.begin(),dot_mu.end());
    std::sort(dot_nu.begin(),dot_nu.end());
    ints plan(N);
    for (auto j : range(N))
        plan[dot_mu[j].second] = dot_nu[j].second;
    return plan;
}





}

#endif // SLICED_H
