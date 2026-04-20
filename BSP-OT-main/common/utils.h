#pragma once
#include "types.h"


template<class C,class T>
inline std::vector<std::pair<scalar,size_t>> label(const C& x) {
    std::vector<std::pair<T,size_t>> L(x.size());
    for (auto i  = 0;i<x.size();i++)
        L[i] = {x(i),i};
    return L;
}


template<class T>
inline std::vector<std::vector<T>> getPermutations(std::vector<T> C) {
    std::vector<std::vector<T>> rslt;
    do
    {
        rslt.push_back(C);
    }
    while (std::next_permutation(C.begin(), C.end()));
    return rslt;
}

template <typename T>
std::vector<std::vector<T>> getCyclicPermutations(std::vector<T> vec) {
    std::vector<std::vector<T>> result;
    size_t n = vec.size();

    for (size_t i = 0; i < n; ++i) {
        result.push_back(vec);
        std::rotate(vec.begin(), vec.begin() + 1, vec.end()); // Rotate left
    }

    return result;
}
