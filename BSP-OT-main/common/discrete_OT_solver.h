#ifndef DISCRETE_OT_SOLVER_H
#define DISCRETE_OT_SOLVER_H

#include "network_simplex_simple.h"
#include "types.h"
#include <iostream>

#include <set>

namespace OT {

using namespace lemon;
using namespace BSPOT;

// all types should be signed
typedef int64_t arc_id_type; // {short, int, int64_t} ; should be able to handle (n1*n2+n1+n2) with n1 and n2 the number of nodes (int_max = 46340^2, i64_max = 3037000500^2)
typedef double supply_type; // {float, double, int, int64_t} ; should be able to handle the sum of supplies and *should be signed* (a demand is a negative supply)
typedef double cost_type;  // {float, double, int, int64_t} ; should be able to handle (number of arcs * maximum cost) and *should be signed*

struct tsflow {
    int from, to;
    supply_type amount;
};

template<class T>
inline ints solve_transport_bijection(const std::vector<T>& G1,const std::vector<T>& G2,const std::function<scalar(const T&,const T&)>& cost,scalar& emd) {

    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

    arc_id_type n1 = G1.size(), n2 = G2.size(); // just demo for the case n1=n2 ; adapt otherwise
    std::vector<supply_type> weights1(n1), weights2(n2); // works faster with integer weights though

    Digraph di(n1, n2);
    NetworkSimplexSimple<Digraph, supply_type, cost_type, arc_id_type> net(di, true, n1 + n2, n1*n2);

    arc_id_type idarc = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            cost_type d = cost(G1[i],G2[j]);
            net.setCost(di.arcFromId(idarc), d);
            idarc++;
        }
    }

    scalar s = 0;
    scalar W = 1./n1;
    // random supplies and demands
    for (int i = 0; i < n1; i++) {
        weights1[di.nodeFromId(i)] = W;
        s += W;
    }
    for (int i = 0; i < n1; i++)
        weights1[di.nodeFromId(i)] /= s;

    s = 0;

    for (int i = 0; i < n2; i++) {
        weights2[di.nodeFromId(i)] = -W;   // targets should be negative
        s += W;
    }
    for (int i = 0; i < n2; i++)
        weights2[di.nodeFromId(i)] /= s;

    net.supplyMap(&weights1[0], n1, &weights2[0], n2);

    net.run();

    emd = net.totalCost();  // resultdist is the EMD

    //std::cout<<"time: "<< (double)std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.<<std::endl;

    ints mapping(n1);
    emd = 0;

    std::set<int> assigned;

    for (arc_id_type i = 0; i < n1; i++) {
        scalar largest = 0;
        int best = 0;
        for (arc_id_type j = 0; j < n2; j++)
        {
            scalar rho = net.flow(di.arcFromId(i*n2 + j));
            emd += rho*(cost(G1[i],G2[j]));
            if (rho > largest) {
                largest = rho;
                best = j;
            }
        }
        if (assigned.contains(best))
            std::cerr << "non-bijective mapping" << std::endl;
        else assigned.insert(best);

        mapping[i] = best;
    }
    return mapping;
}

inline ints solve_transport_bijection(int n1,int n2,const std::function<scalar(size_t,size_t)>& cost) {

    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

    std::vector<supply_type> weights1(n1), weights2(n2); // works faster with integer weights though

    Digraph di(n1, n2);
    NetworkSimplexSimple<Digraph, supply_type, cost_type, arc_id_type> net(di, true, n1 + n2, n1*n2);

    arc_id_type idarc = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            cost_type d = cost(i,j);
            net.setCost(di.arcFromId(idarc), d);
            idarc++;
        }
    }

    scalar s = 0;
    scalar W = 1./n1;
    // random supplies and demands
    for (int i = 0; i < n1; i++) {
        weights1[di.nodeFromId(i)] = W;
        s += W;
    }
    for (int i = 0; i < n1; i++)
        weights1[di.nodeFromId(i)] /= s;

    s = 0;

    for (int i = 0; i < n2; i++) {
        weights2[di.nodeFromId(i)] = -W;   // targets should be negative
        s += W;
    }
    for (int i = 0; i < n2; i++)
        weights2[di.nodeFromId(i)] /= s;

    net.supplyMap(&weights1[0], n1, &weights2[0], n2);

    net.run();

    scalar emd = net.totalCost();  // resultdist is the EMD

    //std::cout<<"time: "<< (double)std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.<<std::endl;

    ints mapping(n1);
    emd = 0;

    std::set<int> assigned;

    for (arc_id_type i = 0; i < n1; i++) {
        scalar largest = 0;
        int best = 0;
        for (arc_id_type j = 0; j < n2; j++)
        {
            scalar rho = net.flow(di.arcFromId(i*n2 + j));
            if (rho > largest) {
                largest = rho;
                best = j;
            }
        }
        if (assigned.contains(best))
            std::cerr << "non-bijective mapping" << std::endl;
        else assigned.insert(best);

        mapping[i] = best;
    }
    return mapping;
}


template<class T>
inline ints solve_transport_bijection(const std::vector<T>& G1,const std::vector<T>& G2,const std::function<scalar(const T&,const T&)>& cost) {
    scalar tmp;
    return solve_transport_bijection<T>(G1,G2,cost,tmp);
}

template<class T>
inline ints solve_transport_bijection(const std::vector<T>& G1,const std::vector<T>& G2,const std::function<scalar(const size_t&,const size_t&)>& cost,scalar& emd) {

    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

    arc_id_type n1 = G1.size(), n2 = G2.size(); // just demo for the case n1=n2 ; adapt otherwise
    std::vector<supply_type> weights1(n1), weights2(n2); // works faster with integer weights though

    Digraph di(n1, n2);
    NetworkSimplexSimple<Digraph, supply_type, cost_type, arc_id_type> net(di, true, n1 + n2, n1*n2);

    arc_id_type idarc = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            cost_type d = cost(i,j);
            net.setCost(di.arcFromId(idarc), d);
            idarc++;
        }
    }

    scalar s = 0;
    scalar W = 1./n1;
    // random supplies and demands
    for (int i = 0; i < n1; i++) {
        weights1[di.nodeFromId(i)] = W;
        s += W;
    }
    for (int i = 0; i < n1; i++)
        weights1[di.nodeFromId(i)] /= s;

    s = 0;

    for (int i = 0; i < n2; i++) {
        weights2[di.nodeFromId(i)] = -W;   // targets should be negative
        s += W;
    }
    for (int i = 0; i < n2; i++)
        weights2[di.nodeFromId(i)] /= s;

    net.supplyMap(&weights1[0], n1, &weights2[0], n2);

    net.run();

    emd = net.totalCost();  // resultdist is the EMD

    //std::cout<<"time: "<< (double)std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.<<std::endl;

    ints mapping(n1);
    emd = 0;

    std::set<int> assigned;

    for (arc_id_type i = 0; i < n1; i++) {
        scalar largest = 0;
        int best = 0;
        for (arc_id_type j = 0; j < n2; j++)
        {
            scalar rho = net.flow(di.arcFromId(i*n2 + j));
            emd += rho*(cost(i,j));
            if (rho > largest) {
                largest = rho;
                best = j;
            }
        }
        if (assigned.contains(best))
            std::cerr << "non-bijective mapping" << std::endl;
        else assigned.insert(best);

        mapping[i] = best;
    }
    return mapping;
}

template<class T>
inline Eigen::SparseMatrix<scalar,Eigen::RowMajor> solve_transport(const std::vector<T>& G1,const std::vector<T>& G2,const std::function<scalar(const size_t&,const size_t&)>& cost) {

    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

    arc_id_type n1 = G1.size(), n2 = G2.size(); // just demo for the case n1=n2 ; adapt otherwise
    std::vector<supply_type> weights1(n1), weights2(n2); // works faster with integer weights though

    Digraph di(n1, n2);
    NetworkSimplexSimple<Digraph, supply_type, cost_type, arc_id_type> net(di, true, n1 + n2, n1*n2);

    arc_id_type idarc = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            cost_type d = cost(i,j);
            net.setCost(di.arcFromId(idarc), d);
            idarc++;
        }
    }

    scalar s = 0;
    scalar W = 1./n1;
    // random supplies and demands
    for (int i = 0; i < n1; i++) {
        weights1[di.nodeFromId(i)] = W;
        s += W;
    }
    for (int i = 0; i < n1; i++)
        weights1[di.nodeFromId(i)] /= s;

    s = 0;

    for (int i = 0; i < n2; i++) {
        weights2[di.nodeFromId(i)] = -W;   // targets should be negative
        s += W;
    }
    for (int i = 0; i < n2; i++)
        weights2[di.nodeFromId(i)] /= s;

    net.supplyMap(&weights1[0], n1, &weights2[0], n2);

    net.run();

    scalar emd = net.totalCost();  // resultdist is the EMD

    //std::cout<<"time: "<< (double)std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.<<std::endl;

    Eigen::SparseMatrix<scalar,Eigen::RowMajor> mapping(n1,n2);
    std::vector<triplet> triplets;

    std::set<int> assigned;

    for (arc_id_type i = 0; i < n1; i++) {
        for (arc_id_type j = 0; j < n2; j++)
        {
            scalar rho = net.flow(di.arcFromId(i*n2 + j));
            if (rho > 1e-12)
                triplets.push_back({(int)i,(int)j,rho});
        }
    }
    mapping.setFromTriplets(triplets.begin(),triplets.end());
    return mapping;
}

}


#endif // DISCRETE_OT_SOLVER_H
