#ifndef COUPLING_H
#define COUPLING_H

#include "BSPOT.h"

namespace BSPOT {

using Coupling = Eigen::SparseMatrix<scalar,Eigen::RowMajor>;

scalar EvalCoupling(const Coupling& pi,const cost_function& cost);

template<int D>
Points<D> CouplingToGrad(const Coupling& pi,const Points<D>& A,const Points<D>& B) {
    Points<D> Grad = Points<D>::Zero(A.rows(),A.cols());
    for (int k = 0;k<pi.outerSize();k++)
        for (Coupling::InnerIterator it(pi,k);it;++it)
            Grad.col(it.row()) += (B.col(it.col()) - A.col(it.row()))*it.value();
    return Grad;
}

struct Atom {
    scalar mass;
    int id;
    bool operator<(const Atom& other) const {
        return dot < other.dot;
    }
    scalar dot;
};


using Atoms = std::vector<Atom>;

inline Vec Mass(const Atoms& A) {
    Vec M(A.size());
    for (auto i : range(A.size()))
        M[i] = A[i].mass;
    return M;
}

inline Atoms FromMass(const Vec& x) {
    Atoms rslt(x.size());
    for (auto i : range(x.size())) {
        rslt[i].mass = x[i];
        rslt[i].id = i;
    }
    return rslt;
}

inline Atoms UniformMass(int n) {
    Atoms rslt(n);
    for (auto i : range(n)) {
        rslt[i].mass = 1./n;
        rslt[i].id = i;
    }
    return rslt;
}



struct arrow {
    scalar mass;
    scalar cost;
};

using mapping = std::unordered_map<int,arrow>;

struct CouplingMerger {

    cost_function cost;

    CouplingMerger(const cost_function& cost) : cost(cost) {}
    CouplingMerger() {}


    bool rotateIfUpdate(std::vector<mapping>& pi,std::vector<mapping>& piI,int a,int b,int ap,int bp) {
        if (a == ap || b == bp)
            return false;
        // if (!pi[a].contains(b) || !pi[ap].contains(bp) || !piI[b].contains(a) || !piI[bp].contains(ap)){
        //     spdlog::error("wrong vertices");
        //     return false;
        // }
        const auto& T = pi[a][b];
        const auto& Tp = pi[ap][bp];
        const scalar rho1 = T.mass;
        const scalar rho2 = Tp.mass;
        if (rho1 < 1e-8 || rho2 < 1e-8)
            return false;
        const scalar rho = std::min(rho1,rho2);
        const scalar curr_cost = T.cost*rho1 + Tp.cost*rho2;
        scalar cabp = cost(a,bp);
        scalar capb = cost(ap,b);
        const scalar new_cost = T.cost*(rho1 - rho) + Tp.cost*(rho2-rho) + (cabp + capb)*rho;
        if (new_cost < curr_cost) {

            if (rho1 < rho2) {
                // a-b is deleted
                pi[a].erase(b);
                piI[b].erase(a);

                pi[ap][bp].mass -= rho;
                piI[bp][ap].mass -= rho;
            } else {
                pi[ap].erase(bp);
                piI[bp].erase(ap);

                pi[a][b].mass -= rho;
                piI[b][a].mass -= rho;
            }

            pi[a][bp].mass += rho;
            pi[a][bp].cost = cabp;
            piI[bp][a].mass += rho;
            piI[bp][a].cost = cabp;

            pi[ap][b].mass += rho;
            pi[ap][b].cost = capb;
            piI[b][ap].mass += rho;
            piI[b][ap].cost = capb;
            // spdlog::info("old cost {} new cost {}",curr_cost,new_cost);
            return true;
        }

        return false;
    }

    //connect two portions of the tree by an edge
    void connectTree(std::vector<int>& forest, int tip, int parent, int from) {
        //assumes from is an ancestor of tip
        //flips all edges on the path from tip to from
        //connects tip to its new parent
        //beware that this removes the last edge on the path from tip to from
        int previous = parent ;
        int current = tip ;
        while(current != from) {
            int next = forest[current] ;
            forest[current] = previous ;
            previous = current ;
            current = next ;
        }
    }

    void findLoop(const std::vector<int>& forest, int n1, int n2, std::vector<int>& loop) {
        int size = forest.size() ;
        //static marks buffer to find the forest loop
        //TODO benchmark the utility of the static, not thread safe
        static std::vector<int> marked(size, size) ;
        //mark for this run
        //FIXME this may break if more than 2^32 calls are made
        static int mark = 0 ;
        ++mark ;
        //determine the loop between the source and the target
        //TODO benchmark the utility of the static, not thread safe
        static std::vector<int> loop_buf ;
        loop.resize(0) ;
        loop_buf.resize(0) ;
        loop.push_back(n1) ;
        loop_buf.push_back(n2) ;
        marked[n1] = mark ;
        marked[n2] = mark ;
        while(true) {
            int next = forest[loop.back()] ;
            if(next != size) {
                //this side of the path has not reached the root
                if(marked[next] == mark) {
                    //the loop is found, trim the other portion of the loop
                    while(loop_buf.back() != next) {
                        //safety check, ensure the loop is well formed
                        assert(!loop_buf.empty()) ;
                        loop_buf.pop_back() ;
                    }
                    break ;
                } else {
                    marked[next] = mark ;
                }
            } else {
                if(loop_buf.back() == size) {
                    //the edge creates no loop
                    loop.resize(0) ;
                    return ;
                }
            }
            //no loop found yet, grow the loop and swap the portion to grow
            loop.push_back(next) ;
            if(loop_buf.back() != size) {
                loop.swap(loop_buf) ;
            }
        }
        //finalize the loop in a single vector
        if(loop[0] != n1) loop.swap(loop_buf) ;
        for(int node : loop_buf | std::views::reverse) {
            loop.push_back(node) ;
        }
    }

    //mutate the tree to improve the coupling
    void forestImproveLoop(Coupling& coupling, std::vector<int>& forest, std::vector<int>& loop) {
        //problem dimensions
        int n = coupling.rows() ;
        int m = coupling.cols() ;

        //source and target
        int source = loop[0] ;
        int target = loop.back() - n ;

        //safety check, the loop should alternate source and target in equal numbers
        assert(loop.size() % 2 == 0) ;

        //change in transport cost when rotating mass around the loop
        scalar factor = cost(source, target) ;

        //bottlenecks when rotating mass
        //0 => adding mass transfer between loop extremities (always possible)
        //1 => decreasing mass transfer between loop extremities (only if edge already in the coupling)
        scalar bottleneck[2] = {
                                std::numeric_limits<scalar>::infinity(),
            coupling.coeff(source, target)
        } ;
        int bottleneck_edge[2] = {n+m, n+m} ;
        int bottleneck_start[2] = {n+m, n+m} ;
        //iterate over loop edges
        for(std::size_t i = 0; i < loop.size() - 1; ++i) {
            //extremitiex of the edge
            int v1 = loop[i] ;
            int v2 = loop[i+1] ;
            //alternate adding / removing
            scalar c = 2*(i%2) ;
            c -= 1 ; //beware adding -1 above yields havoc because i is unsigned
            //determine whether extremities are sources or targets
            //get transport cost and currently transiting mass
            scalar m = std::numeric_limits<scalar>::infinity() ;
            if(v2 > v1) {
                c *= cost(v1, v2-n) ;
                m = coupling.coeff(v1, v2-n) ;
            } else {
                c *= cost(v2, v1-n) ;
                m = coupling.coeff(v2, v1-n) ;
            }
            //update bottlenecks
            if(m < bottleneck[i%2]) {
                bottleneck[i%2] = m ;
                if(v2 == forest[v1]) {
                    //the bottleneck is such that there is a path source -> ... -> v1 -> v2
                    bottleneck_edge[i%2] = v2 ;
                    bottleneck_start[i%2] = source ;
                } else {
                    //the bottleneck is such that there is a path target -> ... -> v2 -> v1
                    bottleneck_edge[i%2] = v1 ;
                    bottleneck_start[i%2] = target + n ;
                }
            }
            //contribute to the global cost
            factor += c ;
        }

        //determine how mass should rotate around the loop to yield an improvement
        int index = factor > 0 ;
        int direction = -2*index + 1 ;
        if(bottleneck[index] > 0) {
            //improvement when increasing transfer between loop extremities
            //rotate mass in the coupling
            for(std::size_t i = 0; i < loop.size() - 1; ++i) {
                //extremitiex of the edge
                int v1 = loop[i] ;
                int v2 = loop[i+1] ;
                //alternate adding / removing
                scalar c = 2*(i%2) ;
                c -= 1 ;
                c *= direction ;
                if(v2 > v1) {
                    coupling.coeffRef(v1, v2-n) += c*bottleneck[index] ;
                    assert(coupling.coeffRef(v1, v2-n) >= 0) ;
                } else {
                    coupling.coeffRef(v2, v1-n) += c*bottleneck[index] ;
                    assert(coupling.coeffRef(v2, v1-n) >= 0) ;
                }
            }

            //insert the new edge in the coupling
            coupling.coeffRef(source, target) += direction * bottleneck[index] ;

            //update the forest inserting the edge if it is not the bottleneck
            if(bottleneck_edge[index] != n+m) {
                connectTree(forest,
                        bottleneck_start[index],
                        source + target + n - bottleneck_start[index],
                        bottleneck_edge[index]) ;
            }
            //checkForest(forest, n) ;
        }
    }

    void forestTryEdge(Coupling& coupling, std::vector<int>& forest, int source, int target) {
        //problem dimensions
        int n = coupling.rows() ;
        int m = coupling.cols() ;

        //check whether the edge creates a loop
        //TODO benchmark the utility of the static
        static std::vector<int> loop ;
        findLoop(forest, source, target + n, loop) ;

        if(loop.empty()) {
            //no loop created, add the edge
            connectTree(forest, source, target + n, n+m) ;
            //checkForest(forest, n) ;
            return ;
        }

        if(loop.size() == 2) {
            //the edge is already present in the forest
            return ;
        }

        //a loop is created, try improving it
        forestImproveLoop(coupling, forest, loop) ;
    }

    //build a tree from a coupling
    void buildForest(Coupling& coupling, std::vector<int>& forest) {
        //problem dimensions
        int n = coupling.rows() ;
        int m = coupling.cols() ;
        //the forest stores the parents
        //clear provided vector
        forest.resize(0) ;
        //when no parent use n+m
        forest.resize(n+m, n+m) ;

        //list edges to avoid iterator invalidation
        std::vector<std::tuple<int, int, scalar>> edges ;
        std::vector<scalar> max_edge(n, 0) ;
        edges.reserve(coupling.nonZeros()) ;
        for(int source = 0; source < coupling.outerSize(); ++source) {
            for(Coupling::InnerIterator it(coupling, source); it; ++it) {
                edges.emplace_back(source, it.col(), it.value()) ;
                max_edge[source] = std::max(max_edge[source], it.value()) ;
            }
        }

        //sorting directly by decreasing edge values yields better results
        //but it becomes much slower probably because sorting edges by source
        //has a much better memory access pattern
        std::sort(edges.begin(), edges.end(), 
            [&] (auto const& e1, auto const& e2) { 
              auto [s1, t1, v1] = e1 ;
              auto [s2, t2, v2] = e2 ;
              if(s1 == s2) return v1 > v2 ;
              return max_edge[s1] > max_edge[s2] ; 
            }
            ) ;

        for(auto [source, target, value] : edges) {
          //spdlog::info("trying edge {} -> {} with value {}", source, target, -value) ;
          //edge vertices belong to trees
          //if its the same tree, adding the edge may create a loop
          //if a loop exists, it is deleted, improving transport cost
          forestTryEdge(coupling, forest, source, target) ;
        }
    }

    void improveQuads(Coupling& coupling, std::vector<int>& forest) {
      //store neighborhoods
      std::vector<int> source_neighbors ;
      std::vector<int> source_offsets ;
      std::vector<int> target_neighbors ;
      std::vector<int> target_offsets ;

      source_neighbors.reserve(coupling.nonZeros()) ;
      source_offsets.reserve(coupling.outerSize() + 1) ;
      source_offsets.push_back(0) ;
      target_neighbors.resize(coupling.nonZeros()) ;
      target_offsets.resize(coupling.innerSize() + 1, 0) ;

      //source -> target
      for(int source = 0; source < coupling.outerSize(); ++source) {
        for(Coupling::InnerIterator it(coupling, source); it; ++it) {
          int target = it.col() ;
          source_neighbors.push_back(target) ;
          ++target_offsets[target] ;
        }
        source_offsets.push_back(source_neighbors.size()) ;
      }

      //target->source
      for(int target = 1; target < target_offsets.size(); ++target) {
        target_offsets[target] += target_offsets[target-1] ;
      }
      for(int source = 0; source < coupling.outerSize(); ++source) {
        for(Coupling::InnerIterator it(coupling, source); it; ++it) {
          int target = it.col() ;
          --target_offsets[target] ;
          target_neighbors[target_offsets[target]] = source ;
        }
      }

      //list quad edges
      for(int source = 0; source < coupling.outerSize(); ++source) {
        for(Coupling::InnerIterator it(coupling, source); it; ++it) {
          int target = it.col() ;
          //we have a source->target edge
          //try every edge between their respective neighbors
          for(int i = target_offsets[target]; i < target_offsets[target+1]; ++i) {
            for(int j = source_offsets[source]; j < source_offsets[source+1]; ++j) {
              forestTryEdge(coupling, forest, target_neighbors[i], source_neighbors[j]) ;
            }
          }
        }
      }

      //cleanup zeros in the sparse matrix
      coupling = coupling.pruned() ;
    }

    //safety check the tree
    void checkForest(const std::vector<int>& forest, int target_start) {
        int size = forest.size() ;
        //ensure no loop happens
        std::vector<int> marked(size, size) ;
        for(int i = 0; i < size; ++i) {
            int current = i ;
            marked[i] = i ;
            while(forest[current] < size) {
                current = forest[current] ;
                //assert the graph has no loop
                assert(marked[current] != i) ;
                marked[current] = i ;
            }
        }
        //ensure all edges are from source to target
        for(int i = 0; i < size; ++i) {
            int parent = forest[i] ;
            if(parent < size) {
                if(i < target_start) {
                    assert(parent >= target_start) ;
                } else {
                    assert(parent < target_start) ;
                }
            }
        }
    }

    Coupling forestMerge(const std::vector<Coupling>& couplings) {
        Coupling result = couplings[0] ;
        // spdlog::info("initial coupling cost is {}",eval(A,B,result));

        //source size
        int n = result.rows() ;

        //build initial tree
        std::vector<int> forest ;
        buildForest(result, forest) ;
        //checkForest(forest, n) ;

        //merge the other couplings
        for(std::size_t i = 1; i < couplings.size(); ++i) {
            const Coupling& coupling = couplings[i] ;
            //spdlog::info("merging cost {}",eval(A,B,coupling));
            for(int source = 0; source < coupling.outerSize(); ++source) {
                for (Coupling::InnerIterator it(coupling, source); it; ++it) {
                    int target = it.col() ;
                    forestTryEdge(result, forest, source, target) ;
                }
            }
            //spdlog::info("coupling cost is now {}",eval(A,B,result));
            //checkForest(forest, n) ;
        }

        return result.pruned() ;
    }

    Coupling CycleMerge(const std::vector<Coupling>& couplings) {
        std::vector<bool> visited;

        auto pi1 = couplings[0];

        int n = pi1.rows();
        int m = pi1.cols();

        std::vector<std::unordered_map<int,arrow>> edges(n);
        std::vector<std::unordered_map<int,arrow>> edgesI(m);
        for (auto i = 0;i<pi1.outerSize();i++){
            for (Coupling::InnerIterator it(pi1,i);it;++it) {
                int j = it.col();
                scalar c = cost(i,j);
                edges[i][j] = {it.value(),c};
                edgesI[j][i] = {it.value(),c};
            }
        }
        for (auto i : range(1,couplings.size())){
            const auto& pip = couplings[i];
            //spdlog::info("merging cost {}",eval(A,B,pip));
            for (auto a = 0;a<n;a++){
                for (Coupling::InnerIterator it(pip,a);it;++it) {
                    int bp = it.col();
                    bool ok;
                    do {
                        ok = true;
                        for (auto b : edges[a]) {
                            for (auto ap : edgesI[bp]) {
                                if (rotateIfUpdate(edges,edgesI,a,b.first,ap.first,bp)) {
                                    ok = false;
                                    break;
                                }
                            }
                            if (!ok)
                                break;
                        }
                    } while (!ok);
                }
            }
        }

        std::vector<triplet> triplets;
        for (auto i = 0;i<edges.size();i++){
            for (auto j : edges[i]){
                if (j.second.mass > 1e-8) {
                    triplet t(i,j.first,j.second.mass);
                    triplets.push_back(t);
                }
            }
        }
        Coupling pi(n,m);
        pi.setFromTriplets(triplets.begin(),triplets.end());
        return pi;
    }
};


}

#endif // COUPLING_H
