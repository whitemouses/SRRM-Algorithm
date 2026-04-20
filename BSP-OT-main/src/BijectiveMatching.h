#ifndef BIJECTIVEMATCHING_H
#define BIJECTIVEMATCHING_H

#include "BSPOT.h"
#include "data_structures.h"
#include "sampling.h"
#include <fstream>

namespace BSPOT {


class BijectiveMatching
{
public:
    using TransportPlan = ints;

    BijectiveMatching();
    BijectiveMatching(const TransportPlan& T) : plan(T),inverse_plan(getInverse(T)) {}
    BijectiveMatching(const Eigen::Vector<int,-1>& T);

    scalar evalMatching(const cost_function& cost) const;

    template<int D>
    scalar evalMatchingL2(const Points<D>& A,const Points<D>& B) const {
        return (A - B(Eigen::all,plan)).squaredNorm()/A.cols();
    }

    const TransportPlan& getPlan() const;

    size_t operator[](size_t i) const;
    size_t operator()(size_t i) const;
    size_t size() const;
    operator TransportPlan() const;

    BijectiveMatching inverseMatching();

    BijectiveMatching inverseMatching() const;
    bool checkBijectivity() const;

    BijectiveMatching operator()(const BijectiveMatching& other) const;

    template<class T>
    std::vector<T> operator()(const std::vector<T>& X);

    const TransportPlan& getInversePlan();

    bool operator==(const BijectiveMatching& other) const {
        return plan == other.plan;
    }


    static inline bool swapIfUpgrade(ints &T, ints &TI, const ints &TP, int a, const cost_function &cost) {
        int b = T[a];
        int bp = TP[a];
        int ap  = TI[bp];
        if (a == ap || b == bp)
            return false;
        scalar old_cost = cost(a,b) + cost(ap,bp);
        scalar new_cost = cost(a,bp) + cost(ap,b);
        if (new_cost < old_cost) {
            T[a] = bp;
            T[ap] = b;
            TI[bp] = a;
            TI[b] = ap;
            return true;
        }
        return false;
    }

protected:

    BijectiveMatching(const TransportPlan& T,const TransportPlan& TI);
    TransportPlan plan,inverse_plan;

    static TransportPlan getInverse(const TransportPlan& T);
};

BijectiveMatching Merge(const BijectiveMatching &T, const BijectiveMatching &TP, const cost_function &cost,bool verbose = false);

BijectiveMatching MergePlans(const std::vector<BijectiveMatching> &plans,const cost_function &cost,BijectiveMatching T = BijectiveMatching(),bool cycle = true);
BijectiveMatching MergePlansNoPar(const std::vector<BijectiveMatching> &plans,const cost_function &cost,BijectiveMatching T = BijectiveMatching(),bool cycle = true);

bool swapIfUpgradeK(ints &T, ints &TI, const ints &TP, int a,int k, const cost_function &cost);

inline ints rankPlans(const std::vector<BijectiveMatching>& plans,const cost_function& cost) {
    auto start = Time::now();
    std::vector<std::pair<scalar,int>> scores(plans.size());
    for (auto i : range(plans.size())) {
        scores[i].first = plans[i].evalMatching(cost);
        scores[i].second = i;
    }
    std::sort(scores.begin(),scores.end(),[](const auto& a,const auto& b) {
        return a.first < b.first;
    });
    // spdlog::info("sort timing {}",TimeFrom(start));
    ints rslt(scores.size());
    for (auto i : range(scores.size()))
        rslt[i] = scores[i].second;
    return rslt;
}


inline bool checkBijection(const ints& T,const ints& TI) {
    ints I(T.size(),-1);
    for (auto i : range(T.size()))
        I[T[i]] = i;
    bool ok = true;
    for (auto i : range(T.size()))
        if (I[i] == -1){
            spdlog::error("not bijection");
            ok = false;
        }
    for (auto i : range(T.size()))
        if (TI[T[i]] != i){
            spdlog::error("not inverse {} {} {}",i,T[i],TI[T[i]]);
            ok = false;
        }
    return ok;
}

inline void checkBijection(const ints& T) {
    ints I(T.size(),-1);
    for (auto i : range(T.size()))
        I[T[i]] = i;
    for (auto i : range(T.size()))
        if (I[i] == -1)
            spdlog::error("not bijection");
}

BijectiveMatching load_plan(std::string path);

inline void out_plan(std::string out,const BijectiveMatching& T) {
    std::ofstream file(out);
    for (auto i : range(T.size()))
        file << T[i] << "\n";
    file.close();
}


}

#endif // BIJECTIVEMATCHING_H
