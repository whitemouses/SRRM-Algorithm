#ifndef INJECTIVEMATCHING_H
#define INJECTIVEMATCHING_H

#include "BSPOT.h"
#include "data_structures.h"

namespace BSPOT {

class InjectiveMatching
{
public:
    using TransportPlan = ints;
    using InverseTransportPlan = ints;

    int image_domain_size  = -1;

    InjectiveMatching(int m);
    InjectiveMatching();
    InjectiveMatching(const TransportPlan& T,int m);

    scalar evalMatching(const cost_function& cost) const;

    const TransportPlan& getPlan() const;

    size_t operator[](size_t i) const;
    size_t operator()(size_t i) const;
    size_t size() const;
    operator TransportPlan() const;

    InverseTransportPlan inversePlan();
    InverseTransportPlan inversePlan() const;

    static bool swapIfUpgrade(ints& T,ints& TI,const ints& TP,int a,const cost_function& cost);

    static InjectiveMatching Merge(const InjectiveMatching& T1,const InjectiveMatching& T2,const cost_function& cost);

    InverseTransportPlan getInverse() const;


protected:
    InjectiveMatching(const TransportPlan& T,const TransportPlan& TI);
    TransportPlan plan;
    InverseTransportPlan inverse_plan;

    const TransportPlan& getInversePlan();

};

InjectiveMatching MergePlans(const std::vector<InjectiveMatching>& plans,const cost_function& cost,InjectiveMatching T = InjectiveMatching());


}
#endif // INJECTIVEMATCHING_H
