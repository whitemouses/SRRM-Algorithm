#include "coupling.h"


BSPOT::scalar BSPOT::EvalCoupling(const Coupling &pi, const cost_function &cost) {
    scalar W = 0;
    for (int k = 0;k<pi.outerSize();k++)
        for (Coupling::InnerIterator it(pi,k);it;++it)
            W += cost(it.row(),it.col())*it.value();
    return W;
}
