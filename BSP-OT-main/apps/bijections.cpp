#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "../src/BSPOTWrapper.h"
#include "../src/cloudutils.h"
#include "../src/sampling.h"
#include "../src/PointCloudIO.h"
#include "../src/PointCloudIO.h"
#include "../src/plot.h"
#include "../src/sliced.h"
#include <queue>
#include <omp.h>
#include <thread>
#include <chrono>
#include <condition_variable>
#include "../common/discrete_OT_solver.h"
#include "../common/CLI11.hpp"


int N = 1000;
int nb_plans = 16;

constexpr int static_dim = 3;
int dim = static_dim;

using namespace BSPOT;

using Pts = Points<static_dim>;

scalar eval(const Pts& A,const Pts& B,const ints& T) {
    return std::sqrt((A - B(Eigen::all,T)).squaredNorm());
}

Pts A,B;

inline scalar cost(int i,int j) {
    return (A.col(i) - B.col(j)).squaredNorm();
}

BijectiveMatching T,Tex;

void compute() {
    auto plans = std::vector<BijectiveMatching>(nb_plans);
    auto start = Time::now();
    BijectiveBSPMatching<static_dim> BSP(A,B);
#pragma omp parallel for
    for (auto& plan : plans) {
        plan = BSP.computeGaussianMatching();
    }
    spdlog::info("compute time {}",TimeFrom(start));
    start = Time::now();
    T = MergePlansNoPar(plans,cost,BijectiveMatching(),(N < 5e5));
    scalar w = eval(A,B,T);
    spdlog::info("merge time {} cost {}",TimeFrom(start),w);
}

Pts lerp(float t) {
    return Pts(A*(1-t) + t*B(Eigen::all,T));
}


float t = 0;
polyscope::PointCloud* pclerp = nullptr;
void myCallBack() {
    if (ImGui::SliderFloat("lerp",&t,0,1)){
        auto L = lerp(t);
        if (!pclerp)
            pclerp = display<static_dim>("lerp",L);
        else{
            if (L.rows() == 2)
                pclerp->updatePointPositions2D(L.transpose());
            else if (L.rows() == 3)
                pclerp->updatePointPositions(L.transpose());
        }
    }
}

int main(int argc,char** argv) {
    srand(time(NULL));
    if (std::is_same<scalar,double>::value) {
        spdlog::warn("for the bijective matching, compiling using float may lead to a significant speed-up without affecting its quality \n you can change the type in 'common/types.h' ");
    }

    CLI::App app("BSPOT bijection") ;

    app.add_option("--sizes", N, "Number of samples in each cloud (default 1000)");
    app.add_option("--nb_trees", nb_plans, "Number of iterations of refinement (default 100)");

    std::string mu_src;
    std::string nu_src;
    app.add_option("--mu_file", mu_src, "source cloud file (if empty then generated)");
    app.add_option("--nu_file", nu_src, "target cloud file (if empty then generated)");

    std::string plan_src;
    app.add_option("--plan_file", plan_src, "initial transport plan");

    std::string out_src;
    app.add_option("--output_file", out_src, "output transport plan");

    if (static_dim == -1)
        app.add_option("--dim",dim,"dimension of the clouds, required if compiled with static_dim == -1")->required(true);

    bool viz = false;
    app.add_flag("--viz", viz, "use polyscope");
    bool force_size = false;
    app.add_flag("--force_size", force_size, "reduces the size of the measures to sizes if true");

    CLI11_PARSE(app, argc, argv);

    if (!mu_src.empty()){
        A = ReadPointCloud<static_dim>(mu_src);
        spdlog::info("mu | dim : {} size : {}",A.rows(),A.cols());
    } else {
        A = sampleUnitBall<static_dim>(N,dim);
    }

    if (!nu_src.empty()){
        B = ReadPointCloud<static_dim>(nu_src);
    } else {
        B = sampleUnitBall<static_dim>(A.cols(),dim);
    }

    if (!plan_src.empty()){
        T = load_plan(plan_src);
        if (T.size() != A.cols() || !T.checkBijectivity()){
            spdlog::error("invalid plan loaded {}",T.size());
            T = BijectiveMatching();
        }
    }

    if (force_size) {
        A = ForceToSize(A,N);
        B = ForceToSize(B,N);
    } else
        N = A.cols();

    Normalize<static_dim>(A);
    Normalize<static_dim>(B);

    compute();

    if (!out_src.empty())
        out_plan(out_src,T);

    if (viz) {
        polyscope::init();

        display<static_dim>("A",A);
        display<static_dim>("B",B);
        plotMatching("bspot",A,B,T)->setEnabled(false);

        polyscope::state::userCallback = myCallBack;
        polyscope::show();
    }
    return 0;
}
