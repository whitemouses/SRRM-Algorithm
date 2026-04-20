#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "../src/BSPOTWrapper.h"
#include "../src/PartialBSPMatching.h"
#include "../src/InjectiveMatching.h"
#include "../src/cloudutils.h"
#include "../src/sampling.h"
#include "../src/PointCloudIO.h"
#include "../src/plot.h"
#include "../common/CLI11.hpp"

int NA = 1000;
int NB = 1000;
int nb_plans = 16;

constexpr int static_dim = 3;
int dim = static_dim;

using namespace BSPOT;


using Pts = Points<static_dim>;

scalar eval(const Pts& A,const Pts& B,const ints& T) {
    return (A - B(Eigen::all,T)).squaredNorm()/A.cols();
}

Pts A,B;

ints T;

inline scalar cost(int i,int j) {
    return (A.col(i) - B.col(j)).squaredNorm();
}

void compute() {
    auto start = Time::now();
    PartialBSPMatching BSP(A,B,cost);
    std::vector<InjectiveMatching> plans(nb_plans);
#pragma omp parallel for
    for (auto i : range(nb_plans)) {
        Eigen::Matrix<scalar,static_dim,static_dim> Q = sampleUnitGaussianMat(dim,dim);
        Q = Q.fullPivHouseholderQr().matrixQ();
        plans[i] = BSP.computePartialMatching(Q,false);
    }
    InjectiveMatching plan = MergePlans(plans,cost);
    spdlog::info("compute time {}",TimeFrom(start));
    T = plan;
    spdlog::info("transport cost {}",eval(A,B,T));
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

    CLI::App app("BSPOT partial transport") ;

    app.add_option("--sizeA", NA, "Number of samples in cloud A (default 1000)");
    app.add_option("--iter", nb_plans, "Number of iterations of refinement (default 100)");

    std::string mu_src;
    std::string nu_src;
    app.add_option("--mu_file", mu_src, "source cloud file (if empty then generated)");
    app.add_option("--nu_file", nu_src, "target cloud file (if empty then generated)");

    if (static_dim == -1)
        app.add_option("--dim",dim,"dimension of the clouds, required if compiled with static_dim == -1")->required(true);

    bool viz = false;
    app.add_flag("--viz", viz, "use polyscope");

    CLI11_PARSE(app, argc, argv);

    if (!mu_src.empty()){
        A = ReadPointCloud<static_dim>(mu_src);
        spdlog::info("mu | dim : {} size : {}",A.rows(),A.cols());
    } else {
        A = sampleUnitBall<static_dim>(NA,dim);
    }

    Normalize<static_dim>(A);

    if (!nu_src.empty()){
        B = ReadPointCloud<static_dim>(nu_src);
        Normalize<static_dim>(B);
    } else {
        auto copy = A;
        copy.colwise() += Vector<static_dim>(1,0,0);
        B = concat(A,copy);
        Pts noise = Pts::Random(static_dim,1000);
        B = concat(B,noise);
        //A.colwise() += Vector<static_dim>(0.5,0,0);
    }

    compute();

    if (viz) {
        polyscope::init();

        display<static_dim>("A",A);
        display<static_dim>("B",B);
        plotMatching("partial",A,B,T);

        polyscope::state::userCallback = myCallBack;
        polyscope::show();
    }
    return 0;
}
