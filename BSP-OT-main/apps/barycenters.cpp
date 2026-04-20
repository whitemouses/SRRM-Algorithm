#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "../src/BSPOTWrapper.h"
#include "../src/cloudutils.h"
#include "../src/sampling.h"
#include "../src/PointCloudIO.h"
#include "../src/plot.h"
#include "../src_images/image_as_measure.h"
#include "../common/CLI11.hpp"


int N = 1000;
int nb_plans = 64;

constexpr int static_dim = 3;
int dim = static_dim;

using namespace BSPOT;

using Pts = Points<static_dim>;

polyscope::PointCloud* displayAtPos(std::string label, const Pts& X,const vec& x) {
    auto pc = display(label,X);
    glm::mat4 T(1);
    T = glm::translate(T,glm::vec3(x(0),x(1),x(2)));
    pc->setTransform(T);
    return pc;
}


Pts grid(int N) {
    Pts rslt = Pts::Zero(static_dim,N*N);
    for (auto i : range(N)) {
        for (auto j : range(N)) {
            // rslt.col(i*N+j) = vec2(i,j);
        }
    }
    // rslt = normalize(rslt);
    return rslt;
}

class LinearizedWassersteinBarycenters {
    Pts pivot;
    std::vector<Pts> distributions;
    std::vector<BijectiveMatching> plans;

public:

    LinearizedWassersteinBarycenters(){
    }

    void setPivot() {
        pivot = distributions[0];
        // pivot = grid(256);
        // Normalize(pivot);
    }

    void computePlans() {
        plans.resize(distributions.size());
        for (auto k : range(plans.size())) {
            BijectiveBSPMatching<static_dim> BSP(pivot,distributions[k]);
            auto C = [&] (int i,int j) {
                return (pivot.col(i) - distributions[k].col(j)).squaredNorm();
            };
            plans[k] = computeGaussianBSPOT(pivot,distributions[k],nb_plans,C);
        }
    }

    void addDistribution(const Pts& D) {
        distributions.push_back(D);
    }

    BijectiveMatching toPivot(const Pts& X,int nb,BijectiveMatching T = BijectiveMatching()) const {
        auto cost = [&] (int i,int j) {
            return (X.col(i) - pivot.col(j)).squaredNorm();
        };
        return computeBijectiveBSPOT(X,pivot,nb,cost,T);
    }

    BijectiveMatching toDistribution(const BijectiveMatching& T,int id) const {
        return plans[id](T);
    }

    Pts computeBarycenter(const Vec& w,int iter) {
        if (w.size() != plans.size()) {
            spdlog::error("weights size {} != plans size {}",w.size(),plans.size());
            return Pts();
        }
        for (auto i : range(plans.size()))
            if (std::abs(w(i) - 1) < 1e-6)
                return distributions[i];
        Pts barycenter = Pts::Zero(static_dim,pivot.cols());
        for (auto i : range(plans.size()))
            barycenter += w(i)*distributions[i](Eigen::all,plans[i]);

        BijectiveMatching T;
        for (auto i : range(iter)) {
            T = toPivot(barycenter,nb_plans,T);
            barycenter = Pts::Zero(dim,pivot.cols());
            for (auto j : range(plans.size()))
                barycenter += w(j)*distributions[j](Eigen::all,toDistribution(T,j));
        }

        return barycenter;
    }

    scalar r = 3;

    void plotDistributions() {

        for (auto i : range(distributions.size())) {
            Vec w = Vec::Zero(distributions.size());
            w(i) = 1;
            vec p = screenposFromWeight(w);
            displayAtPos("mesure" + std::to_string(i),distributions[i],p);
        }

    }

    vec screenposFromWeight(const Vec& w) {
        vec rslt(0,0,0);

        for (auto i : range(distributions.size())) {
            scalar th = 2*M_PI*scalar(i+1)/distributions.size();
            rslt += w(i)*vec(r*cos(th),r*sin(th),0);
        }
        return rslt;
    }
};

LinearizedWassersteinBarycenters LWB;

void compute() {

    LWB.setPivot();

    auto start = Time::now();

    LWB.computePlans();
    spdlog::info("compute time pivot plans {}",TimeFrom(start));

    scalar total_time = TimeFrom(start);
    int c = 0;
    for (auto i : range(5))
        for (auto j : range(5)) {
            Vec W = Vec::Zero(4);
            scalar x = scalar(i)/4.;
            scalar y = scalar(j)/4.;


            W << (1-x)*(1-y), x*(1-y), (1-x)*y, x*y;
            // std::cout << W.transpose() << std::endl;

            // W /= W.sum();
            auto start = Time::now();
            Pts B0 = LWB.computeBarycenter(W,0);
            total_time += TimeFrom(start);
            displayAtPos("barycenter " + std::to_string(c++),B0,vec(x,y,0)*4);

            std::string filename("/tmp/barycenterBSPOT_" + std::to_string(i) + "_" + std::to_string(j) + ".pts");
            WritePointCloud(filename,B0);
        }

    spdlog::info("total compute time {}",total_time);


}

float t = 0;
polyscope::PointCloud* pclerp = nullptr;
void myCallBack() {
}

int main(int argc,char** argv) {
    srand(time(NULL));

    CLI::App app("BSPOT linearized wasserstein barycenters");

    app.add_option("--sizes", N, "Number of samples in each cloud (default 1000)");
    app.add_option("--iter", nb_plans, "Number of iterations of refinement (default 100)");


    std::vector<std::string> sources;
    app.add_option("--sources", sources, "List of source files")->expected(-1); // -1 means any number of arguments

    if (static_dim == -1)
        app.add_option("--dim",dim,"dimension of the clouds, required if compiled with static_dim == -1")->required(true);

    bool viz = false;
    app.add_flag("--viz", viz, "use polyscope");

    CLI11_PARSE(app, argc, argv);

    int w,h;
    for (const auto& s : sources) {
        auto D = LoadImageRGB(s,w,h).cast<scalar>();
        // auto D = BSPOT::LoadImageAsMeasure(s).drawFrom(N);
        // Normalize(D);
        LWB.addDistribution(D);
    }

    LWB.setPivot();
    LWB.computePlans();

    auto Mean = LWB.computeBarycenter(Vec::Ones(sources.size())/scalar(sources.size()),0);


    exportImageRGB("/tmp/mean_colors.jpg",Mean,w,h);


    if (viz) {
        polyscope::init();

        // compute();

        // LWB.plotDistributions();

        polyscope::state::userCallback = myCallBack;
        polyscope::show();
    }
    return 0;
}
