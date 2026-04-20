#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "../src/BSPOTWrapper.h"
#include "../src/sampling.h"
#include "../src/PointCloudIO.h"
#include "../src/plot.h"
#include "../src/cloudutils.h"
#include "../src_images/image_as_measure.h"
#include "../common/CLI11.hpp"

int N = 1000;
int nb_marginals = 2;
int nb_plans = 64;

constexpr int total_dim = 3;
constexpr int sub_dim = total_dim -1;

using namespace BSPOT;

using Pts = Points<total_dim>;
using ProjPts = Points<sub_dim>;
using projector = std::function<ProjPts(const Pts&)>;
using unprojector = std::function<Pts(const ProjPts&)>;
using Matching = BijectiveMatching;

scalar eval(const Pts& A,const Pts& B,const ints& T) {
    return (A - B(Eigen::all,T)).squaredNorm()/A.cols();
}
scalar eval(const ProjPts& A,const ProjPts& B,const ints& T) {
    return (A - B(Eigen::all,T)).squaredNorm()/A.cols();
}

cost_function make_cost(const ProjPts& X,const ProjPts& Y) {
    return [&X,&Y](int i,int j) {
        return (X.col(i) - Y.col(j)).squaredNorm();
    };
}

int count_diff(const ints& T1,const ints& T2) {
    int count = 0;
    for (int i = 0;i<T1.size();i++) {
        if (T1[i] != T2[i])
            count++;
    }
    return count;
}

std::ofstream out("/tmp/nb_change.data");
std::ofstream out_cost("/tmp/W2.data");

struct MultiMarginalsSover {
    Pts mu;
    std::vector<ProjPts> marginals;
    std::vector<BijectiveMatching> plans;
    int nb;

    std::vector<projector> projectors;
    std::vector<unprojector> unprojectors;


    MultiMarginalsSover() {}


    MultiMarginalsSover(const std::vector<ProjPts>& marginals) : marginals(marginals) {
        nb = marginals.size();
        plans.resize(nb);

        projectors.resize(nb);
        unprojectors.resize(nb);

        for (auto& x : this->marginals)
            x *= 2;


        for (auto i = 0;i<nb;i++) {
            projectors[i] = [i] (const Pts& X) -> ProjPts {
                ProjPts Y(X.rows()-1,X.cols());
                Y.setZero();
                for (int j = 0;j<X.cols();j++) {
                    for (int k = 0;k<X.rows();k++) {
                        if (k < i)
                            Y(k,j) = X(k,j);
                        else if (k > i)
                            Y(k-1,j) = X(k,j);
                    }
                }
                return Y;
            };
            unprojectors[i] = [i] (const ProjPts& X) -> Pts {
                Pts Y(X.rows()+1,X.cols());
                Y.setZero();
                for (int j = 0;j<X.cols();j++) {
                    for (int k = 0;k<X.rows();k++) {
                        if (k < i)
                            Y(k,j) = X(k,j);
                        else if (k >= i)
                            Y(k+1,j) = X(k,j);
                    }
                }
                return Y;
            };
        }
        mu = unprojectors[0](marginals[0]);
        plans[0] = rangeVec(mu.cols());
        // Normalize(mu);
    }

    const Pts& getPoints() const {return mu;}

    StopWatch profiler;

    scalar compute_time = 0;
    int nb_iter = 0;

    void iterate(int nb_iters) {
        nb_iter += nb_iters;
        auto time = Time::now();

        for (int i = 0;i<nb_iters;i++) {
            Pts Grad = Pts::Zero(total_dim,N);
            scalar W2 = 0;
            for (int j = 0;j<nb;j++) {
                // profiler.start();
                ProjPts Pimu = projectors[j](mu);
//                profiler.tick("project");

                const auto& nu = marginals[j];
                auto cost = make_cost(Pimu,nu);
                // auto new_plan = computeGaussianBSPOT(Pimu,nu,nb_plans,cost,plans[j]);
                plans[j] = computeGaussianBSPOT(Pimu,nu,nb_plans,cost,plans[j]);
//                spdlog::info("score {}:{}",j,eval(Pimu,nu,plans[j]));
                // W2 += eval(Pimu,nu,plans[j]);
                // profiler.tick("compute plan");
                // if (plans[j].size() != 0 && j == 0){
                //     spdlog::info("plan {} : nb diff {}",j,count_diff(new_plan,plans[j]));
                //     out << count_diff(new_plan,plans[j]) << std::endl;
                // }
                // plans[j] = new_plan;
                Grad += unprojectors[j](nu(Eigen::all,plans[j]) - Pimu)*0.5;
//                profiler.tick("unproj");
            }
            mu += Grad;
            // out_cost << W2 << std::endl;
        }
        compute_time += TimeFrom(time);
        // profiler.profile();
    }
};

MultiMarginalsSover MMS;
std::string nu_src1,nu_src2;

const int res = 512;
polyscope::PointCloud* pcmu = nullptr;


void displayMarginals() {
    for (auto i : range(MMS.nb)) {
        auto mui = MMS.unprojectors[i](MMS.projectors[i](MMS.getPoints()));
        display("marginal " + std::to_string(i),mui);
    }
}

void compute() {
    ProjPts nu1 = LoadImageAsMeasure(nu_src1,res,res).drawFrom(N);
    ProjPts nu2 = LoadImageAsMeasure(nu_src2,res,res).drawFrom(N);
    std::vector<ProjPts> marginals = {nu1,nu2};
    MMS = MultiMarginalsSover(marginals);
    auto start = Time::now();
    MMS.iterate(10);
    auto total_time = TimeFrom(start);
    spdlog::info("total compute time {}",MMS.compute_time);

    pcmu = display("X",MMS.getPoints());
    displayMarginals();
}

void exportPoints() {
    for (auto i : range(MMS.nb))
        WritePointCloud("/tmp/nu"+std::to_string(i+1)+".pts",MMS.unprojectors[i](MMS.projectors[i](MMS.getPoints())));
    WritePointCloud("/tmp/MMS.pts",MMS.getPoints());
}


bool run = false;

void myCallBack() {
    if (run) {
        auto start = Time::now();
        MMS.iterate(1);
        pcmu->updatePointPositions(MMS.getPoints().transpose());
        displayMarginals();
        spdlog::info("total compute time {}",MMS.compute_time);
    }
    if (ImGui::Button("export")) {
        exportPoints();
    }
    ImGui::Checkbox("run",&run);
}

int main(int argc,char** argv) {
    srand(time(NULL));
    polyscope::init();

    CLI::App app("BSPOT multimarginals") ;

    app.add_option("--sizes", N, "Number of samples in each cloud (default 1000)");
    app.add_option("--nu1", nu_src1, "first marginals");
    app.add_option("--nu2", nu_src2, "second marginals");
    app.add_option("--nb_trees", nb_plans, "nb trees");

    CLI11_PARSE(app, argc, argv);

    polyscope::init();

    compute();

    polyscope::state::userCallback = myCallBack;
    polyscope::show();
    return 0;
}
