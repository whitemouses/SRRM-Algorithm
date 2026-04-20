#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "../src/BSPOT.h"
#include "../src/BSPOTWrapper.h"
#include "../src/PartialBSPMatching.h"
#include "../src/plot.h"
#include "../src/PointCloudIO.h"
#include "../src/cloudutils.h"
#include "../common/discrete_OT_solver.h"
#include "../common/CLI11.hpp"

// #define DIM 3

constexpr int static_dim = 2;
int dim = static_dim;

using namespace BSPOT;

using Pts = Points<static_dim>;
using Cov = Eigen::Matrix<scalar,static_dim,static_dim>;

Pts Asrc,A,B;
InjectiveMatching T;

#include <fstream>

float alpha = 0.0;

Pts getRigidTransform(Pts X,const Pts& Y,const InjectiveMatching& T) {
    Vector<static_dim> muX = Mean(X);

    Pts Yt = Y(Eigen::all,T);
    Vector<static_dim> muYt = Mean(Yt);

    Cov W = Covariance(X,Yt);
    Eigen::JacobiSVD<Cov> svd(W,Eigen::ComputeFullU | Eigen::ComputeFullV);
    svd.computeU();
    svd.computeV();
    Cov U = svd.matrixU();
    Cov V = svd.matrixV();
    Cov S = Mat::Identity(dim,dim);
    Cov D = svd.singularValues().asDiagonal();

    Pts X0 = X;

    if ((U*V).determinant() < 0) {
        S(dim-1,dim-1) = -1;
    }
    scalar sig2 = (X.colwise()-muX).colwise().squaredNorm().mean();
    scalar scaling = (D*S).trace()/sig2;
    Cov R = V*S*U.transpose();
    R /= R.determinant();

    X.colwise() -= muX;
    X = R*X*scaling;
    X.colwise() += muYt;

    return X0 * alpha + (1-alpha) *X;
}

scalar relative_error(scalar a,scalar b) {
    return (a - b)/b;
}

scalar cost(size_t i,size_t j) {
    return (A.col(i) - B.col(j)).squaredNorm();;
}

int steps = 16;

bool preserve = false;

InjectiveMatching getAssignation(const Pts& X,const Pts& Y) {
    if (preserve)
        return computePartialBSPOT(X,Y,steps,cost,T);
    else
        return computePartialBSPOT(X,Y,steps,cost);//,T);
}

polyscope::PointCloud* pcA;

int nb_iter = 0;



scalar total_time = 0;

void iter() {
    auto start = Time::now();
    nb_iter++;
    T = getAssignation(A,B);
    //T = InjectiveMatching(OT::solve_transport_bijection(A.cols(),B.cols(),cost),A.cols());
    A = getRigidTransform(A,B,T);
    total_time += TimeFrom(start);

    plotPartialMatching("matching",A,B,T.getPlan());
    if (dim == 3)
        pcA->updatePointPositions(A.transpose());
    else if (dim == 2)
        pcA->updatePointPositions2D(A.transpose());
}

Pts lerp(float t) {
    return Pts(A*(1-t) + t*B(Eigen::all,T));
}

bool run = false;

int noise = 0;

Pts addNoise(const Pts& X,int n) {
    if (n == 0)
        return X;
    Pts noise = sampleUnitBall(n,dim);
    return concat(X,noise);
}

Pts ApplyRigid(Pts X,scalar s,const Vector<static_dim>& t) {
    Normalize<static_dim>(X);

    X.colwise() -= X.rowwise().mean();
    X = addNoise(X,noise);
    X *= s;
    X.colwise() += t;
    return X;
}


Pts RandomRigid(Pts X) {
    Cov R;
// #if static_dim == 3
    // vec axis = sampleUnitGaussian<static_dim>(dim).normalized();
    // auto AA = Eigen::AngleAxisd(polyscope::randomReal(-1,1)/3.*M_PI,axis);
    // auto AA = Eigen::AngleAxisf(polyscope::randomReal(-1,1)/2.*M_PI,axis);
    // R = AA.toRotationMatrix();
// #endif
// #if static_dim == 2
    R = Eigen::Rotation2D(polyscope::randomReal(-1,1)/3.*M_PI).toRotationMatrix();
// #endif


    scalar s = polyscope::randomUnit()*2;

    Normalize<static_dim>(X);

    Vector<static_dim> t = Vector<static_dim>::Random(dim);
    X.colwise() -= X.rowwise().mean();
    X = addNoise(X,noise);
    X = R*X*s;
    X.colwise() += t;
    return X;
}


float x,y=0,z = 0,s = 1;
void myCallBack() {
    ImGui::Checkbox("run",&run);
    if (run){
        iter();
        spdlog::info("nb iter {} total time {}",nb_iter,total_time);
    }
    ImGui::SliderInt("steps",&steps,1,64);
    ImGui::SliderFloat("alpha",&alpha,0,1);
    if (ImGui::Button("restart")){
        A = RandomRigid(Asrc);
        T = InjectiveMatching();
        pcA = display<static_dim>("A",A);
        nb_iter = 0;
        total_time = 0;
    }
    ImGui::SliderFloat("x",&x,-2,2);
    ImGui::SliderFloat("y",&y,-2,2);
    ImGui::SliderFloat("z",&z,-2,2);
    ImGui::SliderFloat("s",&s,0,2);
    ImGui::Checkbox("preserve",&preserve);
    if (ImGui::Button("export")) {
        WritePointCloud("/tmp/source.pts",A);
        WritePointCloud("/tmp/target.pts",B);
    }
    // if (dim == 2 && ImGui::Button("apply rigid")){
    //     A = ApplyRigid(Asrc,s,vec2(x,y));
    //     T = InjectiveMatching();
    //     pcA = display<static_dim>("A",A);
    // }
    // else if (dim == 3) {
        // A = ApplyRigid(Asrc,s,vec(x,y,z));
    // }
}

void init () {

    //A = RandomRigid(A);

    pcA = display<static_dim>("A",A);
    display<static_dim>("B",B);
}



int main(int argc,char** argv) {
    polyscope::init();
    CLI::App app("scale rigid matching") ;

    int N;

    app.add_option("--sizes", N, "Number of samples in each cloud (default 1000)");

    std::string mu_src;
    std::string nu_src = "/home/baptiste-genest/dev/BSPOT/data/point_clouds/star_70k.pts";
    app.add_option("--mu_file", mu_src, "source cloud file (if empty then generated)")->required();
    app.add_option("--nu_file", nu_src, "target cloud file (if empty then generated)")->required();

    app.add_option("--noise", noise, "target cloud file (if empty then generated)");//->required();

    if (static_dim == -1)
        app.add_option("--dim",dim,"dimension of the clouds, required if compiled with static_dim == -1")->required(true);

    bool viz = false;
    app.add_flag("--viz", viz, "use polyscope");

    CLI11_PARSE(app, argc, argv);
    Asrc = ReadPointCloud<static_dim>(mu_src);
    // Normalize<static_dim>(Asrc);
    B = ReadPointCloud<static_dim>(nu_src);
    // Normalize<static_dim>(B);
    B = addNoise(B,noise);

    // A = RandomRigid(Asrc);
    A = Asrc;
    spdlog::info("A size {} dim {}",A.cols(),A.rows());
    spdlog::info("B size {} dim {}",B.cols(),B.rows());

    init();
    polyscope::state::userCallback = myCallBack;
    polyscope::show();
    return 0;
}
