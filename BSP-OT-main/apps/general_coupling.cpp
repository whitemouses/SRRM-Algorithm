#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "../src/BSPOTWrapper.h"
#include "../src/cloudutils.h"
#include "../src/sampling.h"
#include "../src/PointCloudIO.h"
#include "../src/plot.h"
#include "../src_meshes/Mesh.h"
#include "../common/CLI11.hpp"

#include <limits>
#include <ranges>

#define TREE_MERGE

int NA = 1000;
int NB = 1000;
int nb_plans = 100;

constexpr int dim = 3;

using namespace BSPOT;

using Pts = Points<dim>;
using Matching = BijectiveMatching;

Vec mass_mu,mass_nu;

Pts A,B;

inline scalar cost(int i,int j) {
    return (A.col(i) - B.col(j)).squaredNorm();
}

scalar eval(const Pts& A,const Pts& B,const Coupling& pi) {
    scalar C = 0;
    for (auto i = 0;i<pi.outerSize();i++){
        for (Coupling::InnerIterator it(pi,i);it;++it) {
            int j = it.col();
            C += cost(i,j)*it.value();
        }
    }
    return C;
}



void checkMarginals(const Coupling& pi) {
    Vec sumA = pi*Vec::Ones(pi.cols()) - mass_mu;
    // spdlog::info("sumA {}",sumA.norm());

    Vec sumB = pi.transpose()*Vec::Ones(pi.rows()) - mass_nu;
    //print which doesn't respect marginals
    for (int i = 0; i < sumB.size(); ++i){
        if (std::abs(sumB(i)) > 1e-6)
            spdlog::info("sumB[{}] = {} != {}",i,sumB(i),mass_nu(i));
    }

    // spdlog::info("sumB {}",sumB.norm());
}


Pts Grad;

Vec randomMass(int n) {
    Vec m = Vec::Random(n)*0.2;
    m += Vec::Ones(n);
    return m/m.sum();
}

Atoms mu,nu;


std::pair<Vec,Pts> toMeasure(const Mesh& M) {
    int n = M.topology->nFaces();
    Pts X = Mat::Zero(dim,n);
    Vec mass = Vec::Zero(n);
    auto A = M.faceAreas();
    scalar S = 0;
    for (auto i : range(n)){
        X.col(i) = M.faceBarycenter(M.topology->face(i));
        mass[i] = A[i];
        S += A[i];
    }
    return {mass/S,X};
}

std::string mu_mesh_src = "/home/nivoliev/Taf/Baptiste/BSPOT/data/meshes/spot.obj",nu_mesh_src = "/home/nivoliev/Taf/Baptiste/BSPOT/data/meshes/bunny.obj";

CouplingMerger merger;

void compute() {

    std::tie(mass_mu,A) = toMeasure(Mesh(mu_mesh_src));
    std::tie(mass_nu,B) = toMeasure(Mesh(nu_mesh_src));

    Normalize(A);
    Normalize(B);

    // mass_mu = randomMass(NA);
    // mass_nu = randomMass(NB);

    mu = FromMass(mass_mu);
    nu = FromMass(mass_nu);

    display<dim>("A",A,mass_mu);
    display<dim>("B",B,mass_nu);
    std::vector<Coupling> couplings(nb_plans);
    auto start = Time::now();
#pragma omp parallel for
    for (auto i : range(nb_plans)) {
        GeneralBSPMatching<dim> BSP(A,mu,B,nu);
        mat Q = mat::Random().fullPivHouseholderQr().matrixQ();
        couplings[i] = BSP.computeCoupling();
        spdlog::info("cost coupling {}",eval(A,B,couplings[i]));
    }
    spdlog::info("compute plan time {}",TimeFrom(start));

    Coupling piMForest ;
    Coupling piMCentroid ;
    Coupling piMQuad ;
    Coupling piMPruned ;
    merger = CouplingMerger(cost);

    //{
    //  start = Time::now();

    //  piMForest = merger.forestMerge(couplings);

    //  spdlog::info("forest merge time {}",TimeFrom(start));
    //  spdlog::info("cost merged {}",eval(A,B,piMForest));
    //}

    {
      start = Time::now();

      std::vector<Eigen::Triplet<scalar>> triplets ;
      for(std::size_t c = 0; c < couplings.size(); ++c) {
        for (auto i = 0;i<couplings[c].outerSize();i++){
            for (Coupling::InnerIterator it(couplings[c],i);it;++it) {
              triplets.emplace_back(it.row(), it.col(), it.value() / couplings.size()) ;
            }
        }
      }
      piMCentroid.resize(couplings[0].rows(), couplings[0].cols()) ;
      piMCentroid.setFromTriplets(triplets.begin(), triplets.end()) ;

      std::vector<int> forest ;
      merger.buildForest(piMCentroid, forest) ;
      Coupling pruned = piMCentroid.pruned() ;

      spdlog::info("centroid merge time {}",TimeFrom(start));
      spdlog::info("cost centroid {}",eval(A,B,piMCentroid));
      piMCentroid.swap(pruned) ;

      start = Time::now();

      merger.improveQuads(piMCentroid, forest) ;

      spdlog::info("quad improvement time {}",TimeFrom(start));
      spdlog::info("cost centroid {}",eval(A,B,piMCentroid));
    }

    //{
    //  start = Time::now();

    //  piMQuad = merger.CycleMerge(couplings);

    //  spdlog::info("cycle merge time {}",TimeFrom(start));
    //  spdlog::info("cost merged {}",eval(A,B,piMQuad));
    //  spdlog::info("edge count {}",piMQuad.nonZeros());
    //}

    //{
    //  start = Time::now();

    //  piMPruned = piMQuad ;
    //  std::vector<int> forest ;
    //  merger.buildForest(piMPruned, forest) ;
    //  piMPruned = piMPruned.pruned() ;

    //  spdlog::info("cycle deletion time {}",TimeFrom(start));
    //  spdlog::info("cost improved {}",eval(A,B,piMPruned));
    //  spdlog::info("edge count {}",piMPruned.nonZeros());
    //}

    //checkMarginals(piMForest);
    checkMarginals(piMCentroid);
    //checkMarginals(piMQuad);
    //checkMarginals(piMPruned);

    Coupling piM = piMCentroid ;

    Grad = Pts::Zero(dim,A.cols());;
    for (auto i = 0;i<piM.outerSize();i++){
        for (Coupling::InnerIterator it(piM,i);it;++it) {
            int j = it.col();
            Grad.col(i) += (B.col(j) - A.col(i))*it.value();
        }
        Grad.col(i) /= mass_mu(i);
    }
    // Grad.array().colwise() /= mass_mu.array();

    plotCoupling("piM",A,B,piM);
}

Pts lerp(float t) {
    return Pts(A + t*Grad);
}

void flow() {
    GeneralBSPMatching<dim> BSP(A,mu,B,nu);
    A += 0.1*BSP.computeOrthogonalTransportGradient();
    //A += 0.1*computeBSPOTGradient(A,mu,B,nu,16);
}

float t = 0;
polyscope::PointCloud* pclerp = nullptr;
polyscope::PointCloud* pcflow = nullptr;
bool run = false;
void myCallBack() {
    if (ImGui::SliderFloat("lerp",&t,0,1)){
        auto L = lerp(t);
        if (!pclerp)
            pclerp = BSPOT::display<dim>("lerp",L,mass_mu);
        else{
            if (dim == 2)
                pclerp->updatePointPositions2D(L.transpose());
            else if (dim == 3)
                pclerp->updatePointPositions(L.transpose());
        }
    }
    ImGui::Checkbox("run",&run);
    if (run) {
        flow();
        if (!pcflow)
            pcflow = display<dim>("flow",A,mass_mu);
        else{
            if (dim == 2)
                pcflow->updatePointPositions2D(A.transpose());
            else if (dim == 3)
                pcflow->updatePointPositions(A.transpose());
        }
    }
}

int main(int argc,char** argv) {
    srand(time(NULL));
    polyscope::init();

    CLI::App app("BSPOT bijection") ;

    app.add_option("--size_mu", NA, "Number of samples in cloud mu (default 1000)");
    app.add_option("--size_nu", NB, "Number of samples in cloud nu (default 1000)");
    app.add_option("--iter", nb_plans, "Number of iterations of refinement (default 100)");

    std::string mu_src;
    std::string nu_src;
    app.add_option("--mu_file", mu_src, "source cloud file (if empty then generated)");
    app.add_option("--nu_file", nu_src, "target cloud file (if empty then generated)");

    CLI11_PARSE(app, argc, argv);

    if (!mu_src.empty()){
        A = ReadPointCloud<dim>(mu_src);
        spdlog::info("mu | dim : {} size : {}",A.rows(),A.cols());
    } else {
        A = Pts::Random(dim,NA);
    }

    if (!nu_src.empty()){
        B = ReadPointCloud<dim>(nu_src);
    } else {
        B = Pts::Random(dim,NB);
        //B = sampleUnitBall<dim>(NB);
    }

    Normalize<dim>(A);
    Normalize<dim>(B);

    polyscope::init();

    compute();

    polyscope::state::userCallback = myCallBack;
    polyscope::show();
    return 0;
}
