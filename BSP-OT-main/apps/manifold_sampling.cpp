#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "../src/BSPOTWrapper.h"
#include "../src/cloudutils.h"
#include "../src/sampling.h"
#include "../src/PointCloudIO.h"
#include "../src/plot.h"
#include <queue>
#include <omp.h>
#include <thread>
#include <chrono>
#include <condition_variable>
#include "../common/discrete_OT_solver.h"
#include "../common/CLI11.hpp"

#include "../src_meshes/Mesh.h"
#include "../src_meshes/MeshSampling.h"
#include "../src_meshes/SpectralEmbedding.h"

#include <geometrycentral/surface/heat_method_distance.h>
#include <geometrycentral/surface/vector_heat_method.h>


int N = 1000;
int iter_advect = 16;

constexpr int static_dim = 3;
int dim = static_dim;

constexpr int embed_dim = 64;

using namespace BSPOT;
using namespace geometrycentral;
using namespace geometrycentral::surface;

Mesh mesh;

SurfacePoints MU,NU;

using SpectralPts = Points<embed_dim>;

std::shared_ptr<HeatMethodDistanceSolver> solver;
std::shared_ptr<VectorHeatMethodSolver> solver_vector;

Vector3 pos(const SurfacePoint& x) {return x.interpolate(mesh.geometry->vertexPositions);}
Vector3 pos(const Vertex& v) {return mesh.geometry->vertexPositions[v];}


Vec toVec(const BSPOT::VertexData<double>& V) {
    Vec X(mesh.topology->nVertices());
    for (auto v : mesh.topology->vertices())
        X(v.getIndex()) = V[v];
    return X;
}

scalars face_weight_geodesic_gaussian(Vertex v0,scalar sigma = 1) {
    using namespace geometrycentral;
    auto D = toVec(heatMethodDistance(*mesh.geometry,v0));
    D /= D.maxCoeff();
    mesh.geometry->requireCotanLaplacian();
    mesh.geometry->requireVertexGalerkinMassMatrix();
    auto F = mesh.topology->nFaces();
    scalars rslt(F);
    for (auto f : mesh.topology->faces()){
        rslt[f.getIndex()] = 0;
        for (auto v : f.adjacentVertices())
            rslt[f.getIndex()] += mesh.geometry->vertexDualArea(v)*std::exp(-std::pow(D(v.getIndex()),2)*sigma);
//        rslt[f.getIndex()] = ;
    }
    return rslt;
}

scalars face_weight_curvature(scalar offset = 0.01) {
    using namespace geometrycentral;
    mesh.geometry->requireFaceGaussianCurvatures();
    auto F = mesh.topology->nFaces();
    scalars rslt(F);
    for (auto f : mesh.topology->faces()){
        rslt[f.getIndex()] = offset + std::abs(mesh.geometry->faceGaussianCurvatures[f]);
    }
    return rslt;
}

polyscope::SurfaceMesh* loadMeshes(std::string meshname) {
    std::tie(mesh.topology, mesh.geometry) = readManifoldSurfaceMesh(meshname);
    return polyscope::registerSurfaceMesh("input_mesh",
                                          mesh.geometry->vertexPositions, mesh.topology->getFaceVertexList());
}

BSPOT::VertexData<scalar> toVertexData(const Vec& X) {
    return BSPOT::VertexData<scalar>(*mesh.topology,X);
}


GlobalPointSignature<embed_dim>* GPS;

struct SpectralLloydSampling {
    SurfacePoints mu;
    std::vector<SurfacePoints> nus;
    SpectralPts GPSmu;
    std::vector<SpectralPts> GPSnus;
    std::vector<BijectiveMatching> plans;
    int pps,N;

    SpectralLloydSampling() {}

    template<class generator>
    SpectralLloydSampling(int N_,int pps_,generator gen) :pps(pps_),N(N_) {
        mu = gen(N);
        spdlog::info("embbed GPS");
        GPSmu = GPS->computeEmbedding(mu);
        nus.resize(pps);
        GPSnus.resize(pps);
        for (auto i : range(pps)) {
            nus[i] = gen(N);
            GPSnus[i] = GPS->computeEmbedding(nus[i]);
        }
        plans.resize(pps);
        spdlog::info("done");
    }

    void toMostCentral() {
        #pragma omp parallel for
        for (auto x : range(N)){
            BSPOT::Vector<embed_dim> mean = BSPOT::Vector<embed_dim>::Zero(embed_dim);
            for (auto k : range(pps))
                mean += GPSnus[k].col(plans[k][x])/pps;
            scalar dist = 1e8;
            int c = 0;
            for (auto k : range(pps)) {
                scalar d = (mean-GPSnus[k].col(plans[k][x])).squaredNorm();
                if (Smin(dist,d)) {
                    c = k;
                }
            }
            mu[x] = nus[c][plans[c][x]];
            GPSmu.col(x) = GPSnus[c].col(plans[c][x]);
        }
    }

    void iterate(int nb_iters) {
        std::vector<std::function<scalar(size_t,size_t)>> costs(pps);
        for (auto iter : range(nb_iters)) {
            for (auto i : range(pps))
                costs[i] = [&,i](size_t x,size_t y) -> scalar {
                    return (GPSmu.col(x)-GPSnus[i].col(y)).squaredNorm();
                };
            #pragma omp parallel for
            for (int i = 0;i<pps;i++) {
                plans[i] = computeBijectiveOrthogonalBSPOT(GPSmu,GPSnus[i],16,costs[i],plans[i]);
            }
            toMostCentral();
        }
    }
};

int pps;

polyscope::SurfaceMesh* pc;
std::string mesh_name;
SpectralLloydSampling sampler;

polyscope::PointCloud* pcsamples = nullptr;

void export_surface_points(const std::string& filename) {
    std::ofstream out(filename);
    auto pts =  toPositions(mesh,sampler.mu).transpose();
    for (auto i : range(sampler.mu.size())) {
        out << pts.row(i) << std::endl;
    }
}

void compute() {
    pc = loadMeshes(mesh_name);
    spdlog::info("factor VHM solver");
    solver_vector = std::make_shared<VectorHeatMethodSolver>(*mesh.geometry);
    solver = std::make_shared<HeatMethodDistanceSolver>(*mesh.geometry);

    // auto A = mesh.faceAreas();


    GPS = new GlobalPointSignature<embed_dim>(mesh);

    scalars mesh_density(mesh.topology->nFaces(),0);
    srand(time(NULL));

    mesh_density = mesh.faceAreas();//face_weight_curvature();
    // mesh_density = face_weight_geodesic_gaussian(mesh.topology->vertex(6754),20);

    {
        std::ofstream out("/tmp/mesh_density.data");
        Vec density(mesh.topology->nFaces());
        for (auto f : mesh.topology->faces())
            density(f.getIndex()) = mesh_density[f.getIndex()];
        out << density.transpose() << std::endl;
    }

    sampler = SpectralLloydSampling(N,pps,[mesh_density](int n) ->SurfacePoints {
        return sampleMesh(mesh,n,mesh_density);
    });

    pc->addFaceScalarQuantity("nu",mesh_density);

    {
        std::ofstream out("/tmp/mu_dim" + std::to_string(embed_dim) + ".pts");
        out << sampler.GPSmu.transpose() << std::endl;
    }{
        std::ofstream out("/tmp/nu_dim" + std::to_string(embed_dim) + ".pts");
        out << sampler.GPSnus[0].transpose() << std::endl;
    }

    auto bruno = ReadPointCloud<3>("/tmp/sample_bunny.pts");
    display("bruno",bruno);

    //spdlog::info("start lloyd");
    //sampler.iterate(nb_iter);
    //spdlog::info("done");

    spdlog::info("iteration");
    auto start = Time::now();
    sampler.iterate(iter_advect);
    spdlog::info("done in {}",TimeFrom(start));
    // sampler.profiler.profile();

    pcsamples = display("sampling",mesh,sampler.mu);
}

float t = 0;
bool run = false;
void myCallBack() {
    ImGui::Checkbox("run",&run);

    if (run) {
        sampler.iterate(1);
        display("sampling",mesh,sampler.mu);
    }
    if (ImGui::Button("export")){
        export_surface_points("/tmp/manifold_sampling.pts");
    }

}

int main(int argc,char** argv) {
    srand(time(NULL));

    CLI::App app("BSPOT intrinsic manifold sampling") ;

    app.add_option("--sizes", N, "Number of samples");
    app.add_option("--iter", iter_advect, "Number of iterations of refinement (default 100)");
    app.add_option("--pps", pps, "relative size of nu next to mu (integer)");

    // std::string nu_src;
    app.add_option("--mesh_file", mesh_name, "input surface")->required();
    // app.add_option("--nu_file", nu_src, "target cloud file (if empty then generated)");

    std::string out_src;
    app.add_option("--output_file", out_src, "output transport plan");

    if (static_dim == -1)
        app.add_option("--dim",dim,"dimension of the clouds, required if compiled with static_dim == -1")->required(true);

    bool viz = true;
    // app.add_flag("--viz", viz, "use polyscope");
    bool force_size = false;
    app.add_flag("--force_size", force_size, "reduces the size of the measures to sizes if true");

    CLI11_PARSE(app, argc, argv);


    if (viz) {
        polyscope::init();

    compute();


        polyscope::state::userCallback = myCallBack;
        polyscope::show();
    }
    return 0;
}
