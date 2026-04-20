#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/scalar_image_quantity.h"
#include "polyscope/image_quantity.h"
#include "../common/CLI11.hpp"


#include "../src/BSPOTWrapper.h"
#include "../src/cloudutils.h"
#include "../src/sampling.h"
#include "../src/PointCloudIO.h"
#include "../src/plot.h"
#include "../src_images/image_as_measure.h"

int NA = 1000;
int NB = 1000;
int nb_plans = 100;

constexpr int dim = 2;

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

CouplingMerger merger(cost);

struct SplattingGrid {
    Vec values;
    int w,h;
    SplattingGrid() = default;

    SplattingGrid(int w,int h) : w(w),h(h) {
        values = Vec::Zero(w*h);
    }

    void clear() {
        values.setZero();
    }

    void add(int i,int j,scalar v) {
        if (i >= 0 && i < w && j >= 0 && j < h)
            values(j*h + i) += v;
//        else
//            spdlog::error("out of bounds {} {}",i,j);
    }

    void gaussianSplat(const vec2& p,scalar v,int kernel_size = 3) {
        //use bilinear weigths
        //p in [0,1]x[0,1]

        vec2 p2 = p.cwiseProduct(vec2(w-1,h-1));
        int x = p2.x();
        int y = p2.y();

        scalar s = 0;

        std::vector<std::pair<Eigen::Vector2i,scalar>> to_splat;

        //splat in a wider kernel with decaying weights
        for (int i = -kernel_size;i<=kernel_size;i++){
            for (int j = -kernel_size;j<=kernel_size;j++){
                int x2 = x + i;
                int y2 = y + j;
                if (x2 >= 0 && x2 < w && y2 >= 0 && y2 < h) {
                    scalar dx = (i+0.5)/kernel_size;
                    scalar dy = (j+0.5)/kernel_size;
                    scalar w = std::exp(-(dx*dx + dy*dy));
                    s += w;
                    to_splat.push_back({{x2,y2},w});
                }
            }
        }

        for (auto& [p,w] : to_splat){
            add(p.x(),p.y(),w*v/s);
        }



    }

    void splat(const vec2& p,scalar v) {
        //use bilinear weigths
        //p in [0,1]x[0,1]

        vec2 p2 = p.cwiseProduct(vec2(w-1,h-1));
        int x = p2.x();
        int y = p2.y();
        scalar dx = p2.x() - x;
        scalar dy = p2.y() - y;
        scalar dx1 = 1 - dx;
        scalar dy1 = 1 - dy;
        scalar w00 = dx1*dy1;
        scalar w01 = dx1*dy;
        scalar w10 = dx*dy1;
        scalar w11 = dx*dy;

        add(x,y,w00*v);
        add(x+1,y,w01*v);
        add(x,y+1,w10*v);
        add(x+1,y+1,w11*v);
    }

};

Atoms mu,nu;

SplattingGrid SG;
polyscope::ScalarImageQuantity* image;

Coupling pi;

scalar s0 = 1,s1 = 1;

void compute() {

    std::vector<Coupling> couplings(nb_plans);

    spdlog::info("start compute coupling between {} and {} atoms",mu.size(),nu.size());

    auto start = Time::now();
#pragma omp parallel for
    for (auto i : range(nb_plans)) {
        GeneralBSPMatching<dim> BSP(A,mu,B,nu);
        mat2 Q = sampleUnitGaussian<dim>().fullPivHouseholderQr().matrixQ();
//         couplings[i] = BSP.computeCoupling();
        couplings[i] = BSP.computeOrthogonalCoupling(Q);
        spdlog::info("cost coupling {}",eval(A,B,couplings[i]));
    }
    spdlog::info("compute plan time {}",TimeFrom(start));
    start = Time::now();
//    pi = couplings[0];

    auto cmp = [](const Coupling& a,const Coupling& b){
        return eval(A,B,a) < eval(A,B,b);
    };

     std::sort(couplings.begin(),couplings.end(),cmp);

//     pi = *std::min_element(couplings.begin(),couplings.end(),cmp);//CycleMerge(couplings);
    pi = merger.CycleMerge(couplings);
    spdlog::info("merge time {}",TimeFrom(start));
    spdlog::info("cost merged {}",eval(A,B,pi));
    ints forest;
    merger.buildForest(pi,forest);
    spdlog::info("cycle cut time {}",TimeFrom(start));
    spdlog::info("cost acyclic {}",eval(A,B,pi));

//    checkMarginals(piM);

}

void SplatAdvection(const Coupling& pi,scalar t) {
    SG.clear();
    for (auto i = 0;i<pi.outerSize();i++){
        for (Coupling::InnerIterator it(pi,i);it;++it) {
            int j = it.col();
            scalar v = it.value();
            vec2 p = A.col(i) + t*(B.col(j) - A.col(i));
            // SG.splat(p,v);
            SG.gaussianSplat(p,v);
        }
    }
    scalar s = ((1-t)*s0 + t*s1);
//    SG.values *= s/SG.values.maxCoeff();
    image->updateData(SG.values);
//    image->setMapRange({0.,s});
    image->setMapRange({0.,SG.values.maxCoeff()});
}

float t = 0;
void myCallBack() {
    if (ImGui::SliderFloat("lerp",&t,0,1)){
        SplatAdvection(pi,t);
    }
}

int main(int argc,char** argv) {
    srand(time(NULL));
    polyscope::init();

    CLI::App app("image transport splatting") ;

    app.add_option("--iter", nb_plans, "Number of iterations of refinement (default 100)");

    std::string mu_src;
    std::string nu_src;
    app.add_option("--mu_file", mu_src, "source image (if empty then generated)")->required();
    app.add_option("--nu_file", nu_src, "target image (if empty then generated)")->required();

    CLI11_PARSE(app, argc, argv);

    polyscope::init();

    if (!mu_src.empty()){
        int n = 512;
        auto H = LoadImageAsMeasure(mu_src,n,n);
//        H.filterThreshold(1e-6);
        H.mass.array() += 1e-8;
        H.mass /= H.mass.sum();
        s0 = H.mass.maxCoeff();
        A = H.positions;
        mu = H.getAtoms();
        SG = SplattingGrid(H.dimensions(0),H.dimensions(1));
        image = polyscope::addScalarImageQuantity("image",SG.w,SG.h,SG.values,polyscope::ImageOrigin::LowerLeft);
        image->setEnabled(true);
        image->setColorMap("blues");
    }
    if (!nu_src.empty()){
        auto H = LoadImageAsMeasure(nu_src,256,256);
        B = H.positions;
        H.mass.array() += 1e-8;
        H.mass /= H.mass.sum();
        s1 = H.mass.maxCoeff();
        nu = H.getAtoms();
    }







    compute();

    polyscope::state::userCallback = myCallBack;
    polyscope::show();
    return 0;
}
