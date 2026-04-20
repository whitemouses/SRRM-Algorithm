#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "../src/BSPOTWrapper.h"
#include "../src/cloudutils.h"
#include "../src/sampling.h"
#include "../src/PointCloudIO.h"
#include "../src/plot.h"
#include "../src_images/load_image.h"
#include "../common/CLI11.hpp"

int N = 1000;




constexpr int dim = 3;

using namespace BSPOT;

using Pts = Points<dim>;
using Matching = BijectiveMatching;

Pts A,B;

scalar cost(size_t i,size_t j) {
    return (A.col(i) - B.col(j)).squaredNorm();
}

scalar eval(const Pts& A,const Pts& B,const BijectiveMatching& T){
    return (A - B(Eigen::all,T)).squaredNorm();
}

struct dot_id {
    scalar dot;
    int id;
    bool operator<(const dot_id& other) const {
        return dot < other.dot;
    }
};

BijectiveMatching SWGG(const Pts& A,const Pts& B,int nb_slices) {
    auto N = A.cols();
    auto dim = A.rows();
    auto slices = sampleUnitSphere(nb_slices,dim);
    std::vector<BijectiveMatching> plans(nb_slices);
#pragma omp parallel for
    for (auto i : range(nb_slices)) {
        std::vector<dot_id> dot_mu(N),dot_nu(N);
        for (auto j : range(N)) {
            dot_mu[j] = {slices.col(i).dot(A.col(j)),j};
            dot_nu[j] = {slices.col(i).dot(B.col(j)),j};
        }
        std::sort(dot_mu.begin(),dot_mu.end());
        std::sort(dot_nu.begin(),dot_nu.end());
        ints plan(N);
        for (auto j : range(N))
            plan[dot_mu[j].id] = dot_nu[j].id;
        plans[i] = plan;
    }
    auto cost = [&A,&B] (int i,int j) -> scalar {
        return (A.col(i)-B.col(j)).squaredNorm();
    };
    ints I = rankPlans(plans,cost);
    return plans[I[0]];
}


BijectiveMatching T;

Points<5> concatPosition(const Points<3>& colors,scalar scale,int w,int h) {
    Points<5> rslt(5,colors.cols());
    for (auto i : range(w)) {
        for (auto j : range(h)) {
            int id = i + j*w;
            rslt(0,id) = colors(0,id);
            rslt(1,id) = colors(1,id);
            rslt(2,id) = colors(2,id);
            rslt(3,id) = (scalar)i / w * scale;
            rslt(4,id) = (scalar)j / h * scale;
        }
    }
    return rslt;
}

bool gaussian_slicer = false;

void color_transfert(std::string mu_src,std::string nu_src,std::string output,int nb_plans) {
    int wa,ha,wb,hb;

    A = LoadImageRGB(mu_src,wa,ha).cast<scalar>();

    auto nu_resize = ResizeTo(nu_src,wa,ha);
    spdlog::info("resize to {}",nu_resize.c_str());
    B = LoadImageRGB(nu_resize,wb,hb).cast<scalar>();
    if (wa != wb || ha != hb) {
        spdlog::error("images must have the same size");
        return;
    }

    Pts OriginalA = A;

    spdlog::info("compute matching between {} pixels",A.cols());
    auto start = Time::now();
    if (gaussian_slicer)
        T = computeGaussianBSPOT(A,B,nb_plans,cost,T);
    else
        T = computeBijectiveBSPOT(A,B,nb_plans,cost,T);
    spdlog::info("matching done in {} score {}",TimeFrom(start),eval(A,B,T));

    N = A.cols();
    Pts rslt(3,N);
    for (auto i : range(N))
        rslt.col(i) = B.col(T[i]);

    out_plan("/tmp/plan.txt",T.inverseMatching());

    exportImageRGB(output,rslt,wa,ha);
    spdlog::info("export rslt at {}",output);
}

void color_transfert_with_pos(std::string mu_src,std::string nu_src,std::string output,int nb_plans) {
    int wa,ha,wb,hb;

    A = LoadImageRGB(mu_src,wa,ha).cast<scalar>();
    Points<5> Apos = concatPosition(A,0.1,wa,ha);


    auto nu_resize = ResizeTo(nu_src,wa,ha);
    spdlog::info("resize to {}",nu_resize.c_str());
    B = LoadImageRGB(nu_resize,wb,hb).cast<scalar>();
    if (wa != wb || ha != hb) {
        spdlog::error("images must have the same size");
        return;
    }
    Points<5> Bpos = concatPosition(B,0.1,wa,ha);

    // Pts OriginalA = A;

    spdlog::info("compute matching between {} pixels",A.cols());
    auto start = Time::now();
    if (gaussian_slicer)
        T = computeGaussianBSPOT<5>(Apos,Bpos,nb_plans,cost,T);
    else
        T = computeBijectiveBSPOT<5>(Apos,Bpos,nb_plans,cost,T);
    spdlog::info("matching done in {} score",TimeFrom(start));//,eval(Apos,Bpos,T));

    N = A.cols();
    Pts rslt(3,N);
    for (auto i : range(N))
        rslt.col(i) = B.col(T[i]);

    out_plan("/tmp/plan.txt",T.inverseMatching());

    exportImageRGB(output,rslt,wa,ha);
    spdlog::info("export rslt at {}",output);
}


int nb_plans = 100;

std::string to_string_fix(int i,int n) {
    // the total string must be n char long, padded with 0s
    std::string rslt = std::to_string(i);
    while (rslt.size() < n)
        rslt = "0" + rslt;
    return rslt;
}

void color_transfert_movie(std::string mu_src,std::string nu_src,std::string output) {
    std::string command = "convert " + mu_src + " -coalesce /tmp/gif_%05d.png";
    system(command.c_str());
    for (auto i : range(14)) {
        std::string name = "/tmp/gif_" + to_string_fix(i,5) + ".png";
        color_transfert(name,nu_src,"/tmp/rslt" + to_string_fix(i,5) + ".png",nb_plans);
    }
    system("convert /tmp/rslt*.png /tmp/rslt.gif");

}

int main(int argc,char** argv) {
    CLI::App app("BSPOT Color Transfer") ;

    std::string mu_src;
    std::string nu_src,outfile = "/tmp/rslt.png",colorspace="RGB";

    app.add_option("--target_image", mu_src, "source image file");
    app.add_option("--colors", nu_src, "target image file");
    app.add_option("--output", outfile, "output image file");
    app.add_option("--colorspace", colorspace, "RGB OR Lab");
    bool with_pos;

    app.add_option("--iter", nb_plans, "Number of iterations of refinement");

    app.add_flag("--gaussian_slicing", gaussian_slicer, "Use gaussian slicing");
    app.add_flag("--with_pos", with_pos, "penalize pixel pos");
    bool movie = false;
    app.add_flag("--movie", movie, "coherent tranfer between movie frames");

    CLI11_PARSE(app, argc, argv);

    if (!movie) {
        if (with_pos)
            color_transfert_with_pos(mu_src,nu_src,outfile,nb_plans);
        else
            color_transfert(mu_src,nu_src,outfile,nb_plans);
    } else {
        color_transfert_movie(mu_src,nu_src,outfile);

    }

    return 0;
}
