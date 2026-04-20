#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "../src/BSPOTWrapper.h"
#include "../src/cloudutils.h"
#include "../src/sampling.h"
#include "../src/PointCloudIO.h"
#include "../src/plot.h"
#include "../src_images/image_as_measure.h"
#include "../common/CLI11.hpp"

int NA = 1000;
int NB = 1000;
int nb_plans = 100;

constexpr int dim = 2;

using namespace BSPOT;

using Pts = Points<dim>;

Pts A,B;

Vec mass_mu,mass_nu;

Pts Grad;

Atoms mu,nu;

Vector<dim> geometric_median(const std::vector<Vector<dim>>& grads) {
    Vector<dim> median = Vector<dim>::Zero();
    scalar tau = 1e-7;
    scalar j = 2*tau;
    while (j > tau) {
        scalar d = 0;
        scalar w = 0;
        Vector<dim> y = grads[0];//Vector<dim>::Zero();
        for (const auto& x : grads){
            d = tau + (median - x).norm();
            w += 1./d;
            y += x/d;
        }
        y /= w;
        j = (y - median).norm();
        median = y;
    }
    return median;
}

GeneralBSPMatching<dim> *BSP;

void save_svg(const Pts& pts, std::string filename, int W, int H, int N) {
    FILE* f = fopen(filename.c_str(), "w+");
    double r = 2.5 * sqrt(8192.0/N);
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"%u\" height = \"%u\">\n", 1024, 1024);
    fprintf(f, "<g>\n");
    for (int j = 0; j < pts.cols(); j++) {
        fprintf(f, "<circle cx=\" %3.3f\" cy=\"%3.3f\" r=\"%3.3f\" /> \n", (pts.col(j)(0)) * 1024., (1-pts.col(j)(1))*1024., r);
    }
    fprintf(f, "</g>\n");
    fprintf(f, "</svg>\n");
    fclose(f);
}


void save_txt(const Pts& pts, std::string filename) {
  FILE* f = fopen(filename.c_str(), "w+");
  for (int j = 0; j < pts.cols(); j++) {
    fprintf(f, "%f %f\n", (pts.col(j)(0)), (pts.col(j)(1)));
  }
  fclose(f);
}


template<int dim>
Points<dim> computeBSPOTRadon(int nb_plans) {
    Points<dim> Grad = Points<dim>::Zero(A.rows(),A.cols());
    scalar min = 1e10;
    std::vector<Pts> grads(nb_plans);
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++) {
        GeneralBSPMatching BSP(A,mu,B,nu);
        scalar th = (M_PI*i)/nb_plans;
        //Points<dim> Grad_i = BSP.computeOrthogonalTransportGradient(Eigen::Rotation2D<scalar>(th).toRotationMatrix());
        grads[i] = BSP.computeOrthogonalTransportGradient(Eigen::Rotation2D<scalar>(th).toRotationMatrix(),true);
        // grads[i] = BSP.computeTransportGradient(true);
        /*
        #pragma omp critical
        {
            //Grad += Grad_i/nb_plans;
            if (Grad_i.squaredNorm() < min) {
                min = Grad_i.squaredNorm();
                Grad = Grad_i;
            }
        }
*/
    }
#pragma omp parallel for
    for (auto i : range(A.cols())) {
        scalar min = 1e8;
        std::vector<Vector<dim>> grads_i(nb_plans);
        for (auto j : range(nb_plans)) {
            grads_i[j] = grads[j].col(i);
        }
        Grad.col(i) = geometric_median(grads_i);
    }
    return Grad;
}

int radon = 32;
float alpha = 0.1;

polyscope::PointCloud* pcflow = nullptr;

void compute(bool viz) {
    mass_mu = Vec::Ones(NA)/NA;
    mu = FromMass(mass_mu);

    if (viz)
    {
        display<dim>("A",A);
        display<dim>("B",B,Mass(nu));
    }
    BSP= new GeneralBSPMatching<dim>(A,mu,B,nu);
    spdlog::info("start compute");
    auto start = Time::now();
    scalar dt = 0.2;
    for (auto i : range(100)) {
        A += computeBSPOTRadon<dim>(radon)*dt;
        dt *= 0.95;
    }
    if (viz) 
      pcflow = display<dim>("flow",A);
    //Grad = BSP->computeTransportGradient();
    spdlog::info("compute time {}",TimeFrom(start));
}

Pts lerp(float t) {
    return Pts(A + t*Grad);
}

void flow() {
    A += alpha*computeBSPOTRadon<2>(radon);
}

float t = 0;
polyscope::PointCloud* pclerp = nullptr;
bool run = false;
void myCallBack() {
    ImGui::Checkbox("run",&run);
    ImGui::SliderFloat("alpha",&alpha,0,1);
    ImGui::SliderInt("radon",&radon,1,32);
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
    if (ImGui::Button("export SVG")) {
        save_svg(A,"flow.svg",1024,1024,A.cols());
        WritePointCloud("/tmp/stippling.pts",A);
    }
}

int main(int argc,char** argv) {

    if (!std::is_same<scalar,double>::value) {
        spdlog::error("non uniform matchings must be compiled with doubles\n you can change it in common/types.h -> using scalar = double;");
        return 1;
    }

    srand(time(NULL));

    CLI::App app("BSPOT bijection") ;

    app.add_option("--size_mu", NA, "Number of samples in cloud mu (default 1000)");

    std::string nu_src;
    app.add_option("--nu_file", nu_src, "source image")->required();

    int rez = 256;
    app.add_option("--res_grid", rez, "resolution of target grid");

    bool filter_white = false;
    app.add_flag("--filter", filter_white, "filter almost white pixels");

    bool viz = false;
    app.add_flag("--viz", viz, "polyscope");

  std::string output="/tmp/out.pts";
  app.add_option("--output,-o", output, "output point filename");

  
    CLI11_PARSE(app, argc, argv);

    auto path = ResizeTo(nu_src,rez,rez,true);

    auto H = LoadImageAsMeasure(path,-1,-1);
    spdlog::info("size before filter {}",H.positions.cols());
    if (filter_white)
     H.filterThreshold(1e-6);
    spdlog::info("size after filter {}",H.positions.cols());
    B = H.positions;
    // B += Points<2>::Random(2,B.cols())*1e-3;
    nu = H.getAtoms();

    A = H.drawFrom(NA);

    if (viz)
    {
     polyscope::init();
     polyscope::options::ssaaFactor = 3;
    }

    compute(viz);

    if (viz)
    {
      polyscope::state::userCallback = myCallBack;
      polyscope::show();
    }
  
    save_txt(A,output);

    return 0;
}
