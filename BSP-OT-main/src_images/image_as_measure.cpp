#include "image_as_measure.h"
#include <random>


BSPOT::Histogram<2> BSPOT::LoadImageAsMeasure(const std::string &filename, int mw,int mh,scalar threshold)
{
    int w,h;

    auto img = LoadImageRGB(filename,w,h);
    if (w < 0 || h < 0){
        spdlog::error("image not found");
        return Histogram<2>();
    }
    if (w > mw || h > mh){
        auto rs = ResizeTo(filename,mw,mh);
        img = LoadImageRGB(rs,w,h);
    }

    Vec mass = Luminance(img);

    mass.array() += threshold;
    mass = mass.cwiseMin(1);

    mass = 1 - mass.array();
    mass /= mass.sum();


    Histogram<2> hist;
//    hist.weights = FromMass(mass);
    hist.mass = mass;
    hist.positions = Points<2>::Zero(2,mass.size());
#pragma omp parallel for
    for (auto j : range(h)) {
        for (auto i : range(w)) {
            hist.positions(0,i + j*w) = i;
            hist.positions(1,i + j*w) = h-j;
        }
    }

    hist.dimensions(0) = w;
    hist.dimensions(1) = h;


    spdlog::info("image loaded as measure {} {}",w,h);

    hist.positions /= std::max(w,h);
    hist.positions += Points<2>::Random(2,mass.size())*1e-5;
    return hist;
}

BSPOT::Histogram<2> BSPOT::UniformGrid(int n)
{
        Histogram<2> hist;
    hist.dimensions(0) = n;
    hist.dimensions(1) = n;

    hist.positions = Points<2>::Zero(2,n*n);
#pragma omp parallel for
    for (auto j : range(n)) {
        for (auto i : range(n)) {
            hist.positions(0,i + j*n) = i;
            hist.positions(1,i + j*n) = j;
        }
    }

    hist.mass = Vec::Zero(n*n);
    hist.mass.array() += 1.;
    hist.mass /= hist.mass.sum();

    hist.positions /= n;
    hist.positions += Points<2>::Random(2,n*n)*1e-5;

    return hist;
}
