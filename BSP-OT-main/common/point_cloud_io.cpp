#include "point_cloud_io.h"
#include "types.h"



PointCloudIO::vecs PointCloudIO::read_point_cloud(std::string filename)
{
    vecs X;
    std::ifstream file(filename);
    double x,y,z;
    while (file >> x >> y >> z)
        X.push_back(BSPOT::vec(x,y,z));
    return X;
}
