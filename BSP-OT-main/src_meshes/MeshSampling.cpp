#include "MeshSampling.h"
#include <polyscope/polyscope.h>

polyscope::PointCloud *BSPOT::display(std::string label, const Mesh &M, const SurfacePoints &X)
{
    return polyscope::registerPointCloud(label,toPositions(M,X).transpose());
}


BSPOT::SurfacePoints BSPOT::sampleMesh(const BSPOT::Mesh &M, int sampleNum, const scalars &face_weights) {
    SurfacePoints sampleList(sampleNum);

    const auto& pos = M.geometry->vertexPositions;

    int triNum = M.topology->nFaces();
    if (triNum != face_weights.size()){
        spdlog::error("invalid face weights");
        return sampleList;
    }

    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::discrete_distribution<> d(face_weights.begin(), face_weights.end());

    for (int j = 0; j < sampleNum; j++) {
        // double sample;
        size_t faceidx = d(gen);

        //random point generation within previously selected face area
        vecs V;
        for (auto v : M.topology->face(faceidx).adjacentVertices()){
            V.push_back(toVec(pos[v]));
        }



        //resulting sample
        // vec P = a*V[0] + b*V[1] + c*V[2];
        //PointOnMesh pom;
        //sampleList[j] = {P,faceidx,vec(a,b,c)};
        sampleList[j] = SurfacePoint(M.topology->face(faceidx),toVec3(drawBary()));
    }

    return sampleList;
}

BSPOT::SurfacePoints BSPOT::sampleMeshVertexDensity(const Mesh &M, const Vec &V, int n) {
    scalars F(M.topology->nFaces(),0);
    auto A = M.faceAreas();
    for (auto f : M.topology->faces())
        for (auto v : f.adjacentVertices()){
            F[f.getIndex()] += A[f.getIndex()]/3*V(v.getIndex());
        }
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::discrete_distribution<> draw_f(F.begin(),F.end());
    SurfacePoints rslt(n);
    for (auto i : range(n)) {
        auto f = M.topology->face(draw_f(gen));
        int k = 0;
        vec P;
        for (auto v : f.adjacentVertices())
            P[k++] = V[v.getIndex()];
        rslt[i] = SurfacePoint(f,toVec3(drawWeightedBary(P)));
    }
    return rslt;
}
