#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/surface/integer_coordinates_intrinsic_triangulation.h"
#include "geometrycentral/surface/intrinsic_triangulation.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/signpost_intrinsic_triangulation.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_point.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cassert>
#include <fstream>
#include <igl/cotmatrix.h>
#include <igl/decimate.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <igl/readSTL.h>
#include <iostream>

using namespace geometrycentral;
using namespace geometrycentral::surface;

int insertVertex(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int which_face, int which_edge) {
  int v1 = F.row(which_face).col(which_edge).value();
  int v2 = F.row(which_face).col((which_edge + 1) % 3).value();

  int newv = V.rows();
  V.conservativeResize(V.rows() + 1, V.cols());
  V.row(newv) = 0.5 * (V.row(v1) + V.row(v2));
  return newv;
}

void triangulateVertexInFace(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int which_face, int which_edge, int new_vertex) {
  int v1      = F.row(which_face).col(which_edge).value();
  int v2      = F.row(which_face).col((which_edge + 1) % 3).value();
  int v_other = F.row(which_face).col((which_edge + 2) % 3).value();

  int new_face = F.rows();
  F.conservativeResize(F.rows() + 1, F.cols());
  F.row(which_face).col((which_edge + 1) % 3).setConstant(new_vertex);
  F.row(new_face).col(0).setConstant(v2);
  F.row(new_face).col(1).setConstant(v_other);
  F.row(new_face).col(2).setConstant(new_vertex);
}

int main() {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::readOBJ("assets/square.obj", V, F);

  srand(time(NULL));
  // srand(0);

  for (int i = 0; i < 200; i++) {
    int which_face = rand() % F.rows();
    int which_edge = rand() % 3;

    int v1 = F.row(which_face).col(which_edge).value();
    int v2 = F.row(which_face).col((which_edge + 1) % 3).value();
    int v3 = insertVertex(V, F, which_face, which_edge);

    int other_face = -1, other_edge = -1;
    for (int j = 0; j < F.rows(); j++) {
      if (j == which_face)
        continue;

      if (F.row(j).col(0).value() == v1 && F.row(j).col(1).value() == v2) {
        other_face = j;
        other_edge = 0;
        break;
      } else if (F.row(j).col(1).value() == v1 && F.row(j).col(2).value() == v2) {
        other_face = j;
        other_edge = 1;
        break;
      } else if (F.row(j).col(2).value() == v1 && F.row(j).col(0).value() == v2) {
        other_face = j;
        other_edge = 2;
        break;
      } else if (F.row(j).col(0).value() == v2 && F.row(j).col(1).value() == v1) {
        other_face = j;
        other_edge = 0;
        break;
      } else if (F.row(j).col(1).value() == v2 && F.row(j).col(2).value() == v1) {
        other_face = j;
        other_edge = 1;
        break;
      } else if (F.row(j).col(2).value() == v2 && F.row(j).col(0).value() == v1) {
        other_face = j;
        other_edge = 2;
        break;
      }
    }

    triangulateVertexInFace(V, F, which_face, which_edge, v3);
    if (other_face >= 0) {
      triangulateVertexInFace(V, F, other_face, other_edge, v3);
    }
  }

  ManifoldSurfaceMesh mesh(F);
  VertexPositionGeometry geometry(mesh, V);

  mesh.compress();

  // geometry.requireCotanLaplacian();
  // auto laplacian = geometry.cotanLaplacian;

  IntegerCoordinatesIntrinsicTriangulation intTri(mesh, geometry);
  intTri.flipToDelaunay();
  intTri.delaunayRefine(30, 0.01, 10000);
  intTri.requireCotanLaplacian();
  intTri.requireVertexGalerkinMassMatrix();
  auto laplacian = intTri.cotanLaplacian;
  auto galerkin  = intTri.vertexGalerkinMassMatrix;

  CommonSubdivision& subdiv = intTri.getCommonSubdivision();
  subdiv.constructMesh();
  ManifoldSurfaceMesh& subdivMesh = *subdiv.mesh;
  VertexData<Vector3> csPositions = subdiv.interpolateAcrossA(geometry.vertexPositions);
  VertexPositionGeometry subdivGeom(subdivMesh, csPositions);

  intTri.vertexLocations;
  intTri.vertexLocations;
  Eigen::VectorXd gvec, solvec;
  gvec.resize(intTri.vertexLocations.size());
  solvec.resize(intTri.vertexLocations.size());
  gvec.fill(0);
  for (int i = 0; i < intTri.vertexLocations.size(); i++) {
    auto surfpoint = intTri.vertexLocations[i].inSomeFace();
    auto face      = surfpoint.face;
    Vector3 point  = {0, 0, 0};

    int j = 0;
    for (Vertex v : face.adjacentVertices()) {
      point += geometry.vertexPositions[v.getIndex()] * surfpoint.faceCoords[j++];
    }

    gvec[i]   = sin(M_PI * point.x) * sin(M_PI * point.y);
    solvec[i] = sin(M_PI * point.x) * sin(M_PI * point.y) / (2.0 * M_PI * M_PI);
  }

  printf("Laplacian has norm: %f\n", laplacian.norm());

  Eigen::ComputationInfo info;

  // gvec = galerkin * gvec;

  auto fvec = solve(laplacian, gvec);

  VertexData<double> f(intTri.mesh, fvec);
  VertexData<double> g(intTri.mesh, gvec);
  VertexData<double> sol(intTri.mesh, solvec);

  polyscope::init();

  auto plot = polyscope::registerSurfaceMesh("extrinsic mesh", geometry.vertexPositions, mesh.getFaceVertexList());
  auto subdivplot =
      polyscope::registerSurfaceMesh("common subdivision", subdivGeom.vertexPositions, subdivMesh.getFaceVertexList());

  subdivplot->addVertexScalarQuantity("poisson solution", subdiv.interpolateAcrossB(f));
  subdivplot->addVertexScalarQuantity("poisson analytic solution", subdiv.interpolateAcrossB(sol));
  subdivplot->addVertexScalarQuantity("poisson problem", subdiv.interpolateAcrossB(g));

  auto facedat = subdiv.sourceFaceB;
  FaceData<size_t> whichface(subdivMesh);
  for (int i = 0; i < facedat.size(); i++) {
    whichface[i] = facedat[i].getIndex();
  }

  subdivplot->addFaceScalarQuantity("intrinsic triangulation", whichface);

  polyscope::show();

  return 0;
}
