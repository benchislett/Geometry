#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <igl/cotmatrix.h>
#include <igl/decimate.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/readSTL.h>
#include <iostream>

using namespace geometrycentral;
using namespace geometrycentral::surface;

void edge_cond(const int e, const Eigen::MatrixXd& V, const Eigen::MatrixXi& /*F*/, const Eigen::MatrixXi& E,
               const Eigen::VectorXi& /*EMAP*/, const Eigen::MatrixXi& /*EF*/, const Eigen::MatrixXi& /*EI*/,
               double& cost, Eigen::RowVectorXd& p) {
  cost = (V.row(E(e, 0)) - V.row(E(e, 1))).norm();
  // p    = 0.5 * (V.row(E(e, 0)) + V.row(E(e, 1)));
  p = V.row(E(e, 0));
}

int main() {
  Eigen::MatrixXd V, U;
  Eigen::MatrixXi F, G;
  Eigen::VectorXi J, I;
  igl::readOBJ("assets/bunny.obj", V, F);

  int m          = F.rows();
  auto stop_cond = igl::max_faces_stopping_condition(m, m, 500);
  igl::decimate(V, F, edge_cond, stop_cond, U, G, J, I);

  ManifoldSurfaceMesh origMesh(F), coarseMesh(G);
  VertexPositionGeometry origGeometry(origMesh, V), coarseGeometry(coarseMesh, U);

  polyscope::init();

  polyscope::registerSurfaceMesh("fine mesh", origGeometry.vertexPositions, origMesh.getFaceVertexList());
  polyscope::registerSurfaceMesh("coarse mesh", coarseGeometry.vertexPositions, coarseMesh.getFaceVertexList());

  polyscope::show();

  return 0;
}
