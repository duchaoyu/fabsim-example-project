#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include "polyscope/point_cloud.h"


std::vector<int> findBoundaryVertices(fsim::Mat3<int> F) {
  // Edge represented by a pair of vertex indices, with a count of how many faces share the edge
  std::map<std::pair<int, int>, int> edgeCount;

  for(int i = 0; i < F.rows(); ++i){
    auto face = F.row(i);
    int n = face.size();
    for (int i = 0; i < n; ++i) {
      int v1 = face[i];
      int v2 = face[(i + 1) % n]; // Next vertex in the face, wrapping around to the start
      if (v1 > v2) std::swap(v1, v2); // Ensure the first vertex in the pair is the smaller one
      edgeCount[{v1, v2}]++;
    }
  }

  // Set to hold boundary vertices (using a set to avoid duplicates)
  std::set<int> boundaryVertices;

  // Identify boundary edges and their vertices
  for (const auto& edge : edgeCount) {
    if (edge.second == 1) { // Boundary edge found
      boundaryVertices.insert(edge.first.first);
      boundaryVertices.insert(edge.first.second);
    }
  }

  // Convert set to vector and return
  return std::vector<int>(boundaryVertices.begin(), boundaryVertices.end());
}



int main(int argc, char *argv[]) {
  using namespace Eigen;

  // load geometry from OFF mesh file
  fsim::Mat3<double> V0;
  fsim::Mat3<int> F;
//  fsim::readOFF("../data/mesh.off", V0, F);
//  fsim::readOFF("/Users/duch/Downloads/pillow.off", V0, F);
  fsim::readOFF("/Users/duch/Downloads/2part_opt.off", V0, F);
//  fsim::Mat2<double> V = V0.leftCols<2>();
//  V /= 10; // total 5m

  // parameters of the membrane model
  double young_modulus1 = 50000; // 1000 Pa
  double young_modulus2 = 25000;
//  const double young_modulus = 10000000; // 10 MPa
  double thickness = 1.0;  // m
  double poisson_ratio = 0.38;
  double stretch_factor = 1.05;
  double mass = 30; // 1kg per face
  double pressure = 250;

  fsim::OrthotropicStVKMembrane model(V0 , F, thickness, young_modulus1, young_modulus2, poisson_ratio, mass);

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;

  std::vector<int> bdrs = {11, 13, 25, 29, 35, 51, 55, 65, 73, 79, 98, 99, 100, 104, 105, 114, 118, 130, 133, 142, 144, 149, 153, 155, 157, 158, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 186, 188, 200, 204, 210, 226, 230, 247, 274, 275, 284, 288, 302, 311, 313, 314, 322, 324, 326, 327, 332, 333, 334, 336, 337, 338, 339};
  std::sort(bdrs.begin(), bdrs.end());

  for (int bdr : bdrs) {
    solver.options.fixed_dofs.push_back(bdr * 3);
    solver.options.fixed_dofs.push_back(bdr * 3 + 1);
    solver.options.fixed_dofs.push_back(bdr * 3 + 2);
  }
  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
//  solver.options.fixed_dofs = {0 * 3 + 0, 0 * 3 + 1, 0 * 3 + 2, 1 * 3 + 0,
//                               1 * 3 + 1, 1 * 3 + 2, 2 * 3 + 0, 2 * 3 + 1,
//                               2 * 3 + 2, 3 * 3 + 0, 3 * 3 + 1, 3 * 3 + 2};
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be

  // fixed points
  std::vector<glm::vec3> points;

//  for (int i = 0; i < 4; ++i) {
//    points.push_back(glm::vec3(V.row(i)[0], V.row(i)[1], V.row(i)[2]));
//  }
//  // visualize fixes
//  polyscope::PointCloud* psCloud = polyscope::registerPointCloud("Fixed points", points);
//  psCloud->setPointRadius(0.02);
//  psCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);

  // display the mesh
  polyscope::registerSurfaceMesh("mesh", V0, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1});
  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;
  polyscope::init();

  polyscope::state::userCallback = [&]()
  {
      ImGui::PushItemWidth(100);
      if(ImGui::InputDouble("Stretch factor", &stretch_factor, 0, 0, "%.2f"))
        model = fsim::OrthotropicStVKMembrane(V0 / stretch_factor, F, thickness, young_modulus1, young_modulus2, poisson_ratio, mass);

      if(ImGui::InputDouble("Thickness", &thickness, 0, 0, "%.2f"))
        model = fsim::OrthotropicStVKMembrane(V0 / stretch_factor, F, thickness, young_modulus1, young_modulus2, poisson_ratio, mass);

      if(ImGui::InputDouble("Possian", &poisson_ratio, 0, 0, "%.2f"))
//        model = fsim::OrthotropicStVKMembrane(V0 / stretch_factor, F, thickness, young_modulus1, young_modulus2, poisson_ratio, mass);
        model.setPoissonRatio(poisson_ratio);

      if(ImGui::InputDouble("Modulus1", &young_modulus1, 0, 0, "%.2f"))
        model.setE1(young_modulus1);

      if(ImGui::InputDouble("Modulus2", &young_modulus2, 0, 0, "%.2f"))
//        model = fsim::OrthotropicStVKMembrane(V0 / stretch_factor, F, thickness, young_modulus1, young_modulus2, poisson_ratio, mass);
        model.setE2(young_modulus2);

      if(ImGui::InputDouble("Mass", &mass, 0, 0, "%.2f"))
        model.setMass(mass);

      if(ImGui::InputDouble("Pressure", &pressure, 0, 0, "%.2f"))
        model.setPressure(pressure);

      if(ImGui::Button("Solve"))
      {
        // Newton's method: finds a local minimum of the energy (Fval = energy value, Optimality = gradient's norm)
        solver.solve(model, Map<VectorXd>(V0.data(), V0.size()));

        // Display the result of the optimization
        polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
            Map<fsim::Mat3<double>>(solver.var().data(), V0.rows(), 3));
      }
  };
  polyscope::show();
}