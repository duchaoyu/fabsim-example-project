#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include "polyscope/point_cloud.h"

int main(int argc, char *argv[]) {
  using namespace Eigen;

  // load geometry from OFF mesh file
  fsim::Mat3<double> V0;
  fsim::Mat3<int> F;
  fsim::readOFF("../data/mesh.off", V0, F);
//  fsim::Mat2<double> V = V0.leftCols<2>();
//  V /= 10; // total 5m

  // parameters of the membrane model
  double young_modulus1 = 10; // 1000 Pa
  double young_modulus2 = 10;
//  const double young_modulus = 10000000; // 10 MPa
  double thickness = 0.5;  // m
  double poisson_ratio = 0.3;
  double stretch_factor = 1.7;
  double mass = 1; // 1kg per face
  double pressure = 0;

  fsim::OrthotropicStVKMembrane model(V0 , F, thickness, young_modulus1, young_modulus2, poisson_ratio, mass);

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
  solver.options.fixed_dofs = {0 * 3 + 0, 0 * 3 + 1, 0 * 3 + 2, 1 * 3 + 0,
                               1 * 3 + 1, 1 * 3 + 2, 2 * 3 + 0, 2 * 3 + 1,
                               2 * 3 + 2, 3 * 3 + 0, 3 * 3 + 1, 3 * 3 + 2};
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
        model = fsim::OrthotropicStVKMembrane(V0 / stretch_factor, F, thickness, young_modulus1, young_modulus2, poisson_ratio, mass);

      if(ImGui::InputDouble("Modulus1", &young_modulus1, 0, 0, "%.2f"))
        model = fsim::OrthotropicStVKMembrane(V0 / stretch_factor, F, thickness, young_modulus1, young_modulus2, poisson_ratio, mass);

      if(ImGui::InputDouble("Modulus2", &young_modulus2, 0, 0, "%.2f"))
        model = fsim::OrthotropicStVKMembrane(V0 / stretch_factor, F, thickness, young_modulus1, young_modulus2, poisson_ratio, mass);


      if(ImGui::InputDouble("Mass", &mass, 0, 0, "%.2f"))
        model.setMass(mass);

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