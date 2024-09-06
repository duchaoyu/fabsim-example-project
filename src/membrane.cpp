#include <fsim/ElasticMembrane.h>
#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include "polyscope/point_cloud.h"

int main(int argc, char *argv[]) {
  using namespace Eigen;

  // load geometry from OFF mesh file
  fsim::Mat3<double> V;
  fsim::Mat3<int> F;
  fsim::readOFF("../data/mesh.off", V, F);
//  fsim::readOFF("/Users/duch/Downloads/pillow.off", V, F);


  // parameters of the membrane model
  double young_modulus = 10; // 10 MPa
//  const double young_modulus = 10000000; // 10 MPa
  double thickness = 0.05;  // 0.05m
  double poisson_ratio = 0.3;
  double stretch_factor = 1.7;
  double mass = 1; // 1kg per face
  double pressure = 0;


   fsim::StVKMembrane model(V , F, thickness, young_modulus, poisson_ratio, mass, pressure);

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
//  std::vector<int> bdrs = {10, 11, 18, 22, 25, 27, 30, 41, 42, 53, 55, 57, 65, 66, 83, 85, 92, 93, 94, 96, 98, 100, 101, 111, 122, 123, 124, 126, 128, 130, 133, 134, 137, 138, 145, 146, 149, 150, 151, 152, 156, 160, 161, 163, 164, 166};
  std::vector<int> bdrs = {0, 1, 2, 3};
//
  for (int bdr : bdrs) {
    solver.options.fixed_dofs.push_back(bdr * 3);
    solver.options.fixed_dofs.push_back(bdr * 3 + 1);
    solver.options.fixed_dofs.push_back(bdr * 3 + 2);
  }
//  solver.options.fixed_dofs = {0 * 3 + 0, 0 * 3 + 1, 0 * 3 + 2, 1 * 3 + 0,
//                               1 * 3 + 1, 1 * 3 + 2, 2 * 3 + 0, 2 * 3 + 1,
//                               2 * 3 + 2, 3 * 3 + 0, 3 * 3 + 1, 3 * 3 + 2};
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be

  // fixed points
  std::vector<glm::vec3> points;



  for (unsigned int i = 0; i < bdrs.size(); i++) {
    points.push_back(glm::vec3(V.row(bdrs[i])[0], V.row(bdrs[i])[1], V.row(bdrs[i])[2]));
  }
  // visualize fixes
  polyscope::PointCloud* psCloud = polyscope::registerPointCloud("Fixed points", points);
  psCloud->setPointRadius(0.02);
  psCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);

  // display the mesh
  polyscope::registerSurfaceMesh("mesh", V, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1});
  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;
  polyscope::init();

  polyscope::state::userCallback = [&]()
  {
      ImGui::PushItemWidth(100);
      if(ImGui::InputDouble("Stretch factor", &stretch_factor, 0, 0, "%.2f"))
        model = fsim::StVKMembrane(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass, pressure);

      if(ImGui::InputDouble("Thickness", &thickness, 0, 0, "%.2f"))
        model.setThickness(thickness);

      if(ImGui::InputDouble("Possian", &poisson_ratio, 0, 0, "%.2f"))
        model.setPoissonRatio(poisson_ratio);

      if(ImGui::InputDouble("Modulus", &young_modulus, 0, 0, "%.2f"))
        model.setYoungModulus(young_modulus);

      if(ImGui::InputDouble("Mass", &mass, 0, 0, "%.2f"))
        model.setMass(mass);

      if(ImGui::InputDouble("Pressure", &pressure, 0, 0, "%.2f"))
        model.setPressure(pressure);

      if(ImGui::Button("Solve"))
      {
        // Newton's method: finds a local minimum of the energy (Fval = energy value, Optimality = gradient's norm)
        solver.solve(model, Map<VectorXd>(V.data(), V.size()));

        // Display the result of the optimization
        polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
            Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3));
      }
      if(ImGui::Button("Init"))
      {
        solver.init(model, Map<VectorXd>(V.data(), V.size()));
      }
      if(ImGui::Button("Solve one step")){
        solver.solve_one_step();
        polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
            Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3));
      }
  };
  polyscope::show();
}