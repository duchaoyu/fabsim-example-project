#include <fsim/ElasticMembrane.h>
#include "fsim/StVKElement.h"
#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include <iostream>



int main(int argc, char *argv[]) {
  using namespace Eigen;

  // load geometry from OFF mesh file
  fsim::Mat3<double> V;
  fsim::Mat3<int> F;
  fsim::readOFF("/Users/duch/Downloads/pillow.off", V, F);


  // parameters of the membrane model
  const double young_modulus = 10;
  const double thickness = 0.5;
  const double poisson_ratio = 0.2;
  double stretch_factor = 1.7;
  double mass = 1;


  std::vector<double> thicknesses(F.rows(), 0.0); // Assuming F.rows() gives the correct size

  int faceIndices[] = {11, 12, 7, 16, 5, 14, 4, 0, 1, 2, 15,19, 21, 23, 20, 10, 8, 9, 6, 17, 13, 18, 3, 22};
// Use faceIndices directly, assuming the indices are within the bounds of thicknesses
  for (int i = 0; i < sizeof(faceIndices)/sizeof(faceIndices[0]); ++i) {
    int f = faceIndices[i]; // Direct access since it's a plain array
    thicknesses[f] = 1.0;
  }

  // declare StVKMembrane object (could be replaced seamlessly with e.g. NeohookeanMembrane)
  fsim::StVKMembrane model(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);
//     fsim::StVKMembrane model(V / stretch_factor, F, thicknesses, young_modulus, poisson_ratio, mass);


//  fsim::readOFF("/Users/duch/Documents/Github/fabsim-example-project/data/triangle.off", V, F);
//    fsim::StVKMembrane model(V/3 , F, thickness, young_modulus, poisson_ratio, mass);

//    Eigen::VectorXd X = Eigen::Map<Eigen::VectorXd>(V.data(), V.size());
//
//  int i = 0;
//  int j = 0;
//    std::cout << model.gradient(X)(i*3+j) << " gradient 0" << std::endl;
//    std::cout << model.hessian(X) << " hessian 0" << std::endl;
////    std::cout << "full gradient" << std::endl;
////  std::cout << model.gradient(X) << " end" << std::endl;
//
//  double tol = 1e-6;
//    fsim::Mat3<double>  V2 = V;
//    V2(i, j) += tol;
////    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << X << " X" << std::endl;
//
//    std::cout << V2.row(i)[j] - V.row(i)[j]  << " e" << std::endl;
//
//    Eigen::VectorXd X2 = Eigen::Map<Eigen::VectorXd>(V2.data(), V2.size());
//
//
////    std::cout << model.energy(X2) << " energy_e" << std::endl;
//    std::cout << (model.energy(X2) -  model.energy(X)) / tol << " energy_difference / step" << std::endl;
//    std::cout << (model.gradient(X2)(i*3+j) - model.gradient(X)(i*3+j))/tol << "gradient_difference / step" << std::endl;




// declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  std::vector<int> bdrs = {10, 11, 18, 22, 25, 27, 30, 41, 42, 53, 55, 57, 65, 66, 83, 85, 92, 93, 94, 96, 98, 100, 101, 111, 122, 123, 124, 126, 128, 130, 133, 134, 137, 138, 145, 146, 149, 150, 151, 152, 156, 160, 161, 163, 164, 166};

  for (int bdr : bdrs) {
    solver.options.fixed_dofs.push_back(bdr * 3);
    solver.options.fixed_dofs.push_back(bdr * 3 + 1);
    solver.options.fixed_dofs.push_back(bdr * 3 + 2);
  }

  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be

  // display the mesh
  polyscope::registerSurfaceMesh("mesh", V, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1})
      ->setSurfaceColor({0, 1., 1.});
  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;
  polyscope::init();


  bool setThicknessColor = false;
  bool setThickness = false;

  polyscope::state::userCallback = [&]()
  {
      ImGui::PushItemWidth(100);
      if(ImGui::InputDouble("Stretch factor", &stretch_factor, 0, 0, "%.2f"))
        model = fsim::StVKMembrane(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);

      if(ImGui::InputDouble("Mass", &mass, 0, 0, "%.2f"))
        model.setMass(mass);

      if(ImGui::Button("Solve"))
      {
        // Newton's method: finds a local minimum of the energy (Fval = energy value, Optimality = gradient's norm)
        solver.solve(model, Map<VectorXd>(V.data(), V.size()));

        // Display the result of the optimization
        polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
            Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3));

        fsim::saveOBJ("/Users/duch/Downloads/pillow_o.off", Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3), F);
      }

      if (ImGui::Checkbox("Set Thicknesses", &setThickness)) {
        if (setThickness){
          model = fsim::StVKMembrane(V / stretch_factor, F, thicknesses, young_modulus, poisson_ratio, mass);
        }
        else{
          model = fsim::StVKMembrane(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);
        }
      }

      if (ImGui::Checkbox("Use Custom Color", &setThicknessColor)) {
        if (setThicknessColor) {

          Eigen::VectorXd scalarsEigen(thicknesses.size());
          for (size_t i = 0; i < thicknesses.size(); ++i) {
            scalarsEigen(i) = thicknesses[i];
          }

          // Add the scalar quantity to the mesh
          polyscope::getSurfaceMesh("mesh")->addFaceScalarQuantity("Scalar Values", scalarsEigen);

          auto* scalarQ = polyscope::getSurfaceMesh("mesh")->addFaceScalarQuantity("Scalar Values", scalarsEigen);
//            scalarQ->setColormap(polyscope::gl::ColorMapID::COOLWARM); // Example colormap
          scalarQ->setEnabled(true); // Make sure it's enabled for visualization


          polyscope::getSurfaceMesh("mesh")->setSurfaceColor(glm::vec3(1.0, 0.0, 0.0)); // Example: red color
        } else {
          // Set to the default color
          polyscope::getSurfaceMesh("mesh")->setSurfaceColor({0, 1., 1.}); // Use Polyscope's default mesh color
        }
        polyscope::requestRedraw(); // Request a redraw to update the visualization

      }
  };
  polyscope::show();
//  std::cout << model::stress << std::endl;
//  std::cout << V << std::endl;
}
