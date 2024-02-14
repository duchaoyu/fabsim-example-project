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
//  fsim::readOFF("/Users/duch/Downloads/pillow.off", V, F);
  fsim::readOFF("/Users/duch/Downloads/butt_out.off", V, F);
//  V *= 10;



  // parameters of the membrane model
  double young_modulus = 50000;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
  double thickness = 0.4;
  double thickness2 = 1.0;
  double poisson_ratio = 0.38;  // 0.38
  double stretch_factor = 1.1;
  double mass = 30; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
  double pressure = 250; // pressure per area N/m2, Pa, air-supported structure 200-300 Pa


  std::vector<double> thicknesses(F.rows(), thickness); // Assuming F.rows() gives the correct size

//  int faceIndices[] = {11, 12, 7, 16, 5, 14, 4, 0, 1, 2, 15,19, 21, 23, 20, 10, 8, 9, 6, 17, 13, 18, 3, 22};
  int faceIndices[] = {297, 296, 299, 298, 168,169, 170, 171, 77, 76, 49, 48, 30, 31, 243, 242, 180, 181, 314, 315, 317, 316, 112, 113, 272, 273, 221,
                       220, 218, 219, 217, 216, 121, 120, 2, 3, 0, 1, 74, 75, 225, 224, 7, 6, 4, 5, 244, 245,
                       240, 241, 44, 45, 148, 149, 19, 18, 42, 43, 22, 23, 334, 335, 263,262,
                       158, 159, 161, 160, 132, 133, 134,135, 269, 268, 260 ,261,
                       318, 319, 237, 236, 226,227, 228, 229, 186, 187, 231, 230,
                       248, 249, 128, 129, 238,239, 11, 10, 13, 12,
                       190, 191, 109, 108, 89, 88, 92, 93, 185, 184};
// Use faceIndices directly, assuming the indices are within the bounds of thicknesses
  for (int i = 0; i < sizeof(faceIndices)/sizeof(faceIndices[0]); ++i) {
    int f = faceIndices[i]; // Direct access since it's a plain array
    thicknesses[f] = thickness2;
  }

  // declare StVKMembrane object (could be replaced seamlessly with e.g. NeohookeanMembrane)
  fsim::StVKMembrane model(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass, pressure);
//     fsim::StVKMembrane model(V / stretch_factor, F, thicknesses, young_modulus, poisson_ratio, mass);


// declare NewtonSolver object
  optim::NewtonSolver<double> solver;
//  std::vector<int> bdrs = {10, 11, 18, 22, 25, 27, 30, 41, 42, 53, 55, 57, 65, 66, 83, 85, 92, 93, 94, 96, 98, 100, 101, 111, 122, 123, 124, 126, 128, 130, 133, 134, 137, 138, 145, 146, 149, 150, 151, 152, 156, 160, 161, 163, 164, 166};
  std::vector<int> bdrs = {0, 1, 3, 10, 16, 17, 18, 25, 39, 41, 52, 53, 62, 63, 65, 66, 67, 76, 85, 88, 90, 91, 94, 95, 102, 103, 104, 106, 108, 111, 113, 115, 119, 120, 122, 139, 140, 141, 144, 150, 151, 153, 154, 165, 166, 171, 173, 177, 178, 181, 189, 191};

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
  bool setThicknesses = false;
  bool showStrain = false;
  bool showStress = false;
  bool showDeviation = false;
  bool showRef = false;

  Eigen::VectorXd xTarget(V.size());
  for(int i = 0; i < V.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      xTarget(3 * i + j) = V(i, j);


  polyscope::state::userCallback = [&]()
  {
      ImGui::PushItemWidth(100);
      if(ImGui::InputDouble("Stretch factor", &stretch_factor, 0, 0, "%.2f"))
        model = fsim::StVKMembrane(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass, pressure);

      if(ImGui::InputDouble("Thickness", &thickness, 0, 0, "%.2f")){
        for (int i = 0; i < F.rows(); ++i) {
          thicknesses[i] = thickness;
        }
      model = fsim::StVKMembrane(V / stretch_factor, F, thicknesses, young_modulus, poisson_ratio, mass, pressure);}

      if(ImGui::InputDouble("Thickness2", &thickness2, 0, 0, "%.2f")){
        for (int i = 0; i < sizeof(faceIndices)/sizeof(faceIndices[0]); ++i) {
          int f = faceIndices[i]; // Direct access since it's a plain array
          thicknesses[f] = thickness2;
        }
        model = fsim::StVKMembrane(V / stretch_factor, F, thicknesses, young_modulus, poisson_ratio, mass, pressure);
      }

//        model.setThickness(thickness);

      if(ImGui::InputDouble("Possian", &poisson_ratio, 0, 0, "%.2f"))
        model.setPoissonRatio(poisson_ratio);

      if(ImGui::InputDouble("Modulus", &young_modulus, 0, 0, "%.2f"))
        model.setYoungModulus(young_modulus);

      if(ImGui::InputDouble("Mass", &mass, 0, 0, "%.3f"))
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

        if (showDeviation) {

//          Eigen::MatrixXd VTarget = xTarget.reshaped<Eigen::RowMajor>(V.rows(), 3);
//          Eigen::MatrixXd Vsolve = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3))
          fsim::Mat3<double> VTarget = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
          Eigen::VectorXd d = (V - VTarget).cwiseProduct((V - VTarget)).rowwise().sum();
          d = d.array().sqrt();
          std::cout << "Avg distance = "
                    << 100 * d.sum() / d.size() / (VTarget.colwise().maxCoeff() - VTarget.colwise().minCoeff()).norm()
                    << "\n";
          std::cout << "Max distance = "
                    << 100 * d.lpNorm<Eigen::Infinity>() /
                       (VTarget.colwise().maxCoeff() - VTarget.colwise().minCoeff()).norm()
                    << "\n";
          polyscope::getSurfaceMesh("mesh")->addVertexScalarQuantity("Distance", d)->setEnabled(true);
        }

        fsim::saveOBJ("/Users/duch/Downloads/pillow_o.off", Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3), F);
      }

//      if (ImGui::Checkbox("Show Strain", &showStrain)) {
//        if (showStrain){
//          for (int i = 0; i < F.rows(); i++) {
//            Eigen::Vector3i faceV = F.row(i);
//            fsim::StVKElement element(solver.var().data(), faceV, thicknesses[i]);
//          }
//        }
//      }

      if (ImGui::Checkbox("Show Stress", &showStress)) {
        if (showStress){

        }
      }

      if(ImGui::Button("Init"))
      {
        solver.init(model, Map<VectorXd>(V.data(), V.size()));
      }
      if(ImGui::Button("Solve one step")){
        solver.solve_one_step();
        polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
            Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3));

        if (showDeviation) {

//          Eigen::MatrixXd VTarget = xTarget.reshaped<Eigen::RowMajor>(V.rows(), 3);
//          Eigen::MatrixXd Vsolve = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3))
          fsim::Mat3<double> VTarget = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
          Eigen::VectorXd d = (V - VTarget).cwiseProduct((V - VTarget)).rowwise().sum();
          d = d.array().sqrt();
          std::cout << "Avg distance = "
                    << 100 * d.sum() / d.size() / (VTarget.colwise().maxCoeff() - VTarget.colwise().minCoeff()).norm()
                    << "\n";
          std::cout << "Max distance = "
                    << 100 * d.lpNorm<Eigen::Infinity>() /
                       (VTarget.colwise().maxCoeff() - VTarget.colwise().minCoeff()).norm()
                    << "\n";
          polyscope::getSurfaceMesh("mesh")->addVertexScalarQuantity("Distance", d)->setEnabled(true);
        }

      }

      if (ImGui::Checkbox("Set Thicknesses", &setThicknesses)) {
        if (setThicknesses){
          model = fsim::StVKMembrane(V / stretch_factor, F, thicknesses, young_modulus, poisson_ratio, mass, pressure);
        }
        else{
          model = fsim::StVKMembrane(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass, pressure);
        }
      }
      if (ImGui::Checkbox("Set Deviation", &showDeviation)) {
      }

      if (ImGui::Checkbox("Use Custom Color", &setThicknessColor)) {
        if (setThicknessColor) {

          Eigen::VectorXd scalarsEigen(thicknesses.size());
          for (size_t i = 0; i < thicknesses.size(); ++i) {
            scalarsEigen(i) = thicknesses[i];
          }

          // Add the scalar quantity to the mesh
          polyscope::getSurfaceMesh("mesh")->addFaceScalarQuantity("Scalar Values", scalarsEigen);

          auto *scalarQ = polyscope::getSurfaceMesh("mesh")->addFaceScalarQuantity("Scalar Values", scalarsEigen);
//            scalarQ->setColormap(polyscope::gl::ColorMapID::COOLWARM); // Example colormap
          scalarQ->setEnabled(true); // Make sure it's enabled for visualization

        }
      }
      if (ImGui::Checkbox("Show Reference Mesh", &showRef)) {
        if (showRef){
          polyscope::registerSurfaceMesh("refmesh", V, F)
              ->setEdgeWidth(1)
              ->setEdgeColor({0.1, 0.1, 0.1})
              ->setSurfaceColor({0, 1., 1.});
        }
      }

  };
  polyscope::show();
//  std::cout << model::stress << std::endl;
//  std::cout << V << std::endl;
}
