#include <fsim/ElasticMembrane.h>
#include "fsim/StVKElement.h"
#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include <iostream>
#include <fstream>



int main(int argc, char *argv[]) {
    using namespace Eigen;

    // load geometry from OFF mesh file
    fsim::Mat3<double> V;
    fsim::Mat3<int> F;
    fsim::readOFF("/Users/duch/Downloads/pillow_out.off", V, F);

    std::cout << V.rows()<< ", " << F.rows()<< std::endl;


    // parameters of the membrane model
    const double young_modulus = 10;
    const double thickness = 0.5;
    const double poisson_ratio = 0.2;
    double stretch_factor = 1.7;
    double mass = 1.0;


  std::vector<double> thicknesses(F.rows(), 0.0); // Assuming F.rows() gives the correct size
  std::vector<int> faceIndices;
  for (int face = 0; face < F.rows(); face++){
    int f1 = F.row(face)[0];
    int f2 = F.row(face)[1];
    int f3 = F.row(face)[2];
    Vector3d v1 = V.row(f1);
    Vector3d v2 = V.row(f2);
    Vector3d v3 = V.row(f3);
    Vector3d face_cen = (v1+v2+v3)/3;
//    std::cout << v1.transpose() << ", "<< face_cen.transpose() << std::endl;

    if ((face_cen[1]>1.5 and face_cen[1]<1.8) or (face_cen[0]>1.5 and face_cen[0]<1.8)){
      faceIndices.push_back(face);
    }
  }

  std::cout << "faceIndices: ";
  for (int i = 0; i < faceIndices.size(); ++i) {
    std::cout << faceIndices[i] << " ";
  }
  std::cout << std::endl;

//  int faceIndices[] = {11, 12, 7, 16, 5, 14, 4, 0, 1, 2, 15,19, 21, 23, 20, 10, 8, 9, 6, 17, 13, 18, 3, 22};
// Use faceIndices directly, assuming the indices are within the bounds of thicknesses
//  for (int i = 0; i < sizeof(faceIndices)/sizeof(faceIndices[0]); ++i)
    for (int i = 0; i < faceIndices.size(); ++i) {
    int f = faceIndices[i]; // Direct access since it's a plain array
      thicknesses[f] = 2.0;
  }

    // declare StVKMembrane object (could be replaced seamlessly with e.g. NeohookeanMembrane)
    fsim::StVKMembrane model(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);


// declare NewtonSolver object
    optim::NewtonSolver<double> solver;

    std::ifstream bdr_file("/Users/duch/Downloads/pillow_out_bdr.txt");
    std::vector<int> bdrs;
    // Read each line and convert it to an integer
    int num;
    while (bdr_file >> num) {
      bdrs.push_back(num);
    }

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

        // for debug: visaulise the thickened faces
        if (ImGui::Checkbox("Show Thickness Color", &setThicknessColor)) {
          Eigen::VectorXd scalarsEigen(thicknesses.size());
          for (size_t i = 0; i < thicknesses.size(); ++i) {
            scalarsEigen(i) = thicknesses[i];
          }
          auto* scalarQ = polyscope::getSurfaceMesh("mesh")->addFaceScalarQuantity("Scalar Values", scalarsEigen);
          if (setThicknessColor) {
            scalarQ->setEnabled(true);
          } else{
            scalarQ->setEnabled(false);
          }
          polyscope::requestRedraw(); // Request a redraw to update the visualization
        }
    };

    polyscope::show();

}
