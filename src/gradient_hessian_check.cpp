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
  //    fsim::readOFF("/Users/duch/Downloads/pillow.off", V, F);
  fsim::readOFF("/Users/duch/Documents/Github/fabsim-example-project/data/triangle.off", V, F);



  // parameters of the membrane model
  const double young_modulus = 10;
  const double thickness = 0.5;
  const double poisson_ratio = 0.3;
  double stretch_factor = 1.7;
  double mass = 1;

  // declare StVKMembrane object (could be replaced seamlessly with e.g. NeohookeanMembrane)
  //    fsim::StVKMembrane model(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);
  fsim::StVKMembrane model(V / 3, F, thickness, young_modulus, poisson_ratio, mass);

  Eigen::VectorXd X = Eigen::Map<Eigen::VectorXd>(V.data(), V.size());

  int i = 0;
  int j = 0;
  std::cout << model.gradient(X)(i * 3 + j) << " gradient 0" << std::endl;
  std::cout << model.hessian(X) << " hessian 0" << std::endl;
//    std::cout << "full gradient" << std::endl;
//  std::cout << model.gradient(X) << " end" << std::endl;

  double tol = 1e-6;
  fsim::Mat3<double> V2 = V;
  V2(i, j) += tol;
//    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << X << " X" << std::endl;

  std::cout << V2.row(i)[j] - V.row(i)[j] << " e" << std::endl;

  Eigen::VectorXd X2 = Eigen::Map<Eigen::VectorXd>(V2.data(), V2.size());


//    std::cout << model.energy(X2) << " energy_e" << std::endl;
  std::cout << (model.energy(X2) - model.energy(X)) / tol << " energy_difference / step" << std::endl;
  std::cout << (model.gradient(X2)(i * 3 + j) - model.gradient(X)(i * 3 + j)) / tol << "gradient_difference / step"
            << std::endl;
}
