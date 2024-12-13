#include <fsim/ElasticMembrane.h>
#include <fsim/OrthotropicStVKMembrane.h>
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
//      fsim::readOFF("/Users/duch/Downloads/pillow.off", V, F);
  fsim::readOFF("/Users/duch/Documents/Github/fabsim-example-project/data/triangle.off", V, F);

  // parameters of the membrane model
  const double young_modulus1 = 10000;
  const double young_modulus2 = 5000;
  const double thickness = 1;
  const double poisson_ratio = 0.3;
  double stretch_factor = 1;
  double mass = 10;
  double pressure = 100;

  // Create face vectors (one per face)
  std::vector<Eigen::Vector3d> face_vectors(F.rows(), Eigen::Vector3d(1.0, 0.0, 0.0));
  std::cout << face_vectors[0] << std::endl;



  // declare StVKMembrane object (could be replaced seamlessly with e.g. NeohookeanMembrane)
//      fsim::StVKMembrane model(V / stretch_factor, F, thickness, young_modulus1, poisson_ratio, mass);
//  fsim::StVKMembrane model(V / 2, F, thickness, young_modulus1, poisson_ratio, mass, pressure);
//
  fsim::OrthotropicStVKMembrane model(V/1.5 , F, thickness, young_modulus1, young_modulus2, poisson_ratio, face_vectors, mass, pressure);

  std::cout << V << " V" << std::endl;
//  std::cout << F << " F" << std::endl;

  Eigen::VectorXd X = Eigen::Map<Eigen::VectorXd>(V.data(), V.size());

  int a = 0;
  int b = 0;
  std::cout << model.gradient(X)(a * 3 + b) << " gradient 0" << std::endl;

  std::cout << model.hessian(X) << " hessian 0" << std::endl;
  std::cout << "full gradient" << std::endl;
  std::cout << model.gradient(X) << " end" << std::endl;

  double tol = 1e-6;
  fsim::Mat3<double> V2=V;

  V2(a, b) += tol;
//  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << X << " X" << std::endl;

  std::cout << V2.row(a)[b] - V.row(a)[b] << " e" << std::endl;

//  Eigen::VectorXd global_X2 = Eigen::Map<Eigen::VectorXd>(V_local_XY2.data(), V_local_XY2.size());
//  std::cout << (model.energy(global_X2) - model.energy(X)) / tol << " energy_difference / step" << std::endl;
Matrix3d energy_difference;
Matrix3d energy_tol_diff;

  for (int p = 0; p <= 2; ++p) {
    for (int q = 0; q <= 2; ++q) {
      fsim::Mat3<double> V_copy = V;
      V_copy(p, q) += tol;
      Eigen::VectorXd Xcopy = Eigen::Map<Eigen::VectorXd>(V_copy.data(), V_copy.size());
      energy_difference.row(p)[q] =  (model.energy( Xcopy) - model.energy(X));
      energy_tol_diff.row(p)[q] = (model.energy( Xcopy) - model.energy(X)) / tol ;
    }
  }
  std::cout << energy_difference << " energy_difference" << std::endl;
  std::cout << energy_tol_diff << " energy_difference / step" << std::endl;


  Eigen::VectorXd X2 = Eigen::Map<Eigen::VectorXd>(V2.data(), V2.size());
  std::cout << (model.energy( X2) - model.energy(X)) << " energy_difference" << std::endl;
  std::cout << (model.energy( X2) - model.energy(X)) / tol << " energy_difference / step" << std::endl;



//  for (int p =0; p<3; p++){
//    for (int q=0; q<3; q++){
//      V2 << V;
//      V2(p, q) += tol;
//      X2 << Eigen::Map<Eigen::VectorXd>(V2.data(), V2.size());
//      std::cout << model.gradient(X)(p * 3 + q) << " gradent 0" << std::endl;
//      std::cout << (model.energy(X2) - model.energy(X)) / tol << " energy_difference / step" << std::endl;
//    }
//  }
//
//  fsim::Mat3<double> V3 = V;
//  fsim::Mat3<double> V4 = V;
//  Eigen::VectorXd X3 = Eigen::Map<Eigen::VectorXd>(V3.data(), V.size());
//  Eigen::VectorXd X4 = Eigen::Map<Eigen::VectorXd>(V4.data(), V.size());
//
//  for (int p =0; p<3; p++){
//    for (int q=0; q<3; q++){
//      for (int m=0; m<3; m++){
//        for (int n=0; n<3; n++){
//          V2 << V;
//          V3 << V;
//          V4 << V;
//          V2(p, q) += tol;
//          V2(m, n) += tol;
//          X2 << Eigen::Map<Eigen::VectorXd>(V2.data(), V.size());
//          V3(m, n) += tol;
//          X3 << Eigen::Map<Eigen::VectorXd>(V3.data(), V.size());
//          V4(p, q) += tol;
//          X4 << Eigen::Map<Eigen::VectorXd>(V4.data(), V.size());
//
//          std::cout << (model.energy(X2) - model.energy(X3) - model.energy(X4) + model.energy(X)) / (tol * tol) << std::endl;
//          std::cout << p << ", " << q << ", " << m << ", " << n << " hessian numerical" << std::endl;
//
//        }
//      }
//
//    }
//  }

  for (int p =0; p<3; p++){
    for (int q=0; q<3; q++){
      V2 << V;
      V2(p, q) += tol;
      X2 << Eigen::Map<Eigen::VectorXd>(V2.data(), V2.size());
      std::cout << (model.gradient(X2)(p * 3 + q) - model.gradient(X)(p * 3 + q)) / tol << ", " << p << ", " << q << ", gradient_difference / step"
                << std::endl;
    }
  }

}
