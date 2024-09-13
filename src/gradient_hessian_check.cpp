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
  const double young_modulus1 = 5000;
  const double young_modulus2 = 2000;
  const double thickness = 10;
  const double poisson_ratio = 0.3;
  double stretch_factor = 1;
  double mass = 10;
  double pressure = 10;



  // declare StVKMembrane object (could be replaced seamlessly with e.g. NeohookeanMembrane)
//      fsim::StVKMembrane model(V / stretch_factor, F, thickness, young_modulus1, poisson_ratio, mass);
//  fsim::StVKMembrane model(V / 2, F, thickness, young_modulus1, poisson_ratio, mass, pressure);
//
  fsim::OrthotropicStVKMembrane model(V/1.5 , F, thickness, young_modulus1, young_modulus2, poisson_ratio, mass, pressure);

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
  {
//
//
//  Vector3d xaxis;
//  xaxis << -1, 0, 0;
//  xaxis.normalize();
//
//  // in the local frame i, j += tol
//  Vector3d e1 = V.row(F(0)) - V.row(F(2));
//  Vector3d e2 = V.row(F(1)) - V.row(F(2));
//
//// construct local frame
//// origin
//  Vector3d sum = V.row(F(0)) + V.row(F(1)) + V.row(F(2));
//  Vector3d origin = sum / 3.0;
////  std::cout << origin << " , origin " << std::endl;
//
//  Vector3d zaxis = e1.cross(e2).normalized();
//
//  double dotProduct = xaxis.dot(zaxis);
//  double tolerance = 1e-6;
//  if (std::abs(dotProduct) < tolerance) {
////    std::cout << "The vector lies on the triangle's plane." << std::endl;
//  } else {
////    std::cout << dotProduct << "The vector does NOT lie on the triangle's plane." << std::endl;
//    Vector3d v_parallel = (xaxis.dot(zaxis)) * zaxis;
//    Vector3d v_projected = xaxis - v_parallel;
//    xaxis << v_projected.normalized();
////    std::cout << xaxis << std::endl;
//  }
//
//  Vector3d yaxis = zaxis.cross(xaxis).normalized();
//
//  Vector3d _zaxis = xaxis.cross(yaxis).normalized();
////  std::cout << zaxis << " , z " << _zaxis << std::endl;
//
//  Matrix3d R;
//  R.col(0) << xaxis;
//  R.col(1) << yaxis;
//  R.col(2) << zaxis;
//
////  std::cout << R << std::endl;
//
//  Matrix4d T = Matrix4d::Identity();
//  T.block<3, 3>(0, 0) = R;
//  T.block<3, 1>(0, 3) = origin;
//  Matrix4d T_inverse = T.inverse();
//  Matrix4d T_mul = T_inverse * Matrix4d::Identity();
//
//  MatrixXd V_local_XY(3,3);
//
//  for (int i = 0; i < 3; ++i) {
//    // Convert each 3D point to homogeneous coordinates (Vector4d)
//    Matrix<double, 1, 4> V_homogeneous_XY;
//    V_homogeneous_XY << X[i*3], X[i*3+1], X[i*3+2], 1.0;
//    Matrix<double, 1, 4>  V_transformed_XY = V_homogeneous_XY * T_mul.transpose() ;
//    V_local_XY.row(i) << V_transformed_XY.head<3>();
//  }
//
//  Matrix4d T_mul_local_global = Matrix4d::Identity().inverse() * T;
//
//  MatrixXd V_global_XY(3,3 );
//
//  fsim::Mat3<double> V_local_XY2 = V_local_XY;
//  V_local_XY2(a, b) += tol;
//
//  for (int i = 0; i < 3; ++i) {
//    // Convert each 3D point to homogeneous coordinates (Vector4d)
//    Matrix<double, 1, 4> V_homogeneous_xy, V_homogeneous_xy2;
//    V_homogeneous_xy << V_local_XY.row(i)[0], V_local_XY.row(i)[1], V_local_XY.row(i)[2], 1.0;
//    V_homogeneous_xy2 << V_local_XY2.row(i)[0], V_local_XY2.row(i)[1], V_local_XY2.row(i)[2], 1.0;
//    Matrix<double, 1, 4>  V_transformed_xy = V_homogeneous_xy * T_mul_local_global.transpose();
//    Matrix<double, 1, 4>  V_transformed_xy2 = V_homogeneous_xy2 * T_mul_local_global.transpose();
//    V_local_XY.row(i) << V_transformed_xy.head<3>();
//    V_local_XY2.row(i) << V_transformed_xy2.head<3>();
//  }
//
//  // Output the result
////  std::cout << "Global coordinates:\n" << V_local_XY << std::endl;
////  std::cout << "Global coordinates2:\n" << V_local_XY2 << std::endl;
//
//
//Matrix3d grad33;
//grad33.row(0) = model.gradient(X).segment<3>(0);
//grad33.row(1) = model.gradient(X).segment<3>(3);
//grad33.row(2) = model.gradient(X).segment<3>(6);
//std::cout <<  "gradient_full \n" << model.gradient(X) << std::endl;
//std::cout <<  "gradient_full33 \n" << grad33 << std::endl;
//
//MatrixXd grad_33_xy(3,3 );
//
//for (int i = 0; i < 3; ++i) {
//    // Convert each 3D point to homogeneous coordinates (Vector4d)
//    Matrix<double, 1, 4> grad14;
//    grad14 << grad33.row(i)[0], grad33.row(i)[1], grad33.row(i)[2], 1.0;
//    Matrix<double, 1, 4>  grad14_xy = grad14 * T_mul_local_global.transpose();
//    grad_33_xy.row(i) << grad14_xy.head<3>();
//  }
//  std::cout <<" gradient transformed back to global \n"  << grad_33_xy <<  std::endl;
//
  }
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
