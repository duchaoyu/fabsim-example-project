#include <Eigen/Dense>
#include <fsim/ElasticMembrane.h>
#include <fsim/util/io.h>

// Created by Chaoyu Du on 06.09.24.
using namespace Eigen;


int main(int argc, char *argv[]) {

  Vector3d xaxis;
  xaxis << -1, -1, 1;
  xaxis.normalize();

  Matrix3d V;
  V.row(0) << 0, 2, 0;
  V.row(1) << 1, 0, 0;
  V.row(2) << 0, 0, 0;


  Matrix3d X;
  X.row(0) << 0, 2, 1;
  X.row(1) << 1, 0, 1;
  X.row(2) << 0, 0, 1;

  Vector3i E;
  E << 0, 1, 2;


  Vector3d e1 = V.row(E(0)) - V.row(E(2));
  Vector3d e2 = V.row(E(1)) - V.row(E(2));

  std::cout << e1 << " ,  " <<  e2 << std::endl;

// construct local frame
// origin
  Vector3d sum = V.row(E(0)) + V.row(E(1)) + V.row(E(2));
  Vector3d origin = sum / 3.0;
  std::cout << origin << " , origin " << std::endl;

  Vector3d zaxis = e1.cross(e2).normalized();

  double dotProduct = xaxis.dot(zaxis);
  double tolerance = 1e-6;
  if (std::abs(dotProduct) < tolerance) {
    std::cout << "The vector lies on the triangle's plane." << std::endl;
  } else {
    std::cout << dotProduct << "The vector does NOT lie on the triangle's plane." << std::endl;
    Vector3d v_parallel = (xaxis.dot(zaxis)) * zaxis;
    Vector3d v_projected = xaxis - v_parallel;
    xaxis << v_projected.normalized();
    std::cout << xaxis << std::endl;
  }


  Vector3d yaxis = zaxis.cross(xaxis).normalized();

  Vector3d _zaxis = xaxis.cross(yaxis).normalized();
  std::cout << zaxis << " , z " << _zaxis << std::endl;

  Matrix3d R;
  R.col(0) << xaxis;
  R.col(1) << yaxis;
  R.col(2) << zaxis;

  std::cout << R << std::endl;

  Matrix4d T = Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = origin;
  Matrix4d T_inverse = T.inverse();
  Matrix4d T_mul = T_inverse * Matrix4d::Identity();


  std::cout << "T_mul" << std::endl;
  std::cout << T_mul << std::endl;
  std::cout << "T_mul_transpose" << std::endl;
  std::cout << T_mul.transpose() << std::endl;

  MatrixXd V_local_xy33(3, 3);
  MatrixXd V_local_XY(3,3 );

  for (int i = 0; i < 3; ++i) {
    // Convert each 3D point to homogeneous coordinates (Vector4d)
//    Vector4d V_homogeneous;
    Matrix<double, 1, 4> V_homogeneous_xy;
    V_homogeneous_xy << V.row(E(i))[0], V.row(E(i))[1], V.row(E(i))[2], 1.0;

    Matrix<double, 1, 4> V_homogeneous_XY;
    V_homogeneous_XY << X.row(i)[0], X.row(i)[1], X.row(i)[2], 1.0;

    Matrix<double, 1, 4>  V_transformed_xy = V_homogeneous_xy * T_mul.transpose() ;
    Matrix<double, 1, 4>  V_transformed_XY = V_homogeneous_XY * T_mul.transpose() ;


    V_local_xy33.row(i) << V_transformed_xy.head<3>();  // Store in the output matrix
    V_local_XY.row(i) << V_transformed_XY.head<3>();
  }



  MatrixXd V_local_xy(3,2);
//  V_local_xy << V_local;
  V_local_xy << V_local_xy33.block<3, 2>(0, 0);




  std::cout << "V_local_xy" << std::endl;
  std::cout << V_local_xy33 << std::endl;
  std::cout << V_local_xy << std::endl;

  std::cout << "V_local_XY" << std::endl;
  std::cout << V_local_XY << std::endl;





  double coeff;
  Eigen::Matrix<double, 3, 2> _R;
  double area;
  double thickness = 1;

  // _R is the initial length
  _R << V_local_xy(1, 1) - V_local_xy(2, 1), V_local_xy(1, 0) - V_local_xy(2, 0),
      V_local_xy(2, 1) - V_local_xy(0, 1), V_local_xy(2, 0) - V_local_xy(0, 0),
      V_local_xy(0, 1) - V_local_xy(1, 1), V_local_xy(0, 0) - V_local_xy(1, 0);

  double d = Vector3d(V_local_xy(0, 0), V_local_xy(1, 0), V_local_xy(2, 0)).dot(_R.col(0)); // area of the triangle

  _R /= d;
  coeff = thickness / 2 * std::abs(d);

  Matrix3d P;
  P.col(0) << V_local_XY.row(0)[0], V_local_XY.row(0)[1], V_local_XY.row(0)[2];
  P.col(1) << V_local_XY.row(1)[0], V_local_XY.row(1)[1], V_local_XY.row(1)[2];
  P.col(2) << V_local_XY.row(2)[0], V_local_XY.row(2)[1], V_local_XY.row(2)[2];
  Matrix<double, 3, 2> F = P * _R;

  std::cout << F << ", F" << std::endl;

  Vector3d res;
  res(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
  res(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
  res(2) = F.col(1).dot(F.col(0));

  std::cout << res << ", Strain" << std::endl;


  Eigen::Matrix3d C;
  double _E1 = 25000;
  double _E2 = 50000;
  double _poisson_ratio = 0.3;

  C << _E1, _poisson_ratio * sqrt(_E1 * _E2), 0,
      _poisson_ratio * sqrt(_E1 * _E2), _E2, 0,
      0, 0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);
  C /= (1 - std::pow(_poisson_ratio, 2));

  Vector3d stress = C * res;
  std::cout << stress << "stress" <<  std::endl;
  double energy;
  double mass= 0.0;
  energy = coeff * (0.5 * res.dot(C * res) + 9.8 * mass * (X.row(0)[2] + X.row(1)[2] + X.row(2)[2]));

  std::cout << energy << "energy" <<  std::endl;












}
