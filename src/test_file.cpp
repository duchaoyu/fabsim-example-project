#include <Eigen/Dense>
#include <fsim/ElasticMembrane.h>
#include <fsim/util/io.h>

// Created by Chaoyu Du on 06.09.24.
using namespace Eigen;

// Function to compute local frame based on points A, B, C
void computeLocalFrame(const Vector3d& A, const Vector3d& B, const Vector3d& C,
                       Matrix3d& rotationMatrix, Vector3d& translation) {
  // Compute the local X-axis (AB)
  Vector3d xLocal = (B - A).normalized();

  // Compute the local Z-axis (normal to triangle ABC)
  Vector3d zLocal = (B - A).cross(C - A).normalized();

  // Compute the local Y-axis (cross product of Z and X)
  Vector3d yLocal = zLocal.cross(xLocal);

  // Construct the rotation matrix
  rotationMatrix.col(0) = xLocal;
  rotationMatrix.col(1) = yLocal;
  rotationMatrix.col(2) = zLocal;

  // Translation (A becomes the origin of the local frame)
  translation = A;
}
// Function to compute the local coordinates of a point in the new frame
Vector3d computeLocalCoordinates(const Vector3d& point, const Matrix3d& rotationMatrix, const Vector3d& translation) {
  return rotationMatrix.transpose() * (point - translation);
}

int main(int argc, char *argv[]) {

  Vector3d xaxis;
  xaxis << -1, -1, 1;
  xaxis.normalize();

  Matrix3d V;
  V.row(0) << 0, 2, 0;
  V.row(1) << 1, 0, 0;
  V.row(2) << 0, 0, 0;


  Matrix3d X;
  X.row(0) << 0, 0, 1;
  X.row(1) << 1, 2, 1;
  X.row(2) << 0, 2, 1;

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

  std::cout << T_mul << std::endl;
  std::cout << T_mul.transpose() << std::endl;

  MatrixXd V_local(3, 3);

  for (int i = 0; i < 3; ++i) {
    // Convert each 3D point to homogeneous coordinates (Vector4d)
//    Vector4d V_homogeneous;
    Matrix<double, 1, 4> V_homogeneous;
    V_homogeneous << V.row(E(i))[0], V.row(E(i))[1], V.row(E(i))[2], 1.0;
    Matrix<double, 1, 4>  V_transformed = V_homogeneous * T_mul.transpose() ;
    V_local.row(i) << V_transformed.head<3>();  // Store in the output matrix
  }

  std::cout << V_local << std::endl;


  MatrixXd V_local_xy(3,2);
//  V_local_xy << V_local;
  V_local_xy << V_local.block<3, 2>(0, 0);


  Matrix3d rotationMatrix;
  Vector3d translation;

  // Compute the local frame based on A, B, C
  computeLocalFrame( V.row(E(0)),  V.row(E(1)),  V.row(E(2)), rotationMatrix, translation);
  // Print the rotation matrix and translation vector
  std::cout << "Rotation Matrix (world to local):\n" << rotationMatrix << std::endl;
  std::cout << "Translation (origin):\n" << translation.transpose() << std::endl;

  Vector3d A_local = computeLocalCoordinates(V.row(E(0)), rotationMatrix, translation);
  Vector3d B_local = computeLocalCoordinates(V.row(E(1)), rotationMatrix, translation);
  Vector3d C_local = computeLocalCoordinates(V.row(E(2)), rotationMatrix, translation);

  // Print the local coordinates
  std::cout << "Local coordinates of A: " << A_local.transpose() << std::endl;
  std::cout << "Local coordinates of B: " << B_local.transpose() << std::endl;
  std::cout << "Local coordinates of C: " << C_local.transpose() << std::endl;


  Matrix3d rotationMatrixX;
  Vector3d translationX;

  // Compute the local frame based on A, B, C
  computeLocalFrame( X.row(0),  X.row(1),  X.row(2), rotationMatrixX, translationX);
  // Print the rotation matrix and translation vector
  std::cout << "Rotation Matrix (world to local):\n" << rotationMatrixX << std::endl;
  std::cout << "Translation (origin):\n" << translationX.transpose() << std::endl;

  Vector3d A_localX = computeLocalCoordinates(X.row(0), rotationMatrixX, translationX);
  Vector3d B_localX = computeLocalCoordinates(X.row(1), rotationMatrixX, translationX);
  Vector3d C_localX = computeLocalCoordinates(X.row(2), rotationMatrixX, translationX);

  // Print the local coordinates
  std::cout << "Local coordinates of A: " << A_localX.transpose() << std::endl;
  std::cout << "Local coordinates of B: " << B_localX.transpose() << std::endl;
  std::cout << "Local coordinates of C: " << C_localX.transpose() << std::endl;

  Matrix2d T_trans;
  MatrixXd x_matrix(2,2);
  x_matrix.col(0) << B_local[0], B_local[1];
  x_matrix.col(1) << C_local[0],C_local[1];

  MatrixXd X_matrix(2,2);
  X_matrix.col(0) << B_localX[0], B_local[1];
  X_matrix.col(1) << C_localX[0],C_local[1];

  T_trans = X_matrix * x_matrix.inverse();




//  Matrix3d x_matrix;
//  x_matrix.col(0) = V.row(E(0));
//  x_matrix.col(1) = V.row(E(1));
//  x_matrix.col(2) = V.row(E(2));
//
//
//  Matrix3d X_matrix;
//  X_matrix.col(0) = X.row(0);
//  X_matrix.col(1) = X.row(1);
//  X_matrix.col(2) = X.row(2);
//
//  MatrixXd A(3, 4);
//  MatrixXd B(3, 4);
//  Matrix4d T_trans = B * (A.completeOrthogonalDecomposition().pseudoInverse());


//  Matrix3d T_trans = X_matrix * x_matrix.inverse();
//  JacobiSVD<Matrix3d> svd(T_trans, ComputeFullU | ComputeFullV);
//  Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
//
//
  MatrixXd V_local_XY(3, 2);
  V_local_XY = V_local_xy * T_trans;

  std::cout << "T_trans" << std::endl;
  std::cout << T_trans << std::endl;
  std::cout << "V_local_XY" << std::endl;
  std::cout << V_local_XY << std::endl;
  std::cout << "V_local_xy" << std::endl;
  std::cout << V_local_xy << std::endl;





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
  P.col(0) << V_local_XY.row(0)[0], V_local_XY.row(0)[1], 0;
  P.col(1) << V_local_XY.row(1)[0], V_local_XY.row(1)[1], 0;
  P.col(2) << V_local_XY.row(2)[0], V_local_XY.row(2)[1], 0;
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
