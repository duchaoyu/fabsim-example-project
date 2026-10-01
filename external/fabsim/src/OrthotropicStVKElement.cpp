// OrthotropicStVKElement.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#include "fsim/OrthotropicStVKElement.h"
#include <iostream>

using namespace Eigen;
namespace fsim
{

OrthotropicStVKElement::OrthotropicStVKElement(const Eigen::Ref<const Mat3<double>> V,
                                               const Eigen::Vector3i &E,
                                               double thickness,
                                               double E1,
                                               double E2,
                                               double poisson_ratio,
                                               const Eigen::Vector3d face_vector):
    _E1(E1), _E2(E2), _poisson_ratio(poisson_ratio)

{
  using namespace Eigen;

  _nu21 = _poisson_ratio * _E2 / _E1;
  _G12 = _E1 * _E2 / (_E1 + _E2 + 2 * _E2 * _poisson_ratio);
  _C << _E1, _poisson_ratio * sqrt(_E1 * _E2), 0,
      _poisson_ratio * sqrt(_E1 * _E2), _E2, 0,
      0, 0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);
  _C /= (1 - std::pow(_poisson_ratio, 2));


  idx = E;


  // glocal to local frame

  Vector3d xaxis;
  //  xaxis << 0.0, 1.0, 0.0;   // input from the vector field
  xaxis << face_vector;
  xaxis.normalize();

  Vector3d e1 = V.row(idx(0)) - V.row(idx(2));
  Vector3d e2 = V.row(idx(1)) - V.row(idx(2));

  // construct local frame
  // origin
  Vector3d sum = V.row(idx(0)) + V.row(idx(1)) + V.row(idx(2));
  Vector3d origin = sum / 3.0;

  Vector3d zaxis = e1.cross(e2).normalized();

  double dotProduct = xaxis.dot(zaxis);
  double tolerance = 1e-6;
  if (std::abs(dotProduct) < tolerance) {
  } else {
    Vector3d v_parallel = (xaxis.dot(zaxis)) * zaxis;
    Vector3d v_projected = xaxis - v_parallel;
    xaxis << v_projected.normalized();
  }

  Vector3d yaxis = zaxis.cross(xaxis).normalized();
//    Vector3d _zaxis = xaxis.cross(yaxis).normalized();

  Matrix3d R;
  R.col(0) << xaxis;
  R.col(1) << yaxis;
  R.col(2) << zaxis;

  T = Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = origin;
  Matrix4d T_inverse = T.inverse();
  T_mul = T_inverse * Matrix4d::Identity();


  MatrixXd V_local_xy(3,2);
  V_local_xy = local_xy(V);

  // _R is the initial length
  _R << V_local_xy(1, 1) - V_local_xy(2, 1), V_local_xy(1, 0) - V_local_xy(2, 0),
        V_local_xy(2, 1) - V_local_xy(0, 1), V_local_xy(2, 0) - V_local_xy(0, 0),
        V_local_xy(0, 1) - V_local_xy(1, 1), V_local_xy(0, 0) - V_local_xy(1, 0);
//
  double d = Vector3d(V_local_xy(0, 0), V_local_xy(1, 0), V_local_xy(2, 0)).dot(_R.col(0)); // area of the triangle

  _R /= d;
  coeff = thickness / 2 * std::abs(d);
  area = 1.0 / 2 * std::abs(d);



}

  Eigen::MatrixXd OrthotropicStVKElement::local_xy(const Eigen::Ref<const Mat3<double>> V) const {
    using namespace Eigen;
    MatrixXd V_local_xy(3, 3);

    for (int i = 0; i < 3; ++i) {
      // Convert each 3D point to homogeneous coordinates (Vector4d)
      Matrix<double, 1, 4> V_homogeneous_xy;
      V_homogeneous_xy << V.row(idx(i))[0], V.row(idx(i))[1], V.row(idx(i))[2], 1.0;
      Matrix<double, 1, 4>  V_transformed_xy = V_homogeneous_xy * T_mul.transpose();
      V_local_xy.row(i) << V_transformed_xy.head<3>();  // Store in the output matrix
    }

    // remove the zero z axis
    MatrixXd V_local_xy32(3,2);
    V_local_xy32 << V_local_xy.block<3, 2>(0, 0);

    // deformed geometry
  return V_local_xy32;

  }


Eigen::MatrixXd OrthotropicStVKElement::local_XY(const Eigen::Ref<const Eigen::VectorXd> X) const {
  using namespace Eigen;

  MatrixXd V_local_XY(3, 3);

  for (int i = 0; i < 3; ++i) {
    // Convert each 3D point to homogeneous coordinates (Vector4d)
    Matrix<double, 1, 4> V_homogeneous_XY;
    V_homogeneous_XY << X.segment<1>(3 * idx(i) + 0), X.segment<1>(3 * idx(i) + 1), X.segment<1>(3 * idx(i) + 2), 1.0;
    Matrix<double, 1, 4> V_transformed_XY = V_homogeneous_XY * T_mul.transpose();
    V_local_XY.row(i) << V_transformed_XY.head<3>();
  }
  return V_local_XY;
}

  Eigen::Vector3d OrthotropicStVKElement::strain(const Eigen::Ref<const Eigen::VectorXd> X) const {
    using namespace Eigen;
    Matrix3d P;
    Matrix3d V_local_XY = local_XY(X);
    P.col(0) << V_local_XY.row(0)[0], V_local_XY.row(0)[1], V_local_XY.row(0)[2];
    P.col(1) << V_local_XY.row(1)[0], V_local_XY.row(1)[1], V_local_XY.row(1)[2];
    P.col(2) << V_local_XY.row(2)[0], V_local_XY.row(2)[1], V_local_XY.row(2)[2];

    Matrix<double, 3, 2> F = P * _R;

    Vector3d res;
    res(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
    res(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
    res(2) = F.col(1).dot(F.col(0));
//    std::cout << res << " strain"<< std::endl;
    return res;
  }

  Eigen::Vector3d OrthotropicStVKElement::stress(const Eigen::Ref<const Eigen::VectorXd> X, const Eigen::Matrix3d &_C) const {
    using namespace Eigen;
    Vector3d S_E = strain(X);
    return _C * S_E;
  }

  double
  OrthotropicStVKElement::energy(const Eigen::Ref<const Eigen::VectorXd> X, const Eigen::Matrix3d &_C, double mass) const {
    using namespace Eigen;
    Vector3d S_E = strain(X); // elastic potential energy
    if (add_pressure) {
      Vector3d e1 = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
      Vector3d e2 = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
      Vector3d normal = e1.cross(e2);

      Vector3d totalDisplacement = X.segment<3>(3 * idx(0)) + X.segment<3>(3 * idx(1)) + X.segment<3>(3 * idx(2));

//    Vector3d totalDisplacement = X.segment<3>(0) + X.segment<3>(3) + X.segment<3>(6);
      double avgNormalDisplacementArea = totalDisplacement.dot(normal) / 18; // = signed tet volume x0.(x1 x x2)/6

      // Work done by the perpendicular force, work = F * displacement
//        double workDoneByPressure = coeff * (forcePerUnitArea * avgNormalDisplacementArea);
      double workDoneByPressure = forcePerUnitArea * avgNormalDisplacementArea;


//      Matrix3d V_local_XY = local_XY(X);
//      // Calculate area of the triangular element
//      Vector3d e1 = V_local_XY.row(0) - V_local_XY.row(2);
//      Vector3d e2 = V_local_XY.row(1) - V_local_XY.row(2);
//      Vector3d normal = e1.cross(e2);
//
//      Vector3d totalDisplacement = V_local_XY.row(0) + V_local_XY.row(1) + V_local_XY.row(2);
//      double avgNormalDisplacementArea = totalDisplacement.dot(normal) / 6;
//
//      // Work done by the perpendicular force, work = F * displacement
////        double workDoneByPressure = coeff * (forcePerUnitArea * avgNormalDisplacementArea);
//      double workDoneByPressure = forcePerUnitArea * avgNormalDisplacementArea;
      // free energy function for St. Venant-Kirchhoff Material
      double elasticEnergy = coeff * (0.5 * S_E.dot(_C * S_E));
      double workByGravity = area * 9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3;
//    double workByGravity = coeff * (9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3);
      return elasticEnergy + workByGravity - workDoneByPressure;  // Additional gravitational potential energy, +2 for the z-coordinate
    }
//    std::cout << coeff * (0.5 * S_E.dot(_C * S_E)) << " elastic energy" << std::endl;
    return coeff *
           (0.5 * S_E.dot(_C * S_E)) + area * 9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3;
  }

  Vec<double, 9>
  OrthotropicStVKElement::gradient(const Eigen::Ref<const Eigen::VectorXd> X, const Eigen::Matrix3d &_C, double mass) const {
    using namespace Eigen;
    Vector3d S = stress(X, _C);
    Matrix2d SMat = (Matrix2d(2, 2) << S(0), S(2), S(2), S(1)).finished();

    Matrix3d P;
    Matrix3d V_local_XY = local_XY(X);
    P.col(0) << V_local_XY.row(0)[0], V_local_XY.row(0)[1], V_local_XY.row(0)[2];
    P.col(1) << V_local_XY.row(1)[0], V_local_XY.row(1)[1], V_local_XY.row(1)[2];
    P.col(2) << V_local_XY.row(2)[0], V_local_XY.row(2)[1], V_local_XY.row(2)[2];

    Matrix<double, 3, 2> F = P * _R; // deformation gradient

    Matrix3d grad = coeff * F * (SMat * _R.transpose()); // elastic energy gradient in the local frame

    Matrix4d T_mul_local_global = Matrix4d::Identity().inverse() * T;
    Matrix3d grad_33_world;
//    grad_33_world << grad;

    Matrix3d T_mul_local_global_R = T_mul_local_global.block<3, 3>(0, 0);
    grad_33_world = T_mul_local_global_R * grad;

//    for (int i = 0; i < 3; ++i) {
//      // Convert each 3D point to homogeneous coordinates (Vector4d)
//      Matrix<double, 1, 4> grad14;
//      grad14 << grad.col(i)[0], grad.col(i)[1], grad.col(i)[2], 1.0;
//      Matrix<double, 1, 4>  grad14_xy = grad14 * T_mul_local_global.transpose();
//      grad_33_world.col(i) = grad14_xy.head<3>();
//    }
//    std::cout << T << " Transformation matrix" << std::endl;


    grad_33_world.row(2) += Vector3d::Constant(9.8 * area / 3 * mass); // gravity gradient
//    std::cout << grad << " total gradient in local frame" << std::endl;

    Vec<double, 9> grad_flat = Map<Vec<double, 9>>(grad_33_world.data(), 9);
//    std::cout << grad_flat << " flat total gradient in local frame" << std::endl;

    if (add_pressure) {
      Eigen::Vector<double, 9> pressureGrad;

      //matlab
      double x0_x = X(3 * idx(0) + 0);
      double x0_y = X(3 * idx(0) + 1);
      double x0_z = X(3 * idx(0) + 2);
      double x1_x = X(3 * idx(1) + 0);
      double x1_y = X(3 * idx(1) + 1);
      double x1_z = X(3 * idx(1) + 2);
      double x2_x = X(3 * idx(2) + 0);
      double x2_y = X(3 * idx(2) + 1);
      double x2_z = X(3 * idx(2) + 2);

      double t2 = x0_x + x1_x + x2_x;
      double t3 = x0_y + x1_y + x2_y;
      double t4 = x0_z + x1_z + x2_z;
      double t5 = -x1_x;
      double t6 = -x1_y;
      double t7 = -x2_x;
      double t8 = -x1_z;
      double t9 = -x2_y;
      double t10 = -x2_z;
      double t11 = t5 + x0_x;
      double t12 = t7 + x0_x;
      double t13 = t6 + x0_y;
      double t14 = t7 + x1_x;
      double t15 = t9 + x0_y;
      double t16 = t8 + x0_z;
      double t17 = t9 + x1_y;
      double t18 = t10 + x0_z;
      double t19 = t10 + x1_z;
      double t20 = (t12 * t17) / 18.0;
      double t21 = (t14 * t15) / 18.0;
      double t22 = (t12 * t19) / 18.0;
      double t23 = (t14 * t18) / 18.0;
      double t24 = (t15 * t19) / 18.0;
      double t25 = (t17 * t18) / 18.0;
      double t26 = -t21;
      double t27 = -t23;
      double t28 = -t25;

      pressureGrad(0) = forcePerUnitArea * (t24 + t28 + (t4 * t17) / 18.0 - (t3 * t19) / 18.0);
      pressureGrad(1) = -1 * forcePerUnitArea * (t22 + t27 + (t4 * t14) / 18.0 - (t2 * t19) / 18.0);
      pressureGrad(2) = forcePerUnitArea * (t20 + t26 + (t3 * t14) / 18.0 - (t2 * t17) / 18.0);
      pressureGrad(3) = forcePerUnitArea * (t24 + t28 - (t4 * t15) / 18.0 + (t3 * t18) / 18.0);
      pressureGrad(4) = -1 * forcePerUnitArea * (t22 + t27 - (t4 * t12) / 18.0 + (t2 * t18) / 18.0);
      pressureGrad(5) = forcePerUnitArea * (t20 + t26 - (t3 * t12) / 18.0 + (t2 * t15) / 18.0);
      pressureGrad(6) = forcePerUnitArea * (t24 + t28 + (t4 * t13) / 18.0 - (t3 * t16) / 18.0);
      pressureGrad(7) = -1 * forcePerUnitArea * (t22 + t27 + (t4 * t11) / 18.0 - (t2 * t16) / 18.0);
      pressureGrad(8) = forcePerUnitArea * (t20 + t26 + (t3 * t11) / 18.0 - (t2 * t13) / 18.0);

      grad_flat -= pressureGrad;
    }

    return grad_flat;
  }

  Mat<double, 9, 9>
  OrthotropicStVKElement::hessian(const Eigen::Ref<const Eigen::VectorXd> X, const Eigen::Matrix3d &_C, double mass) const {
    using namespace Eigen;
    Vector3d S = stress(X, _C);

    Matrix3d P;
    Matrix3d V_local_XY = local_XY(X);
    P.col(0) << V_local_XY.row(0)[0], V_local_XY.row(0)[1], V_local_XY.row(0)[2];
    P.col(1) << V_local_XY.row(1)[0], V_local_XY.row(1)[1], V_local_XY.row(1)[2];
    P.col(2) << V_local_XY.row(2)[0], V_local_XY.row(2)[1], V_local_XY.row(2)[2];
    Matrix<double, 3, 2> F = P * _R;

    Matrix3d A = _C(0, 0) * F.col(0) * F.col(0).transpose() + S(0) * Matrix3d::Identity() +
                 _C(2, 2) * F.col(1) * F.col(1).transpose();
    Matrix3d B = _C(1, 1) * F.col(1) * F.col(1).transpose() + S(1) * Matrix3d::Identity() +
                 _C(2, 2) * F.col(0) * F.col(0).transpose();
    Matrix3d C = _C(0, 1) * F.col(0) * F.col(1).transpose() + S(2) * Matrix3d::Identity() +
                 _C(2, 2) * F.col(1) * F.col(0).transpose();

    Matrix<double, 9, 9> hess;


    Matrix4d T_mul_local_global = Matrix4d::Identity().inverse() * T;
    Matrix3d T_mul_local_global_R = T_mul_local_global.block<3, 3>(0, 0);

    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j){
        Matrix3d hess_block =
            _R(i, 0) * _R(j, 0) * A + _R(i, 1) * _R(j, 1) * B + _R(i, 0) * _R(j, 1) * C +
            _R(i, 1) * _R(j, 0) * C.transpose();

        Matrix3d hess_block_world;
        hess_block_world = T_mul_local_global_R * hess_block * T_mul_local_global_R.transpose();
//        hess_block_world << hess_block;
//
//        for (int i = 0; i < 3; ++i) {
//          // Convert each 3D point to homogeneous coordinates (Vector4d)
//          Matrix<double, 1, 4> hess14;
//          hess14 << hess_block.col(i)[0], hess_block.col(i)[1], hess_block.col(i)[2], 1.0;
//          Matrix<double, 1, 4>  hess14_xy = hess14 * T_mul_local_global.transpose();
//          hess_block_world.col(i) = hess14_xy.head<3>();
//        }

        hess.block<3, 3>(3 * i, 3 * j) = hess_block_world;
      }

    hess.block<3, 3>(3, 0) = hess.block<3, 3>(0, 3).transpose();
    hess.block<3, 6>(6, 0) = hess.block<6, 3>(0, 6).transpose();
    hess *= coeff;

    if (add_pressure) {
      Matrix<double, 9, 9> pressureHess;
      pressureHess.setZero();

      //matlab
      double x0_x = X(3 * idx(0) + 0);
      double x0_y = X(3 * idx(0) + 1);
      double x0_z = X(3 * idx(0) + 2);
      double x1_x = X(3 * idx(1) + 0);
      double x1_y = X(3 * idx(1) + 1);
      double x1_z = X(3 * idx(1) + 2);
      double x2_x = X(3 * idx(2) + 0);
      double x2_y = X(3 * idx(2) + 1);
      double x2_z = X(3 * idx(2) + 2);

      double t2 = (forcePerUnitArea * x0_x) / 6.0;
      double t3 = (forcePerUnitArea * x0_y) / 6.0;
      double t4 = (forcePerUnitArea * x1_x) / 6.0;
      double t5 = (forcePerUnitArea * x0_z) / 6.0;
      double t6 = (forcePerUnitArea * x1_y) / 6.0;
      double t7 = (forcePerUnitArea * x2_x) / 6.0;
      double t8 = (forcePerUnitArea * x1_z) / 6.0;
      double t9 = (forcePerUnitArea * x2_y) / 6.0;
      double t10 = (forcePerUnitArea * x2_z) / 6.0;
      double t11 = -t2;
      double t12 = -t3;
      double t13 = -t4;
      double t14 = -t5;
      double t15 = -t6;
      double t16 = -t7;
      double t17 = -t8;
      double t18 = -t9;
      double t19 = -t10;

      pressureHess.row(0)[4] = t10;
      pressureHess.row(0)[5] = t18;
      pressureHess.row(0)[7] = t17;
      pressureHess.row(0)[8] = t6;
      pressureHess.row(1)[3] = t19;
      pressureHess.row(1)[5] = t7;
      pressureHess.row(1)[6] = t8;
      pressureHess.row(1)[8] = t13;
      pressureHess.row(2)[3] = t9;
      pressureHess.row(2)[4] = t16;
      pressureHess.row(2)[6] = t15;
      pressureHess.row(2)[7] = t4;
      pressureHess.row(3)[1] = t19;
      pressureHess.row(3)[2] = t9;
      pressureHess.row(3)[7] = t5;
      pressureHess.row(3)[8] = t12;
      pressureHess.row(4)[0] = t10;
      pressureHess.row(4)[2] = t16;
      pressureHess.row(4)[6] = t14;
      pressureHess.row(4)[8] = t2;
      pressureHess.row(5)[0] = t18;
      pressureHess.row(5)[1] = t7;
      pressureHess.row(5)[6] = t3;
      pressureHess.row(5)[7] = t11;
      pressureHess.row(6)[1] = t8;
      pressureHess.row(6)[2] = t15;
      pressureHess.row(6)[4] = t14;
      pressureHess.row(6)[5] = t3;
      pressureHess.row(7)[0] = t17;
      pressureHess.row(7)[2] = t4;
      pressureHess.row(7)[3] = t5;
      pressureHess.row(7)[5] = t11;
      pressureHess.row(8)[0] = t6;
      pressureHess.row(8)[1] = t13;
      pressureHess.row(8)[3] = t12;
      pressureHess.row(8)[4] = t2;

      hess -= pressureHess;
//      std::cout << pressureHess << " pressure Hessian" << std::endl;
    }


    return hess;

  } // namespace fsim
}