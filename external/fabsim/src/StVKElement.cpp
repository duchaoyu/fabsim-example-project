// StVKElement.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 12/06/21

#include "fsim/StVKElement.h"
#include <iostream>

namespace fsim
{

StVKElement::StVKElement(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector3i &E, double thickness)
{
  using namespace Eigen;
  idx = E;

  Vector3d e1 = V.row(E(0)) - V.row(E(2));
  Vector3d e2 = V.row(E(1)) - V.row(E(2));

  _R.col(0) << e1.squaredNorm(), 0;
  _R.col(1) << e2.dot(e1), e2.cross(e1).norm();
  _R /= e1.norm();
  _R = _R.inverse().eval();

  coeff = thickness / 2 * e1.cross(e2).norm(); // volume
  area = 1.0 / 2 * e1.cross(e2).norm();
}

Eigen::Matrix2d StVKElement::strain(const Eigen::Ref<const Eigen::VectorXd> X) const
{ // green lagrange strain, E = 0.5 * [U**2 - I], U ** 2 = F^T * F
  using namespace Eigen;

  Matrix<double, 3, 2> Ds;
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = Ds * _R;

  return 0.5 * (F.transpose() * F - Matrix2d::Identity());
}

Eigen::Matrix2d StVKElement::stress(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu) const
{
  using namespace Eigen;
  Matrix2d E = strain(X);
  return 2 * mu * E + lambda * E.trace() * Matrix2d::Identity(); // 2nd Piola Kirchhoff stress
}

double StVKElement::energy(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double mass) const
{
  using namespace Eigen;
  Matrix2d E = strain(X);

  if (add_pressure) {
        // Calculate area of the triangular element
        Vector3d e1 = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
        Vector3d e2 = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
        Vector3d normal = e1.cross(e2);

        Vector3d totalDisplacement = X.segment<3>(3 * idx(0)) + X.segment<3>(3 * idx(1)) + X.segment<3>(3 * idx(2));

//    Vector3d totalDisplacement = X.segment<3>(0) + X.segment<3>(3) + X.segment<3>(6);
        double avgNormalDisplacementArea = totalDisplacement.dot(normal) / 18; // = signed tet volume x0.(x1 x x2)/6

        // Work done by the perpendicular force, work = F * displacement
//        double workDoneByPressure = coeff * (forcePerUnitArea * avgNormalDisplacementArea);
        double workDoneByPressure = forcePerUnitArea * avgNormalDisplacementArea;

        // free energy function for St. Venant-Kirchhoff Material
        double elasticEnergy = coeff * (mu * (E * E).trace() + lambda / 2 * pow(E.trace(), 2));
        double workByGravity = area * 9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3;

//    double workByGravity = coeff * (9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3);
    return elasticEnergy + workByGravity - workDoneByPressure;  // Additional gravitational potential energy, +2 for the z-coordinate
  }

  return coeff * (mu * (E * E).trace() + lambda / 2 * pow(E.trace(), 2) +
                  9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3);  // Additional gravitational potential energy, +2 for the z-coordinate
}

Vec<double, 9>
StVKElement::gradient(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double mass) const
{
  using namespace Eigen;
  Matrix2d S = stress(X, lambda, mu);

  Matrix<double, 3, 2> Ds;  // differences in nodal positions
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = Ds * _R;  // deformation gradient 'F', _R is the rotation matrix

  Matrix<double, 3, 2> H = coeff * F * (S * _R.transpose());  // an intermediate matrix H, partial forces

  Vec<double, 9> grad;  // gradient for each node of the element.
  grad.segment<3>(0) = H.col(0);
  grad.segment<3>(3) = H.col(1);
  grad.segment<3>(6) = -H.col(0) - H.col(1);

  grad(2) += 9.8 * area / 3 * mass;
  grad(5) += 9.8 * area / 3 * mass;
  grad(8) += 9.8 * area / 3 * mass;

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

    grad -= pressureGrad;
  }

  return grad;
}

Mat<double, 9, 9>
StVKElement::hessian(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double mass) const
{   // how internal stresses or forces in the material change with respect to changes in displacement.
    // gravity doesn't influence the hessian: gravity's magnitude and direction do not change based on the deformation of the material

  using namespace Eigen;

  Matrix2d S = stress(X, lambda, mu);
  Matrix<double, 3, 2> Ds;
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = Ds * _R; //3*2

  Matrix3d A = (lambda + 2 * mu) * F.col(0) * F.col(0).transpose() + S(0, 0) * Matrix3d::Identity() +
               mu * F.col(1) * F.col(1).transpose();
  Matrix3d B = (lambda + 2 * mu) * F.col(1) * F.col(1).transpose() + S(1, 1) * Matrix3d::Identity() +
               mu * F.col(0) * F.col(0).transpose();
  Matrix3d C = lambda * F.col(0) * F.col(1).transpose() + S(0, 1) * Matrix3d::Identity() +
               mu * F.col(1) * F.col(0).transpose();

  Matrix<double, 9, 9> hess;

  for(int i = 0; i < 2; ++i)
    for(int j = i; j < 2; ++j)
      hess.block<3, 3>(3 * i, 3 * j) =
          _R(i, 0) * _R(j, 0) * A + _R(i, 1) * _R(j, 1) * B + _R(i, 0) * _R(j, 1) * C + _R(i, 1) * _R(j, 0) * C.transpose();
//  std::cout << hess << " , hess0" << std::endl;

  hess.block<3, 3>(3, 0) = hess.block<3, 3>(0, 3).transpose();
  hess.block<6, 3>(0, 6) = -hess.block<6, 3>(0, 0) - hess.block<6, 3>(0, 3);
  hess.block<3, 6>(6, 0) = hess.block<6, 3>(0, 6).transpose();
  hess.block<3, 3>(6, 6) = -hess.block<3, 3>(0, 6) - hess.block<3, 3>(3, 6);
//  std::cout << hess << " , hess1" << std::endl;
  hess *= coeff;
//  std::cout << hess << " , hess2" << std::endl;


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
}
} // namespace fsim
