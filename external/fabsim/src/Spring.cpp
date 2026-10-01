// Spring.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#include <fsim/Spring.h>
#include <Eigen/Dense>
#include <Eigen/SparseCore>


namespace fsim
{

Spring::Spring(int _i, int _j, double length, double k)
    : i(_i), j(_j), rest_length(length), stiffness(k)
{}

double Spring::energy(const Eigen::Ref<const Eigen::VectorXd> pos) const
{
  // E = 0.5 * k * (||u-v|| - L)^2
  double diff = (pos.segment<3>(3 * i) - pos.segment<3>(3 * j)).norm() - rest_length;
  return 0.5 * stiffness * diff * diff;
}

Eigen::Vector3d Spring::force(const Eigen::Ref<const Eigen::VectorXd> pos) const
{
  using namespace Eigen;
  Vector3d u = pos.segment<3>(3 * i);
  Vector3d v = pos.segment<3>(3 * j);
  double len = (u - v).norm();
  double r = rest_length / len;
  return -stiffness * (1 - r) * (u - v);
}

// gradient of energy w.r.t. all DOFs, accumulated into Y
Eigen::Vector3d Spring::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
{
  using namespace Eigen;

  Vector3d u = X.segment<3>(3 * i);
  Vector3d v = X.segment<3>(3 * j);
  double len = (u - v).norm();
  double r = rest_length / len;

  // dE/du =  k*(1-r)*(u-v)
  // dE/dv = -k*(1-r)*(u-v)
  Vector3d grad_i = stiffness * (1 - r) * (u - v);
  Y.segment<3>(3 * i) += grad_i;
  Y.segment<3>(3 * j) -= grad_i;
  return grad_i;
}

Eigen::Matrix3d Spring::hessian(const Eigen::Ref<const Eigen::VectorXd> pos) const
{
  using namespace Eigen;

  Vector3d u = pos.segment<3>(3 * i);
  Vector3d v = pos.segment<3>(3 * j);
  double len = (u - v).norm();
  double r = rest_length / len;

  // d²E/du² = k * [(1-r)*I + r/len² * (u-v)(u-v)^T]
  return stiffness * ((1 - r) * Matrix3d::Identity()
         + r / (len * len) * (u - v) * (u - v).transpose());
}

std::vector<Eigen::Triplet<double>> Spring::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  Vector3d u = X.segment<3>(3 * i);
  Vector3d v = X.segment<3>(3 * j);
  double len = (u - v).norm();
  double r = rest_length / len;

  // 3x3 diagonal block: H_ii = H_jj = k*[(1-r)*I + r/len²*(u-v)(u-v)^T]
  // Off-diagonal block:  H_ij = H_ji = -H_ii
  Matrix3d H = stiffness * ((1 - r) * Matrix3d::Identity()
               + r / (len * len) * (u - v) * (u - v).transpose());

  std::vector<Triplet<double>> trips;
  trips.reserve(36);
  for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    {
      trips.push_back({3*i+a, 3*i+b,  H(a,b)});   // ∂²E/∂uᵢ∂uⱼ
      trips.push_back({3*j+a, 3*j+b,  H(a,b)});   // ∂²E/∂vᵢ∂vⱼ
      trips.push_back({3*i+a, 3*j+b, -H(a,b)});   // ∂²E/∂uᵢ∂vⱼ
      trips.push_back({3*j+a, 3*i+b, -H(a,b)});   // ∂²E/∂vᵢ∂uⱼ
    }
  return trips;
}

} // namespace fsim
