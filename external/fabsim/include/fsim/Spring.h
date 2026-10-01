// Spring.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>

namespace fsim
{

struct Spring
{
  int i, j;
  double rest_length;
  double stiffness;  // spring stiffness k (N/m)

  Spring(int _i, int _j, double length, double k = 1.0);

  double energy(const Eigen::Ref<const Eigen::VectorXd> pos) const;
  Eigen::Vector3d force(const Eigen::Ref<const Eigen::VectorXd> pos) const;
  Eigen::Vector3d gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const;
//  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const;
  Eigen::Matrix3d hessian(const Eigen::Ref<const Eigen::VectorXd> pos) const;
  std::vector<Eigen::Triplet<double>> hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const;

};

} // namespace fsim
