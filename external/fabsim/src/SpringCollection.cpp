// SpringCollection.cpp
//
// Implementation of the Discrete Elastic Rods model of Bergou et al. as presented in
// "Discrete Viscous Threads" (https://doi.org/10.1145/1778765.1778853),
// see also  "A discrete, geometrically exact method for simulating nonlinear, elastic or
// non-elastic beams"  (https://hal.archives-ouvertes.fr/hal-02352879v1)
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/09/20

#include "fsim/SpringCollection.h"

#include "fsim/ElasticRod.h"

namespace fsim
{

double SpringCollection::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  double res = 0;
  for (auto s : _springs) {
    res += s.energy(X);
  }
  return res;
}

//void SpringCollection::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
//{
//  for (auto s : _springs) {
//    s.gradient(X, Y);
//  }
//}

Eigen::VectorXd SpringCollection::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

std::vector<Eigen::Triplet<double>> SpringCollection::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  std::vector<Eigen::Triplet<double>> triplets;
  for (auto model : _springs) {
      auto tripletsI = model.hessianTriplets(X);
      triplets.insert(triplets.end(), tripletsI.begin(), tripletsI.end());
  }
  return triplets;
}

} // namespace fsim
