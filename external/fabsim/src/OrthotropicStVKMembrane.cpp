// OrthotropicStVKMembrane.cpp
//
// StVK version of "Stable Orthotropic Materials" by Li and Barbič (https://doi.org/10.1109/tvcg.2015.2448105)
// Parameterizes the elasticity tensor with 2 Young's moduli and 1 Poisson's ratio
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#include "fsim/OrthotropicStVKMembrane.h"

#include <array>
#include <iostream>

namespace fsim
{

OrthotropicStVKMembrane::OrthotropicStVKMembrane(const Eigen::Ref<const Mat3<double>> V,
                                                     const Eigen::Ref<const Mat3<int>> F,
                                                     double thickness,
                                                     double E1,
                                                     double E2,
                                                     double poisson_ratio,
                                                     const std::vector<Eigen::Vector3d>& face_vectors,
                                                     double mass,
                                                     double pressure
                                                     )
    : OrthotropicStVKMembrane(V, F, std::vector<double>(F.rows(), thickness), E1, E2, poisson_ratio, face_vectors, mass, pressure)
{
  _thickness = thickness;
  // TODO: local frame, e1, e2
}

std::vector<double> nCopies(int n, double x) {
  std::vector<double> res(n, x);
  return res;
}

OrthotropicStVKMembrane::OrthotropicStVKMembrane(const Eigen::Ref<const Mat3<double>> V,
                                                     const Eigen::Ref<const Mat3<int>> F,
                                                     const std::vector<double> &thicknesses,
                                                     double E1,
                                                     double E2,
                                                     double poisson_ratio,
                                                     const std::vector<Eigen::Vector3d>& face_vectors,
                                                     double mass,
                                                     double pressure
                                                     )
    :
//    _poisson_ratio{poisson_ratio},
//    _E1{E1},
//    _E2{E2},
//    _mass{mass},
//    _nu12{poisson_ratio},
//    _nu21{poisson_ratio*E2/E1},
//    _pressure{pressure},
//    _face_vectors(face_vectors)
    OrthotropicStVKMembrane(V, F, thicknesses, nCopies(F.rows(), E1), nCopies(F.rows(), E2), nCopies(F.rows(), poisson_ratio), face_vectors, mass, pressure)
{
  using namespace Eigen;
  //  nV = V.rows();
//
//  _nu21 = _nu12 * _E2 / _E1;
//  _G12 = _E1 * _E2 / (_E1 + _E2 + 2 * _E2 * _nu12); // https://help.solidworks.com/2011/english/solidworks/cworks/legacyhelp/simulation/materials/material_models/linear_elastic_orthotropic_model.html
//
//
//  int nF = F.rows();
//
//  // Ensure that the face_vectors size matches the number of faces
//  if (!_face_vectors.empty() && _face_vectors.size() != static_cast<size_t>(nF))
//  {
//    throw std::invalid_argument("Size of face_vectors must match the number of faces in F.");
//  }
//
//
//  this->_elements.reserve(nF);
//  for(int i = 0; i < nF; ++i) {
//    this->_elements.emplace_back(V, F.row(i), thicknesses[i], _E1, _E2, _nu12, face_vectors[i]);
//
//    if(pressure > 0.1){this->_elements[i].addPressure(pressure);}
//  }
}

    OrthotropicStVKMembrane::OrthotropicStVKMembrane(const Eigen::Ref<const Mat3<double>> V,
                                            const Eigen::Ref<const Mat3<int>> F,
                                            const std::vector<double>& thicknesses,
                                            const std::vector<double>& E1_values,
                                            const std::vector<double>& E2_values,
                                            const std::vector<double>& poisson_ratios,
                                            const std::vector<Eigen::Vector3d>& face_vectors,
                                            double mass,
                                            double pressure)
        : _mass{mass}, _pressure{pressure}, _face_vectors(face_vectors), _E1_values(E1_values), _E2_values(E2_values),
        _poisson_ratios(poisson_ratios)
      {
      using namespace Eigen;

      nV = V.rows();
      int nF = F.rows();

      // Ensure input vectors have the correct size
      if (E1_values.size() != static_cast<size_t>(nF) ||
          E2_values.size() != static_cast<size_t>(nF) ||
          poisson_ratios.size() != static_cast<size_t>(nF) ||
          thicknesses.size() != static_cast<size_t>(nF)) {
        throw std::invalid_argument("Sizes of E1, E2, poisson_ratio, and thicknesses must match the number of faces in F.");
      }

      this->_elements.reserve(nF);

      for (int i = 0; i < nF; ++i) {

        // Initialize the element
        this->_elements.emplace_back(V, F.row(i), thicknesses[i], E1_values[i], E2_values[i], poisson_ratios[i], face_vectors[i]);

        // Add pressure if specified
        if (pressure > 0.1) {
          this->_elements[i].addPressure(pressure);
        }
      }
    }

//double OrthotropicStVKMembrane::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
//{
//  Eigen::Matrix3d C;  // compliance matric / elastic material matrix
//
//  C << _E1, _poisson_ratio * sqrt(_E1 * _E2), 0,
//       _poisson_ratio * sqrt(_E1 * _E2), _E2, 0,
//       0, 0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);
//  C /= (1 - std::pow(_poisson_ratio, 2));
//  return ModelBase<OrthotropicStVKElement>::energy(X, C, _mass);
//}

double OrthotropicStVKMembrane::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
    {
      double total_energy = 0.0;
      for (size_t i = 0; i < _elements.size(); ++i) {
        Eigen::Matrix3d C;
        double E1 = _E1_values[i];
        double E2 = _E2_values[i];
        double nu = _poisson_ratios[i];
        C << E1, nu * sqrt(E1 * E2), 0,
            nu * sqrt(E1 * E2), E2, 0,
            0, 0, 0.5 * sqrt(E1 * E2) * (1 - nu);
        C /= (1 - std::pow(nu, 2));
        total_energy += _elements[i].energy(X, C, _mass);
      }
      return total_energy;
    }

//void OrthotropicStVKMembrane::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
//{
//  Eigen::Matrix3d C;
//  C << _E1, _poisson_ratio * sqrt(_E1 * _E2), 0,
//       _poisson_ratio * sqrt(_E1 * _E2), _E2, 0,
//       0, 0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);
//  C /= (1 - std::pow(_poisson_ratio, 2));
//  ModelBase<OrthotropicStVKElement>::gradient(X, Y, C, _mass);
//}


void OrthotropicStVKMembrane::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
    {
      for (size_t i = 0; i < _elements.size(); ++i) {
        Eigen::Matrix3d C;
        double E1 = _E1_values[i];
        double E2 = _E2_values[i];
        double nu = _poisson_ratios[i];
        C << E1, nu * sqrt(E1 * E2), 0,
            nu * sqrt(E1 * E2), E2, 0,
            0, 0, 0.5 * sqrt(E1 * E2) * (1 - nu);
        C /= (1 - std::pow(nu, 2));

        auto grad = _elements[i].gradient(X, C, _mass);
        int nV = _elements[i].nbVertices();
        for(int j = 0; j < nV; ++j) {
            Y.segment<3>(3 * _elements[i].idx(j)) += grad.template segment<3>(3 * j);
          }
          for(int j = 0; j < _elements[i].idx.size() - nV; ++j)
            Y(_elements[i].idx(nV + j)) += grad(3 * nV + j);
      }
    }

Eigen::VectorXd OrthotropicStVKMembrane::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

//Eigen::SparseMatrix<double> OrthotropicStVKMembrane::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
//{
//  Eigen::Matrix3d C;
//  C << _E1, _poisson_ratio * sqrt(_E1 * _E2), 0,
//       _poisson_ratio * sqrt(_E1 * _E2), _E2, 0,
//       0, 0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);
//  C /= (1 - std::pow(_poisson_ratio, 2));
//  return ModelBase<OrthotropicStVKElement>::hessian(X, C, _mass);
//}

Eigen::SparseMatrix<double> OrthotropicStVKMembrane::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
    {
      using namespace Eigen;
      std::vector<Triplet<double>> triplets = hessianTriplets(X);
      SparseMatrix<double> hess(X.size(), X.size());
      hess.setFromTriplets(triplets.begin(), triplets.end());
      return hess;
    }

//std::vector<Eigen::Triplet<double>>
//OrthotropicStVKMembrane::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
//{
//  Eigen::Matrix3d C;
//  C << _E1, _poisson_ratio * sqrt(_E1 * _E2), 0,
//       _poisson_ratio * sqrt(_E1 * _E2), _E2, 0,
//       0, 0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);
//  C /= (1 - std::pow(_poisson_ratio, 2));
//  return ModelBase<OrthotropicStVKElement>::hessianTriplets(X, C, _mass);
//}
//template <class Element>
std::vector<Eigen::Triplet<double>>
OrthotropicStVKMembrane::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
    {
      int nbDOFs = 3 * 3;
      int n = nbDOFs * (nbDOFs + 3) / 2;
      std::vector<Eigen::Triplet<double>> triplets(n * _elements.size());

      #pragma omp parallel for if(_elements.size() > 1000)

      for (size_t i = 0; i < _elements.size(); ++i) {
        Eigen::Matrix3d C;
        double E1 = _E1_values[i];
        double E2 = _E2_values[i];
        double nu = _poisson_ratios[i];
        C << E1, nu * sqrt(E1 * E2), 0,
            nu * sqrt(E1 * E2), E2, 0,
            0, 0, 0.5 * sqrt(E1 * E2) * (1 - nu);
        C /= (1 - std::pow(nu, 2));

        auto &e = _elements[i];

        auto hess = e.hessian(X, C, _mass);
        int id = 0;
        int nV = 3; // always triangle face
        for(int j = 0; j < nV; ++j)
          for(int k = 0; k < nV; ++k)
            if(e.idx(j) <= e.idx(k))
              for(int l = 0; l < 3; ++l)
                for(int m = 0; m < 3; ++m)
                  triplets[n * i + id++] = Eigen::Triplet<double>(3 * e.idx(j) + l, 3 * e.idx(k) + m, hess(3 * j + l, 3 * k + m));

        for(int j = 0; j < e.idx.size() - nV; ++j)
          for(int k = 0; k < nV; ++k)
            if(3 * e.idx(k) < e.idx(3 + j))
              for(int l = 0; l < 3; ++l)
              {
                triplets[n * i + id++] = Eigen::Triplet<double>(e.idx(nV + j), 3 * e.idx(k) + l, hess(3 * nV + j, 3 * k + l));
                triplets[n * i + id++] = Eigen::Triplet<double>(3 * e.idx(k) + l, e.idx(nV + j), hess(3 * k + l, 3 * nV + j));
              }

        for(int j = 0; j < e.idx.size() - nV; ++j)
          for(int k = 0; k < e.idx.size() - nV; ++k)
            if(e.idx(3 + j) <= e.idx(3 + k))
              triplets[n * i + id++] = Eigen::Triplet<double>(e.idx(nV + j), e.idx(nV + k), hess(3 * nV + j, 3 * nV + k));
      }
      return triplets;
    }

//void OrthotropicStVKMembrane::setPoissonRatio(double poisson_ratio)
//{
//  _poisson_ratio = poisson_ratio;
//  _nu12 = poisson_ratio;
//  _nu21 = poisson_ratio *_E2/_E1;
//}

void OrthotropicStVKMembrane::setPressure(double pressure)
    {
      _pressure = pressure;
      for(auto &elem: this->_elements){
        elem.addPressure(pressure);
      }
    }

//void OrthotropicStVKMembrane:: setE1(double E1){
//  _E1 = E1;
//  _nu21 = _nu12 * _E2 / _E1;
//}
//
//void OrthotropicStVKMembrane:: setE2(double E2){
//  _E2 = E2;
//  _nu21 = _nu12 * _E2 / _E1;
//}

void OrthotropicStVKMembrane::setThickness(double t)
{
  if(_thickness <= 0)
    throw std::runtime_error(
        "Warning: membrane may have a locally varying thickness\nCan't set it to a constant value\n");
  for(auto &elem: this->_elements)
  {
    elem.coeff *= t / _thickness;
  }
  _thickness = t;
}

void OrthotropicStVKMembrane::setMass(double mass)
{
  _mass = mass;
}

} // namespace fsim
