// OrthotropicStVKElement.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#pragma once

#include "ElementBase.h"
#include "util/typedefs.h"

namespace fsim
{

class OrthotropicStVKElement : public ElementBase<3>
{
public:
  /**
   * Constructor for the OrthotropicStVKElement class
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  OrthotropicStVKElement(const Eigen::Ref<const Mat3<double>> V,
                         const Eigen::Vector3i &face,
                         double thickness,
                         double E1,
                         double E2,
                         double poisson_ratio,
                         Eigen::Vector3d face_vector);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of the triangle element for a given material model
   */


  double energy(const Eigen::Ref<const Eigen::VectorXd> X, const Eigen::Matrix3d &_C, double mass) const;

  Eigen::MatrixXd local_xy(const Eigen::Ref<const Mat3<double>> V) const;

  Eigen::MatrixXd local_XY(const Eigen::Ref<const Eigen::VectorXd> X) const;
    /**
   * @param V n by 3 list of vertex positions (each row is a vertex)
   * @param X a flat vector stacking all degrees of freedom
   * @return
    */

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of the energy (9 by 1 vector), derivatives are stacked in the order of the triangle indices
   */
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X, const Eigen::Matrix3d &_C, double mass) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian matrix of the energy w.r.t. all 9 degrees of freedom of the triangle
   */
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X, const Eigen::Matrix3d &_C, double mass) const;

  /**
   * Computes the Green strain tensor E = \frac 1 2 (F^T F - I)  where F is the deformation gradient
   * Uses Voigt's notation to express it as a vector
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return Green strain
   */
  Eigen::Vector3d strain(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * Computes the Second Piola-Kirchhoff stress tensor S = \frac{\partial f}{\partial E} where E is the Green strain
   * Uses Voigt's notation to express it as a vector
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return second Piola-Kirchhoff stress
   */
  Eigen::Vector3d stress(const Eigen::Ref<const Eigen::VectorXd> X, const Eigen::Matrix3d &_C) const;


  double coeff;
  Eigen::Matrix<double, 3, 2> _R;
  double area;
  Eigen::Matrix4d T_mul;
  Eigen::Matrix4d T;

  double _E1;              // Young's modulus in direction 1
  double _E2;              // Young's modulus in direction 2
  double _G12;             // Shear modulus
  double _poisson_ratio;   // Poisson's ratio
  double _nu21;            // Secondary Poisson's ratio
  Eigen::Matrix3d _C;      // Compliance matrix specific to this element
};

} // namespace fsim
