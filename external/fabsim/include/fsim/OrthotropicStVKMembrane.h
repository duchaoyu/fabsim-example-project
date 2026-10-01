// OrthotropicStVKMembrane.h
//
// StVK version of "Stable Orthotropic Materials" by Li and Barbič (https://doi.org/10.1109/tvcg.2015.2448105)
// Parameterizes the elasticity tensor with 2 Young's moduli and 1 Poisson's ratio
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 30/10/21

#include "ModelBase.h"
#include "OrthotropicStVKElement.h"

#include <array>

namespace fsim
{

class OrthotropicStVKMembrane : public ModelBase<OrthotropicStVKElement>
{
public:
  /**
   * constructor for OrthotropicStVKMembrane
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param thicknesses  membrane's thickness (per-triangle value)
   * @param E1  0 degree Young's modulus
   * @param E2  90 degrees Young's modulus
   * @param poisson_ratio  membrane's Poisson's ratio, nu12
   * @param mass  membrane's mass (defaults to 0 to disable gravity)
   * @param pressure pressure inside the membrane
   * @param face_vecotrs the vector in each face that represents the direction of E1 in global direction
   */
  OrthotropicStVKMembrane(const Eigen::Ref<const Mat3<double>> V,
                          const Eigen::Ref<const Mat3<int>> F,
                          double thickness,
                          double E1,
                          double E2,
                          double poisson_ratio,
                          const std::vector<Eigen::Vector3d>& face_vectors,
                          double mass = 0,
                          double pressure = 0
                              );


  OrthotropicStVKMembrane(const Eigen::Ref<const Mat3<double>> V,
                          const Eigen::Ref<const Mat3<int>> F,
                          const std::vector<double> &thicknesses,
                          double E1,
                          double E2,
                          double poisson_ratio,
                          const std::vector<Eigen::Vector3d>& face_vectors,
                          double mass = 0,
                          double pressure = 0
                             );

    /**
     * constructor for OrthotropicStVKMembrane
     * @param V  nV by 3 list of vertex positions
     * @param F  nF by 3 list of face indices
     * @param thicknesses  membrane's thickness (per-triangle value)
     * @param E1  a list of E1, 0 degree Young's modulus
     * @param E2  a list of E2, 90 degrees Young's modulus
     * @param poisson_ratio  a list of membrane's Poisson's ratio, nu12
     * @param mass  membrane's mass (defaults to 0 to disable gravity)
     * @param pressure pressure inside the membrane
     * @param face_vecotrs the vector in each face that represents the direction of E1 in global direction
     */

  OrthotropicStVKMembrane(const Eigen::Ref<const Mat3<double>> V,
                         const Eigen::Ref<const Mat3<int>> F,
                         const std::vector<double>& thicknesses,
                         const std::vector<double>& E1,
                         const std::vector<double>& E2,
                         const std::vector<double>& poisson_ratio,
                         const std::vector<Eigen::Vector3d>& face_vectors,
                         double mass = 0,
                         double pressure = 0);

  /**
   * energy function of this material model   f : \R^n -> \R
   * @param X  a flat vector stacking all degrees of freedom
   * @return  the energy of this model evaluated at X
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * gradient of the energy  \nabla f : \R^n -> \R^n
   * @param X  a flat vector stacking all degrees of freedom
   * @param Y  gradient (or sum of gradients) vector in which we will add the gradient of energy evaluated at X
   * @return Y
   */
  void gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const;
  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * hessian of the energy  \nabla^2 f : \R^n -> \R^{n \times n}
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian of the energy stored in a sparse matrix representation
   */
  Eigen::SparseMatrix<double> hessian(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * (row, column, value) triplets used to build the sparse hessian matrix
   * @param X  a flat vector stacking all degrees of freedom
   * @return  all the triplets needed to build the hessian
   */
  std::vector<Eigen::Triplet<double>> hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const;

  // number of degrees of freedom
  int nbDOFs() const { return 3 * nV; }

  // set Poisson ratio (between 0 and 0.5)
//  void setPoissonRatio(double poisson_ratio);
//  double getPoissonRatio() const { return _poisson_ratio; }

  void setPressure(double pressure);
  double getPressure() const { return _pressure; }

  // set Young's moduli (E1 and E2 respectively control the horizontal and vertical stiffness)

//  void setE1 (double E1);
//  double getE1() const { return _E1; }

//  void setE2 (double E2);
//  double getE2() const {return _E2; }

  double getNu12() const {return _nu12; }
  double getNu21() const {return _nu21; }

//  void setYoungModuli(double E1, double E2);
//  std::array<double, 2> getYoungModuli() const { return {_E1, _E2}; }

  // set thickness of the membrane (controls the amount of stretching and the total weight)
  // negative values are not allowed
  void setThickness(double t);
  double getThickness() const { return _thickness; }

  void setMass(double mass);
  double getMass() const { return _mass; }


private:
  int nV;

  double _thickness = -1;
  double _poisson_ratio;
//  double _E1;
//  double _E2;
  double _mass;
  double _nu12;
  double _nu21;
  double _G12; // shear modulus
  double _pressure;

  std::vector<double> _E1_values;        // Per-face E1 values
  std::vector<double> _E2_values;        // Per-face E2 values
  std::vector<double> _poisson_ratios;
  std::vector<Eigen::Vector3d> _face_vectors; // direciton of E1

};

} // namespace fsim
