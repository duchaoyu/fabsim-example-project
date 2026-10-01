// ElementBase.h
//
// This class describes the basic structure of a stencil element, all stencil classes in fabsim
// derive from it, as ModelBase expect its elements to follow this structure
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 08/27/19

#pragma once

#include "fsim/util/typedefs.h"

#include <Eigen/Dense>
#include <iostream>

namespace fsim
{

/**
 * Base class for all stencil elements
 * @tparam nb_vtx  number of vertices of the stencil
 * @tparam additional_dofs  number of additional degrees of freedom (if any)
 */
template <int nb_vtx, int additional_dofs = 0>
class ElementBase
{
public:
  static const int NB_DOFS = 3 * nb_vtx + additional_dofs;
  static const int NB_VERTICES = nb_vtx;

  typedef Vec<double, NB_DOFS> LocalVector;
  typedef Mat<double, NB_DOFS, NB_DOFS> LocalMatrix;

  Vec<int, nb_vtx + additional_dofs> idx;

  constexpr int nbDOFs() const { return NB_DOFS; }
  constexpr int nbVertices() const { return NB_VERTICES; }

  void addPressure(double forcePerUnitArea_) {
    add_pressure = true;
    forcePerUnitArea = forcePerUnitArea_;
  }

//  void addFaceVector(const Eigen::Vector3d& face_vector_){
//    face_vector = face_vector_.normalized();
//    std::cout << "face_vector model base" << face_vector << " ," << face_vector_ << std::endl;
//  }

protected:
    bool add_pressure = false;
    double forcePerUnitArea = 0;
//    Eigen::Vector3d face_vector;

};

} // namespace fsim
