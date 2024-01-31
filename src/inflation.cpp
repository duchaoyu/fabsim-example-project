#include <fsim/ElasticMembrane.h>
#include "fsim/StVKElement.h"
#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include <iostream>

//
//using Eigen::Matrix2d;
//using Eigen::Matrix;
//using Eigen::Vector3d;
//using std::vector;
//
//double calculateEnergy(const Vector3d& x0, const Vector3d& x1, const Vector3d& x2) {
//  double youngsModulus = 10;
//  double nu = 0.3;
//  double lambda = youngsModulus * nu / (1 - std::pow(nu, 2));
//  double mu = 0.5 * youngsModulus / (1 + nu);
//  double thickness = 0.5;
//  double mass = 1;
//
//  Vector3d e1 = x0 - x2;
//  Vector3d e2 = x1 - x2;
//
//  Matrix2d _R;
//  _R.col(0) << e1.squaredNorm(), 0;
//  _R.col(1) << e2.dot(e1), e2.cross(e1).norm();
//  _R /= e1.norm();
//  _R = _R.inverse().eval();
//
//  double coeff = thickness / 2 * e1.cross(e2).norm();
//
//  Matrix<double, 3, 2> Ds;
//  Ds.col(0) = e1;
//  Ds.col(1) = e2;
//  Matrix<double, 3, 2> F = Ds * _R;
//
//  Matrix2d E = 0.5 * (F.transpose() * F - Matrix2d::Identity());
//
//  bool add_pressure = true;
//  double energy = 0.0;
//
//  if (add_pressure) {
//    Vector3d normal = e1.cross(e2);
//    vector<Vector3d> vectorList = {x0, x1, x2};
//
//    double avgNormalDisplacementArea = 0.0;
//    for (int i = 0; i < 3; ++i) {
//      Vector3d displacement = vectorList[i];
//      avgNormalDisplacementArea += 0.5 * displacement.dot(normal);
//    }
//    avgNormalDisplacementArea /= 3;
//
//    double forcePerUnitArea = 15;
//    double workDoneByPressure = coeff * (forcePerUnitArea * avgNormalDisplacementArea);
//    double elasticEnergy = coeff * (mu * (E * E).trace() + lambda / 2 * pow(E.trace(), 2));
//    double workByGravity = coeff * (9.8 * mass * (x0[2] + x1[2] + x2[2]) / 3);
////    energy = elasticEnergy + workByGravity - workDoneByPressure;
//    energy = elasticEnergy + workByGravity;
//  }
//
//  return energy;
//}
//
//
//
//Eigen::VectorXd calculategradient(const Vector3d& x0, const Vector3d& x1, const Vector3d& x2) {
//
//  double youngsModulus = 10;
//  double nu = 0.3;
//  double lambda = youngsModulus * nu / (1 - std::pow(nu, 2));
//  double mu = 0.5 * youngsModulus / (1 + nu);
//  double thickness = 0.5;
//  double mass = 1;
//
//  Vector3d e1 = x0 - x2;
//  Vector3d e2 = x1 - x2;
//
//  Matrix2d _R;
//  _R.col(0) << e1.squaredNorm(), 0;
//  _R.col(1) << e2.dot(e1), e2.cross(e1).norm();
//  _R /= e1.norm();
//  _R = _R.inverse().eval();
//
//  double coeff = thickness / 2 * e1.cross(e2).norm();
//
//  Matrix<double, 3, 2> Ds;
//  Ds.col(0) = e1;
//  Ds.col(1) = e2;
//  Matrix<double, 3, 2> F = Ds * _R;
//
//  Matrix2d E = 0.5 * (F.transpose() * F - Matrix2d::Identity());
//
//  // gradient
//  Matrix2d S = 2 * mu * E + lambda * E.trace() * Matrix2d::Identity(); // stress
//
//  Matrix<double, 3, 2> H = coeff * F * (S * _R.transpose());  // an intermediate matrix H, partial forces
//
//  Eigen::VectorXd grad(9);    // gradient for each node of the element.
//  grad.segment<3>(0) = H.col(0);
//  grad.segment<3>(3) = H.col(1);
//  grad.segment<3>(6) = -H.col(0) - H.col(1);
//
//  grad(2) += 9.8 * coeff / 3 * mass;
//  grad(5) += 9.8 * coeff / 3 * mass;
//  grad(8) += 9.8 * coeff / 3 * mass;
//
//  double forcePerUnitArea = 15;
//
//  // from matlab
//  double x1_1 = x0[0];
//  double x1_2 = x0[1];
//  double x1_3 = x0[2];
//  double x2_1 = x1[0];
//  double x2_2 = x1[1];
//  double x2_3 = x1[2];
//  double x3_1 = x2[0];
//  double x3_2 = x2[1];
//  double x3_3 = x2[2];
//
//  double t2 = x1_1 + x1_2 + x1_3;
//  double t3 = x2_1 + x2_2 + x2_3;
//  double t4 = x3_1 + x3_2 + x3_3;
//  double t5 = -x1_2;
//  double t6 = -x1_3;
//  double t7 = -x2_2;
//  double t8 = -x2_3;
//  double t9 = -x3_2;
//  double t10 = -x3_3;
//  double t11 = t5 + x1_1;
//  double t12 = t6 + x1_1;
//  double t13 = t6 + x1_2;
//  double t14 = t7 + x2_1;
//  double t15 = t8 + x2_1;
//  double t16 = t8 + x2_2;
//  double t17 = t9 + x3_1;
//  double t18 = t10 + x3_1;
//  double t19 = t10 + x3_2;
//  double t20 = (t11 * t15) / 6.0;
//  double t21 = (t13 * t14) / 6.0;
//  double t22 = (t11 * t18) / 6.0;
//  double t23 = (t13 * t17) / 6.0;
//  double t24 = (t14 * t18) / 6.0;
//  double t25 = (t16 * t17) / 6.0;
//  double t26 = -t21;
//  double t27 = -t22;
//  double t28 = -t25;
//
////  grad(0) -= ( t24 + t28 + (t4 * t16) / 6.0 - (t3 * t19) / 6.0) * coeff * forcePerUnitArea;
////  grad(1) -= (t23 + t27 - (t4 * t13) / 6.0 + (t2 * t19) / 6.0) * coeff * forcePerUnitArea;
////  grad(2) -= (t20 + t26 + (t3 * t13) / 6.0 - (t2 * t16) / 6.0) * coeff * forcePerUnitArea;
////  grad(3) -= (t24 + t28 - (t4 * t15) / 6.0 + (t3 * t18) / 6.0) * coeff * forcePerUnitArea;
////  grad(4) -= ( t23 + t27 + (t4 * t12) / 6.0 - (t2 * t18) / 6.0) * coeff * forcePerUnitArea;
////  grad(5) -= (t20 + t26 - (t3 * t12) / 6.0 + (t2 * t15) / 6.0) * coeff * forcePerUnitArea;
////  grad(6) -= (t24 + t28 + (t4 * t14) / 6.0 - (t3 * t17) / 6.0) * coeff * forcePerUnitArea;
////  grad(7) -= (t23 + t27 - (t4 * t11) / 6.0 + (t2 * t17) / 6.0) * coeff * forcePerUnitArea;
////  grad(8) -= (t20 + t26 + (t3 * t11) / 6.0 - (t2 * t14) / 6.0) * coeff * forcePerUnitArea;
//
//  return grad;
//
//}
//int main(int argc, char *argv[]) {
//  using namespace Eigen;
//
//  Vector3d x0 = {10.00000001, 0, 0};
//  Vector3d x1 = {0, 10, 0};
//  Vector3d x2 = {0, 0, 10};
//
//  double energy = calculateEnergy(x0, x1, x2);
//
//
//  Vector3d x00 = {10, 0, 0};
//  double energy00 = calculateEnergy(x00, x1, x2);
//  std::cout << energy << " energy1" << energy00 << "energy00" << energy - energy00<< "energy difference"<<std::endl;
//
//  Eigen::VectorXd grad00 = calculategradient(x00, x1, x2);
//
//  std::cout << grad00 << "grad" << std::endl;
//
//
//
////    bool add_pressure = true;
////    if (add_pressure) {
////      // Calculate area of the triangular element
////      double area = 0.5 * normal.norm();
////      normal.normalize();
////
////      // Calculate average normal displacement
////      double avgNormalDisplacement = 0.0;
////      for (int i = 0; i < 3; ++i) {
////        Vector3d displacement = vectorList[i];
////        avgNormalDisplacement += displacement.dot(normal);
////      }
////      avgNormalDisplacement /= 3;
////      Vector3d e1e2cross = e1.cross(e2);
////
////      Vector3d unit_vector = Vector3d::Zero();
////      for (int i = 0; i < 3; ++i) {
////        unit_vector(i) = 1;
////      }
////
////      Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
////      Eigen::Matrix3d Ie2;
////
////      // Manually cross each column of the identity matrix with e2
////      Ie2.col(0) = I.col(0).cross(e2);
////      Ie2.col(1) = I.col(1).cross(e2);
////      Ie2.col(2) = I.col(2).cross(e2);
////
////      Eigen::Matrix3d e1I;
////
////      // Manually cross e1 with each column of the identity matrix
////      e1I.col(0) = e1.cross(I.col(0));
////      e1I.col(1) = e1.cross(I.col(1));
////      e1I.col(2) = e1.cross(I.col(2));
////
////      Vector3d dA_dx0 = 0.5 / e1e2cross.norm() * e1e2cross.transpose() * Ie2;
////      Vector3d dA_dx1 = 0.5 / e1e2cross.norm() * e1e2cross.transpose() * e1I;
////      Vector3d dA_dx2 = 0.5 / e1e2cross.norm() * e1e2cross.transpose() * (Ie2 * (-1) - e1I);
////
////      grad.segment<3>(0) -= coeff * forcePerUnitArea * (dA_dx0 * avgNormalDisplacement + area / 3 * normal);
////      grad.segment<3>(3) -= coeff * forcePerUnitArea * (dA_dx1 * avgNormalDisplacement + area / 3 * normal);
////      grad.segment<3>(6) -= coeff * forcePerUnitArea * (dA_dx2 * avgNormalDisplacement + area / 3 * normal);
////
////    }
//
//    }



int main(int argc, char *argv[]) {
    using namespace Eigen;

    // load geometry from OFF mesh file
    fsim::Mat3<double> V;
    fsim::Mat3<int> F;
    //    fsim::readOFF("/Users/duch/Downloads/pillow.off", V, F);
    fsim::readOFF("/Users/duch/Documents/Github/fabsim-example-project/data/triangle.off", V, F);



    // parameters of the membrane model
    const double young_modulus = 10;
    const double thickness = 0.5;
    const double poisson_ratio = 0.3;
    double stretch_factor = 1.7;
    double mass = 1;

    // declare StVKMembrane object (could be replaced seamlessly with e.g. NeohookeanMembrane)
    //    fsim::StVKMembrane model(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);
    fsim::StVKMembrane model(V/3 , F, thickness, young_modulus, poisson_ratio, mass);

    Eigen::VectorXd X = Eigen::Map<Eigen::VectorXd>(V.data(), V.size());

  int i = 0;
  int j = 0;
    std::cout << model.gradient(X)(i*3+j) << " gradient 0" << std::endl;
    std::cout << model.hessian(X) << " hessian 0" << std::endl;
//    std::cout << "full gradient" << std::endl;
//  std::cout << model.gradient(X) << " end" << std::endl;

  double tol = 1e-6;
    fsim::Mat3<double>  V2 = V;
    V2(i, j) += tol;
//    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << X << " X" << std::endl;

    std::cout << V2.row(i)[j] - V.row(i)[j]  << " e" << std::endl;

    Eigen::VectorXd X2 = Eigen::Map<Eigen::VectorXd>(V2.data(), V2.size());


//    std::cout << model.energy(X2) << " energy_e" << std::endl;
    std::cout << (model.energy(X2) -  model.energy(X)) / tol << " energy_difference / step" << std::endl;
    std::cout << (model.gradient(X2)(i*3+j) - model.gradient(X)(i*3+j))/tol << "gradient_difference / step" << std::endl;
//    // declare NewtonSolver object
//    optim::NewtonSolver<double> solver;
//    std::vector<int> bdrs = {10, 11, 18, 22, 25, 27, 30, 41, 42, 53, 55, 57, 65, 66, 83, 85, 92, 93, 94, 96, 98, 100, 101, 111, 122, 123, 124, 126, 128, 130, 133, 134, 137, 138, 145, 146, 149, 150, 151, 152, 156, 160, 161, 163, 164, 166};
//
//    for (int bdr : bdrs) {
//        solver.options.fixed_dofs.push_back(bdr * 3);
//        solver.options.fixed_dofs.push_back(bdr * 3 + 1);
//        solver.options.fixed_dofs.push_back(bdr * 3 + 2);
//    }
//
//    // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
//    solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
//
//    // display the mesh
//    polyscope::registerSurfaceMesh("mesh", V, F)
//            ->setEdgeWidth(1)
//            ->setEdgeColor({0.1, 0.1, 0.1});
//    polyscope::view::upDir = polyscope::view::UpDir::ZUp;
//    polyscope::options::groundPlaneHeightFactor = 0.4;
//    polyscope::init();
//
//    polyscope::state::userCallback = [&]()
//    {
//        ImGui::PushItemWidth(100);
//        if(ImGui::InputDouble("Stretch factor", &stretch_factor, 0, 0, "%.1f"))
//            model = fsim::StVKMembrane(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);
//
//        if(ImGui::InputDouble("Mass", &mass, 0, 0, "%.1f"))
//            model.setMass(mass);
//
//        if(ImGui::Button("Solve"))
//        {
//            // Newton's method: finds a local minimum of the energy (Fval = energy value, Optimality = gradient's norm)
//            solver.solve(model, Map<VectorXd>(V.data(), V.size()));
//
//            // Display the result of the optimization
//            polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
//                    Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3));
//        }
//    };
//    polyscope::show();
////  std::cout << model::stress << std::endl;
////  std::cout << V << std::endl;
}
