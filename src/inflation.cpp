#include <fsim/ElasticMembrane.h>
#include "fsim/StVKElement.h"
#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include <iostream>





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
    fsim::StVKMembrane model(V , F, thickness, young_modulus, poisson_ratio, mass);

    Eigen::VectorXd X = Eigen::Map<Eigen::VectorXd>(V.data(), V.size());
    std::cout << model.energy(X) << "," << model.gradient(X) << std::endl;


    Eigen::MatrixXd V2 = V;
    V2.row(0)[0] += 0.0000001;
    std::cout << V2.row(0)[0] - V.row(0)[0]  << std::endl;
    fsim::StVKMembrane model2(V2 , F, thickness, young_modulus, poisson_ratio, mass);

    Eigen::VectorXd X2 = Eigen::Map<Eigen::VectorXd>(V2.data(), V2.size());
    std::cout << model2.energy(X2) << "," << model.energy(X) -  model2.energy(X2) << std::endl;

}
