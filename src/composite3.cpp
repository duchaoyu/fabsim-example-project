#include <fsim/CompositeModel.h>
#include <fsim/ElasticRod.h>
#include <fsim/ElasticMembrane.h>
#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>
#include <optim/NewtonSolver.h>
#include <polyscope/curve_network.h>
#include <polyscope/surface_mesh.h>
#include <algorithm>

int main(int argc, char *argv[]) {
  using namespace Eigen;

  // load geometry from OFF mesh file
  fsim::Mat3<double> V;
  fsim::Mat3<int> F;
  fsim::readOFF("/Users/duch/Downloads/butt_out.off", V, F);

  // parameters of the membrane model
  double young_modulus = 50000;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
  double thickness = 0.4;
  double thickness2 = 1.0;
  double poisson_ratio = 0.38;  // 0.38
  double stretch_factor = 1.01;
  double mass = 30; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
  double pressure = 250;

  std::vector<double> thicknesses;
  std::string input= "1.02302  1.04777  1.09892  1.05907 0.816632 0.895051 0.900279 0.974469 0.575607 0.638555  1.09821  1.02105   1.0778  1.10588 0.669602 0.655472 0.570462 0.561674  1.07181 0.974657 0.568111 0.522818 0.679791 0.982933 0.531759 0.520171 0.549284 0.568658 0.509252 0.530701 0.846616  1.10867 0.447451 0.449206 0.561093 0.483984 0.438575 0.603301 0.441142 0.437865 0.650288 0.558825 0.973767 0.861985  1.02421  1.02164 0.478363 0.391957  0.80584 0.953766 0.508043 0.404758 0.529082 0.589347 0.498216  0.46223 0.486261 0.428482  0.55517 0.542928 0.606956 0.493717 0.487533 0.461645 0.490472 0.481462 0.428294 0.453694 0.382004 0.375175 0.539285 0.459154 0.372449 0.463366 0.949594 0.943892  1.00132 0.981191   0.5661 0.484588 0.382766 0.317738 0.582981 0.303662 0.508565 0.532228 0.401825 0.404473  1.10418  1.07146  0.61001 0.514849 0.956558  1.07876 0.523468 0.462003 0.461911 0.353092 0.646981 0.318345 0.309516 0.124023 0.315679 0.350342 0.437427  0.31381 0.355343 0.313206   1.0118 0.971943 0.741267 0.377699 0.960215 0.960959 0.409146 0.459244 0.399665 0.348643  0.41503 0.490533  1.00719 0.975454 0.386293 0.369608 0.363168  0.38489 0.205275 0.712923 0.971517  1.06392 0.371197 0.365171  0.99431  1.14141 0.915116 0.927281 0.430392 0.499516 0.239297 0.301811 0.374693 0.459467 0.464505 0.398637 0.536349 0.526641 0.257956 0.407434  1.11036 0.967259 0.326278 0.320242 0.409564 0.469723 0.428103  0.54809 0.458904 0.525865  1.05687  1.06452  1.00726  1.02514 0.501496 0.477221 0.562436 0.617247 0.540053 0.540387  1.16773  1.08834  1.08129   1.0743 0.307516 0.307269 0.437101 0.219919 0.367048 0.437181 0.438626 0.561891  1.11417  1.02859 0.375978 0.389359 0.903533 0.923082 0.955874 0.978956 0.528481 0.444148  1.05883  0.98255 0.882767 0.362639 0.374795 0.384008 0.421806 0.448986 0.524588 0.321359 0.399063 0.387911 0.405102 0.475856 0.356675 0.304901 0.481631 0.519472 0.459902 0.446764 0.574927 0.577946 0.590154 0.534151 0.574459 0.595642 0.971675 0.875466 0.799316 0.804595 0.809511  1.09953 0.449433  0.46192  1.01352 0.998945 0.993558 0.987353 0.882011 0.962604 0.936607 0.846663  0.40419 0.360486 0.609334 0.442783 0.925594 0.908178 0.921698 0.927049  1.01454  1.00374  1.13567  1.09611 0.794308  1.03968 0.388759 0.430648 0.985176 0.930454 0.505983 0.526916 0.510345 0.450845 0.317993 0.419242 0.563853 0.739317 0.393637 0.306697  1.07475  1.08261  1.10147   1.2103 0.371798 0.376676 0.348156 0.410902  1.13437 0.852618 0.300119 0.338259  1.21054  1.20056 0.417809 0.451645 0.276258  0.27499 0.419633 0.401944  0.50248 0.208766 0.548096 0.507433  0.31543 0.576948 0.495005 0.356382 0.459704 0.479574 0.288822 0.872915 0.413925 0.479212 0.531823 0.566585  1.00172 0.968965  1.10938  1.08955 0.402236 0.402779 0.380065 0.463202 0.490375 0.478635 0.510601 0.416691 0.355187 0.356098 0.426383 0.437369 0.491994 0.420628  1.02224  1.05844  1.19694  1.20841 0.891027 0.956284 0.558159 0.391349 0.470355 0.473984  0.47495 0.461602 0.376134 0.387998 0.391375 0.427512  0.47189 0.465937 0.480775 0.498621  1.02737  1.04433";
  // Use a string stream to parse the input string
  std::istringstream iss(input);

  double temp;
  while (iss >> temp) {
    thicknesses.push_back(temp);
  }


  VectorXi indices(11);
  indices << 139, 64, 75, 8, 149, 38, 30, 84, 126, 57, 58;

  fsim::RodParams params = {1, 1.5, 1000 * young_modulus, 1, fsim::CrossSection::Circle, false};

  fsim::CompositeModel composite(
      fsim::StVKMembrane(V / stretch_factor, F, thicknesses, young_modulus, poisson_ratio, mass, pressure),
      fsim::ElasticRod(V, indices, Vector3d(0, 0, 1), params)
);

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  std::vector<int> bdrs = {0, 1, 3, 10, 16, 17, 18, 25, 39, 41, 52, 53, 62, 63, 65, 66, 67, 76, 85, 88, 90, 91, 94, 95, 102, 103, 104, 106, 108, 111, 113, 115, 119, 120, 122, 139, 140, 141, 144, 150, 151, 153, 154, 165, 166, 171, 173, 177, 178, 181, 189, 191};

  for (int bdr : bdrs) {
    solver.options.fixed_dofs.push_back(bdr * 3);
    solver.options.fixed_dofs.push_back(bdr * 3 + 1);
    solver.options.fixed_dofs.push_back(bdr * 3 + 2);
  }

  // // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
  // solver.options.fixed_dofs = {0 * 3 + 0, 0 * 3 + 1, 0 * 3 + 2, 1 * 3 + 0,
  //                              1 * 3 + 1, 1 * 3 + 2, 2 * 3 + 0, 2 * 3 + 1,
  //                              2 * 3 + 2, 3 * 3 + 0, 3 * 3 + 1, 3 * 3 + 2};
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
  solver.options.update_fct = [&](const Ref<const VectorXd> X) {
    // updates twist angles as per [Bergou et al. 2010] (section 6)
    composite.getModel<1>().updateProperties(X); 
  };

  // display the curve network
  fsim::Mat3<double> R(indices.size() - 1, 3);
  fsim::Mat2<int> E(indices.size() - 2, 2);
  for(int j = 0; j < indices.size() - 1; ++j)
  {
    R.row(j) = V.row(indices(j));
  }
  for(int j = 0; j < indices.size() - 2; ++j)
  {
    E.row(j) << j, j + 1;
  }
  polyscope::registerCurveNetwork("rod", R, E);
  
  // display the mesh
  polyscope::registerSurfaceMesh("mesh", V, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1});
  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;
  polyscope::init();

  polyscope::state::userCallback = [&]() 
  {
    // ImGui::PushItemWidth(100);
    // if(ImGui::InputDouble("Stretch factor", &stretch_factor, 0, 0, "%.1f"))
    //   membrane = fsim::StVKMembrane(V.leftCols(2) / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);
    
    // if(ImGui::InputDouble("Mass", &mass, 0, 0, "%.1f"))
    //   membrane.setMass(mass);

    if(ImGui::Button("Solve")) 
    {
      // prepare input variables for the Newton solver
      VectorXd var = VectorXd::Zero(V.size() + V.rows());
      var.head(V.size()) = Map<VectorXd>(V.data(), V.size());

      for(int i = 0; i < V.rows(); ++i)
      {
        // add noise in the Z direction to force the rod out of plane
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dis(-0.1, 0.1);
        var(3 * i + 2) = dis(gen);
      }

      composite.getModel<1>().updateProperties(var); // necessary to do so if it's not the first solve

      // Newton's method: finds a local minimum of the energy (Fval = energy value, Optimality = gradient's norm)
      var = solver.solve(composite, var);

      // Display the result of the optimization
      polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
          Map<fsim::Mat3<double>>(var.data(), V.rows(), 3));

      for(int j = 0; j < indices.size() - 1; ++j)
        R.row(j) = var.segment<3>(3 * indices(j));
      polyscope::getCurveNetwork("rod")->updateNodePositions(R);
    }
  };
  polyscope::show();
}
