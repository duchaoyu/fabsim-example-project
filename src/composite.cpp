#include <fsim/CompositeModel.h>
#include <fsim/ElasticRod.h>
#include <fsim/Spring.h>
#include <fsim/ElasticMembrane.h>
#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>
#include <optim/NewtonSolver.h>
#include <polyscope/curve_network.h>
#include <polyscope/surface_mesh.h>
#include <algorithm>
#include <set>
#include <map>

int main(int argc, char *argv[])
{
  using namespace Eigen;

  // load geometry from OFF mesh file
  fsim::Mat3<double> V;
  fsim::Mat3<int> F;
  fsim::readOFF("../data/mesh.off", V, F);

  // parameters of the membrane model
  const double young_modulus = 10;
  const double thickness = 0.5;
  const double poisson_ratio = 0.3;
  double stretch_factor = 1.2;
  double mass = 0;

  RowVector3d barycenter = V.colwise().sum() / V.rows();
  VectorXi indices;


  // Collect the indices of the boundary vertices to connect with springs
  std::vector<std::pair<int, int>> springIndices;

  for(int k = 0; k < 4; ++k)
  {
    std::vector<int> indicesInEdge;
    for(int i = 0; i < V.rows(); ++i)
      if(fsim::point_in_segment(V.row(i), V.row(k), V.row((k + 1) % 4)))
      {
        indicesInEdge.push_back(i);
      }
    std::sort(indicesInEdge.begin(), indicesInEdge.end(), [&](int a, int b) {
        Vector3d orientation = (V.row(a) - barycenter).cross(V.row(b) - barycenter);
        return orientation(2) > 0;
    });

    for (size_t j = 0; j < indicesInEdge.size() - 1; ++j) {
      springIndices.emplace_back(indicesInEdge[j], indicesInEdge[j + 1]);
    }
  }

  for (const auto& pair : springIndices) {
    std::cout << "(" << pair.first << ", " << pair.second << ")\n";
  }




  // Initialize springs
  std::vector<fsim::Spring> springs;
  for (auto &[start, end] : springIndices) {
    double length = (V.row(start) - V.row(end)).norm();
    springs.emplace_back(start, end, length);
  }

// Initialize the membrane model
  auto membrane = fsim::StVKMembrane(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);

// Create a vector to hold all models
//  std::vector<fsim::StVKMembrane> models;
//  models.push_back(membrane);
//  models.insert(models.end(), springs.begin(), springs.end());

// Initialize the composite model with the vector of models
  fsim::CompositeModel composite(springs, 0);

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
   // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
   solver.options.fixed_dofs = {0 * 3 + 0, 0 * 3 + 1, 0 * 3 + 2, 1 * 3 + 0,
                                1 * 3 + 1, 1 * 3 + 2, 2 * 3 + 0, 2 * 3 + 1,
                                2 * 3 + 2, 3 * 3 + 0, 3 * 3 + 1, 3 * 3 + 2};
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
  solver.options.update_fct = [&](const Ref<const VectorXd> X) {
  };

  // display the curve network
  // Collect unique node indices
  std::set<int> uniqueNodeIndices;
  for (const auto& pair : springIndices) {
    uniqueNodeIndices.insert(pair.first);
    uniqueNodeIndices.insert(pair.second);
  }
  std::map<int, int> indexMapping;
  int newIndex = 0;
  for (int idx : uniqueNodeIndices) {
    indexMapping[idx] = newIndex++;
  }

  std::vector<int> nodeIndicesVec(uniqueNodeIndices.begin(), uniqueNodeIndices.end());

  fsim::Mat3<double> R(uniqueNodeIndices.size(), 3);
  fsim::Mat2<int> E(springIndices.size(), 2);

  int row = 0;
  for (int idx : nodeIndicesVec) {
    R.row(row++) = V.row(idx);
  }

  for(size_t j = 0; j < springIndices.size(); ++j) {
    int idx1 = indexMapping[springIndices[j].first];
    int idx2 = indexMapping[springIndices[j].second];
    E.row(j) << idx1, idx2;
  }

  polyscope::registerCurveNetwork("springs", R, E);

  // display the mesh
  polyscope::registerSurfaceMesh("mesh", V, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1});
  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;
  polyscope::init();

  polyscope::state::userCallback = [&]()
  {
      if(ImGui::Button("Solve"))
      {
        VectorXd var = VectorXd::Zero(V.size());
        var.head(V.size()) = Map<VectorXd>(V.data(), V.size());

        var = solver.solve(composite, var);

        polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
            Map<fsim::Mat3<double>>(var.data(), V.rows(), 3));

        // Update R with new positions
        for (int i = 0; i < R.rows(); ++i) {
          int originalIdx = nodeIndicesVec[i]; // nodeIndicesVec is a vector of uniqueNodeIndices
          R.row(i) = var.segment<3>(3 * originalIdx);
        }

        polyscope::getCurveNetwork("springs")->updateNodePositions(R);
      }
  };
  polyscope::show();
}