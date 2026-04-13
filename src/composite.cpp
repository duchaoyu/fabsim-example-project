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

// ── SpringCollection ──────────────────────────────────────────────────────────
// Wraps a vector<Spring> and provides the energy/gradient/hessian interface
// expected by CompositeModel.
struct SpringCollection
{
  std::vector<fsim::Spring> springs;

  explicit SpringCollection(std::vector<fsim::Spring> s) : springs(std::move(s)) {}

  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    double e = 0;
    for(auto& s : springs) e += s.energy(X);
    return e;
  }

  void gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
  {
    for(auto& s : springs) s.gradient(X, Y);
  }

  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    Eigen::VectorXd Y = Eigen::VectorXd::Zero(X.size());
    gradient(X, Y);
    return Y;
  }

  std::vector<Eigen::Triplet<double>> hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    std::vector<Eigen::Triplet<double>> trips;
    for(auto& s : springs)
    {
      auto t = s.hessianTriplets(X);
      trips.insert(trips.end(), t.begin(), t.end());
    }
    return trips;
  }

  Eigen::SparseMatrix<double> hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    auto trips = hessianTriplets(X);
    Eigen::SparseMatrix<double> H(X.size(), X.size());
    H.setFromTriplets(trips.begin(), trips.end());
    return H;
  }
};

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

  // Steel cable stiffness: k = E*A/L; for a thin steel cable this is typically
  // much stiffer than the membrane. Tune as needed.
  double spring_stiffness = 1000.0;

  RowVector3d barycenter = V.colwise().sum() / V.rows();

  // Collect the indices of the boundary vertices to connect with springs
  std::vector<std::pair<int, int>> springIndices;

  for(int k = 0; k < 4; ++k)
  {
    std::vector<int> indicesInEdge;
    for(int i = 0; i < V.rows(); ++i)
      if(fsim::point_in_segment(V.row(i), V.row(k), V.row((k + 1) % 4)))
        indicesInEdge.push_back(i);

    std::sort(indicesInEdge.begin(), indicesInEdge.end(), [&](int a, int b) {
      Vector3d orientation = (V.row(a) - barycenter).cross(V.row(b) - barycenter);
      return orientation(2) > 0;
    });

    for(size_t j = 0; j + 1 < indicesInEdge.size(); ++j)
      springIndices.emplace_back(indicesInEdge[j], indicesInEdge[j + 1]);
  }

  for(const auto& pair : springIndices)
    std::cout << "(" << pair.first << ", " << pair.second << ")\n";

  // Initialize springs with stiffness
  std::vector<fsim::Spring> springVec;
  for(auto& [start, end] : springIndices)
  {
    double length = (V.row(start) - V.row(end)).norm();
    springVec.emplace_back(start, end, length, spring_stiffness);
  }
  SpringCollection springCol(std::move(springVec));

  // Initialize the membrane model
  auto membrane = fsim::StVKMembrane(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);

  // Composite: membrane + cable springs
  fsim::CompositeModel composite(std::move(membrane), std::move(springCol));

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
  solver.options.fixed_dofs = {0*3+0, 0*3+1, 0*3+2,
                               1*3+0, 1*3+1, 1*3+2,
                               2*3+0, 2*3+1, 2*3+2,
                               3*3+0, 3*3+1, 3*3+2};
  solver.options.threshold = 1e-6;
  solver.options.newton.max = 1e10;

  // Display the curve network
  std::set<int> uniqueNodeIndices;
  for(const auto& pair : springIndices)
  {
    uniqueNodeIndices.insert(pair.first);
    uniqueNodeIndices.insert(pair.second);
  }
  std::map<int, int> indexMapping;
  int newIndex = 0;
  for(int idx : uniqueNodeIndices)
    indexMapping[idx] = newIndex++;

  std::vector<int> nodeIndicesVec(uniqueNodeIndices.begin(), uniqueNodeIndices.end());

  fsim::Mat3<double> R(uniqueNodeIndices.size(), 3);
  fsim::Mat2<int> E(springIndices.size(), 2);

  int row = 0;
  for(int idx : nodeIndicesVec)
    R.row(row++) = V.row(idx);

  for(size_t j = 0; j < springIndices.size(); ++j)
    E.row(j) << indexMapping[springIndices[j].first], indexMapping[springIndices[j].second];

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
    if(ImGui::InputDouble("Spring stiffness", &spring_stiffness, 0, 0, "%.1f"))
    {
      // Rebuild springs with new stiffness
      std::vector<fsim::Spring> newSprings;
      for(auto& [start, end] : springIndices)
      {
        double length = (V.row(start) - V.row(end)).norm();
        newSprings.emplace_back(start, end, length, spring_stiffness);
      }
      composite.getModel<1>() = SpringCollection(std::move(newSprings));
    }

    if(ImGui::Button("Solve"))
    {
      VectorXd var = Map<const VectorXd>(V.data(), V.size());
      solver.solve(composite, var);

      polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
          Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3));

      for(int i = 0; i < R.rows(); ++i)
        R.row(i) = solver.var().segment<3>(3 * nodeIndicesVec[i]);

      polyscope::getCurveNetwork("springs")->updateNodePositions(R);
    }
  };
  polyscope::show();
}
