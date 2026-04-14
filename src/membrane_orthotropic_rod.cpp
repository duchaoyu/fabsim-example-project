// membrane_orthotropic_rod.cpp
//
// Orthotropic membrane + elastic rod (spline/cable) composite simulation,
// with the rod DECOUPLED from membrane vertices so it can slide on the surface.
//
// ── DOF layout ───────────────────────────────────────────────────────────────
//   X[0 ... 3*nV-1]                    membrane vertex positions  (nV = V0.rows())
//   X[3*nV ... 3*(nV+nRod)-1]          rod node positions         (decoupled)
//   X[3*(nV+nRod) ... nDOF-1]          rod twist angles
//
// The rod has its own independent position DOFs initialised to the cable vertex
// positions.  A RodSurfaceContact penalty keeps rod nodes on the membrane
// surface while allowing free tangential (sliding) motion.

#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/ElasticRod.h>
#include <fsim/CompositeModel.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include "polyscope/point_cloud.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include "anisotropic_rest_shape.h"
#include "rod_surface_contact.h"

// ── Helpers ───────────────────────────────────────────────────────────────────
std::vector<int> findBoundaryVertices(fsim::Mat3<int> F)
{
  std::map<std::pair<int,int>, int> edgeCount;
  for(int i = 0; i < F.rows(); ++i)
    for(int k = 0; k < 3; ++k)
    {
      int v1 = F(i,k), v2 = F(i,(k+1)%3);
      if(v1 > v2) std::swap(v1, v2);
      edgeCount[{v1,v2}]++;
    }
  std::set<int> bv;
  for(auto& [e,c] : edgeCount)
    if(c == 1) { bv.insert(e.first); bv.insert(e.second); }
  return {bv.begin(), bv.end()};
}

void projectFaceVectorsToFaces(const fsim::Mat3<double>& V, const fsim::Mat3<int>& F,
                               std::vector<Eigen::Vector3d>& face_vectors)
{
  for(int i = 0; i < F.rows(); ++i)
  {
    Eigen::Vector3d n = (V.row(F(i,1))-V.row(F(i,0))).cross(V.row(F(i,2))-V.row(F(i,0)));
    n.normalize();
    Eigen::Vector3d p = face_vectors[i] - face_vectors[i].dot(n)*n;
    face_vectors[i] = p.norm() < 1e-10
        ? Eigen::Vector3d(V.row(F(i,1))-V.row(F(i,0))).normalized()
        : p.normalized();
  }
}

std::vector<int> readCableIndices(const std::string& filename)
{
  std::ifstream f(filename);
  if(!f.is_open()) return {};
  std::vector<int> idx;
  int v;
  while(f >> v) idx.push_back(v);
  std::cout << "Loaded " << idx.size() << " rod vertices from " << filename << "\n";
  return idx;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
  using namespace Eigen;

  // ── Mesh ───────────────────────────────────────────────────────────────────
  fsim::Mat3<double> V0;
  fsim::Mat3<int> F;
  std::string folder    = "/Users/duch/Documents/PhD/knit/2024_prototypes/2part/";
  std::string mesh_name = "2part_opt_simu_m.off";
  fsim::readOFF(folder + mesh_name, V0, F);
  const int nV = V0.rows();

  // ── Rod vertex indices (into the original mesh) ────────────────────────────
  std::vector<int> cable_indices = readCableIndices(folder + "cable_indices.txt");
  if(cable_indices.empty())
    cable_indices = { 313, 100, 163, 158, 169, 170, 166, 334, 65,
                      167, 165, 333,  98,  99, 149, 336,  79, 130, 327 };
  const int nRod = (int)cable_indices.size();

  for(int idx : cable_indices)
    if(idx < 0 || idx >= nV)
    { std::cerr << "Rod index out of range: " << idx << "\n"; return 1; }

  // ── Extended vertex matrix: membrane rows + rod rows ──────────────────────
  // Rod nodes occupy rows nV … nV+nRod-1 in V0_ext.
  fsim::Mat3<double> V0_ext(nV + nRod, 3);
  V0_ext.topRows(nV) = V0;
  for(int k = 0; k < nRod; ++k)
    V0_ext.row(nV + k) = V0.row(cable_indices[k]);

  // Rod indices into V0_ext
  VectorXi rodIndices(nRod);
  for(int k = 0; k < nRod; ++k)
    rodIndices[k] = nV + k;

  // ── Material parameters ────────────────────────────────────────────────────
  double young_modulus1 = 5000;
  double young_modulus2 = 12507;
  double poisson_ratio  = 0.198;
  double thickness      = 1.0;
  double s_f1           = 1.461;
  double s_f2           = 0.908;
  double mass           = 0.001;
  double pressure       = 1000;
  double k_contact      = 1e5;   // normal penalty stiffness (N/m)

  std::vector<double> young_modulus1s(F.rows(), young_modulus1);
  std::vector<double> young_modulus2s(F.rows(), young_modulus2);
  std::vector<double> poisson_ratios(F.rows(), poisson_ratio);
  std::vector<double> thicknesses(F.rows(), thickness);

  // ── Rod parameters ─────────────────────────────────────────────────────────
  fsim::RodParams rodParams;
  rodParams.thickness    = 0.003;
  rodParams.width        = 0.003;
  rodParams.E            = 200e9;
  rodParams.mass         = 0.0;
  rodParams.crossSection = fsim::CrossSection::Circle;

  // ── Face directions ────────────────────────────────────────────────────────
  std::vector<Eigen::Vector3d> face_vectors(F.rows(), Eigen::Vector3d(0.0, 1.0, 0.0));
  projectFaceVectorsToFaces(V0, F, face_vectors);

  // ── Anisotropic rest shape ─────────────────────────────────────────────────
  std::vector<int> bdrs_for_mod = findBoundaryVertices(F);
  std::vector<double> s1_vec(F.rows(), 1.0/s_f1);
  std::vector<double> s2_vec(F.rows(), 1.0/s_f2);
  fsim::Mat3<double> V0_mod =
      computeAnisotropicRestShape(V0, F, bdrs_for_mod, face_vectors, s1_vec, s2_vec);

  // ── Build initial DOF vector ───────────────────────────────────────────────
  // Layout: [3*nV membrane | 3*nRod rod | nEdge twists]
  // Twists are initialised to zero; rod positions start at the cable vertices.
  Eigen::Vector3d e0 = (V0.row(cable_indices[1]) - V0.row(cable_indices[0])).normalized();
  Eigen::Vector3d n0 = e0.unitOrthogonal();

  // Build rod and composite first (needed for nDOF via nbEdges())
  fsim::OrthotropicStVKMembrane membrane(
      V0_mod, F, thicknesses, young_modulus1s, young_modulus2s,
      poisson_ratios, face_vectors, mass, pressure);

  // Rod uses V0_ext and decoupled rodIndices
  // ElasticRod stores nV = V0_ext.rows() = nV+nRod, so twist DOFs land at
  // X[3*(nV+nRod) + e] automatically.
  fsim::ElasticRod rod(V0_ext, rodIndices, n0, rodParams);

  const int nDOF = 3 * (nV + nRod) + rod.nbEdges();

  VectorXd var = VectorXd::Zero(nDOF);
  var.head(3 * nV) = Map<const VectorXd>(V0.data(), 3 * nV);
  for(int k = 0; k < nRod; ++k)
    var.segment<3>(3 * nV + 3 * k) = V0.row(cable_indices[k]).transpose();

  // ── Contact constraint ─────────────────────────────────────────────────────
  // Seed contact assignments from initial flat configuration.
  RodSurfaceContact contact(nV, nRod, k_contact, F, var);

  // ── Composite model: membrane + rod + contact ──────────────────────────────
  fsim::CompositeModel composite(std::move(membrane), std::move(rod), std::move(contact));

  // ── Solver ─────────────────────────────────────────────────────────────────
  optim::NewtonSolver<double> solver;

  // Fix only membrane boundary vertices; rod DOFs (3*nV onward) are free to slide.
  std::vector<int> bdrs = findBoundaryVertices(F);
  std::sort(bdrs.begin(), bdrs.end());
  for(int b : bdrs)
  {
    solver.options.fixed_dofs.push_back(b * 3);
    solver.options.fixed_dofs.push_back(b * 3 + 1);
    solver.options.fixed_dofs.push_back(b * 3 + 2);
  }
  solver.options.threshold  = 1e-6;
  solver.options.newton.max = 1e10;

  // After each Newton step: update rod material frames AND contact assignments.
  solver.options.update_fct = [&](const Ref<const VectorXd> X)
  {
    composite.getModel<1>().updateProperties(X);
    composite.getModel<2>().updateContacts(X);
  };

  // ── Rebuild helpers ────────────────────────────────────────────────────────
  auto rebuildMembrane = [&]()
  {
    std::fill(s1_vec.begin(), s1_vec.end(), 1.0/s_f1);
    std::fill(s2_vec.begin(), s2_vec.end(), 1.0/s_f2);
    V0_mod = computeAnisotropicRestShape(V0, F, bdrs_for_mod, face_vectors, s1_vec, s2_vec);
    std::fill(young_modulus1s.begin(), young_modulus1s.end(), young_modulus1);
    std::fill(young_modulus2s.begin(), young_modulus2s.end(), young_modulus2);
    std::fill(poisson_ratios.begin(),  poisson_ratios.end(),  poisson_ratio);
    std::fill(thicknesses.begin(),     thicknesses.end(),     thickness);
    composite.getModel<0>() = fsim::OrthotropicStVKMembrane(
        V0_mod, F, thicknesses, young_modulus1s, young_modulus2s,
        poisson_ratios, face_vectors, mass, pressure);
  };

  auto rebuildRod = [&]()
  {
    composite.getModel<1>() = fsim::ElasticRod(V0_ext, rodIndices, n0, rodParams);
  };

  // ── Face vectors for display ──────────────────────────────────────────────
  std::vector<glm::vec3> faceVectors(F.rows());
  for(int i = 0; i < F.rows(); ++i)
    faceVectors[i] = glm::vec3(
        static_cast<float>(face_vectors[i](0)),
        static_cast<float>(face_vectors[i](1)),
        static_cast<float>(face_vectors[i](2)));

  // ── Polyscope ──────────────────────────────────────────────────────────────
  fsim::Mat3<double> rodNodes(nRod, 3);
  fsim::Mat2<int>    rodEdges(nRod - 1, 2);
  for(int k = 0; k < nRod; ++k)
    rodNodes.row(k) = V0.row(cable_indices[k]);
  for(int k = 0; k + 1 < nRod; ++k)
    rodEdges.row(k) << k, k + 1;
  polyscope::registerCurveNetwork("rod", rodNodes, rodEdges)
      ->setColor({1.0, 0.6, 0.0})
      ->setRadius(rodParams.thickness / 2.0);

  polyscope::registerSurfaceMesh("mesh", V0, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1})
      ->setSurfaceColor({0.1, 0.5, 0.9});
  polyscope::getSurfaceMesh("mesh")->addFaceVectorQuantity("Face Vectors", faceVectors)
      ->setVectorColor(glm::vec3(1.0f, 0.0f, 0.0f));

  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;
  polyscope::init();

  polyscope::state::userCallback = [&]()
  {
    ImGui::PushItemWidth(100);

    ImGui::Text("── Membrane ──");
    if(ImGui::InputDouble("SF wale (s_f1)",    &s_f1,           0, 0, "%.4f")) rebuildMembrane();
    if(ImGui::InputDouble("SF course (s_f2)",  &s_f2,           0, 0, "%.4f")) rebuildMembrane();
    if(ImGui::InputDouble("Thickness",         &thickness,      0, 0, "%.2f")) rebuildMembrane();
    if(ImGui::InputDouble("Poisson",           &poisson_ratio,  0, 0, "%.3f")) rebuildMembrane();
    if(ImGui::InputDouble("Modulus1",          &young_modulus1, 0, 0, "%.2f")) rebuildMembrane();
    if(ImGui::InputDouble("Modulus2",          &young_modulus2, 0, 0, "%.2f")) rebuildMembrane();
    if(ImGui::InputDouble("Mass",              &mass,           0, 0, "%.3f"))
      composite.getModel<0>().setMass(mass);
    if(ImGui::InputDouble("Pressure",          &pressure,       0, 0, "%.2f"))
      composite.getModel<0>().setPressure(pressure);

    ImGui::Separator();
    ImGui::Text("── Rod ──");
    if(ImGui::InputDouble("Rod thickness (m)", &rodParams.thickness, 0, 0, "%.4f"))
    { rebuildRod(); polyscope::getCurveNetwork("rod")->setRadius(rodParams.thickness/2.0); }
    if(ImGui::InputDouble("Rod width (m)",     &rodParams.width,     0, 0, "%.4f")) rebuildRod();
    if(ImGui::InputDouble("Rod E (Pa)",        &rodParams.E,         0, 0, "%.0f")) rebuildRod();
    if(ImGui::InputDouble("Contact k (N/m)",   &k_contact,           0, 0, "%.0f"))
      composite.getModel<2>().k = k_contact;

    ImGui::Separator();

    if(ImGui::Button("Solve"))
    {
      // Reset DOF vector: membrane at V0, rod at original cable vertex positions.
      var = VectorXd::Zero(nDOF);
      var.head(3 * nV) = Map<const VectorXd>(V0.data(), 3 * nV);
      for(int k = 0; k < nRod; ++k)
        var.segment<3>(3 * nV + 3 * k) = V0.row(cable_indices[k]).transpose();

      composite.getModel<1>().updateProperties(var);
      composite.getModel<2>().updateContacts(var);

      solver.solve(composite, var);
      var = solver.var();

      // Update membrane display
      polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
          Map<fsim::Mat3<double>>(var.data(), nV, 3));

      // Update rod display (positions are in X[3*nV ... 3*(nV+nRod)-1])
      fsim::Mat3<double> rn(nRod, 3);
      for(int k = 0; k < nRod; ++k)
        rn.row(k) = var.segment<3>(3 * nV + 3 * k).transpose();
      polyscope::getCurveNetwork("rod")->updateNodePositions(rn);

      // Save membrane
      std::ostringstream fn;
      fn << folder + "exp/"
         << "rod_SF1_" << s_f1 << "_SF2_" << s_f2
         << "_E_" << rodParams.E
         << "_d_" << rodParams.thickness
         << "_P_" << pressure;
      fsim::saveOBJ(fn.str(),
          Map<fsim::Mat3<double>>(var.data(), nV, 3), F);

      // Save rod as OBJ
      std::ofstream rod_out(fn.str() + "_rod.obj");
      for(int k = 0; k < nRod; ++k)
      {
        Eigen::Vector3d p = var.segment<3>(3 * nV + 3 * k);
        rod_out << "v " << p(0) << " " << p(1) << " " << p(2) << "\n";
      }
      for(int k = 0; k + 1 < nRod; ++k)
        rod_out << "l " << k+1 << " " << k+2 << "\n";
      std::cout << "Saved rod: " << fn.str() + "_rod.obj\n";
    }

    ImGui::PopItemWidth();
  };

  polyscope::show();
}
