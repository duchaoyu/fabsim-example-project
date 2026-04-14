#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/CompositeModel.h>
#include <fsim/util/io.h>
#include "sliding_cable.h"
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include "polyscope/point_cloud.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "anisotropic_rest_shape.h"


// ── Helpers (same as membrane_orthotropic.cpp) ────────────────────────────────
std::vector<int> findBoundaryVertices(fsim::Mat3<int> F)
{
  std::map<std::pair<int,int>, int> edgeCount;
  for(int i = 0; i < F.rows(); ++i)
  {
    auto face = F.row(i);
    for(int k = 0; k < 3; ++k)
    {
      int v1 = face[k], v2 = face[(k+1)%3];
      if(v1 > v2) std::swap(v1, v2);
      edgeCount[{v1,v2}]++;
    }
  }
  std::set<int> bv;
  for(auto& [e, c] : edgeCount)
    if(c == 1) { bv.insert(e.first); bv.insert(e.second); }
  return {bv.begin(), bv.end()};
}

void projectFaceVectorsToFaces(const fsim::Mat3<double>& V, const fsim::Mat3<int>& F,
                               std::vector<Eigen::Vector3d>& face_vectors)
{
  for(int i = 0; i < F.rows(); ++i)
  {
    Eigen::Vector3d n = (V.row(F(i,1)) - V.row(F(i,0))).cross(V.row(F(i,2)) - V.row(F(i,0)));
    n.normalize();
    Eigen::Vector3d p = face_vectors[i] - face_vectors[i].dot(n)*n;
    face_vectors[i] = p.norm() < 1e-10 ? Eigen::Vector3d(V.row(F(i,1)) - V.row(F(i,0))).normalized()
                                        : p.normalized();
  }
}

// Read a list of vertex indices (one per line, or space-separated) from a file.
// Returns an empty vector if the file cannot be opened.
std::vector<int> readCableIndices(const std::string& filename)
{
  std::ifstream f(filename);
  if(!f.is_open())
  {
    std::cerr << "Cable index file not found: " << filename
              << "\n  → using hardcoded cable_indices vector instead.\n";
    return {};
  }
  std::vector<int> indices;
  int idx;
  while(f >> idx) indices.push_back(idx);
  std::cout << "Loaded " << indices.size() << " cable vertices from " << filename << "\n";
  return indices;
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

  // ── Cable vertex indices ───────────────────────────────────────────────────
  // Option A: read from a text file (one index per line or space-separated).
  //   Create a file at the path below, e.g.:  12\n34\n56\n...
  std::string cable_file = folder + "cable_indices.txt";

  // Option B: hardcode the path here as a fallback.
  std::vector<int> cable_indices = readCableIndices(cable_file);
  if(cable_indices.empty())
  {
    // ── Edit this list to define your cable path ───────────────────────────
    cable_indices = { 313, 100, 163, 158, 169, 170, 166, 334, 65, 167, 165, 333, 98, 99, 149, 336, 79, 130, 327 };
  }

  // Cable axial stiffness EA (N) — independent of segment length.
  // k per segment = EA / L_segment  (longer segments are automatically softer).
  // Steel: E=200GPa, d=1mm → A=0.785mm² → EA ≈ 157,000 N
  // Steel: E=200GPa, d=3mm → A=7.07mm²  → EA ≈ 1,414,000 N
  double cable_EA = 157000.0;  // N  (1mm diameter steel cable)

  // ── Material parameters (same as membrane_orthotropic.cpp) ────────────────
  double young_modulus1 = 5000;
  double young_modulus2 = 12507;
  double poisson_ratio  = 0.198;
  double thickness      = 1.0;
  double s_f1           = 1.461;   // wale stretch factor
  double s_f2           = 0.908;   // course stretch factor
  double mass           = 0.001;
  double pressure       = 1000;

  std::vector<double> young_modulus1s(F.rows(), young_modulus1);
  std::vector<double> young_modulus2s(F.rows(), young_modulus2);
  std::vector<double> poisson_ratios(F.rows(), poisson_ratio);
  std::vector<double> thicknesses(F.rows(), thickness);

  // ── Face directions ────────────────────────────────────────────────────────
  std::vector<Eigen::Vector3d> face_vectors(F.rows(), Eigen::Vector3d(0.0, 1.0, 0.0));
  projectFaceVectorsToFaces(V0, F, face_vectors);

  // ── Anisotropic rest shape ─────────────────────────────────────────────────
  std::vector<int> bdrs_for_mod = findBoundaryVertices(F);
  std::vector<double> s1_vec(F.rows(), 1.0/s_f1);
  std::vector<double> s2_vec(F.rows(), 1.0/s_f2);
  fsim::Mat3<double> V0_mod =
      computeAnisotropicRestShape(V0, F, bdrs_for_mod, face_vectors, s1_vec, s2_vec);

  // ── Validate cable indices ─────────────────────────────────────────────────
  for(int idx : cable_indices)
    if(idx < 0 || idx >= V0.rows())
    { std::cerr << "Cable index out of range: " << idx << "\n"; return 1; }

  // ── Build sliding cable ────────────────────────────────────────────────────
  auto buildCable = [&]() { return SlidingCable(cable_indices, cable_EA, V0); };

  // ── Membrane + cable composite ─────────────────────────────────────────────
  fsim::OrthotropicStVKMembrane membrane(
      V0_mod, F, thicknesses, young_modulus1s, young_modulus2s,
      poisson_ratios, face_vectors, mass, pressure);

  fsim::CompositeModel composite(std::move(membrane), buildCable());

  // ── Solver ─────────────────────────────────────────────────────────────────
  optim::NewtonSolver<double> solver;
  std::vector<int> bdrs = findBoundaryVertices(F);
  std::sort(bdrs.begin(), bdrs.end());
  for(int b : bdrs)
  {
    solver.options.fixed_dofs.push_back(b*3);
    solver.options.fixed_dofs.push_back(b*3+1);
    solver.options.fixed_dofs.push_back(b*3+2);
  }
  solver.options.threshold  = 1e-6;
  solver.options.newton.max = 1e10;

  // ── Rebuild helper ─────────────────────────────────────────────────────────
  auto rebuildModel = [&]()
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

  auto rebuildCable = [&]() { composite.getModel<1>() = buildCable(); };

  // ── Face vectors for display (same as membrane_orthotropic.cpp) ──────────
  std::vector<glm::vec3> faceVectors(F.rows());
  for(int i = 0; i < F.rows(); ++i)
    faceVectors[i] = glm::vec3(
        static_cast<float>(face_vectors[i](0)),
        static_cast<float>(face_vectors[i](1)),
        static_cast<float>(face_vectors[i](2)));

  // ── Polyscope visualisation ────────────────────────────────────────────────
  // Cable curve network
  if(cable_indices.size() >= 2)
  {
    fsim::Mat3<double> cNodes(cable_indices.size(), 3);
    fsim::Mat2<int>    cEdges(cable_indices.size()-1, 2);
    for(size_t k = 0; k < cable_indices.size(); ++k)
      cNodes.row(k) = V0.row(cable_indices[k]);
    for(size_t k = 0; k + 1 < cable_indices.size(); ++k)
      cEdges.row(k) << (int)k, (int)(k+1);
    polyscope::registerCurveNetwork("cable", cNodes, cEdges)
        ->setColor({1.0, 0.6, 0.0})
        ->setRadius(0.005);
  }

  // Membrane mesh
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

    // ── Material params ──
    if(ImGui::InputDouble("SF wale (s_f1)",  &s_f1, 0, 0, "%.4f")) rebuildModel();
    if(ImGui::InputDouble("SF course (s_f2)", &s_f2, 0, 0, "%.4f")) rebuildModel();
    if(ImGui::InputDouble("Thickness",        &thickness,    0, 0, "%.2f")) rebuildModel();
    if(ImGui::InputDouble("Poisson",          &poisson_ratio,0, 0, "%.3f")) rebuildModel();
    if(ImGui::InputDouble("Modulus1",         &young_modulus1,0,0, "%.2f")) rebuildModel();
    if(ImGui::InputDouble("Modulus2",         &young_modulus2,0,0, "%.2f")) rebuildModel();
    if(ImGui::InputDouble("Mass",             &mass,         0, 0, "%.3f"))
      composite.getModel<0>().setMass(mass);
    if(ImGui::InputDouble("Pressure",         &pressure,     0, 0, "%.2f"))
      composite.getModel<0>().setPressure(pressure);

    ImGui::Separator();

    // ── Cable params ──
    if(ImGui::InputDouble("Cable EA (N)", &cable_EA, 0, 0, "%.1f"))
      rebuildCable();  // k per segment = EA / L_segment

    ImGui::Separator();

    if(ImGui::Button("Solve"))
    {
      solver.solve(composite, Map<VectorXd>(V0.data(), V0.size()));

      // Update mesh
      polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
          Map<fsim::Mat3<double>>(solver.var().data(), V0.rows(), 3));

      // Update cable positions
      if(cable_indices.size() >= 2)
      {
        fsim::Mat3<double> cNodes(cable_indices.size(), 3);
        for(size_t k = 0; k < cable_indices.size(); ++k)
          cNodes.row(k) = solver.var().segment<3>(3 * cable_indices[k]);
        polyscope::getCurveNetwork("cable")->updateNodePositions(cNodes);
      }

      // Save mesh
      std::ostringstream fn;
      fn << folder + "exp/"
         << "cable_SF1_" << s_f1 << "_SF2_" << s_f2
         << "_EA_" << cable_EA
         << "_P_" << pressure;
      fsim::saveOBJ(fn.str(),
          Map<fsim::Mat3<double>>(solver.var().data(), V0.rows(), 3), F);

      // Save cable as OBJ with line elements
      if(cable_indices.size() >= 2)
      {
        std::string cable_path = fn.str() + "_cable.obj";
        std::ofstream cable_out(cable_path);
        for(size_t k = 0; k < cable_indices.size(); ++k)
        {
          Eigen::Vector3d p = solver.var().segment<3>(3 * cable_indices[k]);
          cable_out << "v " << p(0) << " " << p(1) << " " << p(2) << "\n";
        }
        for(size_t k = 0; k + 1 < cable_indices.size(); ++k)
          cable_out << "l " << k+1 << " " << k+2 << "\n";
        std::cout << "Saved cable: " << cable_path << "\n";
      }
    }

    ImGui::PopItemWidth();
  };

  polyscope::show();
}
