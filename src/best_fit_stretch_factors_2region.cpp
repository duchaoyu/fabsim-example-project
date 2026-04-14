// best_fit_stretch_factors_2region.cpp
//
// Two-region anisotropic pre-strain optimisation (no cable).
//
//   min_{sf1_A, sf2_A, sf1_B, sf2_B}  Σ_v || V_sim_v − V_target_v ||
//
// The mesh faces are split into two spatially connected regions A and B.
// Region membership is determined by face centroid position along a chosen axis
// (default: Y).  Disconnected fragments are automatically merged into the
// neighbouring connected region so both regions are always simply connected.
//
// Variables: phi = (log(sf1_A), log(sf2_A), log(sf1_B), log(sf2_B))
// Gradient:  finite differences — 4 extra Newton solves per L-BFGS step.
// Timing:    per-iteration cycle time and total optimisation time printed.

#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include <optim/LBFGS.h>
#include "anisotropic_rest_shape.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <vector>

using namespace Eigen;

// ── Globals ───────────────────────────────────────────────────────────────────
fsim::Mat3<double> V0;
fsim::Mat3<int>    F;
fsim::Mat3<double> Vtarget;

std::vector<int>             bdrs;
std::vector<Eigen::Vector3d> face_dirs;
std::vector<int>             face_region;   // 0 = region A, 1 = region B

double E1, E2, nu, thickness, mass, pressure;

int      sim_count = 0;
VectorXd warm_start;

// ── Screenshot state ──────────────────────────────────────────────────────────
polyscope::SurfaceMesh* ps_mesh       = nullptr;
std::string             screenshots_dir;
int                     screenshot_idx = 0;

void takeScreenshot(const fsim::Mat3<double>& V, const std::string& label)
{
  if (!ps_mesh) return;

  // Per-face residual
  VectorXd vres = (V - Vtarget).rowwise().norm();
  std::vector<double> fres(F.rows());
  for (int f = 0; f < F.rows(); ++f)
    fres[f] = (vres(F(f,0)) + vres(F(f,1)) + vres(F(f,2))) / 3.0;

  std::vector<double> face_reg_d(face_region.begin(), face_region.end());

  ps_mesh->updateVertexPositions(V);

  auto* q_region = ps_mesh->addFaceScalarQuantity("region (0=A, 1=B)", face_reg_d);
  q_region->setColorMap("blues");
  auto* q_resid  = ps_mesh->addFaceScalarQuantity("fit residual (m)", fres);
  q_resid->setColorMap("reds");

  // ── Screenshot 1: region colouring ───────────────────────────────────────
  q_region->setEnabled(true);
  q_resid ->setEnabled(false);
  std::ostringstream f1;
  f1 << screenshots_dir << std::setw(3) << std::setfill('0')
     << screenshot_idx << "_" << label << "_regions.png";
  polyscope::screenshot(f1.str(), false);

  // ── Screenshot 2: residual heatmap ───────────────────────────────────────
  q_region->setEnabled(false);
  q_resid ->setEnabled(true);
  std::ostringstream f2;
  f2 << screenshots_dir << std::setw(3) << std::setfill('0')
     << screenshot_idx << "_" << label << "_residual.png";
  polyscope::screenshot(f2.str(), false);

  std::cout << "  screenshots: " << f1.str() << "  +  *_residual.png\n";
  ++screenshot_idx;
}

// ── Mesh utilities ────────────────────────────────────────────────────────────
std::vector<int> findBoundaryVertices(const fsim::Mat3<int>& F)
{
  std::map<std::pair<int,int>, int> cnt;
  for (int f = 0; f < F.rows(); ++f)
    for (int i = 0; i < 3; ++i) {
      int a = F(f,i), b = F(f,(i+1)%3);
      if (a > b) std::swap(a, b);
      cnt[{a,b}]++;
    }
  std::set<int> bv;
  for (auto& [e, c] : cnt)
    if (c == 1) { bv.insert(e.first); bv.insert(e.second); }
  return {bv.begin(), bv.end()};
}

void projectFaceVectors(const fsim::Mat3<double>& V, const fsim::Mat3<int>& F,
                        std::vector<Eigen::Vector3d>& fv)
{
  for (int i = 0; i < F.rows(); ++i) {
    Eigen::Vector3d n = (V.row(F(i,1)) - V.row(F(i,0)))
                        .cross(V.row(F(i,2)) - V.row(F(i,0)));
    n.normalize();
    Eigen::Vector3d p = fv[i] - fv[i].dot(n) * n;
    if (p.norm() < 1e-10)
      fv[i] = Eigen::Vector3d(V.row(F(i,1)) - V.row(F(i,0))).normalized();
    else
      fv[i] = p.normalized();
  }
}

// ── Region computation ────────────────────────────────────────────────────────
// Build face adjacency (shared edge) — used by both region functions.
std::vector<std::vector<int>> buildFaceAdj(const fsim::Mat3<int>& F)
{
  const int nF = F.rows();
  std::map<std::pair<int,int>, std::vector<int>> emap;
  for (int f = 0; f < nF; ++f)
    for (int k = 0; k < 3; ++k) {
      int a = F(f,k), b = F(f,(k+1)%3);
      if (a > b) std::swap(a, b);
      emap[{a,b}].push_back(f);
    }
  std::vector<std::vector<int>> adj(nF);
  for (auto& [e, fs] : emap)
    if (fs.size() == 2) {
      adj[fs[0]].push_back(fs[1]);
      adj[fs[1]].push_back(fs[0]);
    }
  return adj;
}

// Ensure both regions are connected; fragments are merged into the other region.
void enforceConnectivity(std::vector<int>& region,
                         const std::vector<std::vector<int>>& adj)
{
  const int nF = region.size();
  for (int r = 0; r < 2; ++r) {
    std::vector<int> comp(nF, -1);
    int nc = 0;
    for (int start = 0; start < nF; ++start) {
      if (region[start] != r || comp[start] != -1) continue;
      std::queue<int> q;
      q.push(start); comp[start] = nc;
      while (!q.empty()) {
        int f = q.front(); q.pop();
        for (int nb : adj[f])
          if (region[nb] == r && comp[nb] == -1) { comp[nb] = nc; q.push(nb); }
      }
      ++nc;
    }
    std::map<int,int> sizes;
    for (int f = 0; f < nF; ++f) if (comp[f] >= 0) sizes[comp[f]]++;
    int largest = -1, largest_sz = 0;
    for (auto& [c, s] : sizes) if (s > largest_sz) { largest_sz = s; largest = c; }
    int reassigned = 0;
    for (int f = 0; f < nF; ++f)
      if (comp[f] >= 0 && comp[f] != largest) { region[f] = 1 - r; ++reassigned; }
    if (reassigned > 0)
      std::cout << "  Region " << (char)('A'+r) << ": merged " << reassigned
                << " disconnected face(s) into region " << (char)('B'-r) << "\n";
  }
}

// Splits faces into two connected regions by face centroid along `axis`.
// Uses median as threshold, then flood-fills to fix disconnected fragments.
std::vector<int> computeRegions(const fsim::Mat3<double>& V,
                                const fsim::Mat3<int>& F, int axis = 1)
{
  const int nF = F.rows();
  auto adj = buildFaceAdj(F);

  std::vector<double> coord(nF);
  for (int f = 0; f < nF; ++f) {
    Eigen::Vector3d c = (V.row(F(f,0)) + V.row(F(f,1)) + V.row(F(f,2))) / 3.0;
    coord[f] = c[axis];
  }
  std::vector<double> sorted_coord = coord;
  std::nth_element(sorted_coord.begin(), sorted_coord.begin() + nF/2, sorted_coord.end());
  double threshold = sorted_coord[nF / 2];

  std::vector<int> region(nF);
  for (int f = 0; f < nF; ++f)
    region[f] = (coord[f] < threshold) ? 0 : 1;

  enforceConnectivity(region, adj);

  int nA = (int)std::count(region.begin(), region.end(), 0);
  std::cout << "  Region A: " << nA << " faces,  Region B: " << nF - nA << " faces\n";
  return region;
}

// Splits faces into two connected regions based on per-vertex fit residuals.
// Faces with above-median average residual → region A (worse fit, higher priority).
// Flood-fill enforces connectivity.
std::vector<int> computeRegionsByResidual(const fsim::Mat3<double>& Vsim,
                                          const fsim::Mat3<double>& Vtgt,
                                          const fsim::Mat3<int>&    F)
{
  const int nF = F.rows();
  auto adj = buildFaceAdj(F);

  // Per-vertex residuals
  VectorXd vres = (Vsim - Vtgt).rowwise().norm();

  // Per-face residual = average of its three vertex residuals
  std::vector<double> fres(nF);
  for (int f = 0; f < nF; ++f)
    fres[f] = (vres(F(f,0)) + vres(F(f,1)) + vres(F(f,2))) / 3.0;

  std::vector<double> sorted_fres = fres;
  std::nth_element(sorted_fres.begin(), sorted_fres.begin() + nF/2, sorted_fres.end());
  double threshold = sorted_fres[nF / 2];

  // Region A = high-error faces, Region B = low-error faces
  std::vector<int> region(nF);
  for (int f = 0; f < nF; ++f)
    region[f] = (fres[f] >= threshold) ? 0 : 1;

  enforceConnectivity(region, adj);

  int nA = (int)std::count(region.begin(), region.end(), 0);
  double meanA = 0, meanB = 0;
  int cntA = 0, cntB = 0;
  for (int f = 0; f < nF; ++f) {
    if (region[f] == 0) { meanA += fres[f]; ++cntA; }
    else                { meanB += fres[f]; ++cntB; }
  }
  std::cout << "  Region A (high-error): " << nA << " faces"
            << "  mean residual=" << meanA/cntA << " m\n";
  std::cout << "  Region B (low-error):  " << nF-nA << " faces"
            << "  mean residual=" << meanB/cntB << " m\n";
  return region;
}

// ── Newton solve ──────────────────────────────────────────────────────────────
VectorXd newtonSolve(fsim::OrthotropicStVKMembrane& model, const VectorXd& x0)
{
  optim::NewtonSolver<double> solver;
  solver.options.display         = optim::SolverDisplay::quiet;
  solver.options.threshold       = 1e-6;
  solver.options.iteration_limit = 10000;
  for (int b : bdrs) {
    solver.options.fixed_dofs.push_back(b*3);
    solver.options.fixed_dofs.push_back(b*3+1);
    solver.options.fixed_dofs.push_back(b*3+2);
  }
  solver.solve(model, x0);
  return solver.var();
}

// ── Forward simulation ────────────────────────────────────────────────────────
// sf1_A/sf2_A applied to region-A faces, sf1_B/sf2_B to region-B faces.
fsim::Mat3<double> simulateImpl(double sf1_A, double sf2_A,
                                double sf1_B, double sf2_B,
                                bool update_warm)
{
  const int nF = F.rows();
  std::vector<double> s1(nF), s2(nF);
  std::vector<double> E1s(nF, E1), E2s(nF, E2), nus(nF, nu), ths(nF, thickness);

  for (int f = 0; f < nF; ++f) {
    if (face_region[f] == 0) { s1[f] = 1.0/sf1_A; s2[f] = 1.0/sf2_A; }
    else                     { s1[f] = 1.0/sf1_B; s2[f] = 1.0/sf2_B; }
  }

  fsim::Mat3<double> V0_mod =
      computeAnisotropicRestShape(V0, F, bdrs, face_dirs, s1, s2);

  if (warm_start.size() == 0) {
    std::cout << "  [cold start] ramping pressure to " << pressure << " Pa...\n" << std::flush;
    VectorXd x = Map<const VectorXd>(V0.data(), V0.size());
    for (double p : {pressure*0.01, pressure*0.1, pressure*0.5, pressure}) {
      fsim::OrthotropicStVKMembrane m(V0_mod, F, ths, E1s, E2s, nus, face_dirs, mass, p);
      x = newtonSolve(m, x);
    }
    warm_start = x;
    std::cout << "  [cold start] done.\n" << std::flush;
  }

  fsim::OrthotropicStVKMembrane model(V0_mod, F, ths, E1s, E2s, nus, face_dirs, mass, pressure);

  VectorXd result = newtonSolve(model, warm_start);
  ++sim_count;
  if (update_warm) warm_start = result;
  return Map<fsim::Mat3<double>>(result.data(), V0.rows(), 3);
}

// ── Loss ─────────────────────────────────────────────────────────────────────
double fitLoss(const fsim::Mat3<double>& Vsim)
{
  return (Vsim - Vtarget).rowwise().norm().mean();
}

// ── L-BFGS objective and gradient ─────────────────────────────────────────────
// phi = (log(sf1_A), log(sf2_A), log(sf1_B), log(sf2_B))

int lbfgs_iter = 0;
std::chrono::steady_clock::time_point iter_start;

double objective(const VectorXd& phi)
{
  return fitLoss(simulateImpl(std::exp(phi(0)), std::exp(phi(1)),
                              std::exp(phi(2)), std::exp(phi(3)),
                              /*update_warm=*/false));
}

VectorXd gradient(const VectorXd& phi)
{
  iter_start = std::chrono::steady_clock::now();
  const double eps = 1e-4;

  double f0 = fitLoss(simulateImpl(std::exp(phi(0)), std::exp(phi(1)),
                                   std::exp(phi(2)), std::exp(phi(3)),
                                   /*update_warm=*/true));

  // Capture screenshot of this iteration's simulated shape
  {
    fsim::Mat3<double> Vcur =
        Map<const fsim::Mat3<double>>(warm_start.data(), V0.rows(), 3);
    std::ostringstream lbl;
    lbl << "iter" << std::setw(2) << std::setfill('0') << lbfgs_iter;
    takeScreenshot(Vcur, lbl.str());
  }

  VectorXd grad(4);
  for (int i = 0; i < 4; ++i) {
    VectorXd phiP = phi;
    phiP(i) += eps;
    grad(i) = (fitLoss(simulateImpl(std::exp(phiP(0)), std::exp(phiP(1)),
                                    std::exp(phiP(2)), std::exp(phiP(3)),
                                    /*update_warm=*/false)) - f0) / eps;
  }

  double cycle_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - iter_start).count();

  std::cout << "[iter " << lbfgs_iter++ << "]"
            << std::fixed << std::setprecision(5)
            << "  sf1_A=" << std::exp(phi(0)) << "  sf2_A=" << std::exp(phi(1))
            << "  sf1_B=" << std::exp(phi(2)) << "  sf2_B=" << std::exp(phi(3))
            << std::setprecision(6)
            << "  loss=" << f0
            << "  |grad|=" << grad.norm()
            << "  cycle=" << std::fixed << std::setprecision(1) << cycle_s << "s"
            << "  (solves: " << sim_count << ")\n" << std::flush;
  return grad;
}

// ── OFF writer ────────────────────────────────────────────────────────────────
void saveOFF(const std::string& path,
             const fsim::Mat3<double>& V, const fsim::Mat3<int>& F)
{
  std::ofstream out(path);
  out << "OFF\n" << V.rows() << " " << F.rows() << " 0\n";
  out << std::fixed << std::setprecision(8);
  for (int i = 0; i < V.rows(); ++i)
    out << V(i,0) << " " << V(i,1) << " " << V(i,2) << "\n";
  for (int i = 0; i < F.rows(); ++i)
    out << "3 " << F(i,0) << " " << F(i,1) << " " << F(i,2) << "\n";
  std::cout << "  saved: " << path << "\n";
}

// ── Region export ─────────────────────────────────────────────────────────────
void saveRegions(const std::string& path, const std::vector<int>& region)
{
  std::ofstream out(path);
  std::vector<int> A, B;
  for (int f = 0; f < (int)region.size(); ++f)
    (region[f] == 0 ? A : B).push_back(f);

  out << "# Two-region face assignment\n";
  out << "# Total faces: " << region.size() << "\n";
  out << "# Format: face index (0-based)\n\n";

  out << "REGION_A " << A.size() << "\n";
  for (int f : A) out << f << "\n";

  out << "\nREGION_B " << B.size() << "\n";
  for (int f : B) out << f << "\n";

  std::cout << "  saved: " << path
            << "  (A=" << A.size() << " faces, B=" << B.size() << " faces)\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
  // ── Configuration ──────────────────────────────────────────────────────────
  const std::string folder      = "/Users/duch/Documents/PhD/knit/2024_prototypes/2part/";
  const std::string mesh_ref    = folder + "2part_opt_simu_m.off";
  const std::string mesh_target = folder + "2part_opt_simu_m.off";  // change to target!
  const std::string out_dir     = "/Users/duch/Downloads/";

  E1        = 5000.0;
  E2        = 12507.0;
  nu        = 0.198;
  thickness = 1.0;
  mass      = 0.001;
  pressure  = 1000.0;

  // Initial guess from single-region best_fit_stretch_factors result
  double sf1_A_init = 1.29916, sf2_A_init = 0.96030;
  double sf1_B_init = 1.29916, sf2_B_init = 0.96030;
  // ───────────────────────────────────────────────────────────────────────────

  fsim::readOFF(mesh_ref,    V0,      F);
  fsim::readOFF(mesh_target, Vtarget, F);

  if (V0.rows() != Vtarget.rows()) {
    std::cerr << "Error: reference and target meshes have different vertex counts.\n";
    return 1;
  }

  std::cout << "Mesh: " << V0.rows() << " vertices, " << F.rows() << " faces\n";

  bdrs = findBoundaryVertices(F);
  std::cout << "Boundary vertices: " << bdrs.size() << "\n";

  face_dirs.assign(F.rows(), Eigen::Vector3d(0.0, 1.0, 0.0));
  projectFaceVectors(V0, F, face_dirs);

  // ── Spatial region split (Y-axis median) ──────────────────────────────────
  std::cout << "Computing regions (split axis=Y)...\n";
  face_region = computeRegions(V0, F, /*axis=*/1);
  saveRegions(out_dir + "sf_2region_faces.txt", face_region);

  // ── Initial simulation ─────────────────────────────────────────────────────
  std::cout << "\n--- Initial simulation (warm-started from single-region opt) ---\n";
  fsim::Mat3<double> Vsim_init =
      simulateImpl(sf1_A_init, sf2_A_init, sf1_B_init, sf2_B_init, /*update_warm=*/true);
  double loss_init = fitLoss(Vsim_init);
  double max_init  = (Vsim_init - Vtarget).rowwise().norm().maxCoeff();
  std::cout << "Initial mean vertex distance: " << loss_init << " m\n";
  std::cout << "Initial max  vertex distance: " << max_init  << " m\n\n";
  saveOFF(out_dir + "sf_2region_initial.off", Vsim_init, F);

  // ── Polyscope init + initial screenshot ────────────────────────────────────
  screenshots_dir = out_dir + "sf_2region_screenshots/";
  std::system(("mkdir -p \"" + screenshots_dir + "\"").c_str());

  polyscope::init();
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  std::vector<double> face_reg_d(face_region.begin(), face_region.end());
  ps_mesh = polyscope::registerSurfaceMesh("mesh", Vsim_init, F);
  ps_mesh->addFaceScalarQuantity("region (0=A, 1=B)", face_reg_d)
         ->setColorMap("blues")->setEnabled(false);

  takeScreenshot(Vsim_init, "00_initial");

  // ── L-BFGS optimisation ────────────────────────────────────────────────────
  VectorXd phi(4);
  phi(0) = std::log(sf1_A_init);
  phi(1) = std::log(sf2_A_init);
  phi(2) = std::log(sf1_B_init);
  phi(3) = std::log(sf2_B_init);

  std::cout << "--- L-BFGS optimisation (4 DOF) ---\n";
  optim::LBFGSSolver<double> lbfgs;
  lbfgs.options.threshold       = 1e-4;
  lbfgs.options.iteration_limit = 50;

  auto t0 = std::chrono::steady_clock::now();
  VectorXd phi_opt = lbfgs.solve(objective, gradient, phi);
  double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  double sf1_A_opt = std::exp(phi_opt(0)), sf2_A_opt = std::exp(phi_opt(1));
  double sf1_B_opt = std::exp(phi_opt(2)), sf2_B_opt = std::exp(phi_opt(3));

  // ── Final result ───────────────────────────────────────────────────────────
  std::cout << std::defaultfloat
            << "\n--- Optimal parameters ---\n";
  std::cout << std::setprecision(6)
            << "  Region A:  sf1=" << sf1_A_opt << "  sf2=" << sf2_A_opt << "\n";
  std::cout << "  Region B:  sf1=" << sf1_B_opt << "  sf2=" << sf2_B_opt << "\n";

  fsim::Mat3<double> Vsim_opt =
      simulateImpl(sf1_A_opt, sf2_A_opt, sf1_B_opt, sf2_B_opt, /*update_warm=*/true);
  double loss_opt = fitLoss(Vsim_opt);
  double max_opt  = (Vsim_opt - Vtarget).rowwise().norm().maxCoeff();

  std::cout << std::defaultfloat << std::setprecision(6)
            << "  mean vertex distance: " << loss_opt << " m"
            << "  (was " << loss_init << " m, "
            << std::fixed << std::setprecision(2)
            << 100.0*(loss_init - loss_opt)/loss_init << "% improvement)\n";
  std::cout << std::defaultfloat
            << "  max  vertex distance: " << max_opt << " m"
            << "  (was " << max_init << " m)\n";
  std::cout << "\nTotal optimisation time: " << std::fixed << std::setprecision(1)
            << total_s << " s\n";
  std::cout << "Total Newton solves: " << sim_count << "\n";

  saveOFF(out_dir + "sf_2region_result.off", Vsim_opt, F);

  // ── Polyscope visualisation ────────────────────────────────────────────────
  // Final screenshots (region + residual pair)
  takeScreenshot(Vsim_opt, "final");
  std::cout << "  screenshots saved to: " << screenshots_dir << "\n";

  // Per-vertex residuals for the interactive view
  Eigen::VectorXd vres_opt = (Vsim_opt - Vtarget).rowwise().norm();
  ps_mesh->addVertexScalarQuantity("vertex residual (m)",
        std::vector<double>(vres_opt.data(), vres_opt.data() + vres_opt.size()))
        ->setColorMap("reds");
  polyscope::registerSurfaceMesh("Target mesh", Vtarget, F)->setEnabled(false);

  polyscope::show();
  return 0;
}
