// best_fit_stretch_factors_3region.cpp
//
// Three-region anisotropic pre-strain optimisation (no cable).
//
//   min_{sf1_0,sf2_0, sf1_1,sf2_1, sf1_2,sf2_2}  Σ_v || V_sim_v − V_target_v ||
//
// Regions are assigned by BFS from two seed faces:
//   Region 0  — faces within `bfs_radius` hops of seed_face_0
//   Region 1  — faces within `bfs_radius` hops of seed_face_1  (unclaimed by region 0)
//   Region 2  — all remaining faces
//
// Variables: phi = (log(sf1_0), log(sf2_0),
//                   log(sf1_1), log(sf2_1),
//                   log(sf1_2), log(sf2_2))
// Gradient:  finite differences — 6 extra Newton solves per L-BFGS step.
// Timing:    per-iteration cycle time and total optimisation time printed.

#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include <optim/LBFGS.h>
#include "anisotropic_rest_shape.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <vector>

using namespace Eigen;

// ── Globals ───────────────────────────────────────────────────────────────────
fsim::Mat3<double> V0;
fsim::Mat3<int>    F;
fsim::Mat3<double> Vtarget;

std::vector<int>             bdrs;
std::vector<Eigen::Vector3d> face_dirs;
std::vector<int>             face_region;   // 0, 1, or 2

double E1, E2, nu, thickness, mass, pressure;

int      sim_count = 0;
VectorXd warm_start;

// ── Screenshot state ──────────────────────────────────────────────────────────
polyscope::SurfaceMesh* ps_mesh       = nullptr;
std::string             screenshots_dir;
int                     screenshot_idx = 0;

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

// ── Face adjacency ────────────────────────────────────────────────────────────
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

// ── 3-region BFS assignment ───────────────────────────────────────────────────
// Region 0: BFS neighbourhood of seed0, radius hops.
// Region 1: BFS neighbourhood of seed1, radius hops (skips region-0 faces).
// Region 2: everything else.
std::vector<int> computeRegions3BFS(const fsim::Mat3<int>& F,
                                    int seed0, int seed1, int radius)
{
  const int nF = F.rows();
  auto adj = buildFaceAdj(F);

  std::vector<int> region(nF, 2);   // default: region 2

  // BFS expands through all faces but only claims unclaimed (region==2) ones.
  auto floodFill = [&](int seed, int r) {
    std::queue<std::pair<int,int>> q;   // (face, hop_count)
    std::vector<bool> enqueued(nF, false);
    q.push({seed, 0});
    enqueued[seed] = true;
    if (region[seed] == 2) region[seed] = r;
    while (!q.empty()) {
      auto [f, h] = q.front(); q.pop();
      if (h >= radius) continue;
      for (int nb : adj[f]) {
        if (!enqueued[nb]) {
          enqueued[nb] = true;
          if (region[nb] == 2) region[nb] = r;   // claim only unclaimed faces
          q.push({nb, h + 1});                   // expand through all faces
        }
      }
    }
  };

  floodFill(seed0, 0);
  floodFill(seed1, 1);

  int n0 = (int)std::count(region.begin(), region.end(), 0);
  int n1 = (int)std::count(region.begin(), region.end(), 1);
  int n2 = (int)std::count(region.begin(), region.end(), 2);
  std::cout << "  Region 0 (seed face " << seed0 << ", " << radius << " hops): "
            << n0 << " faces\n";
  std::cout << "  Region 1 (seed face " << seed1 << ", " << radius << " hops): "
            << n1 << " faces\n";
  std::cout << "  Region 2 (rest): " << n2 << " faces\n";
  return region;
}

// ── Region export ─────────────────────────────────────────────────────────────
void saveRegions(const std::string& path, const std::vector<int>& region)
{
  std::ofstream out(path);
  std::vector<int> R[3];
  for (int f = 0; f < (int)region.size(); ++f)
    R[region[f]].push_back(f);

  out << "# Three-region face assignment\n";
  out << "# Total faces: " << region.size() << "\n";
  out << "# Format: face index (0-based)\n\n";
  for (int r = 0; r < 3; ++r) {
    out << "REGION_" << r << " " << R[r].size() << "\n";
    for (int f : R[r]) out << f << "\n";
    out << "\n";
  }
  std::cout << "  saved: " << path
            << "  (R0=" << R[0].size() << ", R1=" << R[1].size()
            << ", R2=" << R[2].size() << " faces)\n";
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
// sf1[r], sf2[r] are the stretch factors for region r (r = 0,1,2).
fsim::Mat3<double> simulateImpl(const std::array<double,3>& sf1,
                                const std::array<double,3>& sf2,
                                bool update_warm)
{
  const int nF = F.rows();
  std::vector<double> s1(nF), s2(nF);
  std::vector<double> E1s(nF, E1), E2s(nF, E2), nus(nF, nu), ths(nF, thickness);

  for (int f = 0; f < nF; ++f) {
    int r = face_region[f];
    s1[f] = 1.0 / sf1[r];
    s2[f] = 1.0 / sf2[r];
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

// ── Loss ──────────────────────────────────────────────────────────────────────
double fitLoss(const fsim::Mat3<double>& Vsim)
{
  return (Vsim - Vtarget).rowwise().norm().mean();
}

// ── Screenshot ────────────────────────────────────────────────────────────────
void takeScreenshot(const fsim::Mat3<double>& V, const std::string& label)
{
  if (!ps_mesh) return;

  VectorXd vres = (V - Vtarget).rowwise().norm();
  std::vector<double> fres(F.rows());
  for (int f = 0; f < F.rows(); ++f)
    fres[f] = (vres(F(f,0)) + vres(F(f,1)) + vres(F(f,2))) / 3.0;

  // Normalise region index to [0,1] for colourmap (0→0, 1→0.5, 2→1)
  std::vector<double> face_reg_d(face_region.size());
  for (int f = 0; f < (int)face_region.size(); ++f)
    face_reg_d[f] = face_region[f] / 2.0;

  ps_mesh->updateVertexPositions(V);

  auto* q_region = ps_mesh->addFaceScalarQuantity("region (0/1/2)", face_reg_d);
  q_region->setColorMap("viridis");
  auto* q_resid  = ps_mesh->addFaceScalarQuantity("fit residual (m)", fres);
  q_resid->setColorMap("reds");

  // Screenshot 1: region colouring
  q_region->setEnabled(true);
  q_resid ->setEnabled(false);
  std::ostringstream f1;
  f1 << screenshots_dir << std::setw(3) << std::setfill('0')
     << screenshot_idx << "_" << label << "_regions.png";
  polyscope::screenshot(f1.str(), false);

  // Screenshot 2: residual heatmap
  q_region->setEnabled(false);
  q_resid ->setEnabled(true);
  std::ostringstream f2;
  f2 << screenshots_dir << std::setw(3) << std::setfill('0')
     << screenshot_idx << "_" << label << "_residual.png";
  polyscope::screenshot(f2.str(), false);

  std::cout << "  screenshots: " << f1.str() << "  +  *_residual.png\n";
  ++screenshot_idx;
}

// ── L-BFGS objective and gradient ────────────────────────────────────────────
// phi = (log(sf1_0), log(sf2_0), log(sf1_1), log(sf2_1), log(sf1_2), log(sf2_2))

int lbfgs_iter = 0;
std::chrono::steady_clock::time_point iter_start;

std::array<double,3> sfFromPhi(const VectorXd& phi, int comp)
{
  // comp=0 → sf1, comp=1 → sf2
  return { std::exp(phi(comp)),
           std::exp(phi(2 + comp)),
           std::exp(phi(4 + comp)) };
}

double objective(const VectorXd& phi)
{
  return fitLoss(simulateImpl(sfFromPhi(phi,0), sfFromPhi(phi,1), false));
}

VectorXd gradient(const VectorXd& phi)
{
  iter_start = std::chrono::steady_clock::now();
  const double eps = 1e-4;

  double f0 = fitLoss(simulateImpl(sfFromPhi(phi,0), sfFromPhi(phi,1), true));

  // Capture screenshot at this iteration
  {
    fsim::Mat3<double> Vcur =
        Map<const fsim::Mat3<double>>(warm_start.data(), V0.rows(), 3);
    std::ostringstream lbl;
    lbl << "iter" << std::setw(2) << std::setfill('0') << lbfgs_iter;
    takeScreenshot(Vcur, lbl.str());
  }

  VectorXd grad(6);
  for (int i = 0; i < 6; ++i) {
    VectorXd phiP = phi;
    phiP(i) += eps;
    grad(i) = (fitLoss(simulateImpl(sfFromPhi(phiP,0), sfFromPhi(phiP,1), false)) - f0) / eps;
  }

  double cycle_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - iter_start).count();

  std::cout << "[iter " << lbfgs_iter++ << "]"
            << std::fixed << std::setprecision(5)
            << "  R0: sf1=" << std::exp(phi(0)) << " sf2=" << std::exp(phi(1))
            << "  R1: sf1=" << std::exp(phi(2)) << " sf2=" << std::exp(phi(3))
            << "  R2: sf1=" << std::exp(phi(4)) << " sf2=" << std::exp(phi(5))
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

  // ── 3-region BFS parameters ────────────────────────────────────────────────
  const int seed_face_0 = 68;    // centre of region 0
  const int seed_face_1 = 577;   // centre of region 1
  const int bfs_radius  = 10;    // hops from each seed face

  // ── Initial guess (warm-started from single-region result) ─────────────────
  // All regions start at the single-region optimum; LBFGS will diverge them.
  const double sf1_init = 1.29916, sf2_init = 0.96030;
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

  // ── Region assignment ──────────────────────────────────────────────────────
  std::cout << "Computing 3 regions (BFS from faces " << seed_face_0
            << " and " << seed_face_1 << ", radius=" << bfs_radius << ")...\n";
  face_region = computeRegions3BFS(F, seed_face_0, seed_face_1, bfs_radius);
  saveRegions(out_dir + "sf_3region_faces.txt", face_region);

  // ── Initial simulation ─────────────────────────────────────────────────────
  std::array<double,3> sf1_arr = {sf1_init, sf1_init, sf1_init};
  std::array<double,3> sf2_arr = {sf2_init, sf2_init, sf2_init};

  std::cout << "\n--- Initial simulation (all regions: sf1=" << sf1_init
            << ", sf2=" << sf2_init << ") ---\n";
  fsim::Mat3<double> Vsim_init = simulateImpl(sf1_arr, sf2_arr, true);
  double loss_init = fitLoss(Vsim_init);
  double max_init  = (Vsim_init - Vtarget).rowwise().norm().maxCoeff();
  std::cout << "Initial mean vertex distance: " << loss_init << " m\n";
  std::cout << "Initial max  vertex distance: " << max_init  << " m\n\n";
  saveOFF(out_dir + "sf_3region_initial.off", Vsim_init, F);

  // ── Polyscope init + initial screenshot ────────────────────────────────────
  screenshots_dir = out_dir + "sf_3region_screenshots/";
  std::system(("mkdir -p \"" + screenshots_dir + "\"").c_str());

  polyscope::init();
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  // Normalised region scalar for 3 colours: 0→0, 1→0.5, 2→1
  std::vector<double> face_reg_d(face_region.size());
  for (int f = 0; f < (int)face_region.size(); ++f)
    face_reg_d[f] = face_region[f] / 2.0;

  ps_mesh = polyscope::registerSurfaceMesh("mesh", Vsim_init, F);
  ps_mesh->addFaceScalarQuantity("region (0/1/2)", face_reg_d)
         ->setColorMap("viridis")->setEnabled(true);

  takeScreenshot(Vsim_init, "00_initial");

  // ── L-BFGS optimisation ────────────────────────────────────────────────────
  VectorXd phi(6);
  phi(0) = std::log(sf1_init); phi(1) = std::log(sf2_init);
  phi(2) = std::log(sf1_init); phi(3) = std::log(sf2_init);
  phi(4) = std::log(sf1_init); phi(5) = std::log(sf2_init);

  std::cout << "--- L-BFGS optimisation (6 DOF, 3 regions) ---\n";
  optim::LBFGSSolver<double> lbfgs;
  lbfgs.options.threshold       = 1e-4;
  lbfgs.options.iteration_limit = 50;

  auto t0 = std::chrono::steady_clock::now();
  VectorXd phi_opt = lbfgs.solve(objective, gradient, phi);
  double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::array<double,3> sf1_opt = sfFromPhi(phi_opt, 0);
  std::array<double,3> sf2_opt = sfFromPhi(phi_opt, 1);

  // ── Final result ───────────────────────────────────────────────────────────
  std::cout << std::defaultfloat << std::setprecision(6)
            << "\n--- Optimal parameters ---\n";
  for (int r = 0; r < 3; ++r)
    std::cout << "  Region " << r << ":  sf1=" << sf1_opt[r]
              << "  sf2=" << sf2_opt[r] << "\n";

  fsim::Mat3<double> Vsim_opt = simulateImpl(sf1_opt, sf2_opt, true);
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

  saveOFF(out_dir + "sf_3region_result.off", Vsim_opt, F);

  // ── Final screenshots + interactive view ───────────────────────────────────
  takeScreenshot(Vsim_opt, "final");
  std::cout << "  screenshots saved to: " << screenshots_dir << "\n";

  Eigen::VectorXd vres_opt = (Vsim_opt - Vtarget).rowwise().norm();
  ps_mesh->addVertexScalarQuantity("vertex residual (m)",
      std::vector<double>(vres_opt.data(), vres_opt.data() + vres_opt.size()))
      ->setColorMap("reds");
  polyscope::registerSurfaceMesh("Target mesh", Vtarget, F)->setEnabled(false);

  polyscope::show();
  return 0;
}
