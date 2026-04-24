// best_fit_stretch_factors_3region_adaptive.cpp
//
// Alternating optimisation: jointly optimises stretch factors AND region
// boundaries by interleaving two steps per outer iteration:
//
//   (A) Inner L-BFGS  — fix regions, optimise sf1/sf2 per region
//   (B) Boundary reassignment — fix sf1/sf2, greedily re-assign each
//       boundary face to whichever neighbouring region lowers the loss
//
// Convergence: stop when step (B) swaps zero faces.
//
// Region connectivity is enforced: a face is only swapped if removing it
// from its current region leaves that region still connected (BFS check).
//
// Initial regions: BFS from two seed faces (configurable radius).
// Initial sf values: warm-started from the single-region optimum.

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

// ── BFS region assignment ─────────────────────────────────────────────────────
std::vector<int> computeRegions3BFS(const fsim::Mat3<int>& F,
                                    int seed0, int seed1, int radius)
{
  const int nF = F.rows();
  auto adj = buildFaceAdj(F);
  std::vector<int> region(nF, 2);

  auto floodFill = [&](int seed, int r) {
    std::queue<std::pair<int,int>> q;
    std::vector<bool> enqueued(nF, false);
    q.push({seed, 0}); enqueued[seed] = true;
    if (region[seed] == 2) region[seed] = r;
    while (!q.empty()) {
      auto [f, h] = q.front(); q.pop();
      if (h >= radius) continue;
      for (int nb : adj[f])
        if (!enqueued[nb]) {
          enqueued[nb] = true;
          if (region[nb] == 2) region[nb] = r;
          q.push({nb, h + 1});
        }
    }
  };

  floodFill(seed0, 0);
  floodFill(seed1, 1);

  int n0 = (int)std::count(region.begin(), region.end(), 0);
  int n1 = (int)std::count(region.begin(), region.end(), 1);
  int n2 = (int)std::count(region.begin(), region.end(), 2);
  std::cout << "  Region 0 (seed " << seed0 << ", " << radius << " hops): " << n0 << " faces\n";
  std::cout << "  Region 1 (seed " << seed1 << ", " << radius << " hops): " << n1 << " faces\n";
  std::cout << "  Region 2 (rest): " << n2 << " faces\n";
  return region;
}

// ── Region export ─────────────────────────────────────────────────────────────
void saveRegions(const std::string& path, const std::vector<int>& region)
{
  std::ofstream out(path);
  std::vector<int> R[3];
  for (int f = 0; f < (int)region.size(); ++f) R[region[f]].push_back(f);
  out << "# Three-region face assignment\n# Total faces: " << region.size() << "\n\n";
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
  solver.options.threshold       = 1e-4;
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
    std::cout << "  [cold start] ramping pressure...\n" << std::flush;
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

  std::vector<double> face_reg_d(face_region.size());
  for (int f = 0; f < (int)face_region.size(); ++f)
    face_reg_d[f] = face_region[f] / 2.0;

  ps_mesh->updateVertexPositions(V);
  auto* qr = ps_mesh->addFaceScalarQuantity("region (0/1/2)", face_reg_d);
  qr->setColorMap("viridis");
  auto* qe = ps_mesh->addFaceScalarQuantity("fit residual (m)", fres);
  qe->setColorMap("reds");

  auto shot = [&](const std::string& suffix, bool region_active) {
    qr->setEnabled(region_active);
    qe->setEnabled(!region_active);
    std::ostringstream fn;
    fn << screenshots_dir << std::setw(3) << std::setfill('0')
       << screenshot_idx << "_" << label << suffix;
    polyscope::screenshot(fn.str(), false);
  };
  shot("_regions.png", true);
  shot("_residual.png", false);
  std::cout << "  screenshot: " << screenshots_dir << std::setw(3) << std::setfill('0')
            << screenshot_idx << "_" << label << "_regions/residual.png\n";
  ++screenshot_idx;
}

// ── L-BFGS objective / gradient ───────────────────────────────────────────────
int lbfgs_iter = 0;
std::chrono::steady_clock::time_point iter_start;

// phi = (log_sf1_shared, log_sf2_shared, log_sf1_R2, log_sf2_R2)
// R0 and R1 share the same stretch factors (phi[0], phi[1]).
std::array<double,3> sf1FromPhi(const VectorXd& phi)
{
  return { std::exp(phi(0)), std::exp(phi(0)), std::exp(phi(2)) };
}
std::array<double,3> sf2FromPhi(const VectorXd& phi)
{
  return { std::exp(phi(1)), std::exp(phi(1)), std::exp(phi(3)) };
}

double objective(const VectorXd& phi)
{
  return fitLoss(simulateImpl(sf1FromPhi(phi), sf2FromPhi(phi), false));
}

VectorXd gradient(const VectorXd& phi)
{
  iter_start = std::chrono::steady_clock::now();
  const double eps = 1e-4;
  double f0 = fitLoss(simulateImpl(sf1FromPhi(phi), sf2FromPhi(phi), true));

  // Per-iteration screenshot
  {
    fsim::Mat3<double> Vcur = Map<const fsim::Mat3<double>>(warm_start.data(), V0.rows(), 3);
    std::ostringstream lbl; lbl << "iter" << std::setw(3) << std::setfill('0') << lbfgs_iter;
    takeScreenshot(Vcur, lbl.str());
  }

  VectorXd grad(4);
  for (int i = 0; i < 4; ++i) {
    VectorXd phiP = phi; phiP(i) += eps;
    grad(i) = (fitLoss(simulateImpl(sf1FromPhi(phiP), sf2FromPhi(phiP), false)) - f0) / eps;
  }
  double cs = std::chrono::duration<double>(std::chrono::steady_clock::now()-iter_start).count();
  std::cout << "[iter " << lbfgs_iter++ << "]"
            << std::fixed << std::setprecision(5)
            << "  R0=R1: sf1=" << std::exp(phi(0)) << " sf2=" << std::exp(phi(1))
            << "  R2: sf1=" << std::exp(phi(2)) << " sf2=" << std::exp(phi(3))
            << std::setprecision(6) << "  loss=" << f0
            << "  |grad|=" << grad.norm()
            << std::fixed << std::setprecision(1) << "  cycle=" << cs << "s"
            << "  (solves:" << sim_count << ")\n" << std::defaultfloat << std::flush;
  return grad;
}

// ── Connectivity check ────────────────────────────────────────────────────────
// Returns true if region `r` remains connected when face `f` is removed from it.
bool isConnectedWithout(int f, int r,
                        const std::vector<std::vector<int>>& adj)
{
  const int nF = (int)face_region.size();
  // Collect remaining faces of region r (excluding f)
  int start = -1, count = 0;
  for (int g = 0; g < nF; ++g)
    if (face_region[g] == r && g != f) { if (start < 0) start = g; ++count; }
  if (count == 0) return false;  // would leave region empty

  // BFS from `start` within region r \ {f}
  std::vector<bool> visited(nF, false);
  std::queue<int> q;
  q.push(start); visited[start] = true;
  int reached = 1;
  while (!q.empty()) {
    int g = q.front(); q.pop();
    for (int nb : adj[g])
      if (!visited[nb] && face_region[nb] == r && nb != f) {
        visited[nb] = true; ++reached; q.push(nb);
      }
  }
  return reached == count;
}

// ── Greedy boundary-face reassignment ─────────────────────────────────────────
// For each boundary face (adjacent to a different region), try assigning it to
// each neighbouring region. Keep the assignment with lowest loss, provided it
// does not disconnect the face's original region.
// Uses Jacobi-style evaluation: all trials use the same warm_start snapshot.
// Returns the number of faces that were reassigned.
int reassignBoundaryFaces(const std::array<double,3>& sf1,
                          const std::array<double,3>& sf2,
                          const std::vector<std::vector<int>>& adj)
{
  const int nF = (int)face_region.size();
  double base_loss = fitLoss(simulateImpl(sf1, sf2, /*update_warm=*/false));
  std::cout << "  [reassign] base loss = " << base_loss << " m\n" << std::flush;

  std::vector<int> new_region = face_region;  // will hold accepted swaps
  int n_swapped = 0;

  for (int f = 0; f < nF; ++f) {
    int r_orig = face_region[f];

    // Collect unique regions of f's neighbours
    std::set<int> nbr_regions;
    for (int nb : adj[f])
      if (face_region[nb] != r_orig) nbr_regions.insert(face_region[nb]);
    if (nbr_regions.empty()) continue;  // interior face — skip

    // Connectivity guard: don't remove f if it would disconnect region r_orig
    if (!isConnectedWithout(f, r_orig, adj)) continue;

    double best_loss = base_loss;
    int    best_r    = r_orig;

    for (int r2 : nbr_regions) {
      face_region[f] = r2;                                          // trial
      double trial = fitLoss(simulateImpl(sf1, sf2, false));
      face_region[f] = r_orig;                                      // restore
      if (trial < best_loss - 1e-9) { best_loss = trial; best_r = r2; }
    }

    new_region[f] = best_r;
    if (best_r != r_orig) ++n_swapped;
  }

  face_region = new_region;   // apply all accepted swaps at once

  int n0 = (int)std::count(face_region.begin(), face_region.end(), 0);
  int n1 = (int)std::count(face_region.begin(), face_region.end(), 1);
  int n2 = (int)std::count(face_region.begin(), face_region.end(), 2);
  std::cout << "  [reassign] " << n_swapped << " face(s) swapped"
            << "  → R0=" << n0 << " R1=" << n1 << " R2=" << n2 << " faces\n"
            << std::flush;
  return n_swapped;
}

#include "save_mesh.h"

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
  // ── Configuration ──────────────────────────────────────────────────────────
  const std::string folder      = "/Users/duch/Documents/PhD/knit/2024_prototypes/2part/";
  const std::string mesh_ref    = folder + "2part_opt_simu_m.off";
  const std::string mesh_target = folder + "2part_opt_simu_m.off";
  const std::string out_dir     = "/Users/duch/Downloads/";

  E1 = 5000.0; E2 = 12507.0; nu = 0.198; thickness = 1.0; mass = 0.001; pressure = 1000.0;

  const int seed_face_0  = 68;
  const int seed_face_1  = 577;
  const int bfs_radius   = 10;
  const int max_outer    = 6;       // max alternating iterations

  const double sf1_init = 1.29916, sf2_init = 0.96030;
  // ───────────────────────────────────────────────────────────────────────────

  fsim::readOFF(mesh_ref,    V0,      F);
  fsim::readOFF(mesh_target, Vtarget, F);
  if (V0.rows() != Vtarget.rows()) {
    std::cerr << "Error: mesh vertex count mismatch.\n"; return 1;
  }
  std::cout << "Mesh: " << V0.rows() << " vertices, " << F.rows() << " faces\n";

  bdrs = findBoundaryVertices(F);
  std::cout << "Boundary vertices: " << bdrs.size() << "\n";
  face_dirs.assign(F.rows(), Eigen::Vector3d(0,1,0));
  projectFaceVectors(V0, F, face_dirs);

  // ── Initial region assignment ──────────────────────────────────────────────
  std::cout << "Computing initial 3-region BFS (seeds " << seed_face_0
            << ", " << seed_face_1 << ", radius=" << bfs_radius << ")...\n";
  face_region = computeRegions3BFS(F, seed_face_0, seed_face_1, bfs_radius);
  saveRegions(out_dir + "sf_3region_adaptive_faces.txt", face_region);

  auto adj = buildFaceAdj(F);

  // ── Initial simulation ─────────────────────────────────────────────────────
  std::array<double,3> sf1_arr = {sf1_init, sf1_init, sf1_init};
  std::array<double,3> sf2_arr = {sf2_init, sf2_init, sf2_init};
  std::cout << "\n--- Initial simulation ---\n";
  fsim::Mat3<double> Vsim_init = simulateImpl(sf1_arr, sf2_arr, true);
  double loss_init = fitLoss(Vsim_init);
  double max_init  = (Vsim_init - Vtarget).rowwise().norm().maxCoeff();
  std::cout << "Initial mean distance: " << loss_init << " m\n";
  std::cout << "Initial max  distance: " << max_init  << " m\n\n";
  saveMesh(out_dir + "sf_3region_adaptive_initial.off", Vsim_init, F);

  // ── Polyscope init ─────────────────────────────────────────────────────────
  screenshots_dir = out_dir + "sf_3region_adaptive_screenshots/";
  std::system(("mkdir -p \"" + screenshots_dir + "\"").c_str());
  polyscope::init();
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
  std::vector<double> face_reg_d(face_region.size());
  for (int f = 0; f < (int)face_region.size(); ++f) face_reg_d[f] = face_region[f] / 2.0;
  ps_mesh = polyscope::registerSurfaceMesh("mesh", Vsim_init, F);
  ps_mesh->addFaceScalarQuantity("region (0/1/2)", face_reg_d)->setColorMap("viridis")->setEnabled(true);
  takeScreenshot(Vsim_init, "00_initial");

  // ── Alternating optimisation ───────────────────────────────────────────────
  // 4 DOF: (log_sf1_R0R1, log_sf2_R0R1, log_sf1_R2, log_sf2_R2)
  VectorXd phi(4);
  phi(0)=std::log(sf1_init); phi(1)=std::log(sf2_init);  // shared R0/R1
  phi(2)=std::log(sf1_init); phi(3)=std::log(sf2_init);  // R2

  VectorXd phi_opt = phi;
  double loss_final = loss_init;

  auto t_total = std::chrono::steady_clock::now();

  for (int outer = 0; outer < max_outer; ++outer) {
    std::cout << "\n══════ Outer iteration " << outer
              << " ══════  (regions: R0="
              << std::count(face_region.begin(), face_region.end(), 0) << " R1="
              << std::count(face_region.begin(), face_region.end(), 1) << " R2="
              << std::count(face_region.begin(), face_region.end(), 2) << ")\n";

    // ── (A) Inner L-BFGS ────────────────────────────────────────────────────
    std::cout << "--- (A) L-BFGS inner optimisation ---\n";
    optim::LBFGSSolver<double> lbfgs;
    lbfgs.options.threshold       = 1e-3;
    lbfgs.options.iteration_limit = 10;

    phi_opt = lbfgs.solve(objective, gradient, phi_opt);
    std::array<double,3> sf1_opt = sf1FromPhi(phi_opt);
    std::array<double,3> sf2_opt = sf2FromPhi(phi_opt);

    // Update warm_start to current optimum
    fsim::Mat3<double> Vsim_cur = simulateImpl(sf1_opt, sf2_opt, true);
    loss_final = fitLoss(Vsim_cur);
    std::cout << "  After inner opt: mean=" << loss_final << " m\n";

    std::ostringstream lbl_inner;
    lbl_inner << "outer" << outer << "_after_inner";
    takeScreenshot(Vsim_cur, lbl_inner.str());

    // ── (B) Boundary face reassignment ──────────────────────────────────────
    std::cout << "--- (B) Boundary face reassignment ---\n";
    int n_swapped = reassignBoundaryFaces(sf1_opt, sf2_opt, adj);

    // Save updated regions file
    saveRegions(out_dir + "sf_3region_adaptive_faces.txt", face_region);

    // Update warm_start for new region assignment + current sf
    Vsim_cur = simulateImpl(sf1_opt, sf2_opt, true);
    loss_final = fitLoss(Vsim_cur);

    std::ostringstream lbl_swap;
    lbl_swap << "outer" << outer << "_after_swap";
    takeScreenshot(Vsim_cur, lbl_swap.str());

    std::cout << "  After reassign: mean=" << loss_final << " m"
              << "  max=" << (Vsim_cur - Vtarget).rowwise().norm().maxCoeff() << " m\n";

    if (n_swapped == 0) {
      std::cout << "  → No faces swapped. Converged.\n";
      break;
    }
  }

  double total_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_total).count();

  // ── Final result ───────────────────────────────────────────────────────────
  std::array<double,3> sf1_opt = sf1FromPhi(phi_opt);
  std::array<double,3> sf2_opt = sf2FromPhi(phi_opt);

  fsim::Mat3<double> Vsim_opt = simulateImpl(sf1_opt, sf2_opt, true);
  double loss_opt = fitLoss(Vsim_opt);
  double max_opt  = (Vsim_opt - Vtarget).rowwise().norm().maxCoeff();

  std::cout << std::defaultfloat << std::setprecision(6)
            << "\n═══ Optimal parameters (R0=R1 constrained) ═══\n";
  std::cout << "  Region 0=1:  sf1=" << sf1_opt[0] << "  sf2=" << sf2_opt[0] << "\n";
  std::cout << "  Region 2:    sf1=" << sf1_opt[2] << "  sf2=" << sf2_opt[2] << "\n";
  std::cout << "  mean distance: " << loss_opt << " m"
            << "  (was " << loss_init << " m, "
            << std::fixed << std::setprecision(2)
            << 100.0*(loss_init-loss_opt)/loss_init << "% improvement)\n";
  std::cout << std::defaultfloat
            << "  max  distance: " << max_opt << " m  (was " << max_init << " m)\n";
  std::cout << std::fixed << std::setprecision(1)
            << "\nTotal time: " << total_s << " s\n";
  std::cout << "Total Newton solves: " << sim_count << "\n";

  saveMesh(out_dir + "sf_3region_adaptive_result.off", Vsim_opt, F);
  takeScreenshot(Vsim_opt, "final");
  std::cout << "  screenshots saved to: " << screenshots_dir << "\n";

  // Interactive view
  Eigen::VectorXd vres_opt = (Vsim_opt - Vtarget).rowwise().norm();
  ps_mesh->addVertexScalarQuantity("vertex residual (m)",
      std::vector<double>(vres_opt.data(), vres_opt.data() + vres_opt.size()))
      ->setColorMap("reds");
  polyscope::registerSurfaceMesh("Target mesh", Vtarget, F)->setEnabled(false);
  polyscope::show();
  return 0;
}
