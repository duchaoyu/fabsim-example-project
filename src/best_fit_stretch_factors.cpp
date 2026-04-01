// best_fit_stretch_factors.cpp
//
// Finds the two anisotropic pre-strain factors (s_f1 = wale, s_f2 = course)
// that minimise the vertex-wise distance between the simulated inflated shape
// and a target geometry:
//
//   min_{s_f1, s_f2}  Σ_v || V_sim_v(s_f1, s_f2) − V_target_v ||
//
// V_sim is obtained by:
//   1. Computing V0_mod = computeAnisotropicRestShape(V0, F, bdrs, dirs, 1/s_f1, 1/s_f2)
//   2. Solving the static equilibrium with OrthotropicStVKMembrane(V0_mod, ...)
//      and boundary vertices fixed at V0 positions.
//
// Gradient: finite differences — only 2 extra Newton solves per L-BFGS step.
//
// Material parameters mirror membrane_orthotropic.cpp (material 1).
// Edit the "── Configuration ──" block at the top of main() as needed.

#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include <optim/LBFGS.h>
#include "anisotropic_rest_shape.h"

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <vector>

using namespace Eigen;

// ── Globals ───────────────────────────────────────────────────────────────────
fsim::Mat3<double> V0;
fsim::Mat3<int>    F;
fsim::Mat3<double> Vtarget;

std::vector<int>             bdrs;
std::vector<Eigen::Vector3d> face_dirs;

double E1, E2, nu, thickness, mass, pressure;

int sim_count = 0;   // total Newton solves (for progress reporting)

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

// ── Forward simulation ────────────────────────────────────────────────────────
fsim::Mat3<double> simulate(double sf1, double sf2)
{
  const int nF = F.rows();
  std::vector<double> s1(nF, 1.0/sf1), s2(nF, 1.0/sf2);
  std::vector<double> E1s(nF, E1), E2s(nF, E2), nus(nF, nu), ths(nF, thickness);

  fsim::Mat3<double> V0_mod =
      computeAnisotropicRestShape(V0, F, bdrs, face_dirs, s1, s2);

  fsim::OrthotropicStVKMembrane model(V0_mod, F, ths, E1s, E2s, nus, face_dirs, mass, pressure);

  optim::NewtonSolver<double> solver;
  solver.options.display   = optim::SolverDisplay::quiet;
  solver.options.threshold = 1e-6;
  for (int b : bdrs) {
    solver.options.fixed_dofs.push_back(b*3);
    solver.options.fixed_dofs.push_back(b*3+1);
    solver.options.fixed_dofs.push_back(b*3+2);
  }

  VectorXd x0 = Map<const VectorXd>(V0.data(), V0.size());
  solver.solve(model, x0);
  ++sim_count;
  return Map<fsim::Mat3<double>>(solver.var().data(), V0.rows(), 3);
}

// ── Fit loss: mean per-vertex distance ────────────────────────────────────────
double fitLoss(const fsim::Mat3<double>& Vsim)
{
  return (Vsim - Vtarget).rowwise().norm().mean();
}

// ── L-BFGS objective and gradient ─────────────────────────────────────────────
// Variables: phi = (log(sf1), log(sf2))  →  sf_i = exp(phi_i) > 0 always.

int lbfgs_iter = 0;

double objective(const VectorXd& phi)
{
  return fitLoss(simulate(std::exp(phi(0)), std::exp(phi(1))));
}

VectorXd gradient(const VectorXd& phi)
{
  const double eps = 1e-4;
  double f0 = fitLoss(simulate(std::exp(phi(0)), std::exp(phi(1))));

  VectorXd grad(2);
  for (int i = 0; i < 2; ++i) {
    VectorXd phiP = phi;
    phiP(i) += eps;
    grad(i) = (fitLoss(simulate(std::exp(phiP(0)), std::exp(phiP(1)))) - f0) / eps;
  }

  std::cout << "[iter " << lbfgs_iter++ << "]"
            << "  sf1=" << std::fixed << std::setprecision(5) << std::exp(phi(0))
            << "  sf2=" << std::exp(phi(1))
            << "  loss=" << std::setprecision(6) << f0
            << "  |grad|=" << grad.norm()
            << "  (total Newton solves: " << sim_count << ")\n" << std::flush;
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
  const std::string mesh_ref    = folder + "2part_opt_simu_m.off";   // V0: reference/flat
  const std::string mesh_target = folder + "2part_opt_simu_m.off";   // V_target: change this!
  const std::string out_dir     = "/Users/duch/Downloads/";

  E1        = 5000.0;   // N/m, wale
  E2        = 12507.0;  // N/m, course
  nu        = 0.198;
  thickness = 1.0;
  mass      = 0.001;    // kg/m²
  pressure  = 1000.0;   // Pa — match membrane_orthotropic.cpp

  // Initial guess for stretch factors
  double sf1_init = 1.043;
  double sf2_init = 1.043;
  // ───────────────────────────────────────────────────────────────────────────

  fsim::readOFF(mesh_ref,    V0,      F);
  fsim::readOFF(mesh_target, Vtarget, F);

  if (V0.rows() != Vtarget.rows() || F.rows() != Vtarget.rows()) {
    // Only check vertex count — face topology assumed identical
    if (V0.rows() != Vtarget.rows()) {
      std::cerr << "Error: reference and target meshes have different vertex counts.\n";
      return 1;
    }
  }

  std::cout << "Mesh: " << V0.rows() << " vertices, " << F.rows() << " faces\n";

  bdrs = findBoundaryVertices(F);
  std::cout << "Boundary vertices: " << bdrs.size() << "\n";

  face_dirs.assign(F.rows(), Eigen::Vector3d(0.0, 1.0, 0.0));
  projectFaceVectors(V0, F, face_dirs);

  // ── Initial simulation ─────────────────────────────────────────────────────
  std::cout << "\n--- Initial simulation (sf1=" << sf1_init
            << ", sf2=" << sf2_init << ") ---\n";
  fsim::Mat3<double> Vsim_init = simulate(sf1_init, sf2_init);
  double loss_init = fitLoss(Vsim_init);
  std::cout << "Initial mean vertex distance: " << loss_init << " m\n\n";
  saveOFF(out_dir + "sf_opt_initial.off", Vsim_init, F);

  // ── L-BFGS optimisation ────────────────────────────────────────────────────
  // phi = (log(sf1), log(sf2))  so that sf_i = exp(phi_i) is always positive.
  VectorXd phi(2);
  phi(0) = std::log(sf1_init);
  phi(1) = std::log(sf2_init);

  std::cout << "--- L-BFGS optimisation ---\n";
  optim::LBFGSSolver<double> lbfgs;
  lbfgs.options.threshold       = 1e-5;
  lbfgs.options.iteration_limit = 50;

  VectorXd phi_opt = lbfgs.solve(objective, gradient, phi);

  double sf1_opt = std::exp(phi_opt(0));
  double sf2_opt = std::exp(phi_opt(1));

  // ── Final result ───────────────────────────────────────────────────────────
  std::cout << "\n--- Optimal parameters ---\n";
  std::cout << "  sf1 (wale)   = " << sf1_opt << "\n";
  std::cout << "  sf2 (course) = " << sf2_opt << "\n";

  fsim::Mat3<double> Vsim_opt = simulate(sf1_opt, sf2_opt);
  double loss_opt = fitLoss(Vsim_opt);
  std::cout << "  mean vertex distance: " << loss_opt << " m"
            << "  (was " << loss_init << " m, "
            << std::fixed << std::setprecision(2)
            << 100.0*(loss_init - loss_opt)/loss_init << "% improvement)\n";

  saveOFF(out_dir + "sf_opt_result.off", Vsim_opt, F);

  std::cout << "\nTotal Newton solves: " << sim_count << "\n";
  return 0;
}
