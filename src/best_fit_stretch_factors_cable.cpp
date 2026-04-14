// best_fit_stretch_factors_cable.cpp
//
// Same as best_fit_stretch_factors.cpp but the forward simulation also includes
// a stiffening cable (a chain of springs) built from cable_indices.
//
//   min_{s_f1, s_f2}  Σ_v || V_sim_v(s_f1, s_f2) − V_target_v ||
//
// The cable contributes to equilibrium via CompositeModel<membrane, cable>.
// Only the two stretch factors are optimised; cable_EA is a fixed parameter.

#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/CompositeModel.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include <optim/LBFGS.h>
#include "anisotropic_rest_shape.h"
#include "sliding_cable.h"

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
double cable_EA;
std::vector<int> cable_indices;

int sim_count = 0;
VectorXd warm_start;

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

// ── Cable builder ─────────────────────────────────────────────────────────────
SlidingCable buildCable()
{
  return SlidingCable(cable_indices, cable_EA, V0);
}

// ── Newton solve helpers ──────────────────────────────────────────────────────
// Solver options (fixed DOFs) are common to all solves.
template <class Model>
VectorXd newtonSolve(Model& model, const VectorXd& x0)
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
fsim::Mat3<double> simulateImpl(double sf1, double sf2, bool update_warm)
{
  const int nF = F.rows();
  std::vector<double> s1(nF, 1.0/sf1), s2(nF, 1.0/sf2);
  std::vector<double> E1s(nF, E1), E2s(nF, E2), nus(nF, nu), ths(nF, thickness);

  fsim::Mat3<double> V0_mod =
      computeAnisotropicRestShape(V0, F, bdrs, face_dirs, s1, s2);

  if (warm_start.size() == 0) {
    std::cout << "  [cold start] ramping pressure to " << pressure << " Pa...\n" << std::flush;
    VectorXd x = Map<const VectorXd>(V0.data(), V0.size());
    for (double p : {pressure * 0.01, pressure * 0.1, pressure * 0.5, pressure}) {
      fsim::OrthotropicStVKMembrane m(V0_mod, F, ths, E1s, E2s, nus, face_dirs, mass, p);
      fsim::CompositeModel cm(std::move(m), buildCable());
      x = newtonSolve(cm, x);
    }
    warm_start = x;
    std::cout << "  [cold start] done.\n" << std::flush;
  }

  fsim::OrthotropicStVKMembrane membrane(
      V0_mod, F, ths, E1s, E2s, nus, face_dirs, mass, pressure);
  fsim::CompositeModel composite(std::move(membrane), buildCable());

  VectorXd result = newtonSolve(composite, warm_start);
  ++sim_count;
  if (update_warm)
    warm_start = result;
  return Map<fsim::Mat3<double>>(result.data(), V0.rows(), 3);
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
  return fitLoss(simulateImpl(std::exp(phi(0)), std::exp(phi(1)), /*update_warm=*/false));
}

VectorXd gradient(const VectorXd& phi)
{
  const double eps = 1e-4;
  double f0 = fitLoss(simulateImpl(std::exp(phi(0)), std::exp(phi(1)), /*update_warm=*/true));

  VectorXd grad(2);
  for (int i = 0; i < 2; ++i) {
    VectorXd phiP = phi;
    phiP(i) += eps;
    grad(i) = (fitLoss(simulateImpl(std::exp(phiP(0)), std::exp(phiP(1)), /*update_warm=*/false)) - f0) / eps;
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
  const std::string mesh_ref    = folder + "2part_opt_simu_m.off";
  const std::string mesh_target = folder + "2part_opt_simu_m.off";   // change to target!
  const std::string out_dir     = "/Users/duch/Downloads/";

  E1        = 5000.0;    // N/m, wale
  E2        = 12507.0;   // N/m, course
  nu        = 0.198;
  thickness = 1.0;
  mass      = 0.001;     // kg/m²
  pressure  = 1000.0;    // Pa

  cable_EA = 157000.0;   // N  (1mm diameter steel cable, EA = E*A)

  cable_indices = { 313, 100, 163, 158, 169, 170, 166, 334, 65,
                    167, 165, 333,  98,  99, 149, 336, 79, 130, 327 };

  // Initial guess for stretch factors
  double sf1_init = 1.043;
  double sf2_init = 1.043;
  // ───────────────────────────────────────────────────────────────────────────

  fsim::readOFF(mesh_ref,    V0,      F);
  fsim::readOFF(mesh_target, Vtarget, F);

  if (V0.rows() != Vtarget.rows()) {
    std::cerr << "Error: reference and target meshes have different vertex counts.\n";
    return 1;
  }

  std::cout << "Mesh: " << V0.rows() << " vertices, " << F.rows() << " faces\n";
  std::cout << "Cable: " << cable_indices.size() << " vertices, "
            << cable_indices.size()-1 << " segments, EA=" << cable_EA << " N\n";

  bdrs = findBoundaryVertices(F);
  std::cout << "Boundary vertices: " << bdrs.size() << "\n";

  face_dirs.assign(F.rows(), Eigen::Vector3d(0.0, 1.0, 0.0));
  projectFaceVectors(V0, F, face_dirs);

  // ── Initial simulation ─────────────────────────────────────────────────────
  std::cout << "\n--- Initial simulation (sf1=" << sf1_init
            << ", sf2=" << sf2_init << ") ---\n";
  fsim::Mat3<double> Vsim_init = simulateImpl(sf1_init, sf2_init, /*update_warm=*/true);
  double loss_init = fitLoss(Vsim_init);
  double max_init  = (Vsim_init - Vtarget).rowwise().norm().maxCoeff();
  std::cout << "Initial mean vertex distance: " << loss_init << " m\n";
  std::cout << "Initial max  vertex distance: " << max_init  << " m\n\n";
  saveOFF(out_dir + "sf_cable_opt_initial.off", Vsim_init, F);

  // ── L-BFGS optimisation ────────────────────────────────────────────────────
  VectorXd phi(2);
  phi(0) = std::log(sf1_init);
  phi(1) = std::log(sf2_init);

  std::cout << "--- L-BFGS optimisation ---\n";
  optim::LBFGSSolver<double> lbfgs;
  lbfgs.options.threshold       = 1e-4;
  lbfgs.options.iteration_limit = 30;

  VectorXd phi_opt = lbfgs.solve(objective, gradient, phi);

  double sf1_opt = std::exp(phi_opt(0));
  double sf2_opt = std::exp(phi_opt(1));

  // ── Final result ───────────────────────────────────────────────────────────
  std::cout << "\n--- Optimal parameters ---\n";
  std::cout << "  sf1 (wale)   = " << sf1_opt << "\n";
  std::cout << "  sf2 (course) = " << sf2_opt << "\n";

  fsim::Mat3<double> Vsim_opt = simulateImpl(sf1_opt, sf2_opt, /*update_warm=*/true);
  double loss_opt = fitLoss(Vsim_opt);
  double max_opt  = (Vsim_opt - Vtarget).rowwise().norm().maxCoeff();
  std::cout << "  mean vertex distance: " << loss_opt << " m"
            << "  (was " << loss_init << " m, "
            << std::fixed << std::setprecision(2)
            << 100.0*(loss_init - loss_opt)/loss_init << "% improvement)\n";
  std::cout << std::defaultfloat
            << "  max  vertex distance: " << max_opt << " m"
            << "  (was " << max_init << " m)\n";

  saveOFF(out_dir + "sf_cable_opt_result.off", Vsim_opt, F);

  std::cout << "\nTotal Newton solves: " << sim_count << "\n";
  return 0;
}
