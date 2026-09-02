// seam_imperfection.cpp
//
// How much does the transition between two knitted regions move the shape?
//
// Strategy E of Section 6.4.1 partitions the creased shell into three regions
// and gives each its own pair of stretch factors.  The model says nothing about
// what happens *at* a region boundary: the rest shape is a global least-squares
// fit (anisotropic_rest_shape.h), so the jump in rest metric is smoothed over one
// edge ring and the seam carries no stiffness of its own.  A real transition is a
// seam - a line of fabric that is bonded, cast off and rejoined, or knitted
// through - and it does carry stiffness.  It is also hard to characterise.
//
// This driver puts a non-sliding cable (edge_cable.h) on the region boundary and
// sweeps its axial stiffness EA, measuring how far the equilibrium moves from the
// seamless strategy-E solution.  The EA -> inf
// limit bounds the whole question: if a rigid seam line does not move the
// surface, no finite-width model of the transition can either.
//
// The rest length is read off V0_mod, the pre-strained rest shape the membrane
// itself uses, NOT off the equilibrium.  Referenced to the equilibrium the cable
// would sit exactly taut at zero tension and change nothing at any EA - a null
// experiment.  Referenced to V0_mod, rho = 1 means "the seam is knitted at the
// same rest length as the fabric it joins", which is the honest nominal.
//
// Three seams are tested:
//   R01|R2   the material discontinuity - R0/R1 share stretch factors, R2 differs
//   R0|R1    a partition boundary with no material jump - the control.  Any
//            movement here is pure added stiffness, not the discontinuity.
//   both
//
// Usage:  HEADLESS=1 ./seam_imperfection [out.csv]

#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/CompositeModel.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include "anisotropic_rest_shape.h"
#include "edge_cable.h"
#include "save_mesh.h"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace Eigen;

fsim::Mat3<double> V0, Vtarget;
fsim::Mat3<int>    F;
std::vector<int>   bdrs;
std::vector<Eigen::Vector3d> face_dirs;
std::vector<int>   face_region;
double E1, E2, nu, thickness, mass, pressure;
int    sim_count = 0;

// ── Mesh utilities (as in the strategy-E driver) ─────────────────────────────
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

// ── Region file (written by best_fit_stretch_factors_3region_adaptive) ───────
std::vector<int> loadRegions(const std::string& path, int nF)
{
  std::vector<int> reg(nF, -1);
  std::ifstream in(path);
  if (!in) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
  std::string line; int cur = -1;
  while (std::getline(in, line)) {
    if (line.rfind("REGION_", 0) == 0) { cur = std::stoi(line.substr(7, 1)); continue; }
    if (line.empty() || line[0] == '#') continue;
    reg[std::stoi(line)] = cur;
  }
  for (int f = 0; f < nF; ++f)
    if (reg[f] < 0) { std::cerr << "face " << f << " unassigned\n"; std::exit(1); }
  return reg;
}

// ── Seam extraction: edges whose two incident faces lie in different sets ────
std::vector<std::pair<int,int>> seamEdges(const std::set<int>& A, const std::set<int>& B)
{
  std::map<std::pair<int,int>, std::vector<int>> emap;
  for (int f = 0; f < F.rows(); ++f)
    for (int i = 0; i < 3; ++i) {
      int a = F(f,i), b = F(f,(i+1)%3);
      if (a > b) std::swap(a, b);
      emap[{a,b}].push_back(f);
    }
  std::vector<std::pair<int,int>> out;
  for (auto& [e, fs] : emap) {
    if (fs.size() != 2) continue;
    int ra = face_region[fs[0]], rb = face_region[fs[1]];
    bool ab = A.count(ra) && B.count(rb);
    bool ba = B.count(ra) && A.count(rb);
    if (ab || ba) out.push_back(e);
  }
  return out;
}

// ── Newton solve ─────────────────────────────────────────────────────────────
optim::SolverStatus last_status = optim::SolverStatus::uninitialized;

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
  last_status = solver.info();
  ++sim_count;
  return solver.var();
}

int main(int argc, char** argv)
{
  const std::string folder   = "data/2part/";
  const std::string mesh_ref = folder + "2part_opt_simu_m.off";
  const std::string reg_file = "out/sf_3region_adaptive_faces.txt";
  const std::string csv_path = (argc > 1) ? argv[1] : "out/seam_imperfection.csv";

  E1 = 5000.0; E2 = 12507.0; nu = 0.198; thickness = 1.0; mass = 0.001; pressure = 1000.0;

  // Strategy E converged optimum (R0 and R1 share their factors).
  const std::array<double,3> sf1 = {1.25722, 1.25722, 1.47318};
  const std::array<double,3> sf2 = {0.873827, 0.873827, 0.92008};

  fsim::readOFF(mesh_ref, V0, F);
  Vtarget = V0;                       // rest reference is the design target
  bdrs = findBoundaryVertices(F);
  face_dirs.assign(F.rows(), Eigen::Vector3d(0,1,0));
  projectFaceVectors(V0, F, face_dirs);
  face_region = loadRegions(reg_file, F.rows());

  std::vector<char> is_bdr(V0.rows(), 0);
  for (int b : bdrs) is_bdr[b] = 1;
  const int n_int = (int)std::count(is_bdr.begin(), is_bdr.end(), 0);

  std::cout << "Mesh " << V0.rows() << " v / " << F.rows() << " f, "
            << bdrs.size() << " boundary v, " << n_int << " interior v\n";
  std::cout << "Regions R0=" << std::count(face_region.begin(), face_region.end(), 0)
            << " R1="        << std::count(face_region.begin(), face_region.end(), 1)
            << " R2="        << std::count(face_region.begin(), face_region.end(), 2) << "\n";

  // ── Rest shape: identical to strategy E, and independent of the seam ───────
  const int nF = F.rows();
  std::vector<double> s1(nF), s2(nF);
  std::vector<double> E1s(nF,E1), E2s(nF,E2), nus(nF,nu), ths(nF,thickness);
  for (int f = 0; f < nF; ++f) {
    s1[f] = 1.0 / sf1[face_region[f]];
    s2[f] = 1.0 / sf2[face_region[f]];
  }
  fsim::Mat3<double> V0_mod = computeAnisotropicRestShape(V0, F, bdrs, face_dirs, s1, s2);

  // ── Seams ─────────────────────────────────────────────────────────────────
  auto seam_M    = seamEdges({0,1}, {2});   // material discontinuity
  auto seam_C    = seamEdges({0},   {1});   // control: no material jump
  auto seam_both = seam_M;
  seam_both.insert(seam_both.end(), seam_C.begin(), seam_C.end());

  auto restLen = [&](const std::vector<std::pair<int,int>>& segs) {
    double L = 0; for (auto& s : segs) L += (V0_mod.row(s.second)-V0_mod.row(s.first)).norm();
    return L;
  };
  auto tgtLen = [&](const std::vector<std::pair<int,int>>& segs) {
    double L = 0; for (auto& s : segs) L += (V0.row(s.second)-V0.row(s.first)).norm();
    return L;
  };
  std::cout << std::fixed << std::setprecision(4)
            << "Seam R01|R2 : " << seam_M.size() << " segments, rest " << restLen(seam_M)
            << " m, on target " << tgtLen(seam_M) << " m\n"
            << "Seam R0|R1  : " << seam_C.size() << " segments, rest " << restLen(seam_C)
            << " m, on target " << tgtLen(seam_C) << " m\n";

  // ── Baseline: no seam, i.e. strategy E itself ─────────────────────────────
  VectorXd x = Map<const VectorXd>(V0.data(), V0.size());
  for (double p : {pressure*0.01, pressure*0.1, pressure*0.5, pressure}) {
    fsim::OrthotropicStVKMembrane m(V0_mod, F, ths, E1s, E2s, nus, face_dirs, mass, p);
    x = newtonSolve(m, x);
  }
  const VectorXd x_base = x;
  auto Vb = Map<const fsim::Mat3<double>>(x_base.data(), V0.rows(), 3);
  const double base_fit_mean = (Vb - Vtarget).rowwise().norm().mean();
  const double base_fit_max  = (Vb - Vtarget).rowwise().norm().maxCoeff();
  const double base_crown    = Vb.col(2).maxCoeff();
  std::cout << std::setprecision(6)
            << "Baseline (no seam): fit mean " << base_fit_mean << " m, max "
            << base_fit_max << " m, crown " << base_crown << " m\n";
  saveMesh("out/seam_baseline.off", Vb, F);

  // How stretched is each seam at the baseline, against its V0_mod rest length?
  for (auto [nm, segs] : {std::pair<const char*, std::vector<std::pair<int,int>>*>
                          {"R01|R2", &seam_M}, {"R0|R1", &seam_C}}) {
    int taut = 0; double smin = 1e9, smax = -1e9, smean = 0;
    for (auto& s : *segs) {
      double l0 = (V0_mod.row(s.second)-V0_mod.row(s.first)).norm();
      double l  = (x_base.segment<3>(3*s.second) - x_base.segment<3>(3*s.first)).norm();
      double r = l / l0; smin = std::min(smin,r); smax = std::max(smax,r); smean += r;
      if (r > 1.0) ++taut;
    }
    smean /= segs->size();
    std::cout << "  " << nm << " baseline stretch l/L0: mean " << smean
              << "  range [" << smin << ", " << smax << "]  taut "
              << taut << "/" << segs->size() << "\n";
  }

  // ── Sweep ─────────────────────────────────────────────────────────────────
  std::ofstream csv(csv_path);
  csv << "seam,n_seg,EA,rho,tension_only,L_pos_rms,L_pos_max,fit_mean,fit_max,"
         "crown,d_crown,T_max,T_sum,n_taut,stretch_mean,solves,status\n";
  csv << std::setprecision(10);

  std::vector<double> EAs;
  for (double e : {0.1,0.3,1.0,3.0,10.0,30.0,100.0,300.0,1e3,3e3,1e4,3e4,1e5,3e5,1e6,1e7})
    EAs.push_back(e);
  auto statusName = [](optim::SolverStatus st) {
    switch (st) {
      case optim::SolverStatus::success:                 return "success";
      case optim::SolverStatus::line_search_failed:      return "line_search_failed";
      case optim::SolverStatus::wrong_descent_direction: return "wrong_descent";
      case optim::SolverStatus::regularization_failed:   return "regularization_failed";
      case optim::SolverStatus::iteration_overflow:      return "iteration_overflow";
      case optim::SolverStatus::NaN_error:               return "nan";
      default:                                           return "uninitialized";
    }
  };

  // EA is swept upward with the previous solution as the warm start.  A near-rigid
  // seam is a stiff constraint: started cold from the seamless equilibrium the
  // Newton solve gives up (regularization_failed) and hands back its own input,
  // which reads as "no effect" when it is in fact "no answer".  Continuation in EA
  // keeps every step small enough to converge.
  auto run = [&](const char* name, const std::vector<std::pair<int,int>>& segs,
                 double EA, double rho, bool tens_only, const VectorXd& x_start,
                 VectorXd& x_out)
  {
    EdgeCable cable(segs, EA, V0_mod, rho, tens_only);
    fsim::OrthotropicStVKMembrane membrane(V0_mod, F, ths, E1s, E2s, nus,
                                           face_dirs, mass, pressure);
    fsim::CompositeModel model(std::move(membrane), EdgeCable(cable));
    int s0 = sim_count;
    VectorXd xr = newtonSolve(model, x_start);
    const char* st = statusName(last_status);
    x_out = xr;
    auto Vr = Map<const fsim::Mat3<double>>(xr.data(), V0.rows(), 3);

    // Displacement from the seamless baseline, interior vertices only: the
    // boundary is clamped and averaging it in would deflate every figure.
    double ss = 0.0, mx = 0.0;
    for (int v = 0; v < V0.rows(); ++v) {
      if (is_bdr[v]) continue;
      double d = (Vr.row(v) - Vb.row(v)).norm();
      ss += d*d; mx = std::max(mx, d);
    }
    double rms = std::sqrt(ss / n_int);

    double T_max, T_sum, stretch_mean; int n_taut;
    cable.report(xr, T_max, T_sum, n_taut, stretch_mean);

    double crown = Vr.col(2).maxCoeff();
    csv << name << ',' << segs.size() << ',' << EA << ',' << rho << ','
        << (tens_only ? 1 : 0) << ',' << rms << ',' << mx << ','
        << (Vr - Vtarget).rowwise().norm().mean() << ','
        << (Vr - Vtarget).rowwise().norm().maxCoeff() << ','
        << crown << ',' << (crown - base_crown) << ','
        << T_max << ',' << T_sum << ',' << n_taut << ',' << stretch_mean << ','
        << (sim_count - s0) << ',' << st << '\n';
    csv.flush();
    return std::pair<double,const char*>{rms, st};
  };

  struct SeamDef { const char* name; std::vector<std::pair<int,int>>* segs; };
  std::vector<SeamDef> seams = {{"R01|R2", &seam_M}, {"R0|R1", &seam_C},
                                {"both", &seam_both}};

  auto t0 = std::chrono::steady_clock::now();

  // Stiffness sweep at rho = 1, both slack models, with EA continuation
  std::cout << "\n── EA sweep at rho = 1 ──\n";
  for (auto& sd : seams)
    for (bool to : {true, false}) {
      VectorXd x_prev = x_base, x_cur;
      for (double EA : EAs) {
        auto [r, st] = run(sd.name, *sd.segs, EA, 1.0, to, x_prev, x_cur);
        if (std::string(st) == "success" || std::string(st) == "line_search_failed")
          x_prev = x_cur;                       // only continue from a usable state
        std::cout << "  " << std::setw(7) << sd.name << "  EA=" << std::setw(9)
                  << std::scientific << std::setprecision(1) << EA
                  << (to ? "  tension-only" : "  two-sided  ")
                  << "  L_pos = " << std::fixed << std::setw(7) << std::setprecision(3)
                  << r*1000 << " mm   " << st << "\n" << std::flush;
      }
    }

  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  std::cout << "\n" << sim_count << " Newton solves, " << std::setprecision(1)
            << std::fixed << secs << " s\n  wrote " << csv_path << "\n";

  // ── Two shape dumps for the figure: rigid seam, each seam alone ───────────
  for (auto& sd : seams) {
    VectorXd xr = x_base;
    for (double EA : EAs) {
      EdgeCable cable(*sd.segs, EA, V0_mod, 1.0, false);
      fsim::OrthotropicStVKMembrane membrane(V0_mod, F, ths, E1s, E2s, nus,
                                             face_dirs, mass, pressure);
      fsim::CompositeModel model(std::move(membrane), EdgeCable(cable));
      VectorXd xn = newtonSolve(model, xr);
      if (last_status == optim::SolverStatus::regularization_failed ||
          last_status == optim::SolverStatus::NaN_error) break;
      xr = xn;
    }
    std::string nm(sd.name);
    for (auto& c : nm) if (c == '|') c = '_';
    saveMesh("out/seam_rigid_" + nm + ".off",
             Map<const fsim::Mat3<double>>(xr.data(), V0.rows(), 3), F);
  }
  return 0;
}
