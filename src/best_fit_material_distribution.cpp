// best_fit_material_distribution.cpp
//
// Optimises the per-face distribution of two orthotropic materials so that the
// inflated membrane best fits the target geometry.
//
// Variable: phi_f ∈ ℝ per face, with t_f = sigmoid(phi_f).
//   t=0  →  material A (pattern 1)
//   t=1  →  material B (pattern 2)
//
// Objective = fit(phi) + lambda * smoothness(phi)
//   fit    = Σ_v ||v_sim(phi) - v_target||
//   smooth = Σ_{adjacent f,g} (phi_f - phi_g)²
//
// Gradient: finite differences for fit (one Newton solve per face),
//           analytical for the smoothness term.
// Solver:   LBFGSSolver from the optim library.

#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include <optim/LBFGS.h>

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Eigen;

// ── Globals ────────────────────────────────────────────────────────────────────
fsim::Mat3<double> V0;        // rest shape (= target: we try to recover it after inflation)
fsim::Mat3<int>    F;

std::vector<int>              bdrs;
std::vector<std::vector<int>> adjFaces;

// Simulation parameters (from genetic_algorithm.cpp / membrane_orthotropic.cpp)
const double stretch_factor = 1.05;
const double mass           = 30.0;    // kg/m²
const double pressure       = 250.0;  // Pa

// Material A: knit pattern 1 – wale along X, thinner
// Material B: knit pattern 2 – wale along Y, thicker
// E1/E2 chosen to mirror the original thin(0.40) / thick(0.60) stiffness ratio
// while introducing orthotropic character. Poisson from original calibration.
struct Material { double E1, E2, nu, thickness; Eigen::Vector3d dir; };
const Material matA = {50000.0, 30000.0, 0.38, 0.40, Eigen::Vector3d(1, 0, 0)};
const Material matB = {30000.0, 50000.0, 0.38, 0.60, Eigen::Vector3d(0, 1, 0)};

double lambda_smooth = 0.02;

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

std::vector<std::vector<int>> buildAdjFaces(const fsim::Mat3<int>& F)
{
  std::map<std::pair<int,int>, std::vector<int>> ef;
  for (int f = 0; f < F.rows(); ++f)
    for (int i = 0; i < 3; ++i) {
      int a = F(f,i), b = F(f,(i+1)%3);
      if (a > b) std::swap(a, b);
      ef[{a,b}].push_back(f);
    }
  std::vector<std::vector<int>> adj(F.rows());
  for (auto& [e, faces] : ef)
    if (faces.size() == 2) {
      adj[faces[0]].push_back(faces[1]);
      adj[faces[1]].push_back(faces[0]);
    }
  return adj;
}

// ── Simulation ────────────────────────────────────────────────────────────────
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double logit(double p)   { return std::log(p / (1.0 - p)); }

fsim::Mat3<double> simulate(const VectorXd& phi)
{
  int nF = F.rows();
  std::vector<double> E1s(nF), E2s(nF), nus(nF), thicknesses(nF);
  std::vector<Eigen::Vector3d> dirs(nF);

  for (int f = 0; f < nF; ++f) {
    double t  = sigmoid(phi(f));
    E1s[f]        = (1-t)*matA.E1        + t*matB.E1;
    E2s[f]        = (1-t)*matA.E2        + t*matB.E2;
    nus[f]        = (1-t)*matA.nu        + t*matB.nu;
    thicknesses[f]= (1-t)*matA.thickness + t*matB.thickness;
    dirs[f]       = ((1-t)*matA.dir      + t*matB.dir).normalized();
  }

  fsim::OrthotropicStVKMembrane model(
      V0 / stretch_factor, F, thicknesses, E1s, E2s, nus, dirs, mass, pressure);

  optim::NewtonSolver<double> solver;
  solver.options.display   = optim::SolverDisplay::quiet;
  solver.options.threshold = 1e-6;
  for (int b : bdrs) {
    solver.options.fixed_dofs.push_back(b*3);
    solver.options.fixed_dofs.push_back(b*3+1);
    solver.options.fixed_dofs.push_back(b*3+2);
  }
  solver.solve(model, Map<VectorXd>(V0.data(), V0.size()));
  return Map<fsim::Mat3<double>>(solver.var().data(), V0.rows(), 3);
}

// ── Objective ─────────────────────────────────────────────────────────────────
double fitLoss(const fsim::Mat3<double>& Vsim)
{
  // Target is V0: we want the inflated shape to recover the loaded geometry.
  return (Vsim - V0).rowwise().norm().sum();
}

double smoothLoss(const VectorXd& phi)
{
  double s = 0;
  for (int f = 0; f < F.rows(); ++f)
    for (int g : adjFaces[f])
      s += (phi(f) - phi(g)) * (phi(f) - phi(g));
  return 0.5 * s;
}

int lbfgs_iter = 0;

double totalObjective(const VectorXd& phi)
{
  return fitLoss(simulate(phi)) + lambda_smooth * smoothLoss(phi);
}

VectorXd totalGradient(const VectorXd& phi)
{
  int nF = F.rows();
  VectorXd grad(nF);

  // Finite-difference gradient of fit term (nF Newton solves)
  double f0 = fitLoss(simulate(phi));
  const double eps = 1e-4;
  VectorXd phiP = phi;
  for (int f = 0; f < nF; ++f) {
    phiP(f) += eps;
    grad(f) = (fitLoss(simulate(phiP)) - f0) / eps;
    phiP(f) = phi(f);
  }

  // Analytical gradient of smoothness term
  for (int f = 0; f < nF; ++f)
    for (int g : adjFaces[f])
      grad(f) += lambda_smooth * (phi(f) - phi(g));

  double totalObj = f0 + lambda_smooth * smoothLoss(phi);
  std::cout << "[iter " << lbfgs_iter++ << "]  fit=" << f0
            << "  smooth=" << smoothLoss(phi)
            << "  total=" << totalObj
            << "  |grad|=" << grad.norm() << "\n" << std::flush;
  return grad;
}

// ── Save result as OFF ────────────────────────────────────────────────────────
void saveOFF(const std::string& path, const fsim::Mat3<double>& V, const fsim::Mat3<int>& F)
{
  std::ofstream f(path);
  f << "OFF\n" << V.rows() << " " << F.rows() << " 0\n";
  for (int i = 0; i < V.rows(); ++i)
    f << V(i,0) << " " << V(i,1) << " " << V(i,2) << "\n";
  for (int i = 0; i < F.rows(); ++i)
    f << "3 " << F(i,0) << " " << F(i,1) << " " << F(i,2) << "\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
  fsim::readOFF("/Users/duch/Downloads/2part_opt.off", V0, F);
  int nF = F.rows();
  std::cout << "Mesh loaded: " << V0.rows() << " vertices, " << nF << " faces\n";

  bdrs     = findBoundaryVertices(F);
  adjFaces = buildAdjFaces(F);
  std::cout << "Boundary vertices: " << bdrs.size() << "\n";

  // ── Initial assignment from genetic_algorithm.cpp ─────────────────────────
  // Faces in faceIndices start as material B (t=1), rest as material A (t=0).
  std::vector<double> initT(nF, 0.0);
  int faceIndices[] = {1, 2, 3, 7, 9, 10, 11, 12, 16, 17, 18, 20, 21, 22, 24, 27, 28,
    30, 31, 34, 38, 39, 40, 41, 44, 46, 47, 48, 49, 52, 54, 56, 57, 58, 60, 61, 62, 64,
    65, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 81, 83, 85, 86, 87, 88, 89, 90, 91, 92,
    93, 96, 97, 98, 103, 104, 105, 106, 107, 112, 113, 114, 115, 117, 118, 119, 121, 122,
    123, 124, 125, 126, 128, 129, 130, 131, 134, 139, 140, 141, 142, 147, 149, 150, 152,
    154, 155, 156, 157, 158, 162, 163, 165, 168, 174, 175, 176, 178, 179, 181, 182, 183,
    186, 187, 188, 189, 192, 193, 195, 196, 201, 202, 203, 205, 206, 207, 208, 210, 211,
    212, 213, 214, 215, 220, 221, 223, 224, 225, 228, 231, 234, 236, 237, 238, 240, 242,
    250, 251, 258, 261, 271, 274, 276, 282, 284, 286, 288, 290, 291, 292, 293, 295, 296,
    297, 298, 299, 300, 301, 302, 304, 306, 307, 308, 309, 310, 311, 312, 313, 315, 316,
    318, 319, 320, 324, 326, 327, 328, 329, 333, 334, 335, 337, 338, 339, 341, 344, 345,
    347, 348, 351, 355, 356, 357, 358, 361, 363, 364, 365, 366, 369, 371, 373, 374, 375,
    377, 378, 379, 381, 382, 386, 387, 389, 390, 391, 392, 393, 394, 395, 396, 398, 400,
    402, 403, 404, 405, 406, 407, 408, 409, 410, 413, 414, 415, 420, 421, 422, 423, 424,
    429, 430, 431, 432, 434, 435, 436, 438, 439, 440, 441, 442, 443, 445, 446, 447, 448,
    451, 456, 457, 458, 459, 464, 466, 467, 469, 471, 472, 473, 474, 475, 479, 480, 482,
    485, 491, 492, 493, 495, 496, 498, 499, 500, 503, 504, 505, 506, 509, 510, 512, 513,
    518, 519, 520, 522, 523, 524, 525, 527, 528, 529, 530, 531, 532, 537, 538, 540, 541,
    542, 545, 548, 551, 553, 554, 555, 557, 559, 567, 568, 575, 578, 588, 591, 593, 599,
    601, 603, 605, 607, 608, 609, 610, 612, 613, 614, 615, 616, 617, 618, 619, 621, 623,
    624, 625, 626, 627, 628, 629, 630, 632, 633};
  for (int idx : faceIndices) initT[idx] = 1.0;

  // Soften to avoid logit(0)/logit(1) = ±∞
  VectorXd phi(nF);
  for (int f = 0; f < nF; ++f)
    phi(f) = logit(initT[f] == 0.0 ? 0.05 : 0.95);

  // ── Simulate initial assignment ───────────────────────────────────────────
  std::cout << "\n--- Initial simulation ---\n";
  fsim::Mat3<double> VsimInit = simulate(phi);
  double fitInit = fitLoss(VsimInit);
  std::cout << "Initial fit loss: " << fitInit << "\n\n";
  saveOFF("/Users/duch/Downloads/2part_result_initial.off", VsimInit, F);

  // ── L-BFGS optimisation ───────────────────────────────────────────────────
  // Each iteration requires nF Newton solves for the gradient.
  // With nF~634 and iteration_limit=5 this is ~3200 Newton solves.
  std::cout << "--- L-BFGS optimisation (iteration_limit=5) ---\n";
  optim::LBFGSSolver<double> lbfgs;
  lbfgs.options.threshold       = 1e-3;
  lbfgs.options.iteration_limit = 5;

  VectorXd phiOpt = lbfgs.solve(totalObjective, totalGradient, phi);

  // ── Final binary assignment and simulation ────────────────────────────────
  std::cout << "\n--- Final binary simulation ---\n";
  VectorXd phiBinary(nF);
  std::vector<double> tFinal(nF);
  for (int f = 0; f < nF; ++f) {
    tFinal[f]    = sigmoid(phiOpt(f)) > 0.5 ? 1.0 : 0.0;
    phiBinary(f) = logit(tFinal[f] == 0.0 ? 0.05 : 0.95);
  }
  fsim::Mat3<double> VsimFinal = simulate(phiBinary);
  double fitFinal = fitLoss(VsimFinal);
  std::cout << "Final fit loss (binary): " << fitFinal << "\n";
  std::cout << "Improvement: " << 100.0*(fitInit - fitFinal)/fitInit << "%\n";

  // Count how many faces changed assignment
  int changed = 0;
  for (int f = 0; f < nF; ++f)
    if (std::abs(tFinal[f] - initT[f]) > 0.5) changed++;
  std::cout << "Faces that changed material: " << changed << " / " << nF << "\n";

  saveOFF("/Users/duch/Downloads/2part_result_optimised.off", VsimFinal, F);
  std::cout << "\nResults saved to ~/Downloads/2part_result_initial.off"
               " and ~/Downloads/2part_result_optimised.off\n";
  return 0;
}
