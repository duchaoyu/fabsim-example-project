// parameter_study_circle.cpp
//
// Parameter study: flat circle with fixed boundary, orthotropic knit material 1.
// Varies either inflation pressure or boundary pre-tension (stretch factor).
// Each simulation result is exported as an OFF file.
// A summary CSV is written with key deformation metrics.
//
// Material 1 parameters taken from membrane_orthotropic.cpp:
//   E1 = 5000  N/m   (wale direction, along X)
//   E2 = 12507 N/m   (course direction, along Y)
//   nu = 0.198
//   thickness = 1.0 m
//   mass = 0.001 kg/m²

#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>

using namespace Eigen;

// ── Circular mesh generation ──────────────────────────────────────────────────
// Produces a flat unit-disk mesh centred at the origin.
// N_spokes: number of vertices on each ring (including boundary)
// N_rings:  number of concentric rings (boundary is the outermost)
void generateCircle(double R, int N_spokes, int N_rings,
                    fsim::Mat3<double>& V, fsim::Mat3<int>& F)
{
  std::vector<Vector3d>  verts;
  std::vector<Vector3i>  faces;

  // Centre vertex
  verts.push_back(Vector3d(0, 0, 0));

  // Rings 1 … N_rings
  for (int ring = 1; ring <= N_rings; ++ring) {
    double r = R * double(ring) / double(N_rings);
    for (int j = 0; j < N_spokes; ++j) {
      double theta = 2.0 * M_PI * j / N_spokes;
      verts.push_back(Vector3d(r * std::cos(theta), r * std::sin(theta), 0.0));
    }
  }

  // Fan from centre to ring 1
  for (int j = 0; j < N_spokes; ++j) {
    int v1 = 1 + j;
    int v2 = 1 + (j + 1) % N_spokes;
    faces.push_back(Vector3i(0, v1, v2));
  }

  // Quad strips between consecutive rings
  for (int ring = 1; ring < N_rings; ++ring) {
    int base1 = 1 + (ring - 1) * N_spokes;
    int base2 = 1 + ring       * N_spokes;
    for (int j = 0; j < N_spokes; ++j) {
      int j1 = (j + 1) % N_spokes;
      faces.push_back(Vector3i(base1 + j, base2 + j,  base1 + j1));
      faces.push_back(Vector3i(base2 + j, base2 + j1, base1 + j1));
    }
  }

  V.resize(verts.size(), 3);
  F.resize(faces.size(), 3);
  for (int i = 0; i < (int)verts.size(); ++i)  V.row(i) = verts[i];
  for (int i = 0; i < (int)faces.size(); ++i)  F.row(i) = faces[i];
}

// ── Boundary detection ────────────────────────────────────────────────────────
// For the generated circle, the outermost ring is the boundary.
// N_spokes vertices starting at index (1 + (N_rings-1)*N_spokes).
std::vector<int> outerRing(int N_spokes, int N_rings)
{
  std::vector<int> bdrs;
  int base = 1 + (N_rings - 1) * N_spokes;
  for (int j = 0; j < N_spokes; ++j) bdrs.push_back(base + j);
  return bdrs;
}

// ── OFF writer ────────────────────────────────────────────────────────────────
void saveOFF(const std::string& path,
             const fsim::Mat3<double>& V, const fsim::Mat3<int>& F)
{
  std::ofstream out(path);
  out << "OFF\n" << V.rows() << " " << F.rows() << " 0\n";
  out << std::fixed << std::setprecision(6);
  for (int i = 0; i < V.rows(); ++i)
    out << V(i,0) << " " << V(i,1) << " " << V(i,2) << "\n";
  for (int i = 0; i < F.rows(); ++i)
    out << "3 " << F(i,0) << " " << F(i,1) << " " << F(i,2) << "\n";
  std::cout << "  saved: " << path << "\n";
}

// ── Forward simulation ────────────────────────────────────────────────────────
fsim::Mat3<double> simulate(const fsim::Mat3<double>& V0,
                            const fsim::Mat3<int>&    F,
                            const std::vector<int>&   bdrs,
                            double E1, double E2, double nu,
                            double thickness, double stretch_factor,
                            double mass, double pressure,
                            const std::vector<Eigen::Vector3d>& face_dirs)
{
  int nF = F.rows();
  std::vector<double> E1s(nF, E1), E2s(nF, E2), nus(nF, nu);
  std::vector<double> thicknesses(nF, thickness);

  fsim::OrthotropicStVKMembrane model(
      V0 / stretch_factor, F, thicknesses, E1s, E2s, nus, face_dirs, mass, pressure);

  optim::NewtonSolver<double> solver;
  solver.options.display   = optim::SolverDisplay::quiet;
  solver.options.threshold = 1e-8;
  for (int b : bdrs) {
    solver.options.fixed_dofs.push_back(b * 3);
    solver.options.fixed_dofs.push_back(b * 3 + 1);
    solver.options.fixed_dofs.push_back(b * 3 + 2);
  }

  VectorXd x0 = Map<const VectorXd>(V0.data(), V0.size());
  solver.solve(model, x0);
  return Map<fsim::Mat3<double>>(solver.var().data(), V0.rows(), 3);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
  // ── Mesh ──────────────────────────────────────────────────────────────────
  const double R        = 0.5;   // circle radius [m]
  const int    N_spokes = 24;    // vertices per ring
  const int    N_rings  = 6;     // concentric rings (outermost = boundary)

  fsim::Mat3<double> V0;
  fsim::Mat3<int>    F;
  generateCircle(R, N_spokes, N_rings, V0, F);

  std::vector<int> bdrs = outerRing(N_spokes, N_rings);
  int nV = V0.rows(), nF = F.rows();
  std::cout << "Circle mesh: " << nV << " vertices, " << nF << " faces, "
            << bdrs.size() << " boundary vertices\n\n";

  // Save the flat template mesh
  const std::string outDir = "/Users/duch/Downloads/param_study_circle/";
  // (create the directory manually or change outDir to an existing path)
  saveOFF(outDir + "circle_template.off", V0, F);

  // ── Material 1 (from membrane_orthotropic.cpp) ─────────────────────────────
  const double E1            = 5000.0;   // N/m, wale direction (X)
  const double E2            = 12507.0;  // N/m, course direction (Y)
  const double nu            = 0.198;
  const double thickness     = 1.0;      // m
  const double stretch_factor = 1.043;
  const double mass          = 0.001;    // kg/m²

  // Uniform knitting direction: wale along X for all faces
  std::vector<Eigen::Vector3d> face_dirs(nF, Eigen::Vector3d(1.0, 0.0, 0.0));

  // ── Study 1: vary pressure ────────────────────────────────────────────────
  // Uncomment Study 2 and comment Study 1 to study pre-tension instead.
  std::vector<double> pressures = {300, 500, 700, 900, 1100, 1300, 1500, 1800, 2100, 2500};

  std::cout << "=== Study 1: pressure sweep (stretch_factor=" << stretch_factor << ") ===\n";
  std::cout << std::left
            << std::setw(12) << "pressure"
            << std::setw(14) << "center_z [m]"
            << std::setw(14) << "max_z [m]"
            << std::setw(16) << "avg_z_inner [m]"
            << std::setw(14) << "dome_ratio"
            << "\n"
            << std::string(70, '-') << "\n";

  std::ofstream csv(outDir + "pressure_study.csv");
  csv << "pressure_Pa,center_z_m,max_z_m,avg_z_inner_m,dome_ratio\n";

  for (double p : pressures) {
    fsim::Mat3<double> Vsim = simulate(V0, F, bdrs, E1, E2, nu, thickness,
                                       stretch_factor, mass, p, face_dirs);

    // Deformation metrics (Z column = column 2)
    double center_z   = Vsim(0, 2);                          // centre vertex
    double max_z      = Vsim.col(2).maxCoeff();
    double avg_z      = (Vsim.col(2).sum() - Vsim(0,2) * bdrs.size()) / (nV - (int)bdrs.size());
    // dome_ratio: centre rise / circle radius
    double dome_ratio = center_z / R;

    std::cout << std::left << std::fixed << std::setprecision(4)
              << std::setw(12) << p
              << std::setw(14) << center_z
              << std::setw(14) << max_z
              << std::setw(16) << avg_z
              << std::setw(14) << dome_ratio
              << "\n";

    csv << std::fixed << std::setprecision(6)
        << p << "," << center_z << "," << max_z << ","
        << avg_z << "," << dome_ratio << "\n";

    std::ostringstream fname;
    fname << outDir << "circle_p" << (int)p << ".off";
    saveOFF(fname.str(), Vsim, F);
  }
  csv.close();

  // ── Study 2: vary stretch factor (pre-tension) — uncomment to run ──────────
  /*
  const double fixed_pressure = 1200.0;
  std::vector<double> stretch_factors = {1.00, 1.01, 1.02, 1.03, 1.04, 1.05,
                                          1.06, 1.07, 1.08, 1.09, 1.10};

  std::cout << "\n=== Study 2: pre-tension sweep (pressure=" << fixed_pressure << " Pa) ===\n";
  std::ofstream csv2(outDir + "pretension_study.csv");
  csv2 << "stretch_factor,center_z_m,max_z_m,dome_ratio\n";

  for (double sf : stretch_factors) {
    fsim::Mat3<double> Vsim = simulate(V0, F, bdrs, E1, E2, nu, thickness,
                                        sf, mass, fixed_pressure, face_dirs);
    double center_z   = Vsim(0, 2);
    double max_z      = Vsim.col(2).maxCoeff();
    double dome_ratio = center_z / R;

    std::cout << "sf=" << sf << "  center_z=" << center_z
              << "  max_z=" << max_z << "\n";
    csv2 << sf << "," << center_z << "," << max_z << "," << dome_ratio << "\n";

    std::ostringstream fname;
    fname << outDir << "circle_sf" << std::fixed << std::setprecision(3) << sf << ".off";
    saveOFF(fname.str(), Vsim, F);
  }
  csv2.close();
  */

  std::cout << "\nDone. Results in: " << outDir << "\n";
  return 0;
}
