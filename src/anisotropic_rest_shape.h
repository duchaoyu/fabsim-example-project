// anisotropic_rest_shape.h
//
// Computes a non-uniformly pre-strained rest shape V_mod from a reference mesh V0.
//
// Each face f has its own anisotropic deformation gradient:
//
//   A_f(e) = s1_f * (d1_f · e) * d1_f    ← wale direction
//           + s2_f * (d2_f · e) * d2_f   ← course direction
//           + sn_f * (n_f · e) * n_f     ← normal direction  (sn = (s1+s2)/2)
//
// The normal-direction scale sn = (s1+s2)/2 ensures that when s1=s2=1/sf,
// A_f(e) = e/sf for every edge direction, reproducing the old "V0/stretch_factor"
// behaviour exactly on any mesh geometry (flat or curved).
//
// We find V_mod that minimises the global least-squares problem:
//
//   min Σ_f  Σ_{(i,j) edge of f}  || (V_mod_i − V_mod_j) − A_f*(V_i − V_j) ||²
//   subject to V_mod_i = A_f_avg(V_i) for all boundary vertices
//              (A_f averaged over adjacent faces, applied to position from origin)
//
// The normal equations reduce to a sparse symmetric positive-definite system
// (the combinatorial Laplacian restricted to interior vertices) which is solved
// with Eigen's SimplicialLDLT.
//
// Usage:
//   #include "anisotropic_rest_shape.h"
//
//   std::vector<double> s1(F.rows(), 1.0/stretch_wale);
//   std::vector<double> s2(F.rows(), 1.0/stretch_course);
//
//   fsim::Mat3<double> V0_mod = computeAnisotropicRestShape(V0, F, bdrs, face_dirs, s1, s2);

#pragma once

#include <fsim/util/typedefs.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <set>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// computeAnisotropicRestShape
//
//  V0        : nV×3 reference vertex positions
//  F         : nF×3 face indices
//  bdrs      : boundary vertex indices
//  face_dirs : per-face wale direction vectors (need not be unit or in-plane;
//              they are projected and normalised internally)
//  s1        : per-face scale along the wale direction  (< 1 → compression)
//  s2        : per-face scale along the course direction
//              (s1=s2=1/stretch_factor reproduces the uniform isotropic case)
//
// Returns V_mod (nV×3).
// ─────────────────────────────────────────────────────────────────────────────
inline fsim::Mat3<double> computeAnisotropicRestShape(
    const fsim::Mat3<double>&           V0,
    const fsim::Mat3<int>&              F,
    const std::vector<int>&             bdrs,
    const std::vector<Eigen::Vector3d>& face_dirs,
    const std::vector<double>&          s1,
    const std::vector<double>&          s2)
{
  using namespace Eigen;

  const int nV = V0.rows();
  const int nF = F.rows();

  // ── 1. Build interior vertex map ──────────────────────────────────────────
  std::set<int> bdrSet(bdrs.begin(), bdrs.end());

  std::vector<int> globalToInt(nV, -1);
  std::vector<int> intToGlobal;
  intToGlobal.reserve(nV);
  for (int i = 0; i < nV; ++i)
    if (!bdrSet.count(i)) {
      globalToInt[i] = (int)intToGlobal.size();
      intToGlobal.push_back(i);
    }
  const int nInt = (int)intToGlobal.size();

  // ── 2. Pre-compute per-face frame (n, d1, d2, sn) ────────────────────────
  //
  // sn = (s1+s2)/2 is the normal-direction scale.
  // This ensures A_f(e) = e/sf for all directions when s1=s2=1/sf,
  // matching the old "V0/stretch_factor" behaviour on curved meshes.
  struct FaceFrame {
    Vector3d n, d1, d2;
    double sf1, sf2, sfn;   // sfn = (sf1+sf2)/2
    bool valid;
  };
  std::vector<FaceFrame> frames(nF);

  for (int f = 0; f < nF; ++f) {
    const int idx[3] = { F(f,0), F(f,1), F(f,2) };
    Vector3d v0 = V0.row(idx[0]), v1 = V0.row(idx[1]), v2 = V0.row(idx[2]);

    Vector3d n = (v1 - v0).cross(v2 - v0);
    if (n.norm() < 1e-14) { frames[f].valid = false; continue; }
    n.normalize();

    Vector3d d1 = face_dirs[f] - face_dirs[f].dot(n) * n;
    if (d1.norm() < 1e-10)
      d1 = (v1 - v0).normalized();
    else
      d1.normalize();

    Vector3d d2 = n.cross(d1).normalized();

    const double sf1 = s1[f], sf2 = s2[f];
    frames[f] = { n, d1, d2, sf1, sf2, 0.5*(sf1+sf2), true };
  }


  // Apply A_f to a vector e using frame ff (includes normal-direction scale).
  auto Af = [](const FaceFrame& ff, const Vector3d& e) -> Vector3d {
    return ff.sf1  * ff.d1.dot(e) * ff.d1
         + ff.sf2  * ff.d2.dot(e) * ff.d2
         + ff.sfn  * ff.n.dot(e)  * ff.n;
  };

  // ── 3. Compute modified boundary positions ────────────────────────────────
  //
  // V_mod_b = average of A_f(v_b) over adjacent faces (position from origin).
  // When s1=s2=1/sf: A_f(v_b) = v_b/sf for every face (regardless of geometry).

  std::vector<std::vector<int>> vtxFaces(nV);
  for (int f = 0; f < nF; ++f)
    for (int k = 0; k < 3; ++k)
      vtxFaces[F(f,k)].push_back(f);

  fsim::Mat3<double> Vbdr_mod = V0;
  for (int b : bdrs) {
    Vector3d vb = V0.row(b);
    Vector3d sum = Vector3d::Zero();
    int cnt = 0;
    for (int f : vtxFaces[b]) {
      if (!frames[f].valid) continue;
      sum += Af(frames[f], vb);
      ++cnt;
    }
    if (cnt > 0) Vbdr_mod.row(b) = (sum / cnt).transpose();
  }

  // ── 4. Build sparse Laplacian + RHS for interior vertices ─────────────────
  std::vector<Triplet<double>> trips;
  trips.reserve(nF * 9);
  MatrixXd rhs = MatrixXd::Zero(nInt, 3);

  for (int f = 0; f < nF; ++f) {
    if (!frames[f].valid) continue;
    const FaceFrame& ff = frames[f];

    const int    idx[3] = { F(f,0), F(f,1), F(f,2) };
    const Vector3d v[3] = { V0.row(idx[0]), V0.row(idx[1]), V0.row(idx[2]) };

    for (int a = 0; a < 3; ++a) {
      if (bdrSet.count(idx[a])) continue;
      const int row = globalToInt[idx[a]];

      trips.push_back({row, row, 2.0});

      for (int b = 0; b < 3; ++b) {
        if (b == a) continue;

        if (!bdrSet.count(idx[b])) {
          trips.push_back({row, globalToInt[idx[b]], -1.0});
        } else {
          rhs.row(row) += Vbdr_mod.row(idx[b]);
        }

        rhs.row(row) += Af(ff, v[a] - v[b]).transpose();
      }
    }
  }

  // ── 5. Assemble and solve ──────────────────────────────────────────────────
  SparseMatrix<double> L(nInt, nInt);
  L.setFromTriplets(trips.begin(), trips.end());

  SimplicialLDLT<SparseMatrix<double>> solver;
  solver.compute(L);
  if (solver.info() != Success)
    throw std::runtime_error("computeAnisotropicRestShape: LDLT factorisation failed");

  MatrixXd pInt = solver.solve(rhs);
  if (solver.info() != Success)
    throw std::runtime_error("computeAnisotropicRestShape: solve failed");

  // ── 6. Reconstruct full vertex array ──────────────────────────────────────
  fsim::Mat3<double> Vmod = Vbdr_mod;
  for (int i = 0; i < nInt; ++i)
    Vmod.row(intToGlobal[i]) = pInt.row(i);

  return Vmod;
}
