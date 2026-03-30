// anisotropic_rest_shape.h
//
// Computes a non-uniformly pre-strained rest shape V_mod from a reference mesh V0.
//
// Instead of applying a single scalar stretch_factor uniformly, each face f
// receives its own anisotropic deformation gradient:
//
//   A_f(e) = s1_f * (d1_f · e) * d1_f    ← wale direction
//           + s2_f * (d2_f · e) * d2_f   ← course direction (= n_f × d1_f)
//           + (n_f · e) * n_f             ← normal component unchanged
//
// where d1_f comes from face_dirs (projected onto the face plane).
//
// Boundary vertices have their positions transformed by the average A_f over
// adjacent faces (treating the position vector as originating from the origin).
// When s1=s2=1/sf and the mesh is flat this gives v_bdr/sf, exactly matching
// the old "V0 / stretch_factor" behaviour.
//
// Interior vertices minimise the global least-squares problem:
//
//   min Σ_f  Σ_{(i,j) edge of f}  || (V_mod_i − V_mod_j) − A_f*(V_i − V_j) ||²
//   subject to V_mod_i = A_avg_i(V_i)  for all boundary vertices
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
//   // override per zone if needed:
//   for (int f : zoneB_faces) { s1[f] = 1.0/sf_B_wale; s2[f] = 1.0/sf_B_course; }
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
//  s1        : per-face stretch ratio along the wale direction
//  s2        : per-face stretch ratio along the course direction
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

  // ── 2. Pre-compute per-face frame (n, d1, d2) and A_f ────────────────────
  struct FaceFrame { Vector3d n, d1, d2; double sf1, sf2; bool valid; };
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
    frames[f] = { n, d1, d2, s1[f], s2[f], true };
  }

  // Helper: apply A_f to a vector e using frame ff
  auto Af = [](const FaceFrame& ff, const Vector3d& e) -> Vector3d {
    return ff.sf1 * ff.d1.dot(e) * ff.d1
         + ff.sf2 * ff.d2.dot(e) * ff.d2
         + ff.n.dot(e) * ff.n;
  };

  // ── 3. Compute modified boundary positions ────────────────────────────────
  //
  // For boundary vertex b: V_mod_b = average of A_f(v_b) over adjacent faces,
  // treating v_b as a position vector from the origin.
  //
  // Rationale: when s1=s2=1/sf on a flat mesh,
  //   A_f(v_b) = (1/sf)*v_b_xy + v_b_z  ≈  v_b / sf
  // so the modified boundary positions equal V0/sf, exactly matching the
  // old "V0 / stretch_factor" pre-tension mechanism.

  // vertex → adjacent face list
  std::vector<std::vector<int>> vtxFaces(nV);
  for (int f = 0; f < nF; ++f)
    for (int k = 0; k < 3; ++k)
      vtxFaces[F(f,k)].push_back(f);

  // Vbdr_mod: same as V0 for interior vertices (will be overwritten);
  // boundary vertices get A_f-averaged positions.
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
  //
  // For interior vertex ia and each face f containing ia:
  //   diagonal  L[ia,ia]   += 2
  //   interior  L[ia,ib]   -= 1
  //   boundary  rhs[ia]    += Vbdr_mod[ib]   (modified boundary position)
  //   rhs[ia]  += A_f * (v_ia − v_ib)

  std::vector<Triplet<double>> trips;
  trips.reserve(nF * 9);
  MatrixXd rhs = MatrixXd::Zero(nInt, 3);

  for (int f = 0; f < nF; ++f) {
    if (!frames[f].valid) continue;
    const FaceFrame& ff = frames[f];

    const int idx[3] = { F(f,0), F(f,1), F(f,2) };
    Vector3d  v[3]   = { V0.row(idx[0]), V0.row(idx[1]), V0.row(idx[2]) };

    for (int a = 0; a < 3; ++a) {
      if (bdrSet.count(idx[a])) continue;
      const int row = globalToInt[idx[a]];

      trips.push_back({row, row, 2.0});

      for (int b = 0; b < 3; ++b) {
        if (b == a) continue;

        if (!bdrSet.count(idx[b])) {
          trips.push_back({row, globalToInt[idx[b]], -1.0});
        } else {
          // Boundary neighbour: use modified position
          rhs.row(row) += Vbdr_mod.row(idx[b]);
        }

        // RHS contribution: A_f * (v_a − v_b)
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

  MatrixXd pInt = solver.solve(rhs);    // nInt × 3
  if (solver.info() != Success)
    throw std::runtime_error("computeAnisotropicRestShape: solve failed");

  // ── 6. Reconstruct full vertex array ──────────────────────────────────────
  // Boundary vertices already hold their modified positions in Vbdr_mod.
  fsim::Mat3<double> Vmod = Vbdr_mod;
  for (int i = 0; i < nInt; ++i)
    Vmod.row(intToGlobal[i]) = pInt.row(i);

  return Vmod;
}
