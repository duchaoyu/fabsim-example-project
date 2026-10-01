#pragma once
// Follower pressure on an open membrane.
//
// The fabsim elements apply pressure as the work p·V of the volume the surface
// sweeps against the origin, V = sum_f x0·(x1 x x2)/6.  Its gradient at a vertex
// whose one-ring is closed is the follower force p·(area vector)/3, but at a
// vertex on a FREE boundary the ring is open and the gradient keeps a term that
// depends on where the origin is (up to ~80x the true nodal force, see
// FDM/fea_check).  A clamped boundary hides this because those dofs are fixed.
//
// The physical load is the follower force, which has no potential on an open
// surface.  So this solves the force balance directly:
//
//   r(x) = grad E(x) - f(x) = 0,   f_v = sum_{f ∋ v} p (x1-x0) x (x2-x0) / 6
//
// with E the model built at ZERO pressure (elastic + gravity + cables), by
// Newton on the full non-symmetric Jacobian  J = H - K,
//
//   K = df/dx,  per face:  d f / d x0 = p/6 [x2-x1]x,
//                          d f / d x1 = p/6 [x0-x2]x,
//                          d f / d x2 = p/6 [x1-x0]x      ([a]x b = a x b),
//
// solved with SparseLU, a Levenberg shift when the step is poor, and a
// backtracking line search on |r|^2.

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace follower {

template <class FM>
inline Eigen::VectorXd force(const Eigen::VectorXd& x, const FM& F, double p)
{
  Eigen::VectorXd f = Eigen::VectorXd::Zero(x.size());
  for (int t = 0; t < F.rows(); ++t) {
    const Eigen::Vector3d x0 = x.segment<3>(3 * F(t, 0));
    const Eigen::Vector3d x1 = x.segment<3>(3 * F(t, 1));
    const Eigen::Vector3d x2 = x.segment<3>(3 * F(t, 2));
    const Eigen::Vector3d g = p / 6.0 * (x1 - x0).cross(x2 - x0);
    for (int k = 0; k < 3; ++k) f.segment<3>(3 * F(t, k)) += g;
  }
  return f;
}

inline void addSkew(std::vector<Eigen::Triplet<double>>& T, int row, int col,
                    const Eigen::Vector3d& a, double s)
{
  // s * [a]x  at block (row, col)
  T.emplace_back(row + 0, col + 1, -s * a.z()); T.emplace_back(row + 0, col + 2,  s * a.y());
  T.emplace_back(row + 1, col + 0,  s * a.z()); T.emplace_back(row + 1, col + 2, -s * a.x());
  T.emplace_back(row + 2, col + 0, -s * a.y()); T.emplace_back(row + 2, col + 1,  s * a.x());
}

// Triplets of -K = -df/dx (so they add straight onto the Hessian).
template <class FM>
inline std::vector<Eigen::Triplet<double>>
minusStiffness(const Eigen::VectorXd& x, const FM& F, double p)
{
  std::vector<Eigen::Triplet<double>> T;
  T.reserve(F.rows() * 3 * 3 * 6);
  for (int t = 0; t < F.rows(); ++t) {
    const Eigen::Vector3d x0 = x.segment<3>(3 * F(t, 0));
    const Eigen::Vector3d x1 = x.segment<3>(3 * F(t, 1));
    const Eigen::Vector3d x2 = x.segment<3>(3 * F(t, 2));
    const Eigen::Vector3d d[3] = { x2 - x1, x0 - x2, x1 - x0 };
    for (int a = 0; a < 3; ++a)        // force row: every vertex of the face gets g
      for (int b = 0; b < 3; ++b)      // derivative w.r.t. vertex b
        addSkew(T, 3 * F(t, a), 3 * F(t, b), d[b], -p / 6.0);
  }
  return T;
}

// Full symmetric matrix from a Hessian that may store only its upper triangle.
inline Eigen::SparseMatrix<double> fullFromUpper(const Eigen::SparseMatrix<double>& H)
{
  Eigen::SparseMatrix<double> U = H.triangularView<Eigen::Upper>();
  Eigen::SparseMatrix<double> L = U.transpose();
  Eigen::SparseMatrix<double> S = U + L;
  for (int k = 0; k < S.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(S, k); it; ++it)
      if (it.row() == it.col()) it.valueRef() *= 0.5;
  return S;
}

struct Result {
  Eigen::VectorXd x;
  bool converged = false;
  int iterations = 0;
  double residual = 0.0;   // max |r| on free dofs
};

// model: built at zero pressure.  fixed: fixed vertex indices.
template <class Model, class FM>
Result solve(const Model& model, const FM& F, double p,
             Eigen::VectorXd x, const std::vector<int>& fixed,
             double tol = 1e-6, int max_iter = 200)
{
  const int n = x.size();
  std::vector<char> is_fixed(n, 0);
  for (int v : fixed) for (int k = 0; k < 3; ++k) is_fixed[3 * v + k] = 1;
  std::vector<int> free_idx, map(n, -1);
  for (int i = 0; i < n; ++i) if (!is_fixed[i]) { map[i] = (int)free_idx.size(); free_idx.push_back(i); }
  const int m = free_idx.size();

  auto residual = [&](const Eigen::VectorXd& y) {
    Eigen::VectorXd r = model.gradient(y) - force(y, F, p);
    Eigen::VectorXd rf(m);
    for (int i = 0; i < m; ++i) rf[i] = r[free_idx[i]];
    return rf;
  };

  Result res;
  Eigen::VectorXd r = residual(x);
  double mu = 0.0;
  for (int it = 0; it < max_iter; ++it) {
    res.iterations = it;
    res.residual = r.cwiseAbs().maxCoeff();
    if (res.residual < tol) { res.converged = true; break; }

    Eigen::SparseMatrix<double> H = fullFromUpper(model.hessian(x));
    auto T = minusStiffness(x, F, p);
    for (int k = 0; k < H.outerSize(); ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator e(H, k); e; ++e)
        T.emplace_back(e.row(), e.col(), e.value());

    double diag_scale = 0.0;
    for (int k = 0; k < H.outerSize(); ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator e(H, k); e; ++e)
        if (e.row() == e.col()) diag_scale = std::max(diag_scale, std::abs(e.value()));
    if (diag_scale <= 0.0) diag_scale = 1.0;

    bool stepped = false;
    for (int attempt = 0; attempt < 12 && !stepped; ++attempt) {
      std::vector<Eigen::Triplet<double>> Tf;
      Tf.reserve(T.size() + m);
      for (auto& e : T)
        if (map[e.row()] >= 0 && map[e.col()] >= 0)
          Tf.emplace_back(map[e.row()], map[e.col()], e.value());
      if (mu > 0.0) for (int i = 0; i < m; ++i) Tf.emplace_back(i, i, mu);
      Eigen::SparseMatrix<double> J(m, m);
      J.setFromTriplets(Tf.begin(), Tf.end());
      J.makeCompressed();

      Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
      lu.analyzePattern(J);
      lu.factorize(J);
      if (lu.info() == Eigen::Success) {
        Eigen::VectorXd dx = lu.solve(-r);
        if (lu.info() == Eigen::Success && dx.allFinite()) {
          // backtracking on |r|^2
          const double phi0 = r.squaredNorm();
          double alpha = 1.0;
          for (int ls = 0; ls < 30; ++ls, alpha *= 0.5) {
            Eigen::VectorXd y = x;
            for (int i = 0; i < m; ++i) y[free_idx[i]] += alpha * dx[i];
            Eigen::VectorXd ry = residual(y);
            if (ry.allFinite() && ry.squaredNorm() < (1.0 - 1e-4 * alpha) * phi0) {
              x = y; r = ry; stepped = true; break;
            }
          }
        }
      }
      if (!stepped) mu = (mu == 0.0) ? 1e-8 * diag_scale : mu * 10.0;
    }
    if (!stepped) break;
    mu = (mu > 0.0) ? mu / 10.0 : 0.0;
    if (mu < 1e-12 * diag_scale) mu = 0.0;
  }
  res.x = x;
  res.residual = r.size() ? r.cwiseAbs().maxCoeff() : 0.0;
  if (res.residual < tol) res.converged = true;
  return res;
}

}  // namespace follower
