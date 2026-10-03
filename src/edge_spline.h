// edge_spline.h
//
// A bending-stiff rod ("spline") attached to an ordered chain of mesh vertices,
// e.g. a fibreglass or steel rod in a sleeve along a free edge.  Unlike a
// SlidingCable it is attached at every node, resists compression and resists
// bending.
//
// ── Physics ────────────────────────────────────────────────────────────────
//
// Twist-free isotropic Kirchhoff rod (round section; in a sleeve the twist
// relaxes, so it carries no torsional energy):
//
//   E = Σ_segments  EA / (2 l0_k) (l_k - l0_k)^2
//     + Σ_interior  EI / (2 lbar_i) |κb_i - κb0_i|^2
//
//   κb_i = 2 (e_{i-1} × e_i) / (|e_{i-1}||e_i| + e_{i-1}·e_i)   (discrete
//   curvature binormal of Discrete Elastic Rods, |κb| = 2 tan(φ/2)),
//   lbar_i = (l0_{i-1} + l0_i) / 2.
//
// Rest curvature κb0:
//   * straight rest (κb0 = 0): a bending-active rod bent into place.  Exact and
//     rotation invariant.
//   * pre-formed rest (κb0 = κb of the reference shape): a rod formed to the
//     edge.  κb0 is held in the global frame, which is exact where the rod is
//     locally straight and a small-rotation approximation at kinks; the error
//     is second order in the rotation of the rod, which stays small here
//     because both ends are pinned at supports.
//
// The ends are pinned by the supports, not clamped: there is no bending
// stencil beyond the end nodes, so the end tangents are free.
//
// Gradient analytic; Hessian of the axial term analytic, of each bending
// stencil by central differences of its analytic gradient (9x9, cheap).

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

struct EdgeSpline
{
    std::vector<int> idx;            // ordered vertex indices
    double EA = 0.0, EI = 0.0;       // N, N m^2
    std::vector<double> l0;          // rest segment lengths
    std::vector<Eigen::Vector3d> kb0;  // rest curvature binormal per interior node

    template <typename Derived>
    EdgeSpline(const std::vector<int>& indices, double ea, double ei,
               const Eigen::MatrixBase<Derived>& V0, bool preformed, double length_scale = 1.0)
        : idx(indices), EA(ea), EI(ei)
    {
        const int n = (int)idx.size();
        for (int k = 0; k + 1 < n; ++k)
            l0.push_back(length_scale * (V0.row(idx[k+1]) - V0.row(idx[k])).norm());
        for (int i = 1; i + 1 < n; ++i) {
            Eigen::Vector3d a = (V0.row(idx[i]) - V0.row(idx[i-1])).transpose();
            Eigen::Vector3d b = (V0.row(idx[i+1]) - V0.row(idx[i])).transpose();
            kb0.push_back(preformed ? curvature(a, b) : Eigen::Vector3d::Zero());
        }
    }

    static Eigen::Vector3d curvature(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
    {
        return 2.0 * a.cross(b) / (a.norm() * b.norm() + a.dot(b));
    }

    static Eigen::Matrix3d skew(const Eigen::Vector3d& v)
    {
        Eigen::Matrix3d S;
        S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
        return S;
    }

    Eigen::Vector3d node(const Eigen::Ref<const Eigen::VectorXd>& X, int v) const
    {
        return X.segment<3>(3 * v);
    }

    // bending stencil i (interior node i = 1..n-2) on its three nodes p0,p1,p2
    double bendEnergy(int i, const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                      const Eigen::Vector3d& p2) const
    {
        const double lbar = 0.5 * (l0[i-1] + l0[i]);
        Eigen::Vector3d dk = curvature(p1 - p0, p2 - p1) - kb0[i-1];
        return 0.5 * EI / lbar * dk.squaredNorm();
    }

    Eigen::Matrix<double, 9, 1> bendGradient(int i, const Eigen::Vector3d& p0,
                                             const Eigen::Vector3d& p1,
                                             const Eigen::Vector3d& p2) const
    {
        const double lbar = 0.5 * (l0[i-1] + l0[i]);
        const Eigen::Vector3d a = p1 - p0, b = p2 - p1;
        const double na = a.norm(), nb = b.norm();
        const double d = na * nb + a.dot(b);
        const Eigen::Vector3d c = a.cross(b);
        const Eigen::Vector3d kb = 2.0 * c / d;
        const Eigen::Vector3d r = EI / lbar * (kb - kb0[i-1]);   // dE/dκb
        // dκb/da = 2/d (-[b]x) - 2c/d^2 (dd/da)^T,  dd/da = nb a/na + b
        const Eigen::Vector3d dda = nb / na * a + b, ddb = na / nb * b + a;
        const Eigen::Matrix3d Ja = 2.0 / d * (-skew(b)) - 2.0 / (d * d) * c * dda.transpose();
        const Eigen::Matrix3d Jb = 2.0 / d * skew(a) - 2.0 / (d * d) * c * ddb.transpose();
        const Eigen::Vector3d ga = Ja.transpose() * r, gb = Jb.transpose() * r;
        Eigen::Matrix<double, 9, 1> g;
        g << -ga, ga - gb, gb;
        return g;
    }

    double energy(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        double e = 0.0;
        const int n = (int)idx.size();
        for (int k = 0; k + 1 < n; ++k) {
            const double l = (node(X, idx[k+1]) - node(X, idx[k])).norm();
            e += 0.5 * EA / l0[k] * (l - l0[k]) * (l - l0[k]);
        }
        if (EI > 0.0)
            for (int i = 1; i + 1 < n; ++i)
                e += bendEnergy(i, node(X, idx[i-1]), node(X, idx[i]), node(X, idx[i+1]));
        return e;
    }

    void gradient(const Eigen::Ref<const Eigen::VectorXd>& X, Eigen::Ref<Eigen::VectorXd> Y) const
    {
        const int n = (int)idx.size();
        for (int k = 0; k + 1 < n; ++k) {
            const Eigen::Vector3d e = node(X, idx[k+1]) - node(X, idx[k]);
            const double l = e.norm();
            const Eigen::Vector3d f = EA / l0[k] * (l - l0[k]) * e / l;
            Y.segment<3>(3 * idx[k+1]) += f;
            Y.segment<3>(3 * idx[k])   -= f;
        }
        if (EI > 0.0)
            for (int i = 1; i + 1 < n; ++i) {
                auto g = bendGradient(i, node(X, idx[i-1]), node(X, idx[i]), node(X, idx[i+1]));
                for (int j = 0; j < 3; ++j) Y.segment<3>(3 * idx[i-1+j]) += g.segment<3>(3 * j);
            }
    }

    std::vector<Eigen::Triplet<double>> hessianTriplets(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        std::vector<Eigen::Triplet<double>> T;
        const int n = (int)idx.size();
        for (int k = 0; k + 1 < n; ++k) {
            const Eigen::Vector3d e = node(X, idx[k+1]) - node(X, idx[k]);
            const double l = e.norm();
            const Eigen::Vector3d t = e / l;
            // d2/de2 of EA/(2 l0) (l - l0)^2 ; the geometric part is dropped when
            // the segment is in compression so the block stays positive semi-definite
            Eigen::Matrix3d H = EA / l0[k] * t * t.transpose();
            if (l > l0[k])
                H += EA / l0[k] * (l - l0[k]) / l * (Eigen::Matrix3d::Identity() - t * t.transpose());
            const int a = idx[k], b = idx[k+1];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    T.emplace_back(3*a+r, 3*a+c,  H(r, c));
                    T.emplace_back(3*b+r, 3*b+c,  H(r, c));
                    T.emplace_back(3*a+r, 3*b+c, -H(r, c));
                    T.emplace_back(3*b+r, 3*a+c, -H(r, c));
                }
        }
        if (EI > 0.0)
            for (int i = 1; i + 1 < n; ++i) {
                Eigen::Matrix<double, 9, 1> x;
                x << node(X, idx[i-1]), node(X, idx[i]), node(X, idx[i+1]);
                const double h = 1e-7 * std::max(l0[i-1], l0[i]);
                Eigen::Matrix<double, 9, 9> H;
                for (int c = 0; c < 9; ++c) {
                    Eigen::Matrix<double, 9, 1> xp = x, xm = x;
                    xp[c] += h; xm[c] -= h;
                    H.col(c) = (bendGradient(i, xp.segment<3>(0), xp.segment<3>(3), xp.segment<3>(6)) -
                                bendGradient(i, xm.segment<3>(0), xm.segment<3>(3), xm.segment<3>(6))) / (2 * h);
                }
                H = 0.5 * (H + H.transpose());
                for (int r = 0; r < 9; ++r)
                    for (int c = 0; c < 9; ++c)
                        T.emplace_back(3 * idx[i-1+r/3] + r%3, 3 * idx[i-1+c/3] + c%3, H(r, c));
            }
        return T;
    }
};
