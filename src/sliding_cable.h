// sliding_cable.h
//
// A frictionless sliding cable element.
//
// Models a cable threaded through a channel that can slide freely, so the
// tension T is uniform along the entire cable.  Interior nodes transmit only
// the net curvature force (change in tension direction), NOT the full tension
// per segment independently.  If the cable is locally straight at a node, the
// net force there is zero — the node is free to slide.
//
// ── Physics ────────────────────────────────────────────────────────────────
//
//   E = (EA / 2 L_0) * (L - L_0)^2
//
//   where  L   = sum of current segment lengths  (total arc length)
//          L_0 = sum of rest segment lengths
//          T   = (EA / L_0)(L - L_0)             (uniform tension)
//
// Gradient at vertex v:
//   ∂E/∂x_v = T * ∂L/∂x_v
//            = T * sum_{k: v ∈ seg k} sign(k,v) * t_k
//
//   where sign = -1 if v is the start of segment k, +1 if the end, and
//   t_k is the unit tangent of segment k.
//
//   For an interior vertex connecting segments k-1 and k:
//     ∂E/∂x_v = T * (t_{k-1} - t_k)
//   which is zero when the cable is locally straight → frictionless sliding.
//
// Hessian = material stiffness (EA/L_0)(∇L)(∇L)^T
//         + geometric stiffness T * Σ_k (I - t_k t_k^T) / l_k
//
// ── Interface ──────────────────────────────────────────────────────────────
// Matches SpringCollection / fsim model interface:
//   energy(), gradient() (void and returning VectorXd), hessianTriplets(),
//   hessian().  Can be used directly in fsim::CompositeModel.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

struct SlidingCable
{
    std::vector<int> indices;   // ordered cable vertex indices (path order)
    double EA;                  // axial stiffness (N)
    double L_rest;              // total rest length (m)

    // V0: reference vertex positions — L_rest computed from geometry
    template <typename Derived>
    SlidingCable(const std::vector<int>& idx, double ea,
                 const Eigen::MatrixBase<Derived>& V0)
        : indices(idx), EA(ea), L_rest(0.0)
    {
        for (size_t k = 0; k + 1 < indices.size(); ++k)
            L_rest += (V0.row(indices[k+1]) - V0.row(indices[k])).norm();
    }

    // Direct rest-length constructor — specify total rest length explicitly.
    // Use this to pre-stress the cable: L_rest < geometric length → tension.
    SlidingCable(const std::vector<int>& idx, double ea, double l_rest)
        : indices(idx), EA(ea), L_rest(l_rest) {}

private:
    struct Seg { int a, b; double l; Eigen::Vector3d t; };  // t = unit tangent a→b

    std::vector<Seg> _geometry(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        std::vector<Seg> segs;
        segs.reserve(indices.size() - 1);
        for (size_t k = 0; k + 1 < indices.size(); ++k) {
            int a = indices[k], b = indices[k + 1];
            Eigen::Vector3d d = X.segment<3>(3 * b) - X.segment<3>(3 * a);
            double l = d.norm();
            segs.push_back({a, b, l, d / std::max(l, 1e-12)});
        }
        return segs;
    }

    static double _totalLength(const std::vector<Seg>& segs)
    {
        double L = 0.0;
        for (auto& s : segs) L += s.l;
        return L;
    }

public:
    // ── Energy ───────────────────────────────────────────────────────────────
    double energy(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        auto segs = _geometry(X);
        double dL = _totalLength(segs) - L_rest;
        return (EA / (2.0 * L_rest)) * dL * dL;
    }

    // ── Gradient (in-place, adds to Y) ───────────────────────────────────────
    void gradient(const Eigen::Ref<const Eigen::VectorXd>& X,
                  Eigen::Ref<Eigen::VectorXd> Y) const
    {
        auto segs = _geometry(X);
        // Tension-only: cable cannot push (T < 0 → compression → treat as slack)
        double T = std::max(0.0, (EA / L_rest) * (_totalLength(segs) - L_rest));

        for (auto& s : segs) {
            // ∂l_k/∂x_a = (x_a - x_b)/l = -t_k  →  ∂E/∂x_a += -T * t_k
            // ∂l_k/∂x_b = (x_b - x_a)/l = +t_k  →  ∂E/∂x_b += +T * t_k
            Y.segment<3>(3 * s.a) -= T * s.t;
            Y.segment<3>(3 * s.b) += T * s.t;
        }
    }

    // ── Gradient (returns VectorXd) ───────────────────────────────────────────
    Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        Eigen::VectorXd Y = Eigen::VectorXd::Zero(X.size());
        gradient(X, Y);
        return Y;
    }

    // ── Hessian triplets ─────────────────────────────────────────────────────
    std::vector<Eigen::Triplet<double>>
    hessianTriplets(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        auto segs = _geometry(X);
        double L  = _totalLength(segs);
        double T  = std::max(0.0, (EA / L_rest) * (L - L_rest));  // tension-only
        double kM = (L > L_rest) ? EA / L_rest : 0.0;             // zero stiffness when slack

        // ── ∂L/∂x at each cable vertex (indexed by position in indices[]) ───
        // Segment k: vertex indices[k] is "a" (contributes -t_k),
        //            vertex indices[k+1] is "b" (contributes +t_k).
        const int n = static_cast<int>(indices.size());
        std::vector<Eigen::Vector3d> G(n, Eigen::Vector3d::Zero());
        for (int k = 0; k < static_cast<int>(segs.size()); ++k) {
            G[k]     -= segs[k].t;
            G[k + 1] += segs[k].t;
        }

        std::vector<Eigen::Triplet<double>> trips;

        // ── Material stiffness: kM * (∇L)(∇L)^T ─────────────────────────────
        // Dense over all pairs of cable vertices (n^2 3x3 blocks, n ≤ ~20).
        for (int i = 0; i < n; ++i) {
            int vi = indices[i];
            for (int j = 0; j < n; ++j) {
                int vj = indices[j];
                Eigen::Matrix3d block = kM * G[i] * G[j].transpose();
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        trips.emplace_back(3 * vi + r, 3 * vj + c, block(r, c));
            }
        }

        // ── Geometric stiffness: T * (I - t t^T) / l  per segment ───────────
        for (auto& s : segs) {
            if (s.l < 1e-12) continue;
            Eigen::Matrix3d P  = (Eigen::Matrix3d::Identity() - s.t * s.t.transpose()) / s.l;
            Eigen::Matrix3d TP = T * P;

            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    double v = TP(r, c);
                    trips.emplace_back(3 * s.a + r, 3 * s.a + c,  v);  // (a,a)
                    trips.emplace_back(3 * s.b + r, 3 * s.b + c,  v);  // (b,b)
                    trips.emplace_back(3 * s.a + r, 3 * s.b + c, -v);  // (a,b)
                    trips.emplace_back(3 * s.b + r, 3 * s.a + c, -v);  // (b,a)
                }
        }

        return trips;
    }

    // ── Hessian (sparse matrix) ───────────────────────────────────────────────
    Eigen::SparseMatrix<double>
    hessian(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        auto trips = hessianTriplets(X);
        Eigen::SparseMatrix<double> H(X.size(), X.size());
        H.setFromTriplets(trips.begin(), trips.end());
        return H;
    }
};
