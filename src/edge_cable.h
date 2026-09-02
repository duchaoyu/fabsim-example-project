// edge_cable.h
//
// A non-sliding cable: a chain of independent axial segments.
//
// Where SlidingCable (sliding_cable.h) threads a cable through a channel so the
// tension is uniform along the whole run, this element clamps the cable to the
// mesh at every node.  Material cannot flow between segments, so each segment
// carries its own tension, set by its own stretch:
//
//   E   = sum_k  (EA / 2 L0_k) (l_k - L0_k)^2
//   T_k = (EA / L0_k) (l_k - L0_k)
//
// That is the right element for a seam or a transition strip between two knitted
// regions, which is bonded to the fabric along its whole length rather than free
// to run through it.  It is also what an edge of finite axial stiffness reduces
// to when its width is small against the mesh.
//
// tension_only (default true): a slack segment carries nothing, as a fabric strip
// that buckles would.  Set false to model a seam that also resists compression -
// a bonded tape, or simply the upper bound on any axial seam model.
//
// Interface matches SlidingCable, so it composes in fsim::CompositeModel.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

struct EdgeCable
{
    std::vector<std::pair<int,int>> segments;  // vertex index pairs
    std::vector<double>             L0;        // per-segment rest length
    double EA;
    bool   tension_only;

    EdgeCable(const std::vector<std::pair<int,int>>& segs,
              const std::vector<double>&             rest,
              double ea, bool tens_only = true)
        : segments(segs), L0(rest), EA(ea), tension_only(tens_only) {}

    // Rest lengths read off a reference mesh, optionally scaled by rho.
    template <typename Derived>
    EdgeCable(const std::vector<std::pair<int,int>>& segs, double ea,
              const Eigen::MatrixBase<Derived>& Vrest, double rho = 1.0,
              bool tens_only = true)
        : segments(segs), EA(ea), tension_only(tens_only)
    {
        L0.reserve(segs.size());
        for (auto& s : segs)
            L0.push_back(rho * (Vrest.row(s.second) - Vrest.row(s.first)).norm());
    }

private:
    // Per-segment tension and material stiffness, with the slack branch applied.
    void _seg(const Eigen::Ref<const Eigen::VectorXd>& X, int k,
              double& l, Eigen::Vector3d& t, double& T, double& kM) const
    {
        int a = segments[k].first, b = segments[k].second;
        Eigen::Vector3d d = X.segment<3>(3 * b) - X.segment<3>(3 * a);
        l = d.norm();
        t = d / std::max(l, 1e-12);
        bool slack = tension_only && (l <= L0[k]);
        T  = slack ? 0.0 : (EA / L0[k]) * (l - L0[k]);
        kM = slack ? 0.0 : EA / L0[k];
    }

public:
    double energy(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        double e = 0.0;
        for (size_t k = 0; k < segments.size(); ++k) {
            double l, T, kM; Eigen::Vector3d t;
            _seg(X, (int)k, l, t, T, kM);
            if (kM == 0.0) continue;
            double dl = l - L0[k];
            e += 0.5 * (EA / L0[k]) * dl * dl;
        }
        return e;
    }

    void gradient(const Eigen::Ref<const Eigen::VectorXd>& X,
                  Eigen::Ref<Eigen::VectorXd> Y) const
    {
        for (size_t k = 0; k < segments.size(); ++k) {
            double l, T, kM; Eigen::Vector3d t;
            _seg(X, (int)k, l, t, T, kM);
            if (T == 0.0) continue;
            Y.segment<3>(3 * segments[k].first)  -= T * t;
            Y.segment<3>(3 * segments[k].second) += T * t;
        }
    }

    Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        Eigen::VectorXd Y = Eigen::VectorXd::Zero(X.size());
        gradient(X, Y);
        return Y;
    }

    std::vector<Eigen::Triplet<double>>
    hessianTriplets(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(segments.size() * 36);
        for (size_t k = 0; k < segments.size(); ++k) {
            double l, T, kM; Eigen::Vector3d t;
            _seg(X, (int)k, l, t, T, kM);
            if (kM == 0.0 || l < 1e-12) continue;
            int a = segments[k].first, b = segments[k].second;
            // material  kM t t^T   +   geometric  T (I - t t^T)/l
            Eigen::Matrix3d B = kM * t * t.transpose()
                              + (T / l) * (Eigen::Matrix3d::Identity() - t * t.transpose());
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    double v = B(r, c);
                    trips.emplace_back(3 * a + r, 3 * a + c,  v);
                    trips.emplace_back(3 * b + r, 3 * b + c,  v);
                    trips.emplace_back(3 * a + r, 3 * b + c, -v);
                    trips.emplace_back(3 * b + r, 3 * a + c, -v);
                }
        }
        return trips;
    }

    Eigen::SparseMatrix<double>
    hessian(const Eigen::Ref<const Eigen::VectorXd>& X) const
    {
        auto trips = hessianTriplets(X);
        Eigen::SparseMatrix<double> H(X.size(), X.size());
        H.setFromTriplets(trips.begin(), trips.end());
        return H;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────
    void report(const Eigen::Ref<const Eigen::VectorXd>& X,
                double& T_max, double& T_sum, int& n_taut, double& stretch_mean) const
    {
        T_max = 0.0; T_sum = 0.0; n_taut = 0; stretch_mean = 0.0;
        for (size_t k = 0; k < segments.size(); ++k) {
            double l, T, kM; Eigen::Vector3d t;
            _seg(X, (int)k, l, t, T, kM);
            T_max = std::max(T_max, std::abs(T));
            T_sum += std::abs(T);
            if (l > L0[k]) ++n_taut;
            stretch_mean += l / L0[k];
        }
        stretch_mean /= (double)segments.size();
    }
};
