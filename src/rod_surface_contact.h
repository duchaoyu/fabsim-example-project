// rod_surface_contact.h
//
// Penalty-based constraint that keeps decoupled rod nodes on the membrane surface,
// allowing frictionless tangential sliding.
//
// ── DOF layout assumed ────────────────────────────────────────────────────────
//   X[0 ... 3*nV_mem-1]                    membrane vertex positions
//   X[3*nV_mem ... 3*(nV_mem+nV_rod)-1]    rod node positions   (decoupled)
//   X[3*(nV_mem+nV_rod) ...]               rod twist angles     (not used here)
//
// ── Physics ───────────────────────────────────────────────────────────────────
// For each rod node i at position r_i, find its nearest membrane face f.
// Let n_f be the face unit normal and c_f its centroid.
//   d_i = (r_i - c_f) · n_f           (signed normal distance)
//   E_i = (k/2) * d_i^2
//
// The gradient:
//   ∂E_i/∂r_i          = k * d_i * n_f          (pushes rod onto surface)
//   ∂E_i/∂x_{f,j}      = k * d_i * (-n_f / 3)   (equal reaction on 3 face vertices)
//
// Hessian uses n_f frozen (constant normal approximation) — valid near equilibrium.
//   H(r_i, r_i)        =  k * n_f n_f^T
//   H(x_{f,j}, x_{f,j})=  k/9 * n_f n_f^T
//   H(r_i, x_{f,j})    = -k/3 * n_f n_f^T   (and transpose)
//
// Contact assignments (nearest face per rod node) are updated via updateContacts(),
// which must be called in the Newton update_fct after each step.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <vector>

struct RodSurfaceContact
{
    int    nV_mem;   // number of membrane vertices
    int    nV_rod;   // number of rod nodes
    double k;        // penalty stiffness (N/m)

    // Row-major Nx3 face-index matrix (copy of mesh topology)
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> F;

    // Nearest face index for each rod node — updated by updateContacts()
    mutable std::vector<int> nearest_face;

    // ── Constructor ───────────────────────────────────────────────────────────
    // X0: initial DOF vector (used to seed the contact assignments)
    template <typename DerivedF>
    RodSurfaceContact(int nV_mem_, int nV_rod_, double k_penalty,
                      const Eigen::MatrixBase<DerivedF> &F_,
                      const Eigen::Ref<const Eigen::VectorXd> &X0)
        : nV_mem(nV_mem_), nV_rod(nV_rod_), k(k_penalty),
          F(F_), nearest_face(nV_rod_)
    {
        updateContacts(X0);
    }

    // ── Contact update (call from Newton update_fct) ──────────────────────────
    void updateContacts(const Eigen::Ref<const Eigen::VectorXd> &X) const
    {
        for (int i = 0; i < nV_rod; ++i)
        {
            Eigen::Vector3d r = X.segment<3>(3 * (nV_mem + i));
            int   best_f  = 0;
            double best_d2 = std::numeric_limits<double>::infinity();
            for (int f = 0; f < F.rows(); ++f)
            {
                Eigen::Vector3d c = (X.segment<3>(3 * F(f, 0)) +
                                     X.segment<3>(3 * F(f, 1)) +
                                     X.segment<3>(3 * F(f, 2))) / 3.0;
                double d2 = (r - c).squaredNorm();
                if (d2 < best_d2) { best_d2 = d2; best_f = f; }
            }
            nearest_face[i] = best_f;
        }
    }

private:
    // Per-contact geometry: normal, centroid, signed distance, face vertex DOF offsets
    struct Geom
    {
        Eigen::Vector3d n, c;
        double          d;
        int             dof_va, dof_vb, dof_vc;  // 3 * vertex_index
    };

    Geom _geom(int i, const Eigen::Ref<const Eigen::VectorXd> &X) const
    {
        int f   = nearest_face[i];
        int va  = F(f, 0), vb = F(f, 1), vc = F(f, 2);
        Eigen::Vector3d x0 = X.segment<3>(3 * va);
        Eigen::Vector3d x1 = X.segment<3>(3 * vb);
        Eigen::Vector3d x2 = X.segment<3>(3 * vc);

        Eigen::Vector3d nraw = (x1 - x0).cross(x2 - x0);
        double len = nraw.norm();
        Eigen::Vector3d n;
        if (len < 1e-12) n = Eigen::Vector3d::UnitZ();
        else             n = nraw / len;

        Eigen::Vector3d c = (x0 + x1 + x2) / 3.0;
        Eigen::Vector3d r = X.segment<3>(3 * (nV_mem + i));
        double d = (r - c).dot(n);

        return {n, c, d, 3 * va, 3 * vb, 3 * vc};
    }

public:
    // ── Energy ───────────────────────────────────────────────────────────────
    double energy(const Eigen::Ref<const Eigen::VectorXd> &X) const
    {
        double E = 0.0;
        for (int i = 0; i < nV_rod; ++i)
        {
            auto g = _geom(i, X);
            E += 0.5 * k * g.d * g.d;
        }
        return E;
    }

    // ── Gradient (adds into Y) ────────────────────────────────────────────────
    void gradient(const Eigen::Ref<const Eigen::VectorXd> &X,
                  Eigen::Ref<Eigen::VectorXd> Y) const
    {
        for (int i = 0; i < nV_rod; ++i)
        {
            auto g = _geom(i, X);
            double coeff = k * g.d;

            // Rod node: full normal force
            Y.segment<3>(3 * (nV_mem + i)) += coeff * g.n;

            // Face vertices: equal and opposite reaction, split 1/3 each
            Eigen::Vector3d dv = coeff * (-g.n / 3.0);
            Y.segment<3>(g.dof_va) += dv;
            Y.segment<3>(g.dof_vb) += dv;
            Y.segment<3>(g.dof_vc) += dv;
        }
    }

    Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> &X) const
    {
        Eigen::VectorXd Y = Eigen::VectorXd::Zero(X.size());
        gradient(X, Y);
        return Y;
    }

    // ── Hessian triplets (n frozen — valid near equilibrium) ──────────────────
    std::vector<Eigen::Triplet<double>>
    hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> &X) const
    {
        std::vector<Eigen::Triplet<double>> trips;

        for (int i = 0; i < nV_rod; ++i)
        {
            auto g = _geom(i, X);
            Eigen::Matrix3d nnT = k * g.n * g.n.transpose();

            int rod3 = 3 * (nV_mem + i);
            int mem3[3] = {g.dof_va, g.dof_vb, g.dof_vc};

            // Helper: add a scaled 3x3 nnT block at (row_base, col_base)
            auto addBlock = [&](int r3, int c3, double scale)
            {
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        trips.emplace_back(r3 + r, c3 + c, scale * nnT(r, c));
            };

            addBlock(rod3, rod3, 1.0);           // (rod,  rod)
            for (int m : mem3)
            {
                addBlock(rod3, m,  -1.0 / 3.0);  // (rod,  v_j)
                addBlock(m, rod3, -1.0 / 3.0);   // (v_j,  rod)
            }
            for (int m1 : mem3)
                for (int m2 : mem3)
                    addBlock(m1, m2, 1.0 / 9.0); // (v_j,  v_k)
        }

        return trips;
    }

    Eigen::SparseMatrix<double>
    hessian(const Eigen::Ref<const Eigen::VectorXd> &X) const
    {
        auto trips = hessianTriplets(X);
        Eigen::SparseMatrix<double> H(X.size(), X.size());
        H.setFromTriplets(trips.begin(), trips.end());
        return H;
    }
};
