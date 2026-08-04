// stress_analysis.h
//
// Per-element membrane stress for the batch drivers (fem_batch_sensitivity,
// fem_batch_nregion).
//
// RECONSTRUCTED.  The original header was never committed and was lost from the
// working tree, which left both batch drivers unbuildable.  This version is
// validated against the cached CSVs the original produced: see
// sensitivity_analysis/check_stress_reconstruction.py, which re-runs archived
// samples and diffs every column of <id>_stress.csv.  Do not change the stress
// measure or the column set without re-running that check — max_stress and
// mean_stress are study outputs and all cached data depends on them.
//
// Units: OrthotropicStVKElement builds its elasticity matrix straight from E1,
// E2 and nu, which are membrane moduli in N/m, so S = C : E comes out as a force
// per unit length (N/m) and no thickness factor is applied.  That is why the CSV
// names the in-plane components T_wale_Nm / T_course_Nm.

#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

struct ElementStress
{
    int    face;
    double S11;            // 2nd Piola-Kirchhoff, material frame, N/m
    double S22;
    double S12;
    double von_mises;
    double principal_1;    // larger principal value
    double principal_2;
    double T_wale_Nm;      // = S11, the wale direction is the element's E1 axis
    double T_course_Nm;    // = S22
};

// Plane-stress von Mises equivalent of a 2-D stress state.
inline double vonMises2D(double S11, double S22, double S12)
{
    return std::sqrt(S11 * S11 - S11 * S22 + S22 * S22 + 3.0 * S12 * S12);
}

// Principal values of a symmetric 2-D stress state, larger first.
inline void principal2D(double S11, double S22, double S12,
                        double& p1, double& p2)
{
    const double mid = 0.5 * (S11 + S22);
    const double dev = 0.5 * (S11 - S22);
    const double rad = std::sqrt(dev * dev + S12 * S12);
    p1 = mid + rad;
    p2 = mid - rad;
}

/**
 * Per-element stress of an orthotropic StVK membrane at the configuration x.
 *
 * Each element carries the elasticity matrix built from its own material
 * parameters, so the stress comes from element.stress(x, element._C); the E1, E2,
 * nu and thickness arguments are accepted for call-site compatibility and are
 * deliberately unused. Passing them per-element instead would silently ignore
 * any spatially varying material field the model was constructed with.
 */
template <class Model>
std::vector<ElementStress>
computeElementStresses(Model& model,
                       const Eigen::Ref<const Eigen::VectorXd> x,
                       double /*E1*/, double /*E2*/,
                       double /*nu*/, double /*thickness*/)
{
    auto elements = model.getElements();

    std::vector<ElementStress> out;
    out.reserve(elements.size());

    for (std::size_t i = 0; i < elements.size(); ++i)
    {
        const Eigen::Vector3d S = elements[i].stress(x, elements[i]._C);

        ElementStress e;
        e.face = static_cast<int>(i);
        e.S11  = S(0);
        e.S22  = S(1);
        e.S12  = S(2);
        e.von_mises = vonMises2D(e.S11, e.S22, e.S12);
        principal2D(e.S11, e.S22, e.S12, e.principal_1, e.principal_2);
        e.T_wale_Nm   = e.S11;
        e.T_course_Nm = e.S22;
        out.push_back(e);
    }
    return out;
}

inline void saveStressCSV(const std::string& path,
                          const std::vector<ElementStress>& st)
{
    std::ofstream out(path);
    out << "face,S11,S22,S12,von_mises,principal_1,principal_2,"
           "T_wale_Nm,T_course_Nm\n"
        << std::fixed << std::setprecision(4);
    for (const auto& e : st)
        out << e.face        << ","
            << e.S11         << ","
            << e.S22         << ","
            << e.S12         << ","
            << e.von_mises   << ","
            << e.principal_1 << ","
            << e.principal_2 << ","
            << e.T_wale_Nm   << ","
            << e.T_course_Nm << "\n";
}
