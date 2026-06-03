// fem_batch_sensitivity.cpp
//
// Headless FEM runner for sensitivity analysis.
//
// Usage:
//   ./fem_batch_sensitivity <mesh_path> <sf_wale> <sf_course> <knit_dir_deg>
//                           <pressure> <motif>
//                           <cable_wale_json_or_none> <cable_course_json_or_none>
//                           <output_prefix>
//                           [fixed_vertices]
//                           [material_json_or_none]
//
//   Each cable JSON: {"indices":[...], "EA":150000.0, "L_rest":1.196}
//     L_rest is required — set it to fraction * geometric_length for pre-tension.
//   Pass "none" for either cable to omit it.
//
//   material_json: {"E1":5000.0, "r":4.0, "nu":0.7}
//     E2 is computed as E1/r. Overrides the motif material params.
//     Pass "none" (or omit) to use motif params.
//     Requires fixed_vertices to also be specified (use "auto" for default).
//
//   motif 1: E1=5000,  E2=12507  (course-stiff, E2/E1=2.50)
//   motif 2: E1=5000,  E2=8000   (mild aniso,   E2/E1=1.60)
//   motif 3: E1=5000,  E2=5000   (isotropic)
//   motif 4: E1=8000,  E2=5000   (wale-stiff,   E1/E2=1.60)
//   motif 5: E1=12507, E2=5000   (wale-stiff,   E1/E2=2.50)
//
// Outputs:
//   <prefix>_verts.csv   — deformed vertex positions (vid,x,y,z)
//   <prefix>_stress.csv  — per-face von Mises stresses
//   <prefix>_scalars.csv — crown_height, max_stress, mean_stress,
//                          cable_wale_tension, cable_course_tension,
//                          boundary_reaction_mean

#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/CompositeModel.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include "anisotropic_rest_shape.h"
#include "stress_analysis.h"
#include "sliding_cable.h"

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace Eigen;

// ── Material parameters per motif ─────────────────────────────────────────────
struct MotifParams { double E1, E2, nu, thickness, mass; };

static MotifParams motifParams(int motif)
{
    if (motif == 1) return {5000.0, 12507.0, 0.198, 1.0, 0.001};
    if (motif == 2) return {5000.0,  8000.0, 0.198, 1.0, 0.001};
    if (motif == 3) return {5000.0,  5000.0, 0.198, 1.0, 0.001};
    if (motif == 4) return {8000.0,  5000.0, 0.198, 1.0, 0.001};
    if (motif == 5) return {12507.0, 5000.0, 0.198, 1.0, 0.001};
    throw std::invalid_argument("motif must be 1–5");
}

// ── Cable specification ───────────────────────────────────────────────────────
struct CableSpec {
    std::vector<int> indices;
    double EA     = 150000.0;
    double L_rest = -1.0;    // metres; < 0 → compute from reference geometry
    bool active() const { return !indices.empty(); }
};

// ── Globals ───────────────────────────────────────────────────────────────────
fsim::Mat3<double> V0;
fsim::Mat3<int>    F;
std::vector<int>             bdrs;
std::vector<Eigen::Vector3d> face_dirs;

// ── Mesh utilities ─────────────────────────────────────────────────────────────
static std::vector<int> findBoundaryVertices(const fsim::Mat3<int>& F)
{
    std::map<std::pair<int,int>, int> cnt;
    for (int f = 0; f < F.rows(); ++f)
        for (int i = 0; i < 3; ++i) {
            int a = F(f,i), b = F(f,(i+1)%3);
            if (a > b) std::swap(a,b);
            cnt[{a,b}]++;
        }
    std::set<int> bv;
    for (auto& [e,c] : cnt)
        if (c == 1) { bv.insert(e.first); bv.insert(e.second); }
    return {bv.begin(), bv.end()};
}

static void projectFaceVectors(const fsim::Mat3<double>& V, const fsim::Mat3<int>& F,
                                std::vector<Eigen::Vector3d>& fv)
{
    for (int i = 0; i < F.rows(); ++i) {
        Eigen::Vector3d n = (V.row(F(i,1)) - V.row(F(i,0)))
                             .cross(V.row(F(i,2)) - V.row(F(i,0)));
        n.normalize();
        Eigen::Vector3d p = fv[i] - fv[i].dot(n) * n;
        fv[i] = (p.norm() < 1e-10)
               ? Eigen::Vector3d(V.row(F(i,1)) - V.row(F(i,0))).normalized()
               : p.normalized();
    }
}

// ── Newton solve ──────────────────────────────────────────────────────────────
template <class Model>
static VectorXd newtonSolve(Model& model, const VectorXd& x0)
{
    optim::NewtonSolver<double> solver;
    solver.options.display         = optim::SolverDisplay::quiet;
    solver.options.threshold       = 1e-6;
    solver.options.iteration_limit = 10000;
    for (int b : bdrs) {
        solver.options.fixed_dofs.push_back(b*3);
        solver.options.fixed_dofs.push_back(b*3+1);
        solver.options.fixed_dofs.push_back(b*3+2);
    }
    solver.solve(model, x0);
    return solver.var();
}

// ── Parse cable JSON ──────────────────────────────────────────────────────────
// Expected: {"indices":[1,2,3,...], "EA":150000.0, "L_rest":1.196}
static bool parseCableJson(const std::string& path, CableSpec& spec)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string s((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());

    auto eaPos = s.find("\"EA\"");
    if (eaPos == std::string::npos) eaPos = s.find("\"ea\"");
    if (eaPos != std::string::npos) {
        auto colon = s.find(':', eaPos);
        spec.EA = std::stod(s.substr(colon + 1));
    }

    auto lrPos = s.find("\"L_rest\"");
    if (lrPos == std::string::npos) lrPos = s.find("\"l_rest\"");
    if (lrPos != std::string::npos) {
        auto colon = s.find(':', lrPos);
        spec.L_rest = std::stod(s.substr(colon + 1));
    }

    auto open  = s.find('[');
    auto close = s.find(']');
    if (open == std::string::npos || close == std::string::npos) return false;
    std::string arr = s.substr(open + 1, close - open - 1);
    std::istringstream ss(arr);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(0, tok.find_first_not_of(" \t\n\r"));
        if (!tok.empty()) spec.indices.push_back(std::stoi(tok));
    }
    return !spec.indices.empty();
}

// ── Build SlidingCable from spec ──────────────────────────────────────────────
static SlidingCable makeCable(const CableSpec& spec)
{
    SlidingCable c(spec.indices, spec.EA, V0);  // compute geometric L_rest
    if (spec.L_rest > 0.0) c.L_rest = spec.L_rest;  // override for pre-tension
    return c;
}

// ── Cable deformed length ─────────────────────────────────────────────────────
static double cableDeformedLength(const CableSpec& spec, const fsim::Mat3<double>& Vdef)
{
    double L = 0.0;
    for (size_t k = 0; k + 1 < spec.indices.size(); ++k)
        L += (Vdef.row(spec.indices[k+1]) - Vdef.row(spec.indices[k])).norm();
    return L;
}

static double cableTension(const CableSpec& spec, const fsim::Mat3<double>& Vdef)
{
    SlidingCable c = makeCable(spec);
    double L = cableDeformedLength(spec, Vdef);
    return std::max(0.0, (c.EA / c.L_rest) * (L - c.L_rest));
}

// ── Save scalars CSV ──────────────────────────────────────────────────────────
static void saveScalarsCSV(const std::string& path,
                            double crown_height,
                            double max_stress, double mean_stress,
                            double cable_wale_tension,
                            double cable_course_tension,
                            double boundary_reaction_mean)
{
    std::ofstream out(path);
    out << "crown_height,max_stress,mean_stress,"
           "cable_wale_tension,cable_course_tension,boundary_reaction_mean\n";
    out << std::fixed << std::setprecision(8)
        << crown_height         << ","
        << max_stress           << ","
        << mean_stress          << ","
        << cable_wale_tension   << ","
        << cable_course_tension << ","
        << boundary_reaction_mean << "\n";
}

// ── Save vertex positions CSV ─────────────────────────────────────────────────
static void saveVertsCSV(const std::string& path, const VectorXd& x)
{
    fsim::Mat3<double> V = Map<const fsim::Mat3<double>>(x.data(), V0.rows(), 3);
    std::ofstream out(path);
    out << "vid,x,y,z\n" << std::fixed << std::setprecision(8);
    for (int i = 0; i < V.rows(); ++i)
        out << i << "," << V(i,0) << "," << V(i,1) << "," << V(i,2) << "\n";
}

// ── Boundary reaction ─────────────────────────────────────────────────────────
static double boundaryReactionMean(const VectorXd& grad)
{
    double sum = 0.0;
    for (int b : bdrs) {
        double fx = grad(b*3), fy = grad(b*3+1), fz = grad(b*3+2);
        sum += std::sqrt(fx*fx + fy*fy + fz*fz);
    }
    return bdrs.empty() ? 0.0 : sum / bdrs.size();
}

// ── Simulate ──────────────────────────────────────────────────────────────────
static VectorXd simulate(double sf_wale, double sf_course, double pressure,
                          const MotifParams& mp,
                          const CableSpec& wale_cable,
                          const CableSpec& course_cable)
{
    const int nF = F.rows();
    std::vector<double> s1v(nF, 1.0/sf_wale), s2v(nF, 1.0/sf_course);
    std::vector<double> E1s(nF, mp.E1), E2s(nF, mp.E2);
    std::vector<double> nus(nF, mp.nu),  ths(nF, mp.thickness);

    fsim::Mat3<double> V0_mod =
        computeAnisotropicRestShape(V0, F, bdrs, face_dirs, s1v, s2v);

    VectorXd x = Map<const VectorXd>(V0.data(), V0.size());

    const bool has_wale   = wale_cable.active();
    const bool has_course = course_cable.active();

    for (double p : {pressure*0.01, pressure*0.1, pressure*0.5, pressure}) {
        fsim::OrthotropicStVKMembrane m(V0_mod, F, ths, E1s, E2s, nus,
                                         face_dirs, mp.mass, p);
        if (has_wale && has_course) {
            auto cw = makeCable(wale_cable);
            auto cc = makeCable(course_cable);
            fsim::CompositeModel cm(std::move(m), std::move(cw), std::move(cc));
            x = newtonSolve(cm, x);
        } else if (has_wale) {
            auto cw = makeCable(wale_cable);
            fsim::CompositeModel cm(std::move(m), std::move(cw));
            x = newtonSolve(cm, x);
        } else if (has_course) {
            auto cc = makeCable(course_cable);
            fsim::CompositeModel cm(std::move(m), std::move(cc));
            x = newtonSolve(cm, x);
        } else {
            x = newtonSolve(m, x);
        }
    }
    return x;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    if (argc < 10) {
        std::cerr << "Usage: fem_batch_sensitivity <mesh_path> <sf_wale> <sf_course>"
                     " <knit_dir_deg> <pressure> <motif>"
                     " <cable_wale_json_or_none> <cable_course_json_or_none>"
                     " <output_prefix> [fixed_vertices]\n";
        return 1;
    }

    const std::string mesh_path        = argv[1];
    double sf_wale                     = std::atof(argv[2]);
    double sf_course                   = std::atof(argv[3]);
    double knit_dir_deg                = std::atof(argv[4]);
    double pressure                    = std::atof(argv[5]);
    int    motif                       = std::atoi(argv[6]);
    const std::string cable_wale_arg   = argv[7];
    const std::string cable_course_arg = argv[8];
    const std::string prefix           = argv[9];

    MotifParams mp = motifParams(motif);

    fsim::readOFF(mesh_path, V0, F);
    bdrs = findBoundaryVertices(F);

    // Optional: override fixed vertices from file
    if (argc >= 11) {
        const std::string fv_path = argv[10];
        if (fv_path != "auto") {
            bdrs.clear();
            std::ifstream fv(fv_path);
            int v;
            while (fv >> v) bdrs.push_back(v);
        }
    }

    // Optional: override material params via JSON at argv[11]
    // Format: {"E1":5000.0,"r":4.0,"nu":0.7}  — E2 = E1/r
    if (argc >= 12 && std::string(argv[11]) != "none" && std::string(argv[11]) != "None") {
        std::ifstream mf(argv[11]);
        if (mf.is_open()) {
            std::string ms((std::istreambuf_iterator<char>(mf)),
                            std::istreambuf_iterator<char>());
            auto find_val = [&](const std::string& key) -> double {
                auto pos = ms.find("\"" + key + "\"");
                if (pos == std::string::npos) return -1.0;
                auto colon = ms.find(':', pos);
                return std::stod(ms.substr(colon + 1));
            };
            double E1v = find_val("E1");
            double rv  = find_val("r");
            double nuv = find_val("nu");
            if (E1v > 0 && rv > 0 && nuv > 0) {
                mp.E1 = E1v;
                mp.E2 = E1v / rv;
                mp.nu = nuv;
            }
        }
    }

    double rad = knit_dir_deg * M_PI / 180.0;
    Eigen::Vector3d knit_dir(std::sin(rad), std::cos(rad), 0.0);
    face_dirs.assign(F.rows(), knit_dir);
    projectFaceVectors(V0, F, face_dirs);

    // Parse cables
    CableSpec wale_cable, course_cable;
    if (cable_wale_arg != "none" && cable_wale_arg != "None") {
        if (!parseCableJson(cable_wale_arg, wale_cable)) {
            std::cerr << "Failed to parse wale cable JSON: " << cable_wale_arg << "\n";
            return 1;
        }
    }
    if (cable_course_arg != "none" && cable_course_arg != "None") {
        if (!parseCableJson(cable_course_arg, course_cable)) {
            std::cerr << "Failed to parse course cable JSON: " << cable_course_arg << "\n";
            return 1;
        }
    }

    VectorXd x = simulate(sf_wale, sf_course, pressure, mp, wale_cable, course_cable);

    fsim::Mat3<double> Vdef = Map<const fsim::Mat3<double>>(x.data(), V0.rows(), 3);

    double z_min_rest  = V0.col(2).minCoeff();
    double crown_height = Vdef.col(2).maxCoeff() - z_min_rest;

    // Stresses (reuse the same rest-shape)
    const int nF = F.rows();
    std::vector<double> s1v(nF, 1.0/sf_wale), s2v(nF, 1.0/sf_course);
    std::vector<double> E1s(nF, mp.E1), E2s(nF, mp.E2);
    std::vector<double> nus(nF, mp.nu),  ths(nF, mp.thickness);
    fsim::Mat3<double> V0_mod =
        computeAnisotropicRestShape(V0, F, bdrs, face_dirs, s1v, s2v);
    fsim::OrthotropicStVKMembrane smodel(V0_mod, F, ths, E1s, E2s, nus,
                                          face_dirs, mp.mass, pressure);
    auto st = computeElementStresses(smodel, x, mp.E1, mp.E2, mp.nu, mp.thickness);

    double max_stress = 0.0, sum_stress = 0.0;
    for (auto& s : st) { max_stress = std::max(max_stress, s.von_mises); sum_stress += s.von_mises; }
    double mean_stress = st.empty() ? 0.0 : sum_stress / st.size();

    double cable_wale_tension   = wale_cable.active()   ? cableTension(wale_cable,   Vdef) : 0.0;
    double cable_course_tension = course_cable.active() ? cableTension(course_cable, Vdef) : 0.0;

    VectorXd g = smodel.gradient(x);
    double boundary_reaction = boundaryReactionMean(g);

    saveVertsCSV(prefix + "_verts.csv", x);
    saveStressCSV(prefix + "_stress.csv", st);
    saveScalarsCSV(prefix + "_scalars.csv",
                   crown_height, max_stress, mean_stress,
                   cable_wale_tension, cable_course_tension, boundary_reaction);
    return 0;
}
