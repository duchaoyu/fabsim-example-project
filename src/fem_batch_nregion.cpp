// fem_batch_nregion.cpp
//
// Headless FEM runner with N spatial regions + sliding cables.
// Each region has its own stretch factors and knit direction.
// Designed for the FDM-guided inverse optimisation of B5.
//
// Usage:
//   ./fem_batch_nregion <mesh_path> <region_map.json> <params.json> <output_prefix>
//
// region_map.json  — {"face_regions": [0, 2, 1, 5, ...]}   (one int per face)
// params.json      — {
//                      "pressure": 500.0,
//                      "motif": 1,
//                      "cable_ea": 157000.0,
//                      "cable_paths": [[v0,v1,...], [v0,v1,...], ...],
//                      "regions": [
//                        {"sf_wale": 1.0, "sf_course": 1.0, "knit_dir_deg": 90.0},
//                        ...
//                      ]
//                    }
//
// Outputs:
//   <prefix>_verts.csv    — deformed vertex positions (vid, x, y, z)
//   <prefix>_scalars.csv  — crown_height, max_stress, mean_stress,
//                           boundary_reaction_mean

#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/CompositeModel.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include "anisotropic_rest_shape.h"
#include "sliding_cable.h"
#include "stress_analysis.h"

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
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
    throw std::invalid_argument("motif must be 1-5");
}

// ── Per-region parameters ──────────────────────────────────────────────────────
struct RegionParams {
    double sf_wale    = 1.0;
    double sf_course  = 1.0;
    double knit_dir_deg = 0.0;
};

// ── Multi-cable aggregate model ───────────────────────────────────────────────
// Aggregates any number of SlidingCable objects into a single model
// that satisfies the CompositeModel interface.
struct MultiCableModel {
    std::vector<SlidingCable> cables;

    double energy(const Eigen::Ref<const Eigen::VectorXd>& X) const {
        double e = 0.0;
        for (auto& c : cables) e += c.energy(X);
        return e;
    }

    void gradient(const Eigen::Ref<const Eigen::VectorXd>& X,
                  Eigen::Ref<Eigen::VectorXd> Y) const {
        for (auto& c : cables) c.gradient(X, Y);
    }

    Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd>& X) const {
        Eigen::VectorXd Y = Eigen::VectorXd::Zero(X.size());
        gradient(X, Y);
        return Y;
    }

    std::vector<Eigen::Triplet<double>>
    hessianTriplets(const Eigen::Ref<const Eigen::VectorXd>& X) const {
        std::vector<Eigen::Triplet<double>> trips;
        for (auto& c : cables) {
            auto t = c.hessianTriplets(X);
            trips.insert(trips.end(), t.begin(), t.end());
        }
        return trips;
    }

    Eigen::SparseMatrix<double>
    hessian(const Eigen::Ref<const Eigen::VectorXd>& X) const {
        auto trips = hessianTriplets(X);
        Eigen::SparseMatrix<double> H(X.size(), X.size());
        H.setFromTriplets(trips.begin(), trips.end());
        return H;
    }

    bool empty() const { return cables.empty(); }
};

// ── Globals ───────────────────────────────────────────────────────────────────
fsim::Mat3<double>           V0;
fsim::Mat3<int>              F;
std::vector<int>             bdrs;
std::vector<Eigen::Vector3d> face_dirs;

// ── Mesh utilities ─────────────────────────────────────────────────────────────
static std::vector<int> findBoundaryVertices(const fsim::Mat3<int>& F)
{
    std::map<std::pair<int,int>, int> cnt;
    for (int f = 0; f < F.rows(); ++f)
        for (int k = 0; k < 3; ++k) {
            int a = F(f,k), b = F(f,(k+1)%3);
            if (a > b) std::swap(a, b);
            cnt[{a,b}]++;
        }
    std::set<int> bv;
    for (auto& [e, c] : cnt)
        if (c == 1) { bv.insert(e.first); bv.insert(e.second); }
    return {bv.begin(), bv.end()};
}

static void projectFaceVectors(const fsim::Mat3<double>& V,
                                const fsim::Mat3<int>& F,
                                std::vector<Eigen::Vector3d>& fv)
{
    for (int i = 0; i < F.rows(); ++i) {
        Eigen::Vector3d n = (V.row(F(i,1)) - V.row(F(i,0)))
                             .cross(V.row(F(i,2)) - V.row(F(i,0)));
        n.normalize();
        Eigen::Vector3d p = fv[i] - fv[i].dot(n) * n;
        if (p.norm() < 1e-10)
            fv[i] = Eigen::Vector3d(V.row(F(i,1)) - V.row(F(i,0))).normalized();
        else
            fv[i] = p.normalized();
    }
}

// ── Simple JSON parsing helpers ───────────────────────────────────────────────
static double jsonDouble(const std::string& s, const std::string& key, double def = 0.0)
{
    auto pos = s.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    auto colon = s.find(':', pos);
    if (colon == std::string::npos) return def;
    return std::stod(s.substr(colon + 1));
}

static int jsonInt(const std::string& s, const std::string& key, int def = 0)
{
    auto pos = s.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    auto colon = s.find(':', pos);
    if (colon == std::string::npos) return def;
    return std::stoi(s.substr(colon + 1));
}

// Parse a flat numeric array: "<key>": [a, b, c, ...]
static std::vector<double> jsonDoubleArray(const std::string& s, const std::string& key)
{
    std::vector<double> result;
    auto pos = s.find("\"" + key + "\"");
    if (pos == std::string::npos) return result;
    auto open  = s.find('[', pos);
    auto close = s.find(']', open);
    if (open == std::string::npos || close == std::string::npos) return result;
    std::string arr = s.substr(open + 1, close - open - 1);
    std::istringstream ss(arr);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(0, tok.find_first_not_of(" \t\n\r"));
        if (!tok.empty()) result.push_back(std::stod(tok));
    }
    return result;
}

static std::vector<int> jsonIntArray(const std::string& s, const std::string& key)
{
    std::vector<int> result;
    auto pos = s.find("\"" + key + "\"");
    if (pos == std::string::npos) return result;
    auto open  = s.find('[', pos);
    auto close = s.find(']', open);
    if (open == std::string::npos || close == std::string::npos) return result;
    std::string arr = s.substr(open + 1, close - open - 1);
    std::istringstream ss(arr);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(0, tok.find_first_not_of(" \t\n\r"));
        if (!tok.empty()) result.push_back(std::stoi(tok));
    }
    return result;
}

struct RegionMapData {
    std::vector<int>    face_regions;
    std::vector<double> face_knit_dirs_deg; // empty = use region-level knit_dir_deg
};

static RegionMapData parseRegionMap(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open region_map: " + path);
    std::string s((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
    RegionMapData data;
    data.face_regions      = jsonIntArray(s,    "face_regions");
    data.face_knit_dirs_deg = jsonDoubleArray(s, "face_knit_dirs_deg");
    if (data.face_regions.empty())
        throw std::runtime_error("No 'face_regions' array in region_map JSON");
    return data;
}

static std::vector<RegionParams> parseRegions(const std::string& s)
{
    std::vector<RegionParams> regions;
    auto arr_start = s.find("\"regions\"");
    if (arr_start == std::string::npos)
        throw std::runtime_error("No 'regions' array in params JSON");
    auto open = s.find('[', arr_start);
    // find the matching ] for the regions array (skip nested)
    size_t ca = open;
    int depth = 0;
    for (size_t i = open; i < s.size(); ++i) {
        if (s[i] == '[') depth++;
        else if (s[i] == ']') { depth--; if (depth == 0) { ca = i; break; } }
    }

    size_t pos = open + 1;
    while (pos < ca) {
        auto ob = s.find('{', pos);
        if (ob == std::string::npos || ob >= ca) break;
        auto cb = s.find('}', ob);
        if (cb == std::string::npos) break;
        std::string obj = s.substr(ob, cb - ob + 1);
        RegionParams rp;
        rp.sf_wale      = jsonDouble(obj, "sf_wale",      1.0);
        rp.sf_course    = jsonDouble(obj, "sf_course",    1.0);
        rp.knit_dir_deg = jsonDouble(obj, "knit_dir_deg", 0.0);
        regions.push_back(rp);
        pos = cb + 1;
    }
    return regions;
}

// Parse "cable_paths": [[v0,v1,...], [v0,v1,...], ...]
static std::vector<std::vector<int>> parseCablePaths(const std::string& s)
{
    std::vector<std::vector<int>> result;
    auto key_pos = s.find("\"cable_paths\"");
    if (key_pos == std::string::npos) return result;
    auto open_outer = s.find('[', key_pos);
    if (open_outer == std::string::npos) return result;

    // Find the matching outer ]
    size_t outer_close = open_outer;
    int depth = 0;
    for (size_t i = open_outer; i < s.size(); ++i) {
        if (s[i] == '[') depth++;
        else if (s[i] == ']') { depth--; if (depth == 0) { outer_close = i; break; } }
    }

    // Iterate through inner arrays [v0, v1, ...]
    size_t pos = open_outer + 1;
    while (pos < outer_close) {
        auto ib = s.find('[', pos);
        if (ib == std::string::npos || ib >= outer_close) break;
        auto ie = s.find(']', ib);
        if (ie == std::string::npos) break;
        std::string arr = s.substr(ib + 1, ie - ib - 1);
        std::istringstream ss(arr);
        std::string tok;
        std::vector<int> path;
        while (std::getline(ss, tok, ',')) {
            tok.erase(0, tok.find_first_not_of(" \t\n\r"));
            if (!tok.empty()) path.push_back(std::stoi(tok));
        }
        if (!path.empty()) result.push_back(std::move(path));
        pos = ie + 1;
    }
    return result;
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

// ── Simulate ──────────────────────────────────────────────────────────────────
static VectorXd simulate(const std::vector<RegionParams>& regions,
                         const std::vector<int>& face_reg,
                         const MotifParams& mp,
                         double pressure,
                         const MultiCableModel& cable_template)
{
    const int nF = F.rows();
    std::vector<double> s1v(nF), s2v(nF);
    std::vector<double> E1s(nF, mp.E1), E2s(nF, mp.E2);
    std::vector<double> nus(nF, mp.nu), ths(nF, mp.thickness);

    for (int f = 0; f < nF; ++f) {
        int r = face_reg[f];
        s1v[f] = 1.0 / regions[r].sf_wale;
        s2v[f] = 1.0 / regions[r].sf_course;
    }

    fsim::Mat3<double> V0_mod =
        computeAnisotropicRestShape(V0, F, bdrs, face_dirs, s1v, s2v);

    VectorXd x = Map<const VectorXd>(V0.data(), V0.size());

    for (double p : { pressure*0.01, pressure*0.1, pressure*0.5, pressure }) {
        fsim::OrthotropicStVKMembrane membrane(
            V0_mod, F, ths, E1s, E2s, nus, face_dirs, mp.mass, p);

        if (cable_template.empty()) {
            x = newtonSolve(membrane, x);
        } else {
            fsim::CompositeModel composite(std::move(membrane),
                                           MultiCableModel(cable_template));
            x = newtonSolve(composite, x);
        }
    }
    return x;
}

// ── Save outputs ───────────────────────────────────────────────────────────────
static void saveVertsCSV(const std::string& path, const VectorXd& x)
{
    fsim::Mat3<double> V = Map<const fsim::Mat3<double>>(x.data(), V0.rows(), 3);
    std::ofstream out(path);
    out << "vid,x,y,z\n" << std::fixed << std::setprecision(8);
    for (int i = 0; i < V.rows(); ++i)
        out << i << "," << V(i,0) << "," << V(i,1) << "," << V(i,2) << "\n";
}

static void saveScalarsCSV(const std::string& path, const VectorXd& x,
                            const std::vector<ElementStress>& st,
                            const MotifParams& mp, double pressure)
{
    std::set<int> bdry_set(bdrs.begin(), bdrs.end());
    double crown = 0.0;
    for (int i = 0; i < V0.rows(); ++i)
        if (bdry_set.find(i) == bdry_set.end())
            crown = std::max(crown, x[i*3 + 2]);

    double max_s = 0.0, sum_s = 0.0;
    for (auto& e : st) {
        max_s  = std::max(max_s, e.von_mises);
        sum_s += e.von_mises;
    }
    double mean_s = st.empty() ? 0.0 : sum_s / st.size();

    double bdry_rxn = 0.0;
    for (int b : bdrs) {
        double dx = x[b*3] - V0(b,0);
        double dy = x[b*3+1] - V0(b,1);
        double dz = x[b*3+2] - V0(b,2);
        bdry_rxn += std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    bdry_rxn = bdrs.empty() ? 0.0 : bdry_rxn / bdrs.size();

    std::ofstream out(path);
    out << std::fixed << std::setprecision(8);
    out << "crown_height,max_stress,mean_stress,boundary_reaction_mean\n";
    out << crown << "," << max_s << "," << mean_s << "," << bdry_rxn << "\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr
            << "Usage: fem_batch_nregion <mesh_path> <region_map.json>"
               " <params.json> <output_prefix>\n";
        return 1;
    }

    const std::string mesh_path   = argv[1];
    const std::string map_path    = argv[2];
    const std::string params_path = argv[3];
    const std::string prefix      = argv[4];

    // Load mesh
    fsim::readOFF(mesh_path, V0, F);
    bdrs = findBoundaryVertices(F);

    // Parse region map
    RegionMapData rmd = parseRegionMap(map_path);
    std::vector<int>& face_reg = rmd.face_regions;
    if ((int)face_reg.size() != F.rows()) {
        std::cerr << "region_map has " << face_reg.size()
                  << " entries but mesh has " << F.rows() << " faces\n";
        return 1;
    }

    // Parse params
    std::ifstream pf(params_path);
    if (!pf.is_open()) {
        std::cerr << "Cannot open params: " << params_path << "\n";
        return 1;
    }
    std::string ps((std::istreambuf_iterator<char>(pf)),
                    std::istreambuf_iterator<char>());
    double pressure = jsonDouble(ps, "pressure", 500.0);
    int    motif    = jsonInt(ps,    "motif",     1);
    double cable_ea = jsonDouble(ps, "cable_ea",  157000.0);
    std::vector<RegionParams> regions = parseRegions(ps);
    std::vector<std::vector<int>> cable_paths = parseCablePaths(ps);
    std::vector<double> cable_rest_scales = jsonDoubleArray(ps, "cable_rest_scales");

    if (regions.empty()) {
        std::cerr << "No regions found in params JSON\n";
        return 1;
    }

    int n_regions = (int)regions.size();
    for (int fi = 0; fi < F.rows(); ++fi) {
        if (face_reg[fi] < 0 || face_reg[fi] >= n_regions) {
            std::cerr << "Face " << fi << " has invalid region " << face_reg[fi]
                      << " (n_regions=" << n_regions << ")\n";
            return 1;
        }
    }

    MotifParams mp = motifParams(motif);

    // Per-face knit directions: use field-derived per-face angles if provided,
    // otherwise fall back to the region-level knit_dir_deg.
    const std::vector<double>& face_knit_deg = rmd.face_knit_dirs_deg;
    const bool has_per_face_knit = ((int)face_knit_deg.size() == F.rows());
    if (!face_knit_deg.empty() && !has_per_face_knit)
        std::cerr << "Warning: face_knit_dirs_deg size " << face_knit_deg.size()
                  << " != nF " << F.rows() << "; falling back to region dirs\n";
    face_dirs.resize(F.rows());
    for (int fi = 0; fi < F.rows(); ++fi) {
        double deg = has_per_face_knit ? face_knit_deg[fi]
                                       : regions[face_reg[fi]].knit_dir_deg;
        double rad = deg * M_PI / 180.0;
        face_dirs[fi] = Eigen::Vector3d(std::cos(rad), std::sin(rad), 0.0);
    }
    projectFaceVectors(V0, F, face_dirs);

    // Build cable model
    MultiCableModel cables;
    int cable_idx = 0;
    for (auto& path : cable_paths) {
        if (path.size() < 2) continue;
        bool valid = true;
        for (int v : path)
            if (v < 0 || v >= V0.rows()) { valid = false; break; }
        if (!valid) {
            std::cerr << "Cable path has out-of-range vertex index\n";
            return 1;
        }
        // Optional per-cable rest-length scale: L_rest = scale * L_geom (scale<1 → pre-tensioned)
        if (cable_idx < (int)cable_rest_scales.size()) {
            double L_geom = 0.0;
            for (size_t k = 0; k + 1 < path.size(); ++k)
                L_geom += (V0.row(path[k+1]) - V0.row(path[k])).norm();
            double L_rest = cable_rest_scales[cable_idx] * L_geom;
            cables.cables.emplace_back(path, cable_ea, L_rest);
        } else {
            cables.cables.emplace_back(path, cable_ea, V0);
        }
        cable_idx++;
    }

    std::cerr << "Mesh: " << V0.rows() << "v  " << F.rows() << "f  "
              << bdrs.size() << " boundary  "
              << cables.cables.size() << " cables\n";

    // Run simulation
    VectorXd x = simulate(regions, face_reg, mp, pressure, cables);

    // Compute stresses
    const int nF = F.rows();
    std::vector<double> s1v(nF), s2v(nF);
    for (int fi = 0; fi < nF; ++fi) {
        s1v[fi] = 1.0 / regions[face_reg[fi]].sf_wale;
        s2v[fi] = 1.0 / regions[face_reg[fi]].sf_course;
    }
    std::vector<double> E1s(nF,mp.E1), E2s(nF,mp.E2), nus(nF,mp.nu), ths(nF,mp.thickness);
    fsim::Mat3<double> V0_mod =
        computeAnisotropicRestShape(V0, F, bdrs, face_dirs, s1v, s2v);
    fsim::OrthotropicStVKMembrane smodel(
        V0_mod, F, ths, E1s, E2s, nus, face_dirs, mp.mass, pressure);
    auto st = computeElementStresses(smodel, x, mp.E1, mp.E2, mp.nu, mp.thickness);

    saveVertsCSV(prefix + "_verts.csv", x);
    saveStressCSV(prefix + "_stress.csv", st);
    saveScalarsCSV(prefix + "_scalars.csv", x, st, mp, pressure);

    // Crown from non-boundary vertices
    std::set<int> bdry_set(bdrs.begin(), bdrs.end());
    double crown = 0.0;
    for (int i = 0; i < V0.rows(); ++i)
        if (!bdry_set.count(i))
            crown = std::max(crown, x[i*3 + 2]);

    std::cerr << "OK  crown=" << std::fixed << std::setprecision(4) << crown << "\n";
    return 0;
}
