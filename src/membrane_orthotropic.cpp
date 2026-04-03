#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/util/io.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include "polyscope/point_cloud.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "anisotropic_rest_shape.h"


std::vector<int> findBoundaryVertices(fsim::Mat3<int> F) {
  // Edge represented by a pair of vertex indices, with a count of how many faces share the edge
  std::map<std::pair<int, int>, int> edgeCount;

  for(int i = 0; i < F.rows(); ++i){
    auto face = F.row(i);
    int n = face.size();
    for (int i = 0; i < n; ++i) {
      int v1 = face[i];
      int v2 = face[(i + 1) % n]; // Next vertex in the face, wrapping around to the start
      if (v1 > v2) std::swap(v1, v2); // Ensure the first vertex in the pair is the smaller one
      edgeCount[{v1, v2}]++;
    }
  }

  // Set to hold boundary vertices (using a set to avoid duplicates)
  std::set<int> boundaryVertices;

  // Identify boundary edges and their vertices
  for (const auto& edge : edgeCount) {
    if (edge.second == 1) { // Boundary edge found
      boundaryVertices.insert(edge.first.first);
      boundaryVertices.insert(edge.first.second);
    }
  }

  // Convert set to vector and return
  return std::vector<int>(boundaryVertices.begin(), boundaryVertices.end());
}

void projectFaceVectorsToFaces(const fsim::Mat3<double>& V, const fsim::Mat3<int>& F, std::vector<Eigen::Vector3d>& face_vectors) {
  // Ensure the number of face vectors matches the number of faces
  if (face_vectors.size() != F.rows()) {
    std::cerr << "Error: Number of face vectors does not match the number of faces." << std::endl;
    return;
  }

  for (int i = 0; i < F.rows(); ++i) {
    // Get the vertices of the face
    Eigen::Vector3d v0 = V.row(F(i, 0));
    Eigen::Vector3d v1 = V.row(F(i, 1));
    Eigen::Vector3d v2 = V.row(F(i, 2));

    // Compute two edges of the face
    Eigen::Vector3d edge1 = v1 - v0;
    Eigen::Vector3d edge2 = v2 - v0;

    // Compute the normal of the face
    Eigen::Vector3d faceNormal = edge1.cross(edge2).normalized();

    // Get the original face vector
    Eigen::Vector3d originalVector = face_vectors[i];

    // Compute the projection of the vector onto the plane
    Eigen::Vector3d projection = originalVector - (originalVector.dot(faceNormal)) * faceNormal;

    // Normalize the projected vector
    face_vectors[i] = projection.normalized();
  }
}


std::vector<double> readNumbersFromFile(const std::string& filename) {
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }
  std::vector<double> data;
  std::string line;
  // Read the single line from the file
  if (std::getline(infile, line)) {
    std::istringstream iss(line);
    double num;
    // Parse the space-separated integers
    while (iss >> num) {
      data.push_back(num);
    }
  }
  infile.close();
  return data;
}


int main(int argc, char *argv[]) {
  using namespace Eigen;

  bool showDeviation = false;
  bool showRef = false;
  bool showFixedPoints = false;

  // load geometry from OFF mesh file
  fsim::Mat3<double> V0;
  fsim::Mat3<int> F;
//  fsim::readOFF("../data/mesh.off", V0, F);
//  std::string folder = "/Users/duch/Documents/PhD/knit/2024_prototypes/rectangle/";
//  std::string mesh_name = "gentle_unit_m.off";
  std::string folder = "/Users/duch/Documents/PhD/knit/2024_prototypes/2part/";
  std::string mesh_name = "2part_opt_simu_m.off";
//  std::string folder = "/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/two_patterns/square/";
//  std::string mesh_name = "square.off";
  std::string file_path = folder + mesh_name;
  fsim::readOFF(file_path, V0, F);

  // V0_mod is computed after stretch_factor and face_vectors are both initialised (see below).
  fsim::Mat3<double> V0_mod = V0;   // placeholder


//  fsim::Mat2<double> V = V0.leftCols<2>();
//  V0 /= 100.; // total 5m

  // parameters of the membrane model
  double young_modulus1= 5000;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
  double young_modulus2 = 12507;
  double poisson_ratio = 0.198;

//  std::string E1s_path = folder + "E1.txt";
//  std::vector<double> young_modulus1s = readNumbersFromFile(E1s_path);
//  std::string E2s_path = folder + "E2.txt";
//  std::vector<double> young_modulus2s = readNumbersFromFile(E2s_path);
//  std::string nu1s_path = folder + "nu1.txt";
//  std::vector<double> poisson_ratios = readNumbersFromFile(nu1s_path);
  std::vector<double> young_modulus1s(F.rows(), young_modulus1);
  std::vector<double> young_modulus2s(F.rows(), young_modulus2);
  std::vector<double> poisson_ratios(F.rows(), poisson_ratio);

  double thickness = 1.0;
  std::vector<double> thicknesses(F.rows(), thickness);
  double s_f1 = 1.043;  // wale stretch factor
  double s_f2 = 1.043;  // course stretch factor

  // Per-face anisotropic pre-strain ratios derived from s_f1 / s_f2.
  std::vector<double> s1_vec(F.rows(), 1.0 / s_f1);
  std::vector<double> s2_vec(F.rows(), 1.0 / s_f2);
  double mass = 0.001; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
  double pressure = 1000; // pressure per area N/m2, Pa, air-supported structure 200-300 Pa

//  double young_modulus1 = 6000; // unit: N/m
//  double young_modulus2 = 10000;
////  const double young_modulus = 10000000; // 10 MPa
//  double thickness = 1.0;  // m
//  double poisson_ratio = 0.2;
//  double stretch_factor = 1.06;
//  double mass = 0.001; // / mass per area, density * material thickness
//  double pressure = 850;  // pressure per area N/m2, Pa, air-supported structure 200-300 Pa; fomwork, 0.5-90kN/m2

////   Create face vectors (one per face)
  std::vector<Eigen::Vector3d> face_vectors(F.rows(), Eigen::Vector3d(0.0, 1.0, 0.0));


//  // read face vectors from a txt file
//  std::vector<Eigen::Vector3d> face_vectors;
//  std::string line;
//  std::ifstream infile("/Users/duch/Documents/PhD/knit/2024_prototypes/rectangle/gentle_unit_m_vertex_directional_field.txt");
//  while (std::getline(infile, line))
//  {
//    // Remove any leading/trailing whitespace
//    line.erase(0, line.find_first_not_of(" \t\n\r"));
//    line.erase(line.find_last_not_of(" \t\n\r") + 1);
//
//    // Skip empty lines
//    if (line.empty()) continue;
//
//    // Replace commas with spaces to simplify splitting
//    for (size_t i = 0; i < line.length(); ++i){
//      if (line[i] == ',') line[i] = ' ';}
//
//    // Use a stringstream to parse the numbers
//    std::stringstream ss(line);
//    double x, y, z;
//    if (ss >> x >> y >> z){
//      Eigen::Vector3d vec(x, y, z);
//      face_vectors.push_back(vec);}
//    else{
//      std::cerr << "Warning: Could not parse line: " << line << std::endl;}
//  }

  projectFaceVectorsToFaces(V0, F, face_vectors);

  // Compute anisotropic rest shape now that face_vectors are available.
  std::vector<int> bdrs_for_mod = findBoundaryVertices(F);
  V0_mod = computeAnisotropicRestShape(V0, F, bdrs_for_mod, face_vectors, s1_vec, s2_vec);

  // visualise the vector field, only for TEMPLATE now!!
  fsim::Mat3<double> faceVectorsMatrix(F.rows(), 3);
  for(int i = 0; i < F.rows(); ++i){
    faceVectorsMatrix.row(i) = face_vectors[i];
  }

  // initialise the vector field
  std::vector<glm::vec3> faceVectors(F.rows());

  for(int i = 0; i < F.rows(); ++i){
    faceVectors[i] = glm::vec3(
        static_cast<float>(faceVectorsMatrix(i, 0)),
        static_cast<float>(faceVectorsMatrix(i, 1)),
        static_cast<float>(faceVectorsMatrix(i, 2))
    );
  }


  fsim::OrthotropicStVKMembrane model(V0_mod , F, thicknesses, young_modulus1s, young_modulus2s, poisson_ratios, face_vectors, mass, pressure);

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  std::vector<int> bdrs = findBoundaryVertices(F);
//  std::vector<int> bdrs = {1, 2, 3, 11, 14, 24, 35, 38, 47, 49, 60, 69, 76, 147, 148, 155, 157, 166, 174, 176, 184, 186, 196, 204, 210, 269, 270, 277, 279, 288, 296, 298, 306, 308, 318, 326, 332, 392, 393, 401, 404, 414, 425, 428, 437, 439, 450, 459, 466, 0, 5, 6, 7, 12, 19, 80, 82, 83, 84, 88, 94, 149, 150, 151, 156, 161, 214, 215, 216, 218, 220, 224, 271, 272, 273, 275, 278, 283, 336, 337, 338, 342, 346, 391, 395, 396, 397, 402, 409, 470, 472, 473, 474, 478, 484, 9, 153, 73, 141, 77, 86, 340, 144, 208, 264, 211, 266, 330, 386, 333, 388, 399, 463, 531, 467, 476, 534};
  std::sort(bdrs.begin(), bdrs.end());

  // specify fixed degrees of freedom
  for (int bdr : bdrs) {
    solver.options.fixed_dofs.push_back(bdr * 3);
    solver.options.fixed_dofs.push_back(bdr * 3 + 1);
    solver.options.fixed_dofs.push_back(bdr * 3 + 2);
  }
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be



  // display the mesh
  polyscope::registerSurfaceMesh("mesh", V0, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1});
  polyscope::getSurfaceMesh("mesh")->addFaceVectorQuantity("Face Vectors", faceVectors)
      ->setVectorColor(glm::vec3(1.0f, 0.0f, 0.0f)); // Red color for vectors

  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;
  polyscope::init();

  // Helper: rebuild V0_mod and the model from current s_f1, s_f2 and material params.
  auto rebuildModel = [&]() {
    std::fill(s1_vec.begin(), s1_vec.end(), 1.0 / s_f1);
    std::fill(s2_vec.begin(), s2_vec.end(), 1.0 / s_f2);
    V0_mod = computeAnisotropicRestShape(V0, F, bdrs_for_mod, face_vectors, s1_vec, s2_vec);
    std::fill(young_modulus1s.begin(), young_modulus1s.end(), young_modulus1);
    std::fill(young_modulus2s.begin(), young_modulus2s.end(), young_modulus2);
    std::fill(poisson_ratios.begin(), poisson_ratios.end(), poisson_ratio);
    std::fill(thicknesses.begin(), thicknesses.end(), thickness);
    model = fsim::OrthotropicStVKMembrane(V0_mod, F, thicknesses, young_modulus1s, young_modulus2s, poisson_ratios, face_vectors, mass, pressure);
  };

  polyscope::state::userCallback = [&]()
  {
      ImGui::PushItemWidth(100);

      if(ImGui::InputDouble("SF wale (s_f1)", &s_f1, 0, 0, "%.4f"))
        rebuildModel();

      if(ImGui::InputDouble("SF course (s_f2)", &s_f2, 0, 0, "%.4f"))
        rebuildModel();

      if(ImGui::InputDouble("Thickness", &thickness, 0, 0, "%.2f"))
        rebuildModel();

      if(ImGui::InputDouble("Poisson", &poisson_ratio, 0, 0, "%.3f"))
        rebuildModel();

      if(ImGui::InputDouble("Modulus1", &young_modulus1, 0, 0, "%.2f"))
        rebuildModel();

      if(ImGui::InputDouble("Modulus2", &young_modulus2, 0, 0, "%.2f"))
        rebuildModel();

      if(ImGui::InputDouble("Mass", &mass, 0, 0, "%.3f"))
        model.setMass(mass);

      if(ImGui::InputDouble("Pressure", &pressure, 0, 0, "%.2f"))
        model.setPressure(pressure);

      if(ImGui::Button("Solve"))
      {
        // Newton's method: finds a local minimum of the energy (Fval = energy value, Optimality = gradient's norm)
        // the V0.data() is the initial guess, also the fixed DF is defined here!
        solver.solve(model, Map<VectorXd>(V0.data(), V0.size()));

        // Display the result of the optimization
        polyscope::getSurfaceMesh("mesh")->updateVertexPositions(
            Map<fsim::Mat3<double>>(solver.var().data(), V0.rows(), 3));


        if (showDeviation) {

//          Eigen::MatrixXd VTarget = xTarget.reshaped<Eigen::RowMajor>(V.rows(), 3);
//          Eigen::MatrixXd Vsolve = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3))
          fsim::Mat3<double> Vsolve = Map<fsim::Mat3<double>>(solver.var().data(), V0.rows(), 3);
          Eigen::VectorXd d = (V0 - Vsolve).cwiseProduct((V0 - Vsolve)).rowwise().sum();
          d = d.array().sqrt();
          std::cout << d << std::endl;
          std::cout << "Avg distance = "
                    << 100 * d.sum() / d.size() / (Vsolve.colwise().maxCoeff() - Vsolve.colwise().minCoeff()).norm()
                    << "\n";
          std::cout << "Max distance = "
                    << 100 * d.lpNorm<Eigen::Infinity>() /
                       (Vsolve.colwise().maxCoeff() - Vsolve.colwise().minCoeff()).norm()
                    << "\n";
          polyscope::getSurfaceMesh("mesh")->addVertexScalarQuantity("Distance", d)->setEnabled(true);
        }

        std::ostringstream expfilename;
        expfilename << folder + "exp/"
                 << "YM1_" << young_modulus1
                 << "_YM2_" << young_modulus2
                 << "_PR_" << poisson_ratio
                 << "_SF1_" << s_f1
                 << "_SF2_" << s_f2
//                 << "_M_" << mass
                 << "_P_" << pressure;

        fsim::saveOBJ(expfilename.str(), Map<fsim::Mat3<double>>(solver.var().data(), V0.rows(), 3), F);


      };

    if (ImGui::Checkbox("Set Deviation", &showDeviation)) {
      }

    if (ImGui::Checkbox("Show Reference Mesh", &showRef)) {
      static polyscope::SurfaceMesh *refMesh = polyscope::registerSurfaceMesh("refmesh", V0, F);
      refMesh->setEnabled(showRef);
      if (showRef) {
        refMesh->setEdgeWidth(1);
        refMesh->setEdgeColor({0.1, 0.1, 0.1});
        refMesh->setSurfaceColor({0, 1., 1.});
    }
  }
      if (ImGui::Checkbox("Show Fixed Points", &showFixedPoints)) {
        std::vector<glm::vec3> points(bdrs.size());

        for (int bdr : bdrs) {
          // fixed points
            points.emplace_back(glm::vec3(V0.row(bdr)[0], V0.row(bdr)[1], V0.row(bdr)[2]));
        }
        polyscope::PointCloud* psCloud = polyscope::registerPointCloud("Fixed points", points);
        psCloud->setEnabled(showFixedPoints);

        if (showFixedPoints){
          // visualize fixes
          psCloud->setPointRadius(0.02);
          psCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);
        }
      }
  };
  polyscope::show();
}