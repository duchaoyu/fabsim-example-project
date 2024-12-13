#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include <random>
#include "polyscope/point_cloud.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>

using namespace Eigen;

// Genetic Algorithm Parameters
const int populationSize = 100;
const double mutationRate = 0.05;
const int maxGenerations = 100;

// Data structures for mesh and variables
fsim::Mat3<double> V, VTarget;
fsim::Mat3<int> F;
std::vector<std::vector<int>> adjacentFacesList;

double young_modulus1= 5000;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
double young_modulus2 = 12507;
double thickness = 1.0;
double poisson_ratio = 0.198;
double stretch_factor = 1.043;
double mass = 0.001; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
double pressure = 1199; // pressure per area N/m2, Pa, air-supported structure 200-300 Pa
std::string outputFilePath = "/Users/duch/Documents/PhD/knit/2024_prototypes/4part/1030/half4parts_mm_face_directional_field_mod_ga.txt";


#include <set>

std::set<std::pair<int, int>> excludedEdges = {
    {3, 204}, {3, 326}, {147, 174}, {148, 176}, {148, 186}, {155, 166},
    {155, 184}, {157, 196}, {157, 210}, {166, 186}, {174, 184}, {176, 196},
    {204, 210}, {269, 296}, {270, 298}, {270, 308}, {277, 288}, {277, 306},
    {279, 318}, {279, 332}, {288, 308}, {296, 306}, {298, 318}, {326, 332}
};

// Ensure edges are always stored as (min, max) to avoid duplicates
auto normalizeEdge = [](int v1, int v2) {
    return std::make_pair(std::min(v1, v2), std::max(v1, v2));
};


// Function to save face_vectors to a text file
void saveFaceVectors(const std::vector<Eigen::Vector3d>& face_vectors, const std::string& filePath) {
  std::ofstream outFile(filePath);

  if (!outFile.is_open()) {
    std::cerr << "Error: Unable to open file for writing: " << filePath << std::endl;
    return;
  }

  for (const auto& vec : face_vectors) {
    outFile << vec.x() << ", " << vec.y() << ", " << vec.z() << "\n";
  }

  outFile.close();
  std::cout << "Face vectors saved to " << filePath << std::endl;
}


struct HalfEdge {
    int vertexStart; // Index of the start vertex of the half-edge
    int vertexEnd; // Index of the end vertex of the half-edge
    int face; // Index of the face this half-edge belongs to
    int oppositeHalfEdge; // Index of the opposite half-edge (-1 if unknown)

    HalfEdge(int start, int end, int f) : vertexStart(start), vertexEnd(end), face(f), oppositeHalfEdge(-1) {}
};

// Use a vector to store all half-edges
std::vector<HalfEdge> halfEdges;

#include <unordered_map>
#include <utility>

// Custom hash function for pairs (needed for using pairs as keys in unordered_map)
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &pair) const {
      auto hash1 = std::hash<T1>{}(pair.first);
      auto hash2 = std::hash<T2>{}(pair.second);
      return hash1 ^ hash2;
    }
};

std::unordered_map<std::pair<int, int>, int, pair_hash> edgeToHalfEdgeMap;




// Function to compute deviation
double calculateDeviation(const fsim::Mat3<double>& VSimulated) {
  Eigen::VectorXd d = (VTarget - VSimulated).cwiseProduct((VTarget - VSimulated)).rowwise().sum();
  return d.mean();  // Average deviation
}

// Function to compute smoothness penalty
double calculateSmoothnessPenalty(const std::vector<Eigen::Vector3d>& face_vectors) {
  double penalty = 0.0;
  for (int f = 0; f < F.rows(); ++f) {
    for (int neighbor : adjacentFacesList[f]) {
      double angleDiff = face_vectors[f].dot(face_vectors[neighbor]);
//      std::cout << f << ", " << neighbor << ", " << angleDiff << std::endl;
      penalty += (1.0 - angleDiff);  // Penalize larger angular differences
    }
  }
  return penalty;
}

// Objective Function
double objectiveFunction(const std::vector<Eigen::Vector3d>& face_vectors) {
  // Create and solve the membrane model
  fsim::OrthotropicStVKMembrane model(V / stretch_factor , F, thickness, young_modulus1, young_modulus2, poisson_ratio, face_vectors, mass, pressure);
  optim::NewtonSolver<double> solver;

//  std::vector<int> bdrs = findBoundaryVertices(F);
  std::vector<int> bdrs = {1, 2, 3, 11, 14, 24, 35, 38, 47, 49, 60, 69, 76, 147, 148, 155, 157, 166, 174, 176, 184, 186, 196, 204, 210, 269, 270, 277, 279, 288, 296, 298, 306, 308, 318, 326, 332, 392, 393, 401, 404, 414, 425, 428, 437, 439, 450, 459, 466, 0, 5, 6, 7, 12, 19, 80, 82, 83, 84, 88, 94, 149, 150, 151, 156, 161, 214, 215, 216, 218, 220, 224, 271, 272, 273, 275, 278, 283, 336, 337, 338, 342, 346, 391, 395, 396, 397, 402, 409, 470, 472, 473, 474, 478, 484, 9, 153, 73, 141, 77, 86, 340, 144, 208, 264, 211, 266, 330, 386, 333, 388, 399, 463, 531, 467, 476, 534};

//  std::vector<int> bdrs = {11, 13, 25, 29, 35, 51, 55, 65, 73, 79, 98, 99, 100, 104, 105, 114, 118, 130, 133, 142, 144, 149, 153, 155, 157, 158, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 186, 188, 200, 204, 210, 226, 230, 247, 274, 275, 284, 288, 302, 311, 313, 314, 322, 324, 326, 327, 332, 333, 334, 336, 337, 338, 339};
  std::sort(bdrs.begin(), bdrs.end());

  for (int bdr : bdrs) {
    solver.options.fixed_dofs.push_back(bdr * 3);
    solver.options.fixed_dofs.push_back(bdr * 3 + 1);
    solver.options.fixed_dofs.push_back(bdr * 3 + 2);
  }
  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
//  solver.options.fixed_dofs = {0 * 3 + 0, 0 * 3 + 1, 0 * 3 + 2, 1 * 3 + 0,
//                               1 * 3 + 1, 1 * 3 + 2, 2 * 3 + 0, 2 * 3 + 1,
//                               2 * 3 + 2, 3 * 3 + 0, 3 * 3 + 1, 3 * 3 + 2};
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be

  solver.solve(model, Map<VectorXd>(V.data(), V.size()));

  // Simulated vertex positions
  fsim::Mat3<double> VSimulated = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);

  // Deviation and smoothness penalty
  double deviation = calculateDeviation(VSimulated);
  double smoothness = calculateSmoothnessPenalty(face_vectors);

  return deviation * 100000000 + smoothness;  // Weighted sum
  std::cout << "deviation, " << deviation <<  "smoothness" << smoothness << std::endl;

}

// Individual structure
struct Individual {
    std::vector<Eigen::Vector3d> face_vectors; // Directional fields
    double fitness;

    Individual(int numFaces) : face_vectors(numFaces), fitness(0.0) {
      std::ifstream infile("/Users/duch/Documents/PhD/knit/2024_prototypes/4part/1030/half4parts_mm_face_directional_field_mod_ga.txt");
      if (infile.is_open()) {
        std::string line;
        int count = 0;
        while (std::getline(infile, line)) {
          // Trim whitespace
          line.erase(0, line.find_first_not_of(" \t\n\r"));
          line.erase(line.find_last_not_of(" \t\n\r") + 1);

          // Skip empty lines
          if (line.empty()) continue;

          // Replace commas with spaces to simplify parsing
          for (char& c : line) {
            if (c == ',') c = ' ';
          }

          // Parse the line into x, y, z values
          std::stringstream ss(line);
          double x, y, z;
          if (ss >> x >> y >> z) {
            face_vectors[count] = Eigen::Vector3d(x, y, z).normalized();
            count++;
            if (count >= numFaces) break;  // Stop if we have enough vectors
          } else {
            std::cerr << "Warning: Could not parse line: " << line << std::endl;
          }
        }

        // Handle case where file has fewer vectors than required
        for (int i = count; i < numFaces; ++i) {
          face_vectors[i] = Eigen::Vector3d(1.0, 1.0, 0.0).normalized();
          std::cout << "missing face vector: " << i << std::endl;
        }
      }


      //      for (auto& vec : face_vectors) {
//        vec = Vector3d(1.0, 1.0, 0.0).normalized();  // Initialize with random unit vectors
//      }
    }

    void updateFitness() {
      fitness = objectiveFunction(face_vectors);
    }
};

// Generate a random unit vector
Eigen::Vector3d randomUnitVector() {
  static std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  Eigen::Vector3d vec(dist(rng), dist(rng), dist(rng));
  return vec.normalized();
}

// Crossover between two individuals
Individual crossover(const Individual& parent1, const Individual& parent2) {
  Individual child(parent1.face_vectors.size());
  for (int i = 0; i < child.face_vectors.size(); ++i) {
    child.face_vectors[i] = (std::rand() / (RAND_MAX + 1.0)) < 0.5 ? parent1.face_vectors[i] : parent2.face_vectors[i];
  }
  child.updateFitness();
  return child;
}

// Mutation
void mutate(Individual& individual) {
  for (auto& vec : individual.face_vectors) {
    if ((std::rand() / (RAND_MAX + 1.0)) < mutationRate) {
      vec += randomUnitVector() * 0.1;  // Small random perturbation
      vec.normalize();  // Re-normalize to unit length
    }
  }
  individual.updateFitness();
}

// Genetic Algorithm
void geneticAlgorithm(int numFaces) {
  std::srand(std::time(nullptr));

  // Initialize population
  std::vector<Individual> population;
  for (int i = 0; i < populationSize; ++i) {
    population.emplace_back(numFaces);
    population.back().updateFitness();
  }

  // Evolution loop
  for (int generation = 0; generation < maxGenerations; ++generation) {
    // Sort population by fitness
    std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
        return a.fitness < b.fitness;
    });

    // Print best fitness
    std::cout << "Generation " << generation << ": Best fitness = " << population[0].fitness << "\n";
    saveFaceVectors(population[0].face_vectors, outputFilePath);

//    for(int i = 0; i < F.rows(); ++i){
//      std::cout << population[0].face_vectors[i] << std::endl;
//    }
    // Create next generation
    std::vector<Individual> newGeneration;
    for (int i = 0; i < populationSize / 2; i += 2) {
      Individual parent1 = population[i];
      Individual parent2 = population[i + 1];

      Individual child1 = crossover(parent1, parent2);
      Individual child2 = crossover(parent2, parent1);

      mutate(child1);
      mutate(child2);

      newGeneration.push_back(parent1);
      newGeneration.push_back(parent2);
      newGeneration.push_back(child1);
      newGeneration.push_back(child2);
    }

    population = std::move(newGeneration);
  }

  // Output best solution
  const auto& bestIndividual = population[0];
  std::cout << "Best solution found with fitness: " << bestIndividual.fitness << "\n";

  // Update Polyscope display
  fsim::OrthotropicStVKMembrane model(V / stretch_factor , F, thickness, young_modulus1, young_modulus2, poisson_ratio, bestIndividual.face_vectors, mass, pressure);

  optim::NewtonSolver<double> solver;
  solver.solve(model, Map<VectorXd>(V.data(), V.size()));
  fsim::Mat3<double> VSimulated = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
  polyscope::getSurfaceMesh("mesh")->updateVertexPositions(VSimulated);

  // visualise the vector field, only for TEMPLATE now!!
  fsim::Mat3<double> faceVectorsMatrix(F.rows(), 3);
  for(int i = 0; i < F.rows(); ++i){
    faceVectorsMatrix.row(i) = bestIndividual.face_vectors[i];
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
  polyscope::getSurfaceMesh("mesh")->addFaceVectorQuantity("Face Vectors", faceVectors)
      ->setVectorColor(glm::vec3(1.0f, 0.0f, 0.0f)); // Red color for vectors



}

int main() {
  // Load mesh and target
  fsim::readOFF("/Users/duch/Documents/PhD/knit/2024_prototypes/4part/1030/4parts_unit_m.off", V, F);
  fsim::readOFF("/Users/duch/Documents/PhD/knit/2024_prototypes/4part/1030/4parts_unit_m.off", VTarget, F);



  // find adjacent faces
  for (int f = 0; f < F.rows(); ++f) {
    const auto& face = F.row(f);
    for (int i = 0; i < 3; ++i) {
      int start = face[i];
      int end = face[(i + 1) % face.size()]; // Wrap around to the first vertex
      halfEdges.push_back(HalfEdge(start, end, f));
    }
  }

  adjacentFacesList.resize(F.rows());

  for (int i = 0; i < halfEdges.size(); ++i) {
    const auto& he = halfEdges[i];
    edgeToHalfEdgeMap[std::make_pair(he.vertexStart, he.vertexEnd)] = i;
  }

  for (int f = 0; f < F.rows(); ++f) {
    const auto& face = F.row(f);
    for (int i = 0; i < 3; ++i) { // Since each face has 3 vertices
      int start = face[i];
      int end = face[(i + 1) % 3]; // Next vertex in the face, wrapping around

      // Normalize the edge (smallest vertex first)
      auto edge = normalizeEdge(start, end);

      // Check if the edge is excluded
      if (excludedEdges.find(edge) != excludedEdges.end()) {
        continue; // Skip this edge
      }

      auto oppositeHeIt = edgeToHalfEdgeMap.find(std::make_pair(end, start));
      if (oppositeHeIt != edgeToHalfEdgeMap.end()) {
        int oppositeHeIndex = oppositeHeIt->second;

        // Get the face that the opposite half-edge belongs to
        int adjacentFaceIndex = halfEdges[oppositeHeIndex].face;

        // Avoid adding the face itself and duplicates
        if(adjacentFaceIndex != f && std::find(adjacentFacesList[f].begin(), adjacentFacesList[f].end(), adjacentFaceIndex) == adjacentFacesList[f].end()) {
          adjacentFacesList[f].push_back(adjacentFaceIndex);
        }
      }
    }
  }

  // Run the genetic algorithm


//
//  std::vector<Eigen::Vector3d> face_vectors(F.rows(), Eigen::Vector3d(1.0, 1.0, 0.0).normalized());
//
//  fsim::OrthotropicStVKMembrane model(V / stretch_factor , F, thickness, young_modulus1, young_modulus2, poisson_ratio, face_vectors, mass, pressure);
//  optim::NewtonSolver<double> solver;
//
////  std::vector<int> bdrs = findBoundaryVertices(F);
//  std::vector<int> bdrs = {1, 2, 3, 11, 14, 24, 35, 38, 47, 49, 60, 69, 76, 147, 148, 155, 157, 166, 174, 176, 184, 186, 196, 204, 210, 269, 270, 277, 279, 288, 296, 298, 306, 308, 318, 326, 332, 392, 393, 401, 404, 414, 425, 428, 437, 439, 450, 459, 466, 0, 5, 6, 7, 12, 19, 80, 82, 83, 84, 88, 94, 149, 150, 151, 156, 161, 214, 215, 216, 218, 220, 224, 271, 272, 273, 275, 278, 283, 336, 337, 338, 342, 346, 391, 395, 396, 397, 402, 409, 470, 472, 473, 474, 478, 484, 9, 153, 73, 141, 77, 86, 340, 144, 208, 264, 211, 266, 330, 386, 333, 388, 399, 463, 531, 467, 476, 534};
//
////  std::vector<int> bdrs = {11, 13, 25, 29, 35, 51, 55, 65, 73, 79, 98, 99, 100, 104, 105, 114, 118, 130, 133, 142, 144, 149, 153, 155, 157, 158, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 186, 188, 200, 204, 210, 226, 230, 247, 274, 275, 284, 288, 302, 311, 313, 314, 322, 324, 326, 327, 332, 333, 334, 336, 337, 338, 339};
//  std::sort(bdrs.begin(), bdrs.end());
//
//  for (int bdr : bdrs) {
//    solver.options.fixed_dofs.push_back(bdr * 3);
//    solver.options.fixed_dofs.push_back(bdr * 3 + 1);
//    solver.options.fixed_dofs.push_back(bdr * 3 + 2);
//  }
//  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
////  solver.options.fixed_dofs = {0 * 3 + 0, 0 * 3 + 1, 0 * 3 + 2, 1 * 3 + 0,
////                               1 * 3 + 1, 1 * 3 + 2, 2 * 3 + 0, 2 * 3 + 1,
////                               2 * 3 + 2, 3 * 3 + 0, 3 * 3 + 1, 3 * 3 + 2};
//  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
//
//
//  solver.solve(model, Map<VectorXd>(V.data(), V.size()));
//
//  // Simulated vertex positions
//  fsim::Mat3<double> VSimulated = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
//
//  // Deviation and smoothness penalty
//  double deviation = calculateDeviation(VSimulated);
//  double smoothness = calculateSmoothnessPenalty(face_vectors);
//
//  std::cout << deviation << ", " << smoothness << std::endl;

  // Register Polyscope visualization
  polyscope::registerSurfaceMesh("mesh", V, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1});

  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;

  polyscope::init();





  geneticAlgorithm(F.rows());
  polyscope::show();
  return 0;
}
