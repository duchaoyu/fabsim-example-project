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
#include <iostream>

using namespace Eigen;

fsim::Mat3<double> V;
fsim::Mat3<int> F;
double young_modulus1;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
double young_modulus2;
double thickness;
double poisson_ratio;  // 0.38
double stretch_factor;
double mass; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
double pressure; // pressure per area N/m2, Pa, air-supported structure 200-300 Pa
std::vector<Eigen::Vector3d> face_vectors;

std::vector<int> bdrs;
std::vector<std::vector<int>> adjacentFacesList;

std::vector<int> non_fixed_vertices;
fsim::Mat3<double> V_ref; // Matrix to store the reference mesh vertices
fsim::Mat3<int> F_ref;    // We may not need the faces, but we'll read them anyway
double lambda;
std::vector<std::vector<int>> adjacencyList;


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


std::vector<std::vector<int>> buildVertexAdjacencyList(const fsim::Mat3<int>& F, int numVertices) {
    std::vector<std::set<int>> adjacencySet(numVertices);

    for (int i = 0; i < F.rows(); ++i) {
        int v0 = F(i, 0);
        int v1 = F(i, 1);
        int v2 = F(i, 2);

        adjacencySet[v0].insert(v1);
        adjacencySet[v0].insert(v2);

        adjacencySet[v1].insert(v0);
        adjacencySet[v1].insert(v2);

        adjacencySet[v2].insert(v0);
        adjacencySet[v2].insert(v1);
    }

    // Convert sets to vectors
    std::vector<std::vector<int>> adjacencyList(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        adjacencyList[i] = std::vector<int>(adjacencySet[i].begin(), adjacencySet[i].end());
    }

    return adjacencyList;
}




fsim::Mat3<double> solve(const fsim::Mat3<double>& V_modified, const std::vector<Eigen::Vector3d>& face_vectors) {
  fsim::OrthotropicStVKMembrane model(V_modified / stretch_factor, F, thickness, young_modulus1, young_modulus2, poisson_ratio, face_vectors, mass, pressure);
  optim::NewtonSolver<double> solver;

  for (int bdr : bdrs) {
    solver.options.fixed_dofs.push_back(bdr * 3);
    solver.options.fixed_dofs.push_back(bdr * 3 + 1);
    solver.options.fixed_dofs.push_back(bdr * 3 + 2);
  }

  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be

  solver.solve(model, Map<VectorXd>(V.data(), V.size()));

  fsim::Mat3<double> Vsolve = Map<fsim::Mat3<double>>(solver.var().data(), V_modified.rows(), 3);
  return Vsolve;
}

double objectiveFunction(const fsim::Mat3<double>& Vsolve, const fsim::Mat3<double>& V_ref,
                         const std::vector<std::vector<int>>& adjacencyList,
                         double lambda) {
    double sumDeviation = 0.0;

    // Data term: Deviation from reference mesh
    for (int i = 0; i < Vsolve.rows(); ++i) {
        Eigen::RowVector3d vertex = Vsolve.row(i);
        double minDistSquared = std::numeric_limits<double>::max();

        // Loop over each vertex in V_ref to find the closest one
        for (int j = 0; j < V_ref.rows(); ++j) {
            Eigen::RowVector3d refVertex = V_ref.row(j);
            double distSquared = (vertex - refVertex).squaredNorm();
            if (distSquared < minDistSquared) {
                minDistSquared = distSquared;
            }
        }

        sumDeviation += minDistSquared;
    }

    // Laplacian smoothing term
    double laplacianSum = 0.0;

    for (int i = 0; i < Vsolve.rows(); ++i) {
        const std::vector<int>& neighbors = adjacencyList[i];
        if (neighbors.empty()) continue;

        Eigen::RowVector3d neighborAvg = Eigen::RowVector3d::Zero();
        for (int neighborIdx : neighbors) {
            neighborAvg += Vsolve.row(neighborIdx);
        }
        neighborAvg /= static_cast<double>(neighbors.size());

        Eigen::RowVector3d laplacian = Vsolve.row(i) - neighborAvg + Vsolve.row(i) - V.row(i);

        laplacianSum += laplacian.squaredNorm();
    }

    // Total objective function with weighting factor lambda
    double totalEnergy = sumDeviation + lambda * laplacianSum;

    return totalEnergy;
}



struct Individual {
    std::vector<double> z_coordinates_non_fixed;
    double fitness;
    fsim::Mat3<double> Vsolve;

    // Constructor to initialize with random z-coordinates within bounds
    Individual(double z_min, double z_max, const fsim::Mat3<double>& V_ref,
               const std::vector<Eigen::Vector3d>& face_vectors,
               const std::vector<int>& non_fixed_vertices) {
        int num_non_fixed = non_fixed_vertices.size();
        z_coordinates_non_fixed.resize(num_non_fixed);
        for (int i = 0; i < num_non_fixed; ++i) {
            int vertex_index = non_fixed_vertices[i];
            z_coordinates_non_fixed[i] = V(vertex_index, 2);
        }
        updateFitness(V_ref, face_vectors, non_fixed_vertices, adjacencyList, lambda);
    }

    // Comparator remains the same
    bool operator<(const Individual& rhs) const {
        return fitness < rhs.fitness; // For minimization problem
    }

    void updateFitness(const fsim::Mat3<double>& V_ref,
                   const std::vector<Eigen::Vector3d>& face_vectors,
                   const std::vector<int>& non_fixed_vertices,
                   const std::vector<std::vector<int>>& adjacencyList,
                   double lambda) {
    fsim::Mat3<double> V_modified = V;

    // Update z-coordinates of non-fixed vertices
    for (size_t idx = 0; idx < non_fixed_vertices.size(); ++idx) {
        int vertex_index = non_fixed_vertices[idx];
        V_modified(vertex_index, 2) = z_coordinates_non_fixed[idx];
    }

    Vsolve = solve(V_modified, face_vectors);
    fitness = objectiveFunction(Vsolve, V_ref, adjacencyList, lambda);
    }

};


// Genetic algorithm parameters
const int populationSize = 100;
const double mutationRate = 0.03;
const double crossoverRate = 0.7;

// Define bounds for z
double z_min = 0.0;
double z_max = 0.2;

const int maxGenerations = 1000;



Individual crossover(const Individual& parent1, const Individual& parent2,
                     const fsim::Mat3<double>& V_ref,
                     const std::vector<Eigen::Vector3d>& face_vectors,
                     const std::vector<int>& non_fixed_vertices,
                     const std::vector<std::vector<int>>& adjacencyList,
                     double lambda) {
    Individual child = parent1;

    int num_non_fixed = non_fixed_vertices.size();
    for (int i = 0; i < num_non_fixed; ++i) {
        if ((std::rand() / (double)RAND_MAX) < 0.5) {
            child.z_coordinates_non_fixed[i] = parent1.z_coordinates_non_fixed[i];
        } else {
            child.z_coordinates_non_fixed[i] = parent2.z_coordinates_non_fixed[i];
        }
    }

    child.updateFitness(V_ref, face_vectors, non_fixed_vertices, adjacencyList, lambda);
    return child;
}


void mutate(Individual& individual, double z_min, double z_max,
            const fsim::Mat3<double>& V_ref,
            const std::vector<Eigen::Vector3d>& face_vectors,
            const std::vector<int>& non_fixed_vertices,
            const std::vector<std::vector<int>>& adjacencyList,
            double lambda) {
    int num_non_fixed = non_fixed_vertices.size();
    for (int i = 0; i < num_non_fixed; ++i) {
        if ((std::rand() / (double)RAND_MAX) < mutationRate) {
            double delta = ((std::rand() / (double)RAND_MAX) - 0.5) * (z_max - z_min) * 0.1;
            individual.z_coordinates_non_fixed[i] += delta;
            individual.z_coordinates_non_fixed[i] = std::min(std::max(individual.z_coordinates_non_fixed[i], z_min), z_max);
        }
    }
    individual.updateFitness(V_ref, face_vectors, non_fixed_vertices, adjacencyList, lambda);
}





// Main Genetic Algorithm function
fsim::Mat3<double>  geneticAlgorithm(const fsim::Mat3<double>& V_ref,
                                     const std::vector<Eigen::Vector3d>& face_vectors,
                                     const std::vector<int>& non_fixed_vertices,
                                     const std::vector<std::vector<int>>& adjacencyList,
                                    double lambda) {
    std::srand(std::time(nullptr)); // Seed for random number generation

      // Initialize population
    std::vector<Individual> population;
    for (int i = 0; i < populationSize; ++i) {
        population.push_back(Individual(z_min, z_max, V_ref, face_vectors, non_fixed_vertices));
    }

 // Create the modified V with updated z-coordinates
    fsim::Mat3<double> best_V_modified = V;


  // Evolution loop
  for (int generation = 0; generation < maxGenerations; ++generation) {
    std::sort(population.begin(), population.end()); // Sort population based on fitness
        // Create next generation
        std::vector<Individual> newGeneration;

   // Elitism: Keep the best individuals
        int eliteSize = populationSize * 0.1; // Keep top 10%
        for (int i = 0; i < eliteSize; ++i) {
            newGeneration.push_back(population[i]);
        }

// Generate new individuals through crossover and mutation
        while (newGeneration.size() < populationSize) {
            // Selection (e.g., tournament selection)
            Individual parent1 = population[std::rand() % populationSize];
            Individual parent2 = population[std::rand() % populationSize];

            Individual child = crossover(parent1, parent2, V_ref, face_vectors, non_fixed_vertices, adjacencyList, lambda);
            mutate(child, z_min, z_max, V_ref, face_vectors, non_fixed_vertices, adjacencyList, lambda);

            newGeneration.push_back(child);
        }

      // Replace old generation with new generation
        population = newGeneration;

      // print solution
          std::cout << "Generation " << generation << ", Best fitness: " << population[0].fitness << std::endl;

          // Output the best solution found
    std::sort(population.begin(), population.end());
    std::cout << "Best fitness: " << population[0].fitness << std::endl;
    fsim::saveOBJ("/Users/duch/Documents/PhD/knit/2024_prototypes/rectangle/gentle_exp.off", best_V_modified, F);


    best_V_modified = V;

    // Update z-coordinates of non-fixed vertices
    for (size_t idx = 0; idx < non_fixed_vertices.size(); ++idx) {
        int vertex_index = non_fixed_vertices[idx];
        best_V_modified(vertex_index, 2) = population[0].z_coordinates_non_fixed[idx];
    }

    // Write the updated mesh to an OBJ file

  }

    return best_V_modified;

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




int main() {
  using namespace Eigen;

  // load geometry from OFF mesh file
  fsim::readOFF("/Users/duch/Documents/PhD/knit/2024_prototypes/rectangle/gentle_unit_m.off", V, F);
  // Load the reference mesh
  fsim::readOFF("/Users/duch/Documents/PhD/knit/2024_prototypes/rectangle/gentle_unit_m_dense.off", V_ref, F_ref);



  young_modulus1= 5000;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
  young_modulus2 = 12507;
  thickness = 1.0;
  poisson_ratio = 0.198;
  stretch_factor = 1.043;
  mass = 0.001; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
  pressure = 1200; // pressure per area N/m2, Pa, air-supported structure 200-300 Pa

  for (int i = 0; i < F.rows(); ++i) {
    face_vectors.push_back(Eigen::Vector3d(1.0, 1.0, 0.0));
  }

// Compute the adjacency list
adjacencyList = buildVertexAdjacencyList(F, V.rows());
lambda = 0.005;

  // fix boundary vertices
  std::vector<int> bdrs = findBoundaryVertices(F);
//  bdrs = {1, 2, 3, 11, 14, 24, 35, 38, 47, 49, 60, 69, 76, 147, 148, 155, 157, 166, 174, 176, 184, 186, 196, 204, 210, 269, 270, 277, 279, 288, 296, 298, 306, 308, 318, 326, 332, 392, 393, 401, 404, 414, 425, 428, 437, 439, 450, 459, 466, 0, 5, 6, 7, 12, 19, 80, 82, 83, 84, 88, 94, 149, 150, 151, 156, 161, 214, 215, 216, 218, 220, 224, 271, 272, 273, 275, 278, 283, 336, 337, 338, 342, 346, 391, 395, 396, 397, 402, 409, 470, 472, 473, 474, 478, 484, 9, 153, 73, 141, 77, 86, 340, 144, 208, 264, 211, 266, 330, 386, 333, 388, 399, 463, 531, 467, 476, 534};
// Sorting the vector in ascending order explicitly, although it's already sorted
  std::sort(bdrs.begin(), bdrs.end());

  for (int i = 0; i < V.rows(); ++i) {
    if (std::find(bdrs.begin(), bdrs.end(), i) == bdrs.end()) {
        non_fixed_vertices.push_back(i);
    }
}


 // Run the genetic algorithm and get the modified V
fsim::Mat3<double> best_V_modified = geneticAlgorithm(V_ref, face_vectors, non_fixed_vertices, adjacencyList, lambda);


  polyscope::registerSurfaceMesh("modified mesh", best_V_modified, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1})
      ->setSurfaceColor({0, 1., 1.});

  // Optionally, register the reference mesh for comparison
    polyscope::registerSurfaceMesh("Reference Mesh", V_ref, F_ref)
        ->setEdgeWidth(1)
        ->setEdgeColor({0.1, 0.1, 0.1})
        ->setSurfaceColor({1., 0., 0.})
        ->setTransparency(0.5);


  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;
  polyscope::init();


  polyscope::show();
  return 0;
}