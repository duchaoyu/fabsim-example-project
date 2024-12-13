#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <fsim/OrthotropicStVKMembrane.h>

#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include <iostream>

using namespace Eigen;

fsim::Mat3<double> V;
fsim::Mat3<int> F;
int young_modulus1;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
int young_modulus2;
double thickness;
double poisson_ratio;  // 0.38
double stretch_factor;
double mass; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
double pressure; // pressure per area N/m2, Pa, air-supported structure 200-300 Pa
std::vector<Eigen::Vector3d> face_vectors;

std::vector<int> bdrs;
std::vector<std::vector<int>> adjacentFacesList;



// List of pressures
std::vector<double> pressures = {1000, 1100, 1200};

// Vectors to store reference meshes for each pressure
std::vector<fsim::Mat3<double>> V_refs;
std::vector<fsim::Mat3<int>> F_refs;


//fsim::Mat3<double> V_ref; // Matrix to store the reference mesh vertices
//fsim::Mat3<int> F_ref;    // We may not need the faces, but we'll read them anyway




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


fsim::Mat3<double> solve(double modulus1, double modulus2, double poisson_ratio, double stretch_factor, double pressure, const std::vector<Eigen::Vector3d>& face_vectors) {
  fsim::OrthotropicStVKMembrane model(V / stretch_factor, F, thickness, modulus1, modulus2, poisson_ratio, face_vectors, mass, pressure);
  optim::NewtonSolver<double> solver;

  for (int bdr : bdrs) {
    solver.options.fixed_dofs.push_back(bdr * 3);
    solver.options.fixed_dofs.push_back(bdr * 3 + 1);
    solver.options.fixed_dofs.push_back(bdr * 3 + 2);
  }

  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be

  solver.solve(model, Map<VectorXd>(V.data(), V.size()));

  fsim::Mat3<double> Vsolve = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
  return Vsolve;
}

double objectiveFunction(const fsim::Mat3<double>& Vsolve, const fsim::Mat3<double>& V_ref) {
    double sumDeviation = 0.0;

    // Loop over each vertex in Vsolve
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

        // Add the minimum distance to the sum
        sumDeviation += std::sqrt(minDistSquared);
    }

    return sumDeviation;
}


struct Individual {
    double modulus1;
    double modulus2;
    double poisson_ratio;
    double stretch_factor;
    double fitness;
    fsim::Mat3<double> Vsolve;

    // Constructor to initialize with random values within bounds
    Individual(double mod1_min, double mod1_max, double mod2_min, double mod2_max,
               double pr_min, double pr_max, double sf_min, double sf_max,
               const std::vector<double>& pressures,
               const std::vector<fsim::Mat3<double>>& V_refs, const std::vector<Eigen::Vector3d>& face_vectors) {
        modulus1 = mod1_min + (std::rand() / (double)RAND_MAX) * (mod1_max - mod1_min);
        modulus2 = mod2_min + (std::rand() / (double)RAND_MAX) * (mod2_max - mod2_min);
        poisson_ratio = pr_min + (std::rand() / (double)RAND_MAX) * (pr_max - pr_min);
        stretch_factor = sf_min + (std::rand() / (double)RAND_MAX) * (sf_max - sf_min);
        updateFitness(pressures, V_refs, face_vectors);
//        Vsolve = solve(modulus1, modulus2, poisson_ratio, stretch_factor, pressure, face_vectors);
//        fitness = objectiveFunction(Vsolve, V_ref);

    }

    // Comparator remains the same
    bool operator<(const Individual& rhs) const {
        return fitness < rhs.fitness; // For minimization problem
    }

    void updateFitness(const std::vector<double>& pressures, const std::vector<fsim::Mat3<double>>& V_refs, const std::vector<Eigen::Vector3d>& face_vectors) {
//        Vsolve = solve(modulus1, modulus2, poisson_ratio, stretch_factor, pressure, face_vectors);
//        fitness = objectiveFunction(Vsolve, V_ref);
        fitness = 0.0;

        for (size_t i = 0; i < pressures.size(); ++i) {
            double current_pressure = pressures[i];
            const fsim::Mat3<double>& V_ref = V_refs[i];

            Vsolve = solve(modulus1, modulus2, poisson_ratio, stretch_factor, current_pressure, face_vectors);
            double deviation = objectiveFunction(Vsolve, V_ref);

            // Accumulate the fitness (could also weight deviations)
            fitness += deviation;
        }

    }
};


// Genetic algorithm parameters
const int populationSize = 100;
const double mutationRate = 0.03;
const double crossoverRate = 0.7;

// Define bounds for modulus1
double modulus1_min = 5000; // Example values
double modulus1_max = 20000;

// Define bounds for modulus2
double modulus2_min = 5000;
double modulus2_max = 20000;

// Define bounds for Poisson's ratio
double poisson_ratio_min = 0.15;
double poisson_ratio_max = 0.30;

// Define bounds for stretch factor
double stretch_factor_min = 1.001;
double stretch_factor_max = 1.1;

const int maxGenerations = 1000;



Individual crossover(const Individual& parent1, const Individual& parent2,
                     const std::vector<double>& pressures, const std::vector<fsim::Mat3<double>>& V_refs,
                     const std::vector<Eigen::Vector3d>& face_vectors) {
  Individual child = parent1; // Start with parent1's genes

    // For each variable, randomly choose from one of the parents
    child.modulus1 = (std::rand() / (double)RAND_MAX) < 0.5 ? parent1.modulus1 : parent2.modulus1;
    child.modulus2 = (std::rand() / (double)RAND_MAX) < 0.5 ? parent1.modulus2 : parent2.modulus2;
    child.poisson_ratio = (std::rand() / (double)RAND_MAX) < 0.5 ? parent1.poisson_ratio : parent2.poisson_ratio;
    child.stretch_factor = (std::rand() / (double)RAND_MAX) < 0.5 ? parent1.stretch_factor : parent2.stretch_factor;

    child.updateFitness(pressures, V_refs, face_vectors);
    return child;
}


void mutate(Individual& individual,
            const std::vector<double>& pressures,
            const std::vector<fsim::Mat3<double>>& V_refs,
            const std::vector<Eigen::Vector3d>& face_vectors
            ) {
    if ((std::rand() / (double)RAND_MAX) < mutationRate) {
        // Mutate modulus1
        individual.modulus1 += (modulus1_max - modulus1_min) * ((std::rand() / (double)RAND_MAX) - 0.5) * 0.1;
        individual.modulus1 = std::min(std::max(individual.modulus1, modulus1_min), modulus1_max);
    }

    if ((std::rand() / (double)RAND_MAX) < mutationRate) {
        // Mutate modulus2
        individual.modulus2 += (modulus2_max - modulus2_min) * ((std::rand() / (double)RAND_MAX) - 0.5) * 0.1;
        individual.modulus2 = std::min(std::max(individual.modulus2, modulus2_min), modulus2_max);
    }

    if ((std::rand() / (double)RAND_MAX) < mutationRate) {
        // Mutate Poisson's ratio
        individual.poisson_ratio += (poisson_ratio_max - poisson_ratio_min) * ((std::rand() / (double)RAND_MAX) - 0.5) * 0.1;
        individual.poisson_ratio = std::min(std::max(individual.poisson_ratio, poisson_ratio_min), poisson_ratio_max);
    }

    if ((std::rand() / (double)RAND_MAX) < mutationRate) {
        // Mutate stretch factor
        individual.stretch_factor += (stretch_factor_max - stretch_factor_min) * ((std::rand() / (double)RAND_MAX) - 0.5) * 0.1;
        individual.stretch_factor = std::min(std::max(individual.stretch_factor, stretch_factor_min), stretch_factor_max);
    }

    individual.updateFitness(pressures, V_refs, face_vectors);
}



// Main Genetic Algorithm function
void geneticAlgorithm(const std::vector<double>& pressures,
                      const std::vector<fsim::Mat3<double>>& V_refs,
                      const std::vector<Eigen::Vector3d>& face_vectors) {
  std::srand(std::time(nullptr)); // Seed for random number generation

  // Initialize population
  std::vector<Individual> population;
  for (int i = 0; i < populationSize; ++i) {
    population.push_back(Individual(modulus1_min, modulus1_max,
                                        modulus2_min, modulus2_max,
                                        poisson_ratio_min, poisson_ratio_max,
                                        stretch_factor_min, stretch_factor_max,
                                        pressures, V_refs, face_vectors));
  }


  // Evolution loop
  for (int generation = 0; generation < maxGenerations; ++generation) {
    std::sort(population.begin(), population.end()); // Sort population based on fitness
        // Create next generation
        std::vector<Individual> newGeneration;

    // Generate new individuals through crossover and mutation
    for (int i = 0; i < populationSize; i += 2) {
          // Selection (here simply picking adjacent individuals)
          Individual parent1 = population[i];
          Individual parent2 = population[i + 1];

        // Crossover to produce two children
    Individual child1 = crossover(parent1, parent2, pressures, V_refs, face_vectors);
    Individual child2 = crossover(parent2, parent1, pressures, V_refs, face_vectors);

    // Mutation
    mutate(child1, pressures, V_refs, face_vectors);
    mutate(child2, pressures, V_refs, face_vectors);

    newGeneration.push_back(child1);
    newGeneration.push_back(child2);

    }

    // print solution
        std::cout << "Best solution found: " << newGeneration[0].modulus1 << " modulus 1, " <<
        newGeneration[0].modulus2 << " modulus 2, " <<
        newGeneration[0].poisson_ratio << " possion's ratio, " <<
        newGeneration[0].stretch_factor << " stretch factor, " <<
        " with fitness: " << newGeneration[0].fitness << std::endl;



// Replace old generation with new generation
population = newGeneration;
}

  // Output the best solution found
  std::sort(population.begin(), population.end()); // Sort population based on fitness
  std::cout << "Best solution found: " << population[0].modulus1 << " modulus 1, " <<
        population[0].modulus2 << " modulus 2, " <<
        population[0].poisson_ratio << " possion's ratio, " <<
        population[0].stretch_factor << " stretch factor, " <<
        " with fitness: " << population[0].fitness << std::endl;

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
  fsim::readOFF("/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/flat_changing_directions/circlemesh.off", V, F);

  std::vector<std::string> ref_mesh_filenames = {
//    "/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/flat_changing_directions/no_short_row_p10.off",
//    "/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/flat_changing_directions/no_short_row_p11.off",
//    "/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/flat_changing_directions/no_short_row_p12.off"

"/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/flat_changing_directions/changing_p10.off",
    "/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/flat_changing_directions/changing_p11.off",
    "/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/flat_changing_directions/changing_p12.off"


};


for (size_t i = 0; i < pressures.size(); ++i) {
    fsim::Mat3<double> V_ref;
    fsim::Mat3<int> F_ref;
    fsim::readOFF(ref_mesh_filenames[i], V_ref, F_ref);
    V_refs.push_back(V_ref);
    F_refs.push_back(F_ref);
//    std::cout << V_ref << std::endl;
}




//  // Load the reference mesh
//  fsim::readOBJ("/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/flat_changing_directions/no_short_row_p12.obj", V_ref, F_ref);



  young_modulus1= 5000;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
  young_modulus2 = 10000;
  thickness = 1.0;
  poisson_ratio = 0.2;
  stretch_factor = 1.05;
  mass = 0.001; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
//  pressure = 1200; // pressure per area N/m2, Pa, air-supported structure 200-300 Pa

//  // initialise face_vectors
//  for (int i = 0; i < F.rows(); ++i) {
//    face_vectors.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
//  }


// read face vectors from a txt file
  std::vector<Eigen::Vector3d> face_vectors;
  std::string line;
  std::ifstream infile("/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/flat_changing_directions/circle_face_directional_field.txt");
  while (std::getline(infile, line))
  {
    // Remove any leading/trailing whitespace
    line.erase(0, line.find_first_not_of(" \t\n\r"));
    line.erase(line.find_last_not_of(" \t\n\r") + 1);

    // Skip empty lines
    if (line.empty()) continue;

    // Replace commas with spaces to simplify splitting
    for (size_t i = 0; i < line.length(); ++i){
      if (line[i] == ',') line[i] = ' ';}

    // Use a stringstream to parse the numbers
    std::stringstream ss(line);
    double x, y, z;
    if (ss >> x >> y >> z){
      Eigen::Vector3d vec(x, y, z);
      face_vectors.push_back(vec);}
    else{
      std::cerr << "Warning: Could not parse line: " << line << std::endl;}
  }


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


  // fix boundary vertices
  bdrs = findBoundaryVertices(F);
// Sorting the vector in ascending order explicitly, although it's already sorted
  std::sort(bdrs.begin(), bdrs.end());

  std::vector<Individual> population;
for (int i = 0; i < populationSize; ++i) {
    population.push_back(Individual(modulus1_min, modulus1_max,
                                    modulus2_min, modulus2_max,
                                    poisson_ratio_min, poisson_ratio_max,
                                    stretch_factor_min, stretch_factor_max,
                                    pressures, V_refs, face_vectors));
}

 geneticAlgorithm(pressures, V_refs, face_vectors);



  polyscope::registerSurfaceMesh("mesh", V, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1})
      ->setSurfaceColor({0, 1., 1.});
  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;
  polyscope::init();


  polyscope::show();
  return 0;
}
