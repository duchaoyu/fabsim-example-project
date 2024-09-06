#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <fsim/ElasticMembrane.h>
#include "fsim/StVKElement.h"
#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>
#include <optim/NewtonSolver.h>
#include <polyscope/surface_mesh.h>
#include <iostream>

using namespace Eigen;

fsim::Mat3<double> V;
fsim::Mat3<int> F;
double young_modulus;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
double thickness;
double thickness2;
double poisson_ratio;  // 0.38
double stretch_factor;
double mass; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
double pressure; // pressure per area N/m2, Pa, air-supported structure 200-300 Pa
std::vector<int> bdrs;
std::vector<std::vector<int>> adjacentFacesList;

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


fsim::Mat3<double> solve(std::vector<double> thicknesses) {
  fsim::StVKMembrane model(V / stretch_factor, F, thicknesses, young_modulus, poisson_ratio, mass, pressure);
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

double objectiveFunction(fsim::Mat3<double>& Vsolve, std::vector<double> values) {

  // deviation
  Eigen::VectorXd d = (V - Vsolve).cwiseProduct((V - Vsolve)).rowwise().sum();
  d = d.array().sqrt();

  double dSum = d.sum();
  if (dSum < 1e-4){
    return 100000000.0;
  }

  // smoothness
  double sSum = 0;
  double smoothness = 0;
  int count;
  for (int f = 0; f < F.rows(); ++f) {
    // nbrs
    std::vector<int> nbrs = adjacentFacesList[f];
    smoothness = 0;
    count = 0;
    for (int nbr : nbrs){
      double difference = std::abs(values[f] - values[nbr]);
      if (difference > 0.3){
        count += 1;
      }
      smoothness += difference;
    }
//    if (count == 1){
//      smoothness -= 1.0; // manual difference adjustion
//    }
    sSum += smoothness;
  }

//  for (int adjFaceIndex : adjacentFaces[i]) {
//    double adjFaceValue = faceValues[adjFaceIndex];
//
//    // Calculate the difference between the face and its neighbor
//    double difference = std::abs(faceValue - adjFaceValue);

  return dSum + sSum;


}


//double objectiveFunctionOld(std::vector<double> thicknesses) {
//
//  fsim::StVKMembrane model(V / stretch_factor, F, thicknesses, young_modulus, poisson_ratio, mass, pressure);
//  optim::NewtonSolver<double> solver;
//
//  for (int bdr : bdrs) {
//    solver.options.fixed_dofs.push_back(bdr * 3);
//    solver.options.fixed_dofs.push_back(bdr * 3 + 1);
//    solver.options.fixed_dofs.push_back(bdr * 3 + 2);
//  }
//
//  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
//  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
//
//  solver.solve(model, Map<VectorXd>(V.data(), V.size()));
//
//  fsim::Mat3<double> Vsolve = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
//  Eigen::VectorXd d = (V - Vsolve).cwiseProduct((V - Vsolve)).rowwise().sum();
//  d = d.array().sqrt();
//
//  double dSum = d.sum();
//  if (dSum < 1e-6){
//    return 1000000;
//  }
//  return dSum;
//}

// Individual structure
struct Individual {
    std::vector<double> values; // Represents a candidate solution with nF variables
    double fitness; // Fitness value of the candidate solution
    fsim::Mat3<double> Vsolve;

    // Constructor to initialize a random individual for problems with nF unknowns
    Individual(int nF, double minBound, double maxBound,  std::vector<double> valuesInit) : values(nF), fitness(0) {
      for (int i = 0; i < nF; ++i) {
        values[i] = valuesInit[i] + (std::rand() / (double)RAND_MAX)*0.5 - 0.25;
      }
      Vsolve = solve(values);
      fitness = objectiveFunction(Vsolve, values);
    }

    explicit Individual(int nF) : values(nF), fitness(0) {
    }

    // Comparator to sort individuals based on their fitness
    bool operator<(const Individual& rhs) const {
      return fitness < rhs.fitness; // For minimization problem
      // Use '>' for maximization problem
    }

    void updateFitness(){
      Vsolve = solve(values);
      fitness = objectiveFunction(Vsolve, values);
    }
};

// Genetic algorithm parameters
const int populationSize = 100;
const double mutationRate = 0.03;
const double crossoverRate = 0.7;
const double minBound = 0.2; // Minimum bound of search space
const double maxBound = 5;  // Maximum bound of search space
const int maxGenerations = 1000;



Individual crossover(const Individual& parent1, const Individual& parent2, int nF) {
  Individual child(nF);
  for (int i = 0; i < nF; ++i) {
    child.values[i] = (std::rand() / (RAND_MAX + 1.0)) < 0.5 ? parent1.values[i] : parent2.values[i];
  }

  child.updateFitness();
//  child.fitness = objectiveFunction(child.values);
  return child;
}


void mutate(Individual& individual, double minBound, double maxBound, int nF) {
  for (int i = 0; i < nF; ++i) {
    if ((std::rand() / (RAND_MAX + 1.0)) < mutationRate) {
      // Apply mutation logic here, e.g., add a small perturbation
      individual.values[i] += (0.2 * (std::rand() / (RAND_MAX + 1.0)) - 0.1);
      // Ensure values stay within bounds
      individual.values[i] = std::min(std::max(individual.values[i], minBound), maxBound);
    }
  }

  individual.updateFitness();
}

// Main Genetic Algorithm function
void geneticAlgorithm(int nF, std::vector<double> valuesInit) {
  std::srand(std::time(nullptr)); // Seed for random number generation

  // Initialize population
  std::vector<Individual> population;
  for (int i = 0; i < populationSize; ++i) {
    population.push_back(Individual(nF, minBound, maxBound, valuesInit));
  }


  // Evolution loop
  for (int generation = 0; generation < maxGenerations; ++generation) {
    std::sort(population.begin(), population.end()); // Sort population based on fitness
        // Create next generation
        std::vector<Individual> newGeneration;
        for (int i = 0; i < populationSize / 2; i += 2) {
          // Selection (here simply picking adjacent individuals)
          Individual parent1 = population[i];
          Individual parent2 = population[i + 1];

          // Crossover
          Individual child1 = crossover(parent1, parent2, nF);
          Individual child2 = crossover(parent2, parent1, nF);

          // Mutation
          mutate(child1, minBound, maxBound, nF);
          mutate(child2, minBound, maxBound, nF);

          // Add to new generation
          newGeneration.push_back(parent1);
          newGeneration.push_back(parent2);
          newGeneration.push_back(child1);
          newGeneration.push_back(child2);
        }

        // print solution
        Map<Eigen::RowVectorXd> valuesOut(newGeneration[0].values.data(), newGeneration[0].values.size());
        std::cout << "Best solution found: " << valuesOut << " with fitness: " << newGeneration[0].fitness << std::endl;

        // Display the result of the optimization

        if (generation % 10 == 0){
          polyscope::getSurfaceMesh("mesh")->updateVertexPositions(newGeneration[0].Vsolve);
          Eigen::VectorXd d = (V - newGeneration[0].Vsolve).cwiseProduct((V - newGeneration[0].Vsolve)).rowwise().sum();
          d = d.array().sqrt();
          polyscope::getSurfaceMesh("mesh")->addVertexScalarQuantity("Distance", d)->setEnabled(true);
          polyscope::requestRedraw();
          polyscope::show(100);
        }


    // Replace old generation with new generation
        population = newGeneration;
    }

  // Output the best solution found
  std::sort(population.begin(), population.end()); // Sort population based on fitness
  Map<Eigen::VectorXd> valuesOut(population[0].values.data(), population[0].values.size());
  std::cout << "Best solution found: " << valuesOut << " with fitness: " << population[0].fitness << std::endl;
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
  fsim::readOFF("/Users/duch/Downloads/2part_opt.off", V, F);

  young_modulus= 50000;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
  thickness = 0.40;
  thickness2 = 0.60;
  poisson_ratio = 0.38;  // 0.38
  stretch_factor = 1.05;
  mass = 30; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
  pressure = 250; // pressure per area N/m2, Pa, air-supported structure 200-300 Pa
//  bdrs = {0, 1, 3, 10, 16, 17, 18, 25, 39, 41, 52, 53, 62, 63, 65, 66, 67, 76, 85, 88, 90, 91, 94, 95, 102, 103, 104, 106, 108, 111, 113, 115, 119, 120, 122, 139, 140, 141, 144, 150, 151, 153, 154, 165, 166, 171, 173, 177, 178, 181, 189, 191};
  bdrs = findBoundaryVertices(F);
// Sorting the vector in ascending order explicitly, although it's already sorted
  std::sort(bdrs.begin(), bdrs.end());

//  std::vector<double> thicknesses;
//  std::string input = "0.642239 0.595957 0.771409 0.631053 0.591791 0.670683 0.577107 0.580665 0.630719  0.61826 0.599256 0.606116 0.709123 0.685108 0.546458 0.614121 0.618511 0.601329 0.511283 0.568993 0.532019 0.680565 0.512543 0.618909 0.672009  0.55918 0.592027  0.59397 0.608272 0.588229 0.586488 0.580582 0.670911 0.612335 0.573813  0.58749 0.648497 0.595108 0.715657 0.590827  0.58561  0.55434 0.573149 0.665938 0.584834 0.513719 0.563031 0.705991 0.566585 0.601312 0.551114 0.624276 0.595672 0.537954 0.605973 0.572288 0.601761 0.618188 0.660231 0.587485 0.633181 0.569875 0.608252  0.61772   0.6527 0.581781 0.554277 0.586942 0.525961 0.581241 0.591931 0.579563 0.696851 0.556437 0.602596 0.698862    0.662 0.686671 0.615139 0.501638 0.555376 0.557174 0.662482   0.6613 0.642766 0.651421 0.710593 0.601806 0.703207  0.60216 0.587947 0.622834 0.677959    0.592 0.653591 0.643747 0.531551 0.614615 0.733406 0.537841 0.634145 0.524115 0.557515 0.604881 0.567104 0.608773 0.629793 0.553517 0.683925 0.567818 0.672361 0.592163 0.611382 0.622356 0.616807 0.578739 0.606215 0.592765 0.592169  0.66286 0.606479  0.59123 0.692758 0.618344 0.564439 0.612173 0.666845 0.538358 0.542073 0.609273 0.620062 0.640461 0.644108 0.512777 0.575588 0.583452 0.504367 0.827431 0.537358 0.639896 0.594694 0.552776  0.59831 0.691534 0.572186 0.584205 0.308982 0.625983  0.60036 0.632663 0.626311 0.619782  0.72384 0.622156 0.706164 0.693646 0.597445 0.634579 0.561894 0.576613 0.682419 0.648733 0.653537 0.539525  0.51105 0.639844 0.628171 0.591328 0.602607   0.6652 0.677728 0.643812 0.658405 0.595977 0.595196 0.628981 0.588975 0.610471 0.598837 0.534812 0.575494 0.631576 0.614489 0.599228 0.533744 0.586569 0.534528 0.596623 0.600876 0.682582 0.636054  0.59801 0.541182 0.542504 0.683964 0.563102 0.563438 0.526439 0.727509 0.672161 0.602081 0.696335 0.784037 0.523658 0.638888 0.541176 0.566771 0.545767  0.53526 0.649631 0.585231 0.660552  0.61567 0.650046 0.643842  0.57541 0.718707 0.509741 0.610422 0.692026 0.567345 0.543104 0.675365 0.526377 0.631159 0.577819 0.653686 0.663749 0.713325 0.521578 0.597125 0.541486 0.667474 0.599821 0.679809 0.638176 0.641614 0.577296 0.651287 0.587242 0.529994 0.533837  0.60313 0.532279 0.694041 0.555005 0.597383 0.539887 0.481985  0.57677 0.572891 0.590171 0.587376 0.599909 0.528009 0.629172 0.610177  0.58847 0.619884  0.60406 0.735419 0.528615 0.530059 0.665601 0.612777 0.686249  0.55866 0.654415 0.573253 0.570193 0.522929  0.61097 0.589626  0.58298 0.618343 0.523155 0.570128 0.481449 0.646493 0.504184 0.742631  0.61682 0.731707 0.585223 0.700756 0.551661 0.561376 0.639698 0.625634 0.625823 0.637508 0.575054 0.706274 0.579179 0.702798 0.590296 0.565134 0.556083 0.610776 0.602741 0.659551 0.611276 0.646086  0.61988 0.587027 0.619826 0.527133 0.539054 0.527844 0.565074 0.526404 0.687833 0.542467 0.605275 0.549557 0.733391 0.584518 0.656409  0.57555 0.683523 0.657824 0.616409 0.730862 0.591929 0.626992 0.608437 0.509462 0.654273 0.602331 0.699019 0.503547 0.669707 0.619325 0.672494 0.587459 0.592488 0.657965 0.638131 0.694921 0.836971 0.651935 0.624878 0.539808  0.62408 0.582613 0.649787 0.617964 0.677109 0.535245 0.742165 0.544894 0.605215 0.692614 0.645124 0.588498 0.614848 0.613759 0.579305 0.322353 0.579772 0.580891 0.605636 0.561125 0.593182 0.588746 0.539068 0.554625  0.58647 0.580552 0.642062 0.715088 0.586688 0.694538 0.648746 0.621387 0.621343 0.576036 0.502541 0.651021 0.605804 0.663977 0.591662 0.638563 0.599696 0.587032 0.491416 0.672646 0.454534 0.568467  0.58525 0.632708 0.649472 0.588823 0.711871 0.629788 0.663937 0.582649 0.482042 0.657748 0.567479 0.631095 0.640808 0.677766 0.599237 0.543856 0.587503 0.609917 0.744878 0.593189 0.529783 0.599464 0.585512 0.310113 0.568223 0.614348 0.627039 0.623429 0.638325 0.599523 0.541292 0.571893 0.581386 0.574256 0.614335 0.667682 0.670528 0.611389 0.684739  0.67671 0.589179 0.658062 0.590907 0.670034 0.633242 0.576833 0.587889 0.570368 0.699243 0.688418 0.630048 0.560376 0.732017 0.553544 0.659134 0.645084 0.609197 0.645801 0.651161 0.625612 0.593227 0.535597 0.634291 0.579838 0.544465 0.637414 0.629871 0.672568 0.579131 0.618899 0.615743 0.598569 0.628211 0.605811 0.590101 0.682447   0.6063 0.635216  0.57398 0.616423  0.74946 0.461906 0.600941 0.628637 0.612652   0.5896 0.537866 0.651372 0.690692 0.687642 0.664907 0.682019 0.584418 0.641246 0.614699  0.55182 0.631812 0.579167 0.559724 0.543949 0.544575 0.726427 0.574002 0.558971 0.619283 0.654923 0.607789 0.579067 0.657374 0.817773 0.564233 0.654346 0.750782 0.636197 0.528782 0.624802  0.58429 0.599685 0.576746  0.71486 0.649208 0.637702 0.518265 0.599906 0.602144 0.536411 0.677579 0.582563 0.563432  0.72938 0.683629 0.607006 0.431131 0.673352 0.614466 0.680223 0.578391 0.581291 0.644925 0.601477 0.584322 0.565129 0.574833 0.685987 0.564589 0.495884 0.541463    0.602 0.621241 0.545819 0.544373 0.575642 0.687015 0.631362 0.337417 0.458127  0.61541 0.533747 0.559934 0.624424 0.606624 0.601079 0.681841 0.571798 0.584486 0.719171 0.602432 0.629357 0.608498 0.652543  0.59727 0.613429 0.706811 0.541725 0.871045 0.546476 0.527153 0.369106 0.664601 0.893708 0.588284 0.596513  0.50802 0.552486 0.615712 0.633545 0.677982 0.670058 0.453599 0.653309  0.51209 0.535685 0.577126 0.563508 0.585672 0.538889 0.613451 0.511838 0.646628 0.661551 0.498661 0.627919 0.624881 0.581922 0.647944 0.540209 0.550357 0.583115 0.691195 0.626021 0.597069 0.737582 0.610068  0.58309 0.521186  0.61298 0.657781 0.663335 0.610567  0.63131 0.646045 0.539229 0.614961 0.647562 0.545539 0.580911 0.624837 0.579625 0.654282 0.543349 0.657626 0.653141 0.652969 0.624849 0.610006 0.708946 0.609268 0.670915 0.646736 0.712218 0.624567 0.703725 0.667866 0.651081 0.678988";
//  std::istringstream iss(input);
//
//  double temp;
//  while (iss >> temp) {
//    thicknesses.push_back(temp);
//  }
  std::vector<double> thicknesses(F.rows(), thickness); // Assuming F.rows() gives the correct size
  int faceIndices[] =  {1, 2, 3, 7, 9, 10, 11, 12, 16, 17, 18, 20, 21, 22, 24, 27, 28, 30, 31, 34, 38, 39, 40, 41, 44, 46, 47, 48, 49, 52, 54, 56, 57, 58, 60, 61, 62, 64, 65, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 81, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 96, 97, 98, 103, 104, 105, 106, 107, 112, 113, 114, 115, 117, 118, 119, 121, 122, 123, 124, 125, 126, 128, 129, 130, 131, 134, 139, 140, 141, 142, 147, 149, 150, 152, 154, 155, 156, 157, 158, 162, 163, 165, 168, 174, 175, 176, 178, 179, 181, 182, 183, 186, 187, 188, 189, 192, 193, 195, 196, 201, 202, 203, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 220, 221, 223, 224, 225, 228, 231, 234, 236, 237, 238, 240, 242, 250, 251, 258, 261, 271, 274, 276, 282, 284, 286, 288, 290, 291, 292, 293, 295, 296, 297, 298, 299, 300, 301, 302, 304, 306, 307, 308, 309, 310, 311, 312, 313, 315, 316, 318, 319, 320, 324, 326, 327, 328, 329, 333, 334, 335, 337, 338, 339, 341, 344, 345, 347, 348, 351, 355, 356, 357, 358, 361, 363, 364, 365, 366, 369, 371, 373, 374, 375, 377, 378, 379, 381, 382, 386, 387, 389, 390, 391, 392, 393, 394, 395, 396, 398, 400, 402, 403, 404, 405, 406, 407, 408, 409, 410, 413, 414, 415, 420, 421, 422, 423, 424, 429, 430, 431, 432, 434, 435, 436, 438, 439, 440, 441, 442, 443, 445, 446, 447, 448, 451, 456, 457, 458, 459, 464, 466, 467, 469, 471, 472, 473, 474, 475, 479, 480, 482, 485, 491, 492, 493, 495, 496, 498, 499, 500, 503, 504, 505, 506, 509, 510, 512, 513, 518, 519, 520, 522, 523, 524, 525, 527, 528, 529, 530, 531, 532, 537, 538, 540, 541, 542, 545, 548, 551, 553, 554, 555, 557, 559, 567, 568, 575, 578, 588, 591, 593, 599, 601, 603, 605, 607, 608, 609, 610, 612, 613, 614, 615, 616, 617, 618, 619, 621, 623, 624, 625, 626, 627, 628, 629, 630, 632, 633};


  ////// Use faceIndices directly, assuming the indices are within the bounds of thicknesses
  for (int i = 0; i < sizeof(faceIndices)/sizeof(faceIndices[0]); ++i) {
    int f = faceIndices[i]; // Direct access since it's a plain array
    thicknesses[f] = thickness2;
  }

  int nF = F.rows();

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




  polyscope::registerSurfaceMesh("mesh", V, F)
      ->setEdgeWidth(1)
      ->setEdgeColor({0.1, 0.1, 0.1})
      ->setSurfaceColor({0, 1., 1.});
  polyscope::view::upDir = polyscope::view::UpDir::ZUp;
  polyscope::options::groundPlaneHeightFactor = 0.4;
  polyscope::init();


  geneticAlgorithm(nF, thicknesses);
  polyscope::show();
  return 0;
}
