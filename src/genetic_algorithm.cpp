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
      if (difference > 0.5){
        count += 1;
      }
      smoothness += difference;
    }
    if (count == 1){
      smoothness -= 0.5; // manual difference adjustion
    }
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

//  fsim::readOFF("/Users/duch/Downloads/butt_out.off", V, F);
//  fsim::readOFF("/Users/duch/Downloads/pillow_uni.off", V, F);
//  fsim::readOFF("/Users/duch/Downloads/pillow.off", V, F);
//  fsim::readOFF("/Users/duch/Downloads/barrel_vaultz_tri.off", V, F);
  fsim::readOFF("/Users/duch/Downloads/cross_vault_tri.off", V, F);

  young_modulus= 50000;  // knit: ~50kPa https://journals.sagepub.com/doi/pdf/10.1177/0040517510371864
  thickness = 0.5;
  thickness2 = 1.0;
  poisson_ratio = 0.38;  // 0.38
  stretch_factor = 1.01;
  mass = 30; // mass per area, density * material thickness= 1500kg/m3 * 0.02 = 30kg/m2
  pressure = 250; // pressure per area N/m2, Pa, air-supported structure 200-300 Pa
//  bdrs = {0, 1, 3, 10, 16, 17, 18, 25, 39, 41, 52, 53, 62, 63, 65, 66, 67, 76, 85, 88, 90, 91, 94, 95, 102, 103, 104, 106, 108, 111, 113, 115, 119, 120, 122, 139, 140, 141, 144, 150, 151, 153, 154, 165, 166, 171, 173, 177, 178, 181, 189, 191};
  bdrs = findBoundaryVertices(F);
// Sorting the vector in ascending order explicitly, although it's already sorted
  std::sort(bdrs.begin(), bdrs.end());

//  std::vector<double> thicknesses;
//  std::string input = "0.475668  0.48009 0.503872 0.475508 0.481323 0.458427  0.51005 0.493493 0.457393 0.463553 0.497805 0.450101 0.537666 0.520139 0.481111 0.487105 0.463718 0.466573 0.484114 0.464987 0.471623 0.494385 0.469398 0.420512 0.443365 0.469409 0.461897 0.412576 0.402586 0.484296 0.418322 0.457523  0.44543 0.422078 0.498908 0.450707  0.48802  0.49268 0.482988 0.473173 0.407269 0.417656  0.46784 0.474184 0.532896 0.542482 0.512068 0.518282 0.434279 0.419001 0.520024 0.489624 0.419384 0.480866 0.444249 0.437605 0.393761 0.422047 0.421171 0.422563 0.479781 0.530075 0.482113   0.4833 0.437846 0.444545 0.544534 0.546247 0.523763 0.444841 0.528086 0.525143  0.58942   0.5956  0.46079 0.459835 0.430451 0.448206   0.5332 0.523728 0.531109 0.564563 0.453205 0.468307 0.581326 0.538884  0.43158 0.445712  0.42056 0.521747 0.511914 0.508664 0.472495 0.461985 0.580752 0.541124  0.52992 0.515562 0.383118 0.401476  0.36024 0.380876 0.515318 0.506512 0.583113 0.578785 0.489343 0.452969 0.454786 0.402425  0.35323 0.514584 0.513127 0.375759 0.392547 0.415789 0.407625 0.414604 0.550476 0.436234 0.436639 0.435239 0.474802 0.316667 0.424796 0.416294 0.421013 0.432629 0.472032 0.492857 0.506606 0.421573 0.485392 0.485004 0.495342 0.463321 0.438604 0.481661 0.487638 0.548165 0.423444 0.417418 0.385557 0.531299 0.526859 0.486364 0.376979 0.546343 0.431267 0.504757 0.379541 0.404047 0.433816 0.436428 0.415647 0.535527 0.574333 0.511652 0.491827 0.502234 0.460884  0.52579 0.439045 0.380918 0.406427 0.460924 0.470916 0.499706 0.531262 0.472354 0.508175 0.520655 0.449002 0.446219 0.438983 0.403548 0.516887 0.538446 0.517111 0.515937 0.529866 0.516427   0.5157 0.480044 0.530186 0.535145 0.584564 0.599006 0.556146 0.580509 0.496681  0.49815 0.490002 0.505718 0.484941 0.486749 0.515202 0.496986 0.481524 0.491366 0.509156 0.494361 0.546591 0.531448 0.428906 0.402771 0.520563 0.495365 0.586044 0.540371 0.556311 0.582682 0.528872 0.542264 0.543639 0.513071   0.3803 0.374742 0.477268 0.455521 0.449754 0.452434 0.459116 0.458362  0.41678 0.396326 0.481461 0.513697 0.433829 0.430683  0.48385 0.514542 0.562906 0.543574  0.48078 0.518671 0.398769 0.398918 0.423736 0.448091 0.496097 0.408805 0.504467 0.504342 0.398496 0.431493 0.443749 0.473882 0.504246 0.510451 0.427034 0.428125 0.410635 0.391715 0.479127  0.47202 0.473745  0.54965 0.385755 0.365216 0.499407 0.515593 0.561866 0.549656 0.522815  0.51768 0.530588 0.543615 0.533412 0.423816 0.406861 0.542885 0.510159  0.51733 0.502284 0.484145 0.530781 0.539414 0.499914 0.490464 0.425416 0.481949 0.504664 0.489393 0.528407 0.489405  0.39757 0.387963";
//
//  std::istringstream iss(input);
//
//  double temp;
//  while (iss >> temp) {
//    thicknesses.push_back(temp);
//  }
  std::vector<double> thicknesses(F.rows(), thickness); // Assuming F.rows() gives the correct size
  int faceIndices[] =  {140,141,154, 163, 123,160, 165, 164, 250, 280, 120, 119, 110, 28, 150, 151, 200, 248, 159, 134, 58, 88, 125, 126,
                                             258, 176, 128, 118, 27, 101, 142, 146, 159,134, 282, 226, 133, 132, 57, 87, 152,153};
  ////  int faceIndices[] = {11, 12, 7, 16, 5, 14, 4, 0, 1, 2, 15,19, 21, 23, 20, 10, 8, 9, 6, 17, 13, 18, 3, 22};
////  int faceIndices[] = {297, 296, 299, 298, 168,169, 170, 171, 77, 76, 49, 48, 30, 31, 243, 242, 180, 181, 314, 315, 317, 316, 112, 113, 272, 273, 221,
////                       220, 218, 219, 217, 216, 121, 120, 2, 3, 0, 1, 74, 75, 225, 224, 7, 6, 4, 5, 244, 245,
////                       240, 241, 44, 45, 148, 149, 19, 18, 42, 43, 22, 23, 334, 335, 263,262,
////                       158, 159, 161, 160, 132, 133, 134,135, 269, 268, 260 ,261,
////                       318, 319, 237, 236, 226,227, 228, 229, 186, 187, 231, 230,
////                       248, 249, 128, 129, 238,239, 11, 10, 13, 12,
////                       190, 191, 109, 108, 89, 88, 92, 93, 185, 184};
//
//  int faceIndices[] = {21, 20, 61, 60, 62, 63, 790, 791, 143, 142, 788, 789, 748, 749,  144, 145, 1068, 1069,
//                       457, 456, 498, 499, 818, 819, 508, 509, 392, 393, 394, 395, 1102, 1103, 720, 721, 622, 623, 624,625, 890, 891,
//                       884, 885, 726, 727, 1008, 1009, 236,237, 1036, 1037, 238, 239,
//                       1004, 1005, 998, 999, 842, 843, 54, 55, 512, 513, 514, 515, 218, 219,
//                       586, 587, 584, 585, 1104, 1105, 1122, 1123, 1096, 1097, 428, 429, 1098, 1099,
//                       1116, 1117, 1100, 1101, 24, 25, 1112, 1113, 1042, 1043, 26,27, 156, 157, 594, 595,
//                       418, 419, 501, 500, 462, 463, 460, 461, 90, 91, 547, 546, 398, 399, 146, 147,
//                       400, 401, 148, 149, 732, 733, 554, 555, 838, 839, 557, 556,
//                       588, 589, 1034, 1035, 292, 293, 290, 291, 294, 295, 347, 346, 310 ,311 ,
//                       308, 309, 312, 313, 1111, 1110, 802, 803, 1016, 1017, 563, 562,
//                       560, 561, 378, 379, 382, 383, 380, 381, 628, 629,
//                       528, 529, 1150, 1151, 1030, 1031, 230 ,231 ,1032, 1033, 233, 232, 820, 821, 822, 823,
//                       752, 753, 912, 913, 910, 911, 696, 697, 1148, 1149, 698, 699, 1124, 1125, 969, 968};
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
