cmake_minimum_required(VERSION 3.10)
project(example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_subdirectory("external/fabsim")
add_subdirectory("external/optim")
add_subdirectory("external/polyscope")

# c++ flags
set(CMAKE_CXX_STANDARD 17)

# Project files
add_executable(composite_example src/composite.cpp)
add_executable(membrane_example src/membrane.cpp)
add_executable(rod_example src/rod.cpp)
add_executable(inflatable_example src/inflation.cpp)
add_executable(orthotropic_example src/membrane_orthotropic.cpp)
add_executable(gradient src/gradient_hessian_check.cpp)
add_executable(test src/test_file.cpp)
add_executable(best_fit_thickness src/genetic_algorithm.cpp
        src/genetic_algorithm.cpp
        external/fabsim/tests/check_finite_differences.cpp)

target_link_libraries(composite_example fabsim optim polyscope)
target_link_libraries(membrane_example fabsim optim polyscope)
target_link_libraries(rod_example fabsim optim polyscope)
target_link_libraries(inflatable_example fabsim optim polyscope)
target_link_libraries(orthotropic_example fabsim optim polyscope)
target_link_libraries(gradient fabsim optim polyscope)
target_link_libraries(test fabsim optim polyscope)
target_link_libraries(best_fit_thickness fabsim optim polyscope)
