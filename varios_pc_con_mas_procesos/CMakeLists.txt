cmake_minimum_required(VERSION 3.15)
project(knn_mpi LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Soporte requerido para MPI
find_package(MPI REQUIRED)

# Soporte para OpenCV (si está presente en el sistema)
find_package(OpenCV QUIET)

# Ejecutable principal distribuido con MPI
add_executable(knn_mpi main.cc crossValidation.cc mnist.cc knn.cc sample.cc)
target_include_directories(knn_mpi PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(knn_mpi PRIVATE MPI::MPI_CXX)

if(OpenCV_FOUND)
  message(STATUS "OpenCV encontrado: ${OpenCV_VERSION}")
  target_include_directories(knn_mpi PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(knn_mpi PRIVATE ${OpenCV_LIBS})
endif()

# Tests unitarios
include(CTest)
if(BUILD_TESTING)
  add_executable(knn2_tests tests.cc sample.cc knn.cc mnist.cc crossValidation.cc)
  target_link_libraries(knn2_tests PRIVATE MPI::MPI_CXX)
  add_test(NAME knn2_tests COMMAND knn2_tests)
endif()
