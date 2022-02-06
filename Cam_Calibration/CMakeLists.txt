cmake_minimum_required(VERSION 3.10)
project(Calibration)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++14 -O3 -fopenmp -pthread")


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################

#Google
find_package(GFLAGS REQUIRED) 
include_directories(${GFLAGS_INCLUDE_DIRS})
find_package(GLOG REQUIRED) 
include_directories(${LOG_INCLUDE_DIRS})

# Eigen
include_directories("/usr/local/include/eigen3")

# OpenCV
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
find_Package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})
set(G2O_LIBS g2o_core g2o_stuff g2o_types_sba g2o_types_slam3d
             g2o_solver_csparse g2o_csparse_extension cxsparse)

set(THIRD_PARTY_LIBS ${GFLAGS_LIBRARIES}
                     ${GLOG_LIBRARIES}
                     ${OpenCV_LIBS}
                     ${Sophus_LIBRARIES}
                     ${G2O_LIBS}
                     ${CSPARSE_LIBRARY})

################### source and driver ######################

include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(src)
add_subdirectory(app)