cmake_minimum_required(VERSION 3.16)
project(MeshVoxel)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Libigl
option(LIBIGL_WITH_EIGEN             "Use Eigen"          ON)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_CGAL              "Use CGAL"           ON)
option(LIBIGL_USE_STATIC_LIBRARY     "Use libIGL as static librarie" ON)
include(libigl)

#tbb
add_subdirectory(ext/tbb)

#catch2
add_subdirectory(ext/Catch2)

# main code
set(SRC_FILES MeshVoxel.cpp MeshVoxel.h)
add_executable(MeshVoxel ${SRC_FILES} main.cpp)
target_link_libraries(MeshVoxel PUBLIC
        igl::core igl::opengl_glfw igl::cgal
        tbb tbbmalloc)

# test
add_executable(MeshVoxelTest ${SRC_FILES} test.cpp)
target_link_libraries(MeshVoxelTest PUBLIC
        igl::core igl::opengl_glfw igl::cgal
        tbb tbbmalloc Catch2::Catch2WithMain)
