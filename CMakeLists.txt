cmake_minimum_required(VERSION 3.16)
project(MeshVoxel)

if(LINUX)
    set(CMAKE_CXX_STANDARD 17)
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Libigl
option(LIBIGL_WITH_EIGEN             "Use Eigen"          ON)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
include(libigl)

# mosek
find_package(MOSET)

#tbb
add_subdirectory(ext/tbb)

#catch2
add_subdirectory(ext/Catch2)


include_directories(ext/LBFGSpp/include)

#Kintro
find_package(Knitro QUIET)
include_directories(${KNITRO_INCLUDE_DIRS} ${MOSEK_INCLUDE_DIRS})

# main code
set(SRC_FILES
        MeshVoxel.cpp MeshVoxel.h
        SurfaceEvo.cpp SurfaceEvo.h
        SurfaceSlice.h SurfaceSlice.cpp)

add_executable(MeshVoxel ${SRC_FILES} main.cpp)
target_link_libraries(MeshVoxel PUBLIC
        igl::core
        igl::opengl_glfw
        tbb tbbmalloc
        ${MOSEK_LIBRARIES})

# test
add_executable(MeshVoxelTest ${SRC_FILES} test.cpp)
target_link_libraries(MeshVoxelTest PUBLIC
        igl::core igl::opengl_glfw
        tbb tbbmalloc Catch2::Catch2WithMain
        ${MOSEK_LIBRARIES})
