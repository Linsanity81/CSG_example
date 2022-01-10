cmake_minimum_required(VERSION 3.16)
project(MeshVoxel)

if(LINUX)
    set(CMAKE_CXX_STANDARD 17)
endif()

# If you have cmake 3.14 or newer you can even use FetchContent instead of
# add_subdirectory() to pull in dlib as a dependency.  So instead of using the
# above add_subdirectory() command, you could use the following three commands
# to make dlib available:
include(FetchContent)
FetchContent_Declare(dlib
      GIT_REPOSITORY https://github.com/davisking/dlib.git
      GIT_TAG        v19.18)
FetchContent_MakeAvailable(dlib)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Libigl
option(LIBIGL_WITH_EIGEN             "Use Eigen"          ON)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Uuse IMGUI"         ON)
include(libigl)

#tbb
add_subdirectory(ext/tbb)

#catch2
add_subdirectory(ext/Catch2)

#lbfgs
include_directories(ext/LBFGSpp/include)
# main code
set(SRC_FILES
        MeshVoxel.cpp MeshVoxel.h
        MeshVoxelARAP.cpp MeshVoxelARAP.h
        MeshVoxelARAP_Solver.h
        MeshVoxelARAP_dlibSolver.h)

add_executable(MeshVoxel ${SRC_FILES} main.cpp)
target_link_libraries(MeshVoxel PUBLIC
        igl::core
        igl::opengl_glfw
        igl::opengl_glfw_imgui
        tbb tbbmalloc
        dlib::dlib)

# test
add_executable(MeshVoxelTest ${SRC_FILES} test.cpp)
target_link_libraries(MeshVoxelTest PUBLIC
        igl::core
        igl::opengl_glfw
        igl::opengl_glfw_imgui
        tbb tbbmalloc
        Catch2::Catch2WithMain
        dlib::dlib)
