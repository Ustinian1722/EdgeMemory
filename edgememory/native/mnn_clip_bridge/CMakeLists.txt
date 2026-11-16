cmake_minimum_required(VERSION 3.20)
project(hexagonflow_mnn_clip_bridge LANGUAGES CXX)

set(MNN_SOURCE_DIR "" CACHE PATH "MNN 3.6.2 source checkout")
set(MNN_BUILD_DIR "" CACHE PATH "MNN OpenCL-enabled ARM64 build directory")
if(NOT MNN_SOURCE_DIR OR NOT MNN_BUILD_DIR)
  message(FATAL_ERROR "Set MNN_SOURCE_DIR and MNN_BUILD_DIR")
endif()
find_library(MNN_LIBRARY NAMES MNN PATHS "${MNN_BUILD_DIR}" "${MNN_BUILD_DIR}/lib"
  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
if(NOT MNN_LIBRARY)
  message(FATAL_ERROR "libMNN was not found in MNN_BUILD_DIR")
endif()

add_library(hexagonflow_mnn_clip SHARED ../../../src/mnn_clip_bridge.cpp)
target_compile_features(hexagonflow_mnn_clip PRIVATE cxx_std_17)
target_include_directories(hexagonflow_mnn_clip PRIVATE
  "${MNN_SOURCE_DIR}/include"
  "${MNN_SOURCE_DIR}/3rd_party/OpenCLHeaders")
target_link_libraries(hexagonflow_mnn_clip PRIVATE "${MNN_LIBRARY}" ${CMAKE_DL_LIBS})
set_target_properties(hexagonflow_mnn_clip PROPERTIES
  OUTPUT_NAME hexagonflow_mnn_clip
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")
