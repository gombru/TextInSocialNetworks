cmake_minimum_required(VERSION 2.8)
project(TextProposals)

SET(CMAKE_BUILD_TYPE debug)
# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

set (OpenCV_DIR "/usr/local/opencv-3.0.0/")
find_package(OpenCV 3.0.0 REQUIRED PATHS ${OpenCV_DIR})
#find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} core imgproc highgui features2d ml)

#include_directories ("/usr/src/caffe/include/")
#include_directories ("/usr/src/caffe/.build_release/src/")
#include_directories ("/usr/local/cuda-7.5/include/")
#include_directories (${OpenCV_INCLUDE_DIRS})
include_directories ("/usr/local/opencv-3.0.0/include")

message(status ${OpenCV_INCLUDE_DIRS})


#link_directories ("/usr/src/caffe/build/lib/")
#link_directories ("/usr/local/cuda-7.5/lib/")

set (EXTRA_LIBS ${EXTRA_LIBS} libcaffe.so -lprotobuf -lglog -lboost_system)


ADD_EXECUTABLE(img2hierarchy fast_clustering.cpp main.cpp agglomerative_clustering.cpp region.cpp min_bounding_box.cpp)
ADD_EXECUTABLE(evaluate evaluate.cpp fast_clustering.cpp compute_regions.cpp agglomerative_clustering.cpp region.cpp min_bounding_box.cpp)
#ADD_EXECUTABLE(img2hierarchy_cnn fast_clustering.cpp main_cnn.cpp agglomerative_clustering.cpp stopping_rule.cpp region.cpp min_bounding_box.cpp)
#ADD_EXECUTABLE(eval_IC03 fast_clustering.cpp eval_IC03.cpp agglomerative_clustering.cpp stopping_rule.cpp region.cpp min_bounding_box.cpp)
#ADD_EXECUTABLE(eval_IC15 fast_clustering.cpp eval_IC15.cpp agglomerative_clustering.cpp stopping_rule.cpp region.cpp min_bounding_box.cpp)
#ADD_EXECUTABLE(eval_SVT fast_clustering.cpp eval_SVT.cpp agglomerative_clustering.cpp stopping_rule.cpp region.cpp min_bounding_box.cpp)

# FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
  TARGET_LINK_LIBRARIES(img2hierarchy ${OpenCV_LIBS})
  TARGET_LINK_LIBRARIES(evaluate ${OpenCV_LIBS})
  #TARGET_LINK_LIBRARIES(img2hierarchy_cnn ${OpenCV_LIBS} ${EXTRA_LIBS})
  #TARGET_LINK_LIBRARIES(eval_IC03 ${OpenCV_LIBS} ${EXTRA_LIBS} tinyxml)
  #TARGET_LINK_LIBRARIES(eval_IC15 ${OpenCV_LIBS} ${EXTRA_LIBS} tinyxml)
  #TARGET_LINK_LIBRARIES(eval_SVT ${OpenCV_LIBS} ${EXTRA_LIBS} tinyxml)
ENDIF()
