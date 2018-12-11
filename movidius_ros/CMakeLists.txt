cmake_minimum_required(VERSION 2.8.3)
project(movidius_ros)

###########3

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT(BIN_FOLDER))
    if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set (ARCH intel64)
    else()
        set (ARCH ia32)
    endif()

    set (BIN_FOLDER ${ARCH})
endif()

if (NOT(IE_MAIN_SOURCE_DIR))
    # in case if samples are built out of IE repo
    set (IE_MAIN_SAMPLES_DIR ${CMAKE_CURRENT_BINARY_DIR})
else()
    # in case if samples are built from IE repo
    set (IE_MAIN_SAMPLES_DIR ${IE_MAIN_SOURCE_DIR})
endif()

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
set (LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)

##############

# Add plain cmake packages
FIND_PACKAGE(OpenCV REQUIRED)
find_package(catkin_simple REQUIRED)
find_package(InferenceEngine 1.4 REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Werror=return-type ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self -Wmaybe-uninitialized")
################

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
####################################


include_directories (
        ${CMAKE_CURRENT_SOURCE_DIR}/common/format_reader
        ${InferenceEngine_INCLUDE_DIRS}
        $ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine/src/extension
#        /opt/intel/computer_vision_sdk_2018.4.420/deployment_tools/inference_engine/src/extension
        ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/gflags/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

message("Debug")
message( ${CMAKE_CURRENT_SOURCE_DIR} )
message( ${InferenceEngine_LIBRARIES} )
message( ${OpenCV_LIBRARIES} )
#message( ${LIB_FOLDER} )


link_directories(${LIB_FOLDER})

cs_add_library(human_pose_estimator src/human_pose_estimator.cpp)
cs_add_library(human_pose src/human_pose.cpp)
#cs_add_library(render_human_pose src/render_human_pose.cpp)
cs_add_library(movidius_ros src/movidius_ros.cpp)

cs_add_executable(movidius_ros_node src/movidius_ros_node.cpp)
target_link_libraries(movidius_ros_node movidius_ros)
target_link_libraries(movidius_ros_node human_pose_estimator)
target_link_libraries(movidius_ros_node human_pose)
#target_link_libraries(movidius_ros_node render_human_pose)
#target_link_libraries(movidius_ros_node ${InferenceEngine_LIBRARIES} IE::ie_cpu_extension format_reader gflags)
target_link_libraries(movidius_ros_node ${InferenceEngine_LIBRARIES} IE::ie_cpu_extension gflags)
target_link_libraries(movidius_ros_node ${OpenCV_LIBRARIES})

set_target_properties(movidius_ros_node PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
        COMPILE_PDB_NAME movidius_ros_node)

cs_install()
cs_export()

# CMake Indexing
FILE(GLOB_RECURSE LibFiles "include/*")
add_custom_target(headers SOURCES ${LibFiles})
FILE(GLOB_RECURSE ProtoFiles "proto/*")
add_custom_target(protos SOURCES ${ProtoFiles})
