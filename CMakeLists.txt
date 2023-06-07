CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

PROJECT(focus-stacker)
#
#if(WIN32)
#    SET(CMAKE_FIND_LIBRARY_PREFIXES "")
#    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
#endif()

message("library prefix: ${CMAKE_FIND_LIBRARY_PREFIXES}")
message("library suffix: ${CMAKE_FIND_LIBRARY_SUFFIXES}")

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src/main/header/)


# set(OpenCV_STATIC ON)
FIND_PACKAGE( OpenCV 4.1 COMPONENTS core imgproc photo imgcodecs objdetect features2d highgui REQUIRED)
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


## Find and use 'Boost'
#find_package(Boost)
#find_package(Boost COMPONENTS system filesystem REQUIRED)
#include_directories(${BOOST_INCLUDE_DIR})

# Find header-only library 'date'
find_path(DATE_INCLUDE_DIR date/date.h)
include_directories(${DATE_INCLUDE_DIR})
message(STATUS "Using date header library from ${DATE_INCLUDE_DIR}")

# Find and use 'SPDLog'
find_path(SPDLOG_INCLUDE_DIR spdlog/logger.h)
include_directories(${SPDLOG_INCLUDE_DIR})
message(STATUS "Using SPDLog header library from: ${SPDLOG_INCLUDE_DIR}" )

## Find and use 'SPDLog'
#find_path(JSON_INCLUDE_DIR nlohmann/json.hpp)
#include_directories(${JSON_INCLUDE_DIR})
#message(STATUS "Using json header library from: ${JSON_INCLUDE_DIR}" )

# Find and use 'args'
find_path(ARGS_INCLUDE_DIR args.hxx)
include_directories(${ARGS_INCLUDE_DIR})
message(STATUS "Using args header library from: ${ARGS_INCLUDE_DIR}" )

# Find and use 'fmt'
find_package(fmt CONFIG REQUIRED)
include_directories(${FMT_INCLUDE_DIR})


find_package(Threads REQUIRED)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")


set(CMAKE_CXX_STANDARD 20) # C++20...
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

message(STATUS "selected release flags: ${CMAKE_CXX_FLAGS_RELEASE}")


ADD_SUBDIRECTORY(src/main/)
