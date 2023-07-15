 
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
 
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(Eigen3_INCLUDE_DIRS /usr/include/eigen3)
 
project(CODE)

add_executable(pl_icp pl_icp.cpp)

# include_directories(${CMAKE_SOURCE_DIR} include)  # cmakelist 所在目录的include
 
# include_directories(${CMAKE_SOURCE_DIR} src)

include_directories( ${CERES_INCLUDE_DIRS}
                     include)
 
# link_libraries(pthread)

find_package( Ceres REQUIRED)

find_package( PCL 1.8 REQUIRED )

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  
  # For Use Not PreCompiled Features 
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )

  # [Linker]>[Input]>[Additional Dependencies]
  target_link_libraries( pl_icp ${PCL_LIBRARIES} )
endif()

link_libraries(${JSONCPP_LIBRARIES})
target_link_libraries(pl_icp ${JSONCPP_LIBRARIES} ${CERES_LIBRARIES})
 

 

 
