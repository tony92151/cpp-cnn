cmake_minimum_required(VERSION 3.5)

project(Thirdparty)



set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Armadillo REQUIRED)

find_package(Threads)



include_directories(${Boost_INCLUDE_DIR})
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(
  e1 
  e1_plot.cpp
  )

add_executable(
  e2 
  e2_threadTest.cpp
  )
target_link_libraries(e1 LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(e1 LINK_PUBLIC ${ARMADILLO_LIBRARIES})
target_link_libraries(e2 ${CMAKE_THREAD_LIBS_INIT})

#file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


