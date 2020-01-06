cmake_minimum_required (VERSION 2.6)
project (KGRAPH)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -msse2 -O3 -std=c++11 -fPIC")

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# For Debugging
# 
# variable_watch(Python_INCLUDE_DIRS)
# variable_watch(Python_LIBRARIES)
# variable_watch(NUMPY_INCLUDE_DIR)
# variable_watch(Boost_PYTHON_FOUND)
# variable_watch(Boost_PROGRAM_OPTIONS_FOUND)
# variable_watch(Boost_LIBRARIES)
variable_watch(Boost_LIBRARY_DIR_RELEASE)

set (BOOST_ROOT          C:/Boost)
set (BOOST_INCLUDEDIR    D:/Boost/include/boost-1_64/)
set (BOOST_LIBRARYDIR    D:/Boost/lib/)
set (Boost_USE_STATIC_RUNTIME OFF)
set (Boost_USE_DEBUG_RUNTIME OFF)  
set (Boost_DEBUG ON)  #<---------- Real life saver
# set (Boost_NO_SYSTEM_PATHS ON)



find_package(Boost REQUIRED COMPONENTS python python3 timer chrono system program_options)
find_package(Python REQUIRED Development)

set(LIB_SRC kgraph.cpp metric.cpp)
set(MODULE_SRC kgraph.cpp metric.cpp python/pykgraph.cpp)

set(NUMPY_INCLUDE_DIR ${Python_INCLUDE_DIRS}/../lib/site-packages/numpy/core/include)

include_directories(${KGRAPH_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Python_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR_DEBUG})
link_directories(${Boost_LIBRARY_DIR_RELEASE})

add_library(kgraph SHARED ${LIB_SRC})
add_library(kgrapha STATIC ${LIB_SRC})
target_link_libraries(kgraph ${Boost_LIBRARIES})
target_link_libraries(kgrapha ${Boost_LIBRARIES})
set_target_properties(kgrapha PROPERTIES OUTPUT_NAME kgraph)


add_library(pykgraph SHARED ${MODULE_SRC})
target_link_libraries(pykgraph ${Python_LIBRARIES})
target_link_libraries(pykgraph ${Boost_LIBRARIES})
set_target_properties(pykgraph PROPERTIES SUFFIX ".pyd")




# add_executable(index index.cpp)
# add_executable(search search.cpp)
# 
# 
# 
# 
# set(TOOLS index search)
# foreach(TOOL ${TOOLS})
# 	target_link_libraries(${TOOL} kgrapha ${Boost_LIBRARIES} rt)
# endforeach(TOOL)
# install (FILES kgraph.h kgraph-data.h DESTINATION include) 
# install (TARGETS kgraph kgrapha
# 	 LIBRARY DESTINATION lib
# 	 ARCHIVE DESTINATION lib)
# 
# 