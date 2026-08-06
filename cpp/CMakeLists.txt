cmake_minimum_required(VERSION 3.10)
project(r1_arm_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Install unitree_sdk2 first (see README.md in this directory):
#   cd unitree_sdk2 && mkdir build && cd build && cmake .. && sudo make install
find_package(unitree_sdk2 REQUIRED)

add_executable(r1_arm_example r1_arm_example.cpp)
target_link_libraries(r1_arm_example unitree_sdk2)
