cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ptcpp)

find_package(Torch REQUIRED)

add_executable(ptcpp ptcpp.cpp)
target_compile_options(ptcpp PRIVATE -g -O0 )
target_link_libraries(ptcpp "${TORCH_LIBRARIES}")
set_property(TARGET ptcpp PROPERTY CXX_STANDARD 14)
