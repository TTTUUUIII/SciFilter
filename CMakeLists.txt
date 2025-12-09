cmake_minimum_required(VERSION 3.10.0)
project(SciFilter VERSION 0.1.0 LANGUAGES C CXX)

include_directories(include)
add_executable(SciFilter main.cpp SciFilter.cpp)

