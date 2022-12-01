cmake_minimum_required(VERSION 3.23)
project(matrix_transpose CUDA)

add_executable(${PROJECT_NAME} main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
