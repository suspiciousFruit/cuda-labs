cmake_minimum_required(VERSION 3.23)
project(cuda_labs CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(matrix_multiplication)
add_subdirectory(matrix_transpose)
add_subdirectory(device_info)
add_subdirectory(reduction)
add_subdirectory(histogram)
add_subdirectory(many_transactions)
