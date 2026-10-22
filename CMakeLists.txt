cmake_minimum_required(VERSION 3.24)

# Enable C++ and CUDA compilation
project(llm_int8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile with nvcc, despite .cpp extension
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

# Target GPU architecture of build machine
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(llm_int8 src/main.cpp)

# Create a convenience 'run' target
add_custom_target(run
    COMMAND $<TARGET_FILE:llm_int8>
    DEPENDS llm_int8
    USES_TERMINAL
)