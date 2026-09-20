cmake_minimum_required(VERSION 3.14)
project(tensor-calculator-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# header-only library. other cmake projects can link to it with
#   target_link_libraries(my_target PRIVATE tensor)
add_library(tensor INTERFACE)
target_include_directories(tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(tensor INTERFACE cxx_std_17)

# the interactive calculator
add_executable(tensor-calculator main.cpp)
target_link_libraries(tensor-calculator PRIVATE tensor)

# tests (run with: ctest --test-dir build)
enable_testing()
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE tensor)
add_test(NAME tensor COMMAND test_tensor)

if(MSVC)
    target_compile_options(tensor-calculator PRIVATE /W4)
    target_compile_options(test_tensor PRIVATE /W4)
else()
    target_compile_options(tensor-calculator PRIVATE -Wall -Wextra)
    target_compile_options(test_tensor PRIVATE -Wall -Wextra)
endif()