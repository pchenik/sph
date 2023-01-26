#Kubuntu 22.10
cmake_minimum_required(VERSION 3.24)
project(sph)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -fopenmp")

include_directories(/usr/include/)

find_package(OpenCL REQUIRED)
set(DIR /usr/lib/x86_64-linux-gnu/)
set(SOURCES main.cc theora.cc opengl.hh theora.hh vector.hh)
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
target_link_libraries(${PROJECT_NAME} PRIVATE GL)
target_link_libraries(${PROJECT_NAME} PRIVATE GLU)
target_link_libraries(${PROJECT_NAME} PRIVATE GLEW)
target_link_libraries(${PROJECT_NAME} PRIVATE glut)
target_link_libraries(${PROJECT_NAME} PRIVATE X11)

target_link_libraries(${PROJECT_NAME} PRIVATE ${DIR}libtheoraenc.so)
target_link_libraries(${PROJECT_NAME} PRIVATE ${DIR}libtheora.so)
target_link_libraries(${PROJECT_NAME} PRIVATE ${DIR}libtheoradec.so)
target_link_libraries(${PROJECT_NAME} PRIVATE ${DIR}libogg.a)

set_target_properties(${PROJECT_NAME} PROPERTIES C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF)
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=200)
