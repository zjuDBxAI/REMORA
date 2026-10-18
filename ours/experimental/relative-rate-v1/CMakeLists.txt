cmake_minimum_required(VERSION 3.17)
project(remora_relative_rate LANGUAGES CXX)
# Isolated build: never modify the binaries used by the completed suites.
set(REMORA_OURS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
get_filename_component(REMORA_OURS_ROOT "${REMORA_OURS_ROOT}" REALPATH)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(OURS_JI_KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Relative-rate kernel overlay")
# Audit targets live in this parent directory, so import their dependencies
# here as well rather than relying on child-directory imported-target scope.
list(APPEND CMAKE_MODULE_PATH "${REMORA_OURS_ROOT}/transfer/cmake")
find_package(Threads REQUIRED)
find_package(Glog REQUIRED)
find_package(Fmt REQUIRED)
find_package(GFlags REQUIRED)
add_subdirectory("${REMORA_OURS_ROOT}" "${CMAKE_BINARY_DIR}/base")
if(NOT TARGET roar_exp234_audit)
    add_executable(roar_exp234_audit "${REMORA_OURS_ROOT}/test/t2i-1M/cpp/exp234_audit.cpp")
    set_property(TARGET roar_exp234_audit PROPERTY CXX_STANDARD 20)
    target_include_directories(roar_exp234_audit PRIVATE
        "${RABITQ_ROOT}/include" "${SAQ_ROOT}/saqlib")
    target_link_libraries(roar_exp234_audit PRIVATE
        roar_drb_core Threads::Threads glog::glog fmt::fmt ${GFLAGS_LIBRARIES})
    target_compile_options(roar_exp234_audit PRIVATE $<$<CONFIG:Release>:-march=native>)
endif()
