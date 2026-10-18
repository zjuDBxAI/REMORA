cmake_minimum_required(VERSION 3.17)
project(roar_drb_pq LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenMP REQUIRED)
option(OURS_NATIVE_OPTIMIZATION "Compile the search hot path for the current CPU" ON)
# Opt-in overlay: keep frozen/in-progress suites on their original sources.
set(OURS_JI_KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
    "Root containing src/ji_online.cpp, src/primary_ji_search_main.cpp and include/ours/ji_online.h")
foreach(kernel_file src/ji_online.cpp src/primary_ji_search_main.cpp include/ours/ji_online.h)
    if(NOT EXISTS "${OURS_JI_KERNEL_ROOT}/${kernel_file}")
        message(FATAL_ERROR "Missing kernel overlay file: ${OURS_JI_KERNEL_ROOT}/${kernel_file}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(roar_drb_core
    src/drb_model.cpp
    src/exact_vector_store.cpp
    src/generic_primary.cpp
    ${OURS_JI_KERNEL_ROOT}/src/ji_online.cpp
    src/mapped_fbin.cpp
    src/roar_graph.cpp
    src/search_kernel.cpp)
target_include_directories(roar_drb_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(roar_drb_core BEFORE PUBLIC ${OURS_JI_KERNEL_ROOT}/include)
target_compile_options(roar_drb_core PRIVATE -Wall -Wextra -Wpedantic)
if(OURS_NATIVE_OPTIMIZATION)
    target_compile_options(roar_drb_core PRIVATE $<$<CONFIG:Release>:-march=native>)
endif()

add_executable(roar_drb_search src/search_main.cpp)
target_link_libraries(roar_drb_search PRIVATE roar_drb_core Threads::Threads)

add_executable(roar_base_pair_benchmark src/base_pair_benchmark_main.cpp)
target_link_libraries(roar_base_pair_benchmark PRIVATE
    roar_drb_core Threads::Threads Boost::program_options)

add_executable(roar_compare_graphs src/compare_roar_graphs_main.cpp)
target_link_libraries(roar_compare_graphs PRIVATE Boost::program_options)

set(RABITQ_ROOT "/home/xzl/RaBitQ-Library" CACHE PATH
    "Path to the official RaBitQ-Library checkout")
if(EXISTS "${RABITQ_ROOT}/include/rabitqlib/quantization/rabitq.hpp")
    add_executable(roar_export_rabitq_primary src/export_rabitq_primary_main.cpp)
    target_include_directories(roar_export_rabitq_primary PRIVATE
        "${RABITQ_ROOT}/include")
    target_link_libraries(roar_export_rabitq_primary PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(roar_export_rabitq_primary PRIVATE
        -Wall -Wextra -Wpedantic $<$<CONFIG:Release>:-march=native>)
endif()

set(SAQ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/SAQ" CACHE PATH
    "Path to the official SAQ checkout")
if(EXISTS "${RABITQ_ROOT}/include/rabitqlib/quantization/rabitq.hpp" AND
   EXISTS "${SAQ_ROOT}/saqlib/quantization/saq_data.hpp")
    find_package(Glog REQUIRED)
    find_package(Fmt REQUIRED)
    find_package(GFlags REQUIRED)
    add_executable(roar_primary_ji_search ${OURS_JI_KERNEL_ROOT}/src/primary_ji_search_main.cpp)
    set_property(TARGET roar_primary_ji_search PROPERTY CXX_STANDARD 20)
    target_include_directories(roar_primary_ji_search PRIVATE
        "${RABITQ_ROOT}/include"
        "${SAQ_ROOT}/saqlib")
    target_link_libraries(roar_primary_ji_search PRIVATE
        roar_drb_core Threads::Threads glog::glog fmt::fmt ${GFLAGS_LIBRARIES})
    target_compile_options(roar_primary_ji_search PRIVATE
        -Wall -Wextra -Wpedantic $<$<CONFIG:Release>:-march=native>)
    add_executable(roar_terminal_rate_audit test/t2i-1M/cpp/terminal_rate_audit.cpp)
    set_property(TARGET roar_terminal_rate_audit PROPERTY CXX_STANDARD 20)
    target_include_directories(roar_terminal_rate_audit PRIVATE
        "${RABITQ_ROOT}/include" "${SAQ_ROOT}/saqlib")
    target_link_libraries(roar_terminal_rate_audit PRIVATE
        roar_drb_core Threads::Threads glog::glog fmt::fmt ${GFLAGS_LIBRARIES})
    target_compile_options(roar_terminal_rate_audit PRIVATE $<$<CONFIG:Release>:-march=native>)

    # Exp. 2 uses exactly the same fast adapters as the exp. 3/4 search binary.
    get_filename_component(EXP234_KERNEL_REAL "${OURS_JI_KERNEL_ROOT}" REALPATH)
    if(EXP234_KERNEL_REAL STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/experimental/traversal-fast-v1")
        add_executable(roar_exp234_audit test/t2i-1M/cpp/exp234_audit.cpp)
        set_property(TARGET roar_exp234_audit PROPERTY CXX_STANDARD 20)
        target_include_directories(roar_exp234_audit PRIVATE
            "${RABITQ_ROOT}/include" "${SAQ_ROOT}/saqlib")
        target_link_libraries(roar_exp234_audit PRIVATE
            roar_drb_core Threads::Threads glog::glog fmt::fmt ${GFLAGS_LIBRARIES})
        target_compile_options(roar_exp234_audit PRIVATE $<$<CONFIG:Release>:-march=native>)
    endif()

    # Code-native builder for the primary backends used by the JI-Pure suite.
    # Both query--base KNN and base--base topology comparisons consume codes;
    # no N x D decoded FP32 construction file is created.
    add_executable(roar_primary_ji_build
        src/primary_ji_build_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/index.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/index_bipartite.cpp)
    set_property(TARGET roar_primary_ji_build PROPERTY CXX_STANDARD 20)
    target_include_directories(roar_primary_ji_build PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/robin-map/include
        "${RABITQ_ROOT}/include"
        "${SAQ_ROOT}/saqlib")
    target_link_libraries(roar_primary_ji_build PRIVATE
        roar_drb_core Threads::Threads OpenMP::OpenMP_CXX
        glog::glog fmt::fmt ${GFLAGS_LIBRARIES})
    target_compile_options(roar_primary_ji_build PRIVATE
        -Wall -Wextra -Wpedantic $<$<CONFIG:Release>:-march=native>)

    add_executable(roar_native_primary_reader
        src/native_primary_reader_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/index.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/index_bipartite.cpp)
    set_property(TARGET roar_native_primary_reader PROPERTY CXX_STANDARD 20)
    target_include_directories(roar_native_primary_reader PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/robin-map/include
        "${RABITQ_ROOT}/include" "${SAQ_ROOT}/saqlib")
    target_link_libraries(roar_native_primary_reader PRIVATE
        roar_drb_core Threads::Threads OpenMP::OpenMP_CXX
        glog::glog fmt::fmt ${GFLAGS_LIBRARIES})
    target_compile_options(roar_native_primary_reader PRIVATE $<$<CONFIG:Release>:-march=native>)
endif()

# Independent code-native construction executable.  The stock RoarGraph
# builder remains unchanged and is still built by the repository root CMake
# project.  This target reuses the topology algorithm, but obtains every
# base--base distance from PQ/RPQ codes rather than an N x D FP32 matrix.
add_executable(roar_quantized_build
    src/quantized_build_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/index_bipartite.cpp)
target_include_directories(roar_quantized_build PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/robin-map/include)
target_link_libraries(roar_quantized_build PRIVATE
    roar_drb_core
    Threads::Threads
    Boost::program_options
    OpenMP::OpenMP_CXX)
target_compile_options(roar_quantized_build PRIVATE -Wall -Wextra -Wpedantic)
if(OURS_NATIVE_OPTIMIZATION)
    target_compile_options(roar_quantized_build PRIVATE
        $<$<CONFIG:Release>:-march=native>)
endif()

enable_testing()
add_executable(roar_raw_score_parity test/t2i-1M/cpp/raw_score_native_parity.cpp)
target_link_libraries(roar_raw_score_parity PRIVATE roar_drb_core)
add_executable(roar_drb_tests tests/test_drb_kernel.cpp)
target_link_libraries(roar_drb_tests PRIVATE roar_drb_core)
add_test(NAME roar_drb_tests COMMAND roar_drb_tests)

add_executable(roar_terminal_selection_tests tests/test_terminal_selection.cpp)
target_link_libraries(roar_terminal_selection_tests PRIVATE roar_drb_core)
add_test(NAME roar_terminal_selection_tests COMMAND roar_terminal_selection_tests)

add_executable(roar_generic_primary_tests tests/test_generic_primary.cpp)
target_link_libraries(roar_generic_primary_tests PRIVATE roar_drb_core)
if(OURS_NATIVE_OPTIMIZATION)
    target_compile_options(roar_generic_primary_tests PRIVATE $<$<CONFIG:Release>:-march=native>)
endif()
add_test(NAME roar_generic_primary_tests COMMAND roar_generic_primary_tests)

if(EXISTS "${OURS_JI_KERNEL_ROOT}/tests/test_ji_batch.cpp")
    add_executable(roar_ji_batch_tests ${OURS_JI_KERNEL_ROOT}/tests/test_ji_batch.cpp)
    target_link_libraries(roar_ji_batch_tests PRIVATE roar_drb_core)
endif()

add_executable(roar_exact_store_tests tests/test_exact_vector_store.cpp)
target_link_libraries(roar_exact_store_tests PRIVATE roar_drb_core)
