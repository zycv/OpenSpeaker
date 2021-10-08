cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(openspeaker VERSION 0.1)

include_directories(
  ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/kaldi # for not changing c++ header
  # names in kaldi source files
)

include(FetchContent)
# third_party: glog
FetchContent_Declare(
  glog
  URL https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH
    SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

# third_party: gtest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.10.0.zip
  URL_HASH
    SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91)
FetchContent_MakeAvailable(googletest)

# utils
add_library(utils STATIC utils/string.cc utils/utils.cc)
target_link_libraries(utils PUBLIC glog::glog)

# frontend
add_library(frontend STATIC frontend/feature_pipeline.cc frontend/fft.cc)

target_link_libraries(frontend PUBLIC utils)

add_executable(fbank_main frontend/fbank_main.cc)
target_link_libraries(fbank_main PUBLIC frontend utils)
