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

# third_party: libtorch 1.9.0, use FetchContent_Declare to download, and use
# find_package to find since libtorch is not a standard cmake project
set(PYTORCH_VERSION "1.9.0")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(LIBTORCH_URL
      "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
  )
  set(URL_HASH
      "SHA256=9bcd5b2703e18f2374413e55907ac20b2e5d8f5ae06a669f2e9905096114b4f1")
  set(CMAKE_BUILD_TYPE "Release")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LIBTORCH_URL
      "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
  )
  set(URL_HASH
      "SHA256=6b99edc046f37ad37a3b04dc86663895f10c362af09fdd10884f30360d0ba023")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(LIBTORCH_URL
      "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${PYTORCH_VERSION}.zip"
  )
  set(URL_HASH
      "SHA256=5343b311201b1bd8e010ce613baedf4e5bf31dd098e1ac550b3c21f3192a4aad")
else()
  message(
    FATAL_ERROR
      "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux' or 'Darwin')"
  )
endif()
FetchContent_Declare(
  libtorch
  URL ${LIBTORCH_URL}
  URL_HASH ${URL_HASH})
set(gtest_force_shared_crt
    ON
    CACHE BOOL "Always use msvcrt.dll" FORCE)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")

# utils
add_library(utils STATIC utils/string.cc utils/utils.cc)
target_link_libraries(utils PUBLIC glog::glog)

# frontend
add_library(frontend STATIC frontend/feature_pipeline.cc frontend/fft.cc)

target_link_libraries(frontend PUBLIC utils)

add_executable(fbank_main frontend/fbank_main.cc)
target_link_libraries(fbank_main PUBLIC frontend utils ${TORCH_LIBRARIES})

add_executable(libtorch_test test_scripts/libtorch_test.cc)
target_link_libraries(libtorch_test PUBLIC frontend utils ${TORCH_LIBRARIES})
