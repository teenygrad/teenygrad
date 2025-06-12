cmake_minimum_required(VERSION 3.28)

include_directories(${PROJECT_SOURCE_DIR})

add_library(teeny STATIC teeny.cpp)

# Install headers
install(DIRECTORY ./
        DESTINATION include/teeny
        FILES_MATCHING PATTERN "*.h"
                      PATTERN "*.hpp"
                      PATTERN "*.inc"
                      PATTERN "*.def"
                      PATTERN "*.pdll"
                      PATTERN "*.td"
                      PATTERN "*.td.h")

# Install the library
install(TARGETS teeny
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)