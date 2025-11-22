cmake_minimum_required(VERSION 3.15)
project(city_designer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# ------------------------------------------------------
# macOS: Add Homebrew search paths for GLFW / libs
# ------------------------------------------------------
if(APPLE)
    message(STATUS "Applying macOS Homebrew search paths...")

    # Homebrew root
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

    # Include + Lib paths (NO semicolon mistakes)
    include_directories("/opt/homebrew/include")
    link_directories("/opt/homebrew/lib")

    # Ensure pkg-config can find Homebrew libraries
    if(NOT DEFINED ENV{PKG_CONFIG_PATH})
        set(ENV{PKG_CONFIG_PATH} "/opt/homebrew/lib/pkgconfig")
    else()
        set(ENV{PKG_CONFIG_PATH}
            "/opt/homebrew/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    endif()
endif()

# ------------------------------------------------------
# Source files
# ------------------------------------------------------
set(SRCS
    src/main.cpp
    src/bresenham.cpp
    src/circle.cpp
    src/shader.h
)

# Optional glad loader (for Linux/Windows)
if(EXISTS "${CMAKE_SOURCE_DIR}/src/glad/glad.c")
    list(APPEND SRCS src/glad/glad.c)
    set(HAS_LOCAL_GLAD ON)
endif()

add_executable(${PROJECT_NAME} ${SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/src")

# ------------------------------------------------------
# macOS OpenGL handling
# ------------------------------------------------------
if(APPLE)
    message(STATUS "Building on macOS — using system OpenGL (no GLAD needed).")
else()
    if(HAS_LOCAL_GLAD)
        target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/src/glad")
        message(STATUS "Using local GLAD.")
    else()
        message(STATUS "No GLAD found. Ensure a GL loader is present for Linux/Windows.")
    endif()
endif()

# ------------------------------------------------------
# Find GLFW (pkg-config → fallback → error)
# ------------------------------------------------------
find_package(PkgConfig QUIET)

set(GLFW_FOUND FALSE)

if(PKG_CONFIG_FOUND)
    pkg_check_modules(GLFW_PKG glfw3)
    if(GLFW_PKG_FOUND)
        message(STATUS "Found GLFW via pkg-config")

        target_include_directories(${PROJECT_NAME}
            PRIVATE ${GLFW_PKG_INCLUDE_DIRS})

        target_link_libraries(${PROJECT_NAME}
            PRIVATE ${GLFW_PKG_LIBRARIES})

        # Add linker flags if pkg-config provided any
        foreach(f ${GLFW_PKG_LDFLAGS})
            string(STRIP "${f}" ftrim)
            if(NOT "${ftrim}" STREQUAL "")
                set_property(TARGET ${PROJECT_NAME}
                    APPEND PROPERTY LINK_FLAGS "${ftrim}")
            endif()
        endforeach()

        set(GLFW_FOUND TRUE)
    endif()
endif()

if(NOT GLFW_FOUND)
    find_package(glfw3 QUIET)
    if(glfw3_FOUND)
        message(STATUS "Found glfw3 via find_package")
        target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
        set(GLFW_FOUND TRUE)
    endif()
endif()

# If still not found → final fallback (Homebrew default)
if(NOT GLFW_FOUND)
    message(WARNING "GLFW NOT FOUND via pkg-config or find_package. Using fallback -lglfw")
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
endif()

# ------------------------------------------------------
# OpenGL library
# ------------------------------------------------------
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)

# ------------------------------------------------------
# macOS Frameworks
# ------------------------------------------------------
if(APPLE)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE "-framework Cocoa"
                 "-framework IOKit"
                 "-framework CoreVideo"
                 "-framework CoreFoundation")
endif()

# Windows OpenGL
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE opengl32)
endif()

# ------------------------------------------------------
# Output directory (bin/)
# ------------------------------------------------------
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ------------------------------------------------------
# Copy assets folder to build directory (if exists)
# ------------------------------------------------------
file(GLOB ASSETS "${CMAKE_SOURCE_DIR}/assets/*")
foreach(a ${ASSETS})
    configure_file(${a} ${CMAKE_BINARY_DIR} COPYONLY)
endforeach()

# ------------------------------------------------------
# Summary messages
# ------------------------------------------------------
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Source dir: ${CMAKE_SOURCE_DIR}")
message(STATUS "Binary dir: ${CMAKE_BINARY_DIR}")
