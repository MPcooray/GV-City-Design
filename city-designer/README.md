# City Designer - Part 1 (Modern OpenGL starter)

## Important: assignment spec (local)
See the assignment PDF uploaded: /mnt/data/Assignment_Release.pdf

## Requirements
- CMake >= 3.15
- GLFW (install via Homebrew / apt / vcpkg / package manager)
- On macOS: no extra GL loader needed (system OpenGL 3.3 headers are used).
- On Windows/Linux: you should have a GL loader (glad) available.
  - Option 1: include `src/glad/` (glad.c / glad.h / khrplatform.h) — I can provide this.
  - Option 2: install via vcpkg: `vcpkg install glad glfw3` and build with `-DCMAKE_TOOLCHAIN_FILE=...`

## Build (macOS / Linux)
```bash
# macOS example (Homebrew)
brew install cmake glfw
mkdir build && cd build
cmake ..
cmake --build . --config Release
./bin/city_designer
