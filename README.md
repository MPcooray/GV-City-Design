# City Designer - Part 1 (Modern OpenGL starter)

This small C++/OpenGL project uses CMake and GLFW. The `CMakeLists.txt` already
handles macOS Homebrew paths and will copy the `assets/` folder into the build
directory so the executable can access runtime assets.

**This README shows step-by-step setup and build instructions for macOS and
Windows (vcpkg / Visual Studio).**

**Quick notes**
- **CMake minimum:** 3.15 (project requires C++17)
- **Build output:** `build/bin/city_designer` (or `city_designer.exe` on Windows)
- **GL loader:** macOS uses system OpenGL headers (no GLAD required). On
  Windows/Linux you either provide a local `src/glad/` (glad.c/.h) or use a
  package manager (vcpkg) to install `glad` and `glfw3`.

---

## macOS (recommended: Homebrew)

1. Install prerequisites (Homebrew must be installed):

```bash
brew install cmake glfw pkg-config
```

2. Create build directory and configure:

```bash
mkdir -p build
cd build
cmake ..
```

If you want a Release build explicitly:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

3. Build and run:

```bash
cmake --build . --config Release
./bin/city_designer
```

Notes:
- On Apple Silicon Homebrew is typically at `/opt/homebrew`. The project's
  `CMakeLists.txt` already adds `/opt/homebrew` to `CMAKE_PREFIX_PATH` and
  include/library search paths.
- If `pkg-config` can't find GLFW, ensure `PKG_CONFIG_PATH` includes Homebrew
  pkgconfig (CMake file already tries to set this).

---

## Windows (recommended: vcpkg + Visual Studio)

Option A — Use `vcpkg` (recommended):

1. Install Visual Studio (2019/2022) with "Desktop development with C++".

2. Clone and bootstrap `vcpkg` (run in PowerShell or cmd):

```powershell
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg.exe install glfw3 glad:x64-windows
```

3. Configure CMake to use the `vcpkg` toolchain (from a developer command
   prompt):

```powershell
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -A x64
cmake --build . --config Release
.\bin\city_designer.exe
```

Replace `C:/path/to/vcpkg` with your `vcpkg` clone path.

Option B — Manual / other package managers:
- Install GLFW and an OpenGL loader (GLAD) and ensure include/lib paths are
  available to CMake. The project will try `pkg-config` and `find_package(glfw3)`
  and fallback to `-lglfw` if necessary.

Local GLAD option (Windows/Linux):
- If you prefer not to use vcpkg, add a local GLAD loader to `src/glad/`.
  Place `glad.c`, `glad.h`, and `khrplatform.h` into `src/glad/`. The CMake file
  automatically detects `src/glad/glad.c` and compiles it into the executable.

---

## Run and assets
- The CMake configuration copies files from `assets/` into the build directory
  (top-level `build/`). Run the binary from `build/bin/` so asset paths resolve:

```bash
./build/bin/city_designer    # macOS / Linux
build\\bin\\city_designer.exe # Windows (PowerShell / cmd)
```

---

## Troubleshooting

- "GLFW NOT FOUND": install via Homebrew (macOS) or `vcpkg install glfw3`
  (Windows). Ensure `pkg-config` can find `glfw3` (`PKG_CONFIG_PATH` on mac).
- "Undefined references to GL functions": you need a GL loader (GLAD) on
  non-macOS platforms — either provide `src/glad/` or install via `vcpkg`.
- If building on Intel macOS where Homebrew is under `/usr/local`, and not
  `/opt/homebrew`, you may need to add `/usr/local` to `CMAKE_PREFIX_PATH`.

---

## Development notes
- The project sets `CMAKE_CXX_STANDARD` to 17 and uses `RUNTIME_OUTPUT_DIRECTORY`
  to place the executable into `build/bin/`.
- `CMakeLists.txt` links system frameworks on macOS and `opengl32` on Windows.

If you'd like, I can:
- Add a `vcpkg.json` manifest for easy reproducible installs.
- Add `glad` sources under `src/glad/` so the repo builds out-of-the-box on
  Windows/Linux without external package installs.

---

Good luck — tell me if you want me to add `src/glad/` into the repo or wire up
`vcpkg` manifest automation.
