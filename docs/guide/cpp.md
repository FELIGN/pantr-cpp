# Using PaNTr from C++

PaNTr's algorithms are a header-only C++20 library; the Python package is a binding over it. A C++
project uses it through CMake's `find_package`, with no Python involved.

## Installing the C++ library

From a source checkout, with CMake 3.28 or later and a C++20 compiler, configure and install:

```bash
cmake -S . -B build -DPANTR_BUILD_TESTS=OFF
cmake --install build --prefix /path/to/prefix
```

The configure step fetches the bundled third-party sources over the network (see
`cmake/PantrDependencies.cmake` for building offline). The install holds pantr's headers, its CMake
package and the third-party headers it bundles (Eigen, Spectra, Kokkos mdspan, Boost.Multiprecision
and Boost.Config). The distributed layer `pantr::mpi` is installed only when CMake finds an MPI at
configure time.

## Building against it

```cmake
cmake_minimum_required(VERSION 3.28)
project(my_app LANGUAGES CXX)

find_package(pantr REQUIRED)                # or: COMPONENTS mpi
add_executable(my_app main.cpp)
target_compile_features(my_app PRIVATE cxx_std_20)
target_link_libraries(my_app PRIVATE pantr::pantr)   # and pantr::mpi for the MPI layer
```

Point CMake at the install with `-DCMAKE_PREFIX_PATH=/path/to/prefix`. Then:

```cpp
#include <pantr/pantr.hpp>

#include <cstdio>

int main() {
  std::printf("pantr %d.%d, up to %zu threads\n", pantr::version.major, pantr::version.minor,
              pantr::parallel::max_threads());
}
```

- **`pantr::pantr`** is every core header (`<pantr/pantr.hpp>` includes them all). No core
  header includes `mpi.h`.
- **`pantr::mpi`** (`find_package(pantr COMPONENTS mpi)`) adds the distributed algorithms over an
  MPI communicator and links the MPI that CMake finds in your project.
- By default `pantr::pantr` compiles for the build machine's instruction set (`-march=native`
  or the compiler's equivalent). Set `PANTR_NATIVE_ARCH` to `OFF` before `find_package` to build
  portable binaries.

## Threads

A C++ program gets a thread pool of the same design as the Python package's (see
{doc}`/guide/parallelism`): `pantr::parallel::max_threads()` is its capacity (the CPUs of the
affinity mask, lowered by `PANTR_NUM_THREADS`), and `pantr::parallel::set_num_threads(n)` sets the calling thread's
parallelism.

## The CMake package in the Python wheel

The `pantr` wheel also carries pantr's headers and its CMake package, whose directory
`pantr.cmake_dir()` returns:

```bash
cmake -S . -B build -Dpantr_DIR="$(python -c 'import pantr; print(pantr.cmake_dir())')"
```

It does not carry the bundled third-party headers, so it provides
`find_package(pantr COMPONENTS mpi-capsule)` only, the interface a separately built pantr
extension module (such as `pantr-mpi`) compiles against, and refuses `pantr::pantr` with a message
saying so. For `pantr::pantr`, install the C++ library with CMake as above.
