# Build Instructions

## Library
In order to build this project's source files, the required library needs to be added and built. If it is not built, follow the instructions [here](../../../docs/library_build.md).


## Source Code
Navigate to the `C++` implementation folder and create a build folder:

```bash
cd Sequential_Implementations/C++
mkdir build
```

Then from inside the build folder, use the project's CMake file to create the needed Makefiles. Execute:

```bash
cd build
cmake ../
```

With the Makefiles created, use them to properly compile the `C++` source code. Execute:

```bash
make
```