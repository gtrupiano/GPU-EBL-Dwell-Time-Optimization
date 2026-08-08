# Using Windows

## Compiling on Windows

1. Open x64 Native Tools Command Prompt for VS and go to build

```bash
cd build
```

2. In order to run CMake using Windows, execute the following command:

```bash
run cmake -G "Unix Makefiles" -D CMAKE_CXX_COMPILER="C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" -D CMAKE_CUDA_HOST_COMPILER="C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" ..
```

*Or for debug*
```bash
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER="C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" -DCMAKE_CUDA_HOST_COMPILER="C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" ..
```


Then to run:
```bash
run make
```

and

```bash
run from GPU-EBL-Dwell-Time-Optimization or just click the play in VS Code
gets input files from /input_ICs and /input_PSFs
outputs CSV to output_data
```


## Libwb
I had some issues with libwb on windows, I made the following updates/fixes to its source code


1. Change `libwb/wbFile.cpp (101)` with:
    ```
    wbLog(ERROR, "Failed to open ", fileName, " in mode ", mode);
    ```

2. Change `libwb/vendor/json11.cpp (175)` with:
    ```
    template <>
        bool Value<Json::NUL, std::nullptr_t>::equals(const JsonValue * other) const {
        return true;
    }
    
    template <>
        bool Value<Json::NUL, std::nullptr_t>::less(const JsonValue * other) const {
        return false;
    }
    ```

3. Change `libwb/wbArg.cpp (71)` with:
    ```
    static int getInputFileCount(char *arg) {
    int count = 1;
    while (*arg != '\0') {   // don't stop early on '-'
        if (*arg == ',') {
        count++;
        }
        arg++;
    }
    return count;
    }
    ```