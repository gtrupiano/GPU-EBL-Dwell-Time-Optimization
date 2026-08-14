# Library Build Instructions
The CUDA and sequential C++ implementations depend on `libwb`. Follow the steps below to clone and build the library from the root of this repository.


## Cloning libwb
From the root of this repository run the following:

```bash
git clone https://github.com/abduld/libwb.git
```

This should create a `libwb/` folder inside the repository.
<br><br>


## Building libwb
From the root of this repository, execute the following:

```bash
cd libwb
mkdir build
cd build
cmake ../
make
```