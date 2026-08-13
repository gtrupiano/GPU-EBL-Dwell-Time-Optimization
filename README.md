# GPU-EBL-Dwell-Time-Optimization

## Final Project Abstract

Electron Beam Lithography is the process of creating nanoscale structures using a focused beam of electrons. Due to its use in semiconductor manufacturing and other cutting edge fields, demand has only increased over time and will continue to do so. A persisting problem with the technology is the scattering of electrons from the beam. At the nanoscale, the electrons do not act like rays, instead they scatter upon contact with the resist and substrate. Any attempt to expose a single point will expose the point and to a lesser degree the area surrounding the point as well. This fact must be taken into account when designing lithographic mask layouts. Previous research has been conducted in parallelizing this operation by altering what parts of the substrate are exposed. This was accomplished by iteratively changing the mask based on whether a location was exposed or not exposed. This paper proposes an incremental improvement by not only altering the locations the beam is applied but also the amount of time the beam dwells at each location. To facilitate this change, locations will be treated at any level of exposure instead of the binary exposed and not exposed with the goal of the entire substrate being perfectly exposed.

<br>

## Setting Up Workspace

The assignments and examples require libwb:

[GitHub Repository for libwb](https://github.com/abduld/libwb)

The source files need to be stored in the main directory of the repository.

```
GPU-EBL-Dwell-Time-Optimization/
├── Sequential_Implementations/
|   ├── MATLAB/
|   ├── Python/
    ├── C++/
├── CUDA_Implementation/
|   ├── build/
|   ├── include/
|   ├── src/
|   CMakeLists.txt
    
```

<br>

### Cloning libwb

From the root of this repository:

```bash
git clone https://github.com/abduld/libwb.git
```

This should create a libwb/ folder inside the repository.
<br><br>

### Building libwb

From the root of this repository, execute the following:

```bash
cd libwb
mkdir build
cd build
cmake ../
make
```